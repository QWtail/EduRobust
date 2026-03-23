"""
experiment_runner.py
Main orchestration loop: model × behavior × language × run_index.
Drives translator → prompt_builder → hf_client → evaluator → result_store.
"""

from __future__ import annotations

import logging
import math
import signal
import sys
from pathlib import Path
from typing import Optional

import yaml
from tqdm import tqdm
from tqdm.contrib.logging import logging_redirect_tqdm

from .config_loader import EduRobustConfig, BehaviorConfig, LanguageConfig, ModelConfig
from .evaluator import Evaluator, EvalResult
from .hf_client import RobustHFClient, HFRateLimitError, HFModelUnavailableError, HFClientError
from .prompt_builder import PromptBuilder
from .result_store import ResultStore, RunRecord
from .translator import TranslationCache

logger = logging.getLogger(__name__)


def _build_client(model_id: str, provider: str, config: EduRobustConfig, model_cfg=None):
    """
    Factory: returns the right client based on provider.

    provider options:
      "ollama"             — model served by local Ollama daemon (ollama pull <id> first)
      "huggingface_local"  — model downloaded and run directly via transformers (no rate limits)
      "huggingface"        — model called via HF Inference API (subject to rate limits)
    """
    if provider == "ollama":
        from .ollama_client import OllamaClient
        return OllamaClient(
            model_id=model_id,
            host=config.master.api.ollama_host,
            timeout=config.master.api.timeout_seconds,
        )
    elif provider == "huggingface_local":
        from .hf_local_client import HFLocalClient
        kwargs = {}
        if model_cfg is not None:
            kwargs["torch_dtype"] = model_cfg.torch_dtype
            kwargs["load_in_8bit"] = model_cfg.load_in_8bit
            kwargs["load_in_4bit"] = model_cfg.load_in_4bit
        return HFLocalClient(
            model_id=model_id,
            hf_token=config.hf_token or None,
            **kwargs,
        )
    else:
        # "huggingface" — remote Inference API
        api_cfg = config.master.api
        return RobustHFClient(
            token=config.hf_token,
            model_id=model_id,
            timeout=api_cfg.timeout_seconds,
            max_retries=api_cfg.max_retries,
            base_backoff=api_cfg.base_backoff_seconds,
            model_loading_wait=api_cfg.model_loading_wait_seconds,
        )


class _ModelAbort(Exception):
    """Raised to break out of all inner loops for a single model."""


class ExperimentRunner:
    """
    Runs the full experiment grid: model × behavior × language × run_index.

    Usage:
        runner = ExperimentRunner(config)
        runner.run_all(resume=True)
    """

    def __init__(
        self,
        config: EduRobustConfig,
        provider_override: Optional[str] = None,
        prompt_variant: str = "baseline",
    ):
        self._cfg = config
        self._provider_override = provider_override
        self._prompt_variant = prompt_variant
        self._prompt_builder = PromptBuilder()

        # Load attack templates
        self._attack_templates = self._load_attack_templates()

        # Translation cache
        self._translator = TranslationCache(
            translations_dir=config.translations_dir,
            fallback=True,
        )

        # Result store
        results_csv = config.results_dir / "runs.csv"
        self._store = ResultStore(results_csv)

        # Clients for each target model (provider-aware; CLI --provider overrides yaml)
        self._clients: dict[str, object] = {
            m.id: _build_client(
                m.id,
                provider_override or m.provider,
                config,
                model_cfg=m,
            )
            for m in config.enabled_models
        }

        # Judge client — provider_override also applies to the judge
        eval_cfg = config.master.evaluation
        judge_client = _build_client(
            eval_cfg.judge_model,
            provider_override or eval_cfg.judge_provider,
            config,
        )

        self._evaluator = Evaluator(
            judge_client=judge_client,
            judge_temperature=eval_cfg.judge_temperature,
            judge_max_tokens=eval_cfg.judge_max_tokens,
            confidence_threshold=eval_cfg.confidence_threshold,
            essay_min_words=eval_cfg.essay_min_words,
        )

        # Register SIGINT handler for graceful shutdown
        signal.signal(signal.SIGINT, self._handle_interrupt)
        self._interrupted = False

    # -----------------------------------------------------------------------
    # Public API
    # -----------------------------------------------------------------------

    def run_all(
        self,
        resume: bool = True,
        models_filter: Optional[list[str]] = None,
        behaviors_filter: Optional[list[str]] = None,
        languages_filter: Optional[list[str]] = None,
        dry_run: bool = False,
    ) -> None:
        """
        Execute the full experiment grid.

        Args:
            resume:           Skip already-completed cells from runs.csv
            models_filter:    Limit to these model names (e.g. ["llama31_8b"])
            behaviors_filter: Limit to these behavior IDs
            languages_filter: Limit to these language codes
            dry_run:          Print the run plan without making API calls
        """
        runs_per_cell = self._cfg.master.experiment.runs_per_cell

        # Filter dimensions
        models = [m for m in self._cfg.enabled_models
                  if models_filter is None or m.name in models_filter]
        behaviors = [b for b in self._cfg.behaviors
                     if behaviors_filter is None or b.id in behaviors_filter]
        # Composite variant: skip behaviors that have no composite defense prompt
        # (e.g. english_only — adding "respond in English" would be redundant)
        if self._prompt_variant == "composite":
            behaviors = [b for b in behaviors if b.defense_prompts.get("composite")]
        languages = [l for l in self._cfg.languages
                     if languages_filter is None or l.code in languages_filter]

        total_cells = len(models) * len(behaviors) * len(languages) * runs_per_cell
        logger.info(
            f"Experiment plan: {len(models)} models × {len(behaviors)} behaviors × "
            f"{len(languages)} languages × {runs_per_cell} runs = {total_cells} cells"
        )

        if dry_run:
            print(f"\n[DRY RUN] Would run {total_cells} cells (variant={self._prompt_variant}):")
            for m in models:
                for b in behaviors:
                    for l in languages:
                        print(f"  {m.name} | {b.id} | {self._prompt_variant} | {l.code} | {runs_per_cell} runs")
            return

        # Pre-translate all attack prompts before starting the experiment
        self._pretranslate_all(behaviors, languages)

        # Deduplicate CSV before loading completed keys — removes re-run rows
        # that accumulated when resume failed to detect completed cells.
        if resume:
            removed = self._store.dedup()
            if removed:
                print(f"[startup] Removed {removed} duplicate rows from runs.csv before resuming.")

        completed = self._store.get_completed_keys() if resume else set()
        logger.info(f"Resuming from {len(completed)} completed cells.")

        max_consecutive_errors = self._cfg.master.api.max_consecutive_errors
        pbar = tqdm(total=total_cells, desc="EduRobust", unit="run")
        pbar.update(len(completed))

        try:
            with logging_redirect_tqdm():
                for model_cfg in models:
                    if self._interrupted:
                        break
                    client = self._clients[model_cfg.id]
                    consecutive_errors = 0

                    try:
                        for behavior_cfg in behaviors:
                            if self._interrupted:
                                break
                            templates = self._attack_templates.get(behavior_cfg.id, [])
                            if not templates:
                                logger.warning(f"No attack templates for behavior '{behavior_cfg.id}'")
                                continue

                            for lang_cfg in languages:
                                if self._interrupted:
                                    break

                                for run_idx in range(runs_per_cell):
                                    if self._interrupted:
                                        break

                                    run_key = (model_cfg.name, behavior_cfg.id,
                                               self._prompt_variant,
                                               lang_cfg.code, run_idx)
                                    if run_key in completed:
                                        continue

                                    status = self._execute_single_run(
                                        model_cfg, behavior_cfg, lang_cfg,
                                        templates, run_idx, client
                                    )
                                    pbar.update(1)

                                    if status == "success":
                                        consecutive_errors = 0
                                    else:
                                        consecutive_errors += 1
                                        if consecutive_errors >= max_consecutive_errors:
                                            logger.error(
                                                f"[{model_cfg.name}] {consecutive_errors} consecutive "
                                                f"errors. Skipping remaining runs for this model."
                                            )
                                            raise _ModelAbort()

                    except _ModelAbort:
                        continue

        finally:
            pbar.close()
            if self._interrupted:
                logger.info("Experiment interrupted by user. Progress saved to CSV.")

    # -----------------------------------------------------------------------
    # Pre-translation
    # -----------------------------------------------------------------------

    def _pretranslate_all(
        self,
        behaviors: list,
        languages: list,
    ) -> None:
        """
        Translate all attack templates for the given behaviors and languages
        before the experiment starts. Already-cached translations are skipped instantly.
        Shows a tqdm progress bar.
        """
        # Build the full list of (behavior_id, lang_code, deep_translator_code, template) work items
        work = []
        for behavior in behaviors:
            templates = self._attack_templates.get(behavior.id, [])
            for lang in languages:
                if lang.code == "en":
                    continue  # English is identity — no translation needed
                for template in templates:
                    work.append((behavior.id, lang.code, lang.deep_translator_code, template))

        if not work:
            return

        print(f"\nPre-translating {len(work)} prompts across "
              f"{len(behaviors)} behaviors × {len(languages)} languages ...")

        skipped = 0
        with logging_redirect_tqdm():
            with tqdm(work, desc="Translating", unit="prompt") as pbar:
                for behavior_id, lang_code, deep_code, template in pbar:
                    pbar.set_postfix(lang=lang_code, behavior=behavior_id)
                    cache_key = (template, deep_code)
                    if cache_key in self._translator._cache:
                        skipped += 1
                    else:
                        self._translator.get(template, deep_code, behavior_id)

        cached_count = len(work) - skipped
        print(f"Translation complete: {cached_count} translated, {skipped} already cached.\n")
        logger.info(
            f"Pre-translation done: {cached_count} new, {skipped} from cache, {len(work)} total."
        )

    # -----------------------------------------------------------------------
    # Single run
    # -----------------------------------------------------------------------

    def _execute_single_run(
        self,
        model_cfg: ModelConfig,
        behavior_cfg: BehaviorConfig,
        lang_cfg: LanguageConfig,
        templates: list[str],
        run_idx: int,
        client: RobustHFClient,
    ) -> None:
        """Execute one (model, behavior, language, run_idx) cell and log results."""

        # Select attack template (round-robin)
        template_idx = run_idx % len(templates)
        template_en = templates[template_idx]

        # Translate template (english_only behavior: prompts are in target lang already)
        if behavior_cfg.id == "english_only":
            # The prompt IS meant to be in the foreign language
            translated = self._translator.get(
                template_en, lang_cfg.deep_translator_code, behavior_cfg.id
            )
        else:
            translated = self._translator.get(
                template_en, lang_cfg.deep_translator_code, behavior_cfg.id
            )

        # Fill {problem} placeholder
        user_msg = self._prompt_builder.build(
            translated_template=translated,
            behavior_id=behavior_cfg.id,
            run_index=run_idx,
        )

        # Resolve system prompt based on variant
        system_prompt = self._resolve_system_prompt(behavior_cfg, lang_cfg)

        # Call target model
        response = ""
        status = "success"
        try:
            response = client.chat(
                system_prompt=system_prompt,
                user_message=user_msg,
                temperature=self._cfg.master.experiment.temperature,
                max_new_tokens=model_cfg.max_new_tokens,
            )
        except HFRateLimitError as e:
            status = "rate_limit_error"
            logger.warning(f"Rate limit error: {e}")
        except HFModelUnavailableError as e:
            status = "model_unavailable"
            logger.warning(f"Model unavailable: {e}")
        except HFClientError as e:
            status = f"api_error"
            logger.warning(f"API error: {e}")
        except Exception as e:
            status = f"unexpected_error"
            logger.error(f"Unexpected error: {e}", exc_info=True)

        # If interrupted after model call, skip the judge call (saves ~5-30s)
        if self._interrupted and status == "success":
            self._store.append(RunRecord(
                timestamp=RunRecord.now(),
                model=model_cfg.name,
                judge_model=self._cfg.master.evaluation.judge_model,
                behavior_id=behavior_cfg.id,
                prompt_variant=self._prompt_variant,
                language_code=lang_cfg.code,
                language_name=lang_cfg.name,
                resource_tier=lang_cfg.resource_tier,
                run_index=run_idx,
                template_index=template_idx,
                attack_template=template_en,
                translated_prompt=user_msg,
                model_response=response,
                asr=float("nan"),
                eval_method="interrupted",
                eval_confidence=0.0,
                eval_reason="Run interrupted before evaluation",
                status="interrupted",
            ))
            return "interrupted"

        # Evaluate
        if status == "success" and response:
            eval_result = self._evaluator.evaluate(
                behavior_id=behavior_cfg.id,
                system_prompt=behavior_cfg.system_prompt,
                user_message=user_msg,
                model_response=response,
                refusal_keywords=behavior_cfg.refusal_keywords,
                bypass_indicators=behavior_cfg.bypass_indicators,
                special_eval=behavior_cfg.special_eval,
            )
        else:
            eval_result = EvalResult(
                asr=float("nan"),
                method="skipped",
                confidence=0.0,
                reason=status,
            )

        # Log result
        self._store.append(RunRecord(
            timestamp=RunRecord.now(),
            model=model_cfg.name,
            judge_model=self._cfg.master.evaluation.judge_model,
            behavior_id=behavior_cfg.id,
            prompt_variant=self._prompt_variant,
            language_code=lang_cfg.code,
            language_name=lang_cfg.name,
            resource_tier=lang_cfg.resource_tier,
            run_index=run_idx,
            template_index=template_idx,
            attack_template=template_en,
            translated_prompt=user_msg,
            model_response=response,
            asr=eval_result.asr,
            eval_method=eval_result.method,
            eval_confidence=eval_result.confidence,
            eval_reason=eval_result.reason,
            status=status,
        ))

        logger.debug(
            f"[{model_cfg.name}][{behavior_cfg.id}][{lang_cfg.code}][run={run_idx}] "
            f"ASR={eval_result.asr:.1f} via {eval_result.method} | status={status}"
        )
        return status

    # -----------------------------------------------------------------------
    # Helper methods
    # -----------------------------------------------------------------------

    def _resolve_system_prompt(
        self,
        behavior_cfg: BehaviorConfig,
        lang_cfg: LanguageConfig,
    ) -> str:
        """
        Return the system prompt to use based on the active prompt_variant.

        Variants:
          "baseline"        → original system prompt from behavior config
          "strategy_aware"  → hardened prompt from behavior_cfg.defense_prompts
          "multilingual"    → translated prompt from prompts/defense_system_prompts/
        """
        variant = self._prompt_variant

        if variant == "baseline":
            return behavior_cfg.system_prompt

        if variant == "strategy_aware":
            prompt = behavior_cfg.defense_prompts.get("strategy_aware")
            if not prompt:
                logger.warning(
                    f"No strategy_aware defense prompt for '{behavior_cfg.id}'. "
                    f"Falling back to baseline."
                )
                return behavior_cfg.system_prompt
            return prompt

        if variant == "composite":
            prompt = behavior_cfg.defense_prompts.get("composite")
            if not prompt:
                # english_only has no composite (would be redundant);
                # fall back to baseline
                logger.info(
                    f"No composite defense prompt for '{behavior_cfg.id}'. "
                    f"Falling back to baseline."
                )
                return behavior_cfg.system_prompt
            return prompt

        if variant == "multilingual":
            # Load from prompts/defense_system_prompts/{behavior_id}/{lang_code}.yaml
            yaml_path = (
                self._cfg.project_root
                / "prompts"
                / "defense_system_prompts"
                / behavior_cfg.id
                / f"{lang_cfg.code}.yaml"
            )
            if yaml_path.exists():
                with open(yaml_path, "r", encoding="utf-8") as f:
                    data = yaml.safe_load(f) or {}
                prompt = data.get("system_prompt", "")
                if prompt:
                    return prompt
            logger.warning(
                f"No multilingual defense prompt for '{behavior_cfg.id}' "
                f"in '{lang_cfg.code}'. Falling back to baseline."
            )
            return behavior_cfg.system_prompt

        logger.warning(f"Unknown prompt variant '{variant}'. Using baseline.")
        return behavior_cfg.system_prompt

    def _load_attack_templates(self) -> dict[str, list[str]]:
        """Load attack_templates.yaml and return {behavior_id: [templates]}."""
        path = self._cfg.attack_templates_path
        with open(path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
        return data.get("attack_prompts", {})

    def _handle_interrupt(self, signum, frame) -> None:
        """
        Ctrl+C handler with two-stage shutdown:
          1st Ctrl+C — set flag, finish current HTTP call, then stop.
          2nd Ctrl+C — force-exit immediately (exit code 130).
        """
        if not self._interrupted:
            self._interrupted = True
            logger.info("Interrupt received. Finishing current model call, then stopping.")
            print(
                "\n[EduRobust] Stopping after current model call finishes. "
                "Press Ctrl+C again to force quit immediately."
            )
        else:
            print("\n[EduRobust] Force quit.")
            sys.exit(130)  # 130 = 128 + SIGINT(2), standard Ctrl+C exit code
