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


def _build_client(model_id: str, provider: str, config: EduRobustConfig):
    """Factory: returns the right client based on provider."""
    if provider == "ollama":
        from .ollama_client import OllamaClient
        return OllamaClient(
            model_id=model_id,
            host=config.master.api.ollama_host,
            timeout=config.master.api.timeout_seconds,
        )
    else:
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

    def __init__(self, config: EduRobustConfig):
        self._cfg = config
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

        # Clients for each target model (provider-aware)
        self._clients: dict[str, object] = {
            m.id: _build_client(m.id, m.provider, config)
            for m in config.enabled_models
        }

        # Judge client (shared across all evaluations)
        eval_cfg = config.master.evaluation
        judge_client = _build_client(
            eval_cfg.judge_model, eval_cfg.judge_provider, config
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
        languages = [l for l in self._cfg.languages
                     if languages_filter is None or l.code in languages_filter]

        total_cells = len(models) * len(behaviors) * len(languages) * runs_per_cell
        logger.info(
            f"Experiment plan: {len(models)} models × {len(behaviors)} behaviors × "
            f"{len(languages)} languages × {runs_per_cell} runs = {total_cells} cells"
        )

        if dry_run:
            print(f"\n[DRY RUN] Would run {total_cells} cells:")
            for m in models:
                for b in behaviors:
                    for l in languages:
                        print(f"  {m.name} | {b.id} | {l.code} | {runs_per_cell} runs")
            return

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
        template_en = templates[run_idx % len(templates)]

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

        # Call target model
        response = ""
        status = "success"
        try:
            response = client.chat(
                system_prompt=behavior_cfg.system_prompt,
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
            behavior_id=behavior_cfg.id,
            language_code=lang_cfg.code,
            language_name=lang_cfg.name,
            resource_tier=lang_cfg.resource_tier,
            run_index=run_idx,
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

    def _load_attack_templates(self) -> dict[str, list[str]]:
        """Load attack_templates.yaml and return {behavior_id: [templates]}."""
        path = self._cfg.attack_templates_path
        with open(path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
        return data.get("attack_prompts", {})

    def _handle_interrupt(self, signum, frame) -> None:
        """Graceful shutdown on Ctrl+C."""
        logger.info("Interrupt signal received. Will stop after current run...")
        self._interrupted = True
