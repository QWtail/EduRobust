#!/usr/bin/env python3
"""
rescore_runs.py
Re-score all rows in results/raw/runs.csv with the updated evaluator
(judge_criteria now included in LLM judge prompt; math_only bypass_indicators removed).

Usage:
    python scripts/rescore_runs.py [--resume] [--dry-run] [--batch-size N]

Options:
    --resume        Skip rows already rescored (tracked in rescore_checkpoint.txt)
    --dry-run       Show what would be done without writing anything
    --batch-size N  Save progress every N rows (default: 500)
"""

import argparse
import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

import pandas as pd
import yaml
from pathlib import Path
from tqdm import tqdm

RUNS_PATH = Path(PROJECT_ROOT) / "results" / "raw" / "runs.csv"
CHECKPOINT_PATH = Path(PROJECT_ROOT) / "results" / "raw" / "rescore_checkpoint.txt"
CONFIG_PATH = Path(PROJECT_ROOT) / "config" / "config.yaml"
BEHAVIORS_PATH = Path(PROJECT_ROOT) / "config" / "behaviors.yaml"
DEFENSE_PROMPTS_DIR = Path(PROJECT_ROOT) / "prompts" / "defense_system_prompts"


def load_config():
    with open(CONFIG_PATH) as f:
        return yaml.safe_load(f)


def load_behaviors():
    """Load behaviors.yaml and index by id."""
    with open(BEHAVIORS_PATH) as f:
        data = yaml.safe_load(f)
    return {b["id"]: b for b in data["behaviors"]}


def get_system_prompt(behavior: dict, prompt_variant: str, language_code: str) -> str:
    """Resolve the system prompt for a given prompt_variant, mirroring experiment_runner logic."""
    if prompt_variant == "baseline":
        return behavior["system_prompt"]

    if prompt_variant == "strategy_aware":
        return behavior.get("defense_prompts", {}).get("strategy_aware") or behavior["system_prompt"]

    if prompt_variant == "composite":
        return behavior.get("defense_prompts", {}).get("composite") or behavior["system_prompt"]

    if prompt_variant == "multilingual":
        yaml_path = DEFENSE_PROMPTS_DIR / behavior["id"] / f"{language_code}.yaml"
        if yaml_path.exists():
            with open(yaml_path, encoding="utf-8") as f:
                data = yaml.safe_load(f) or {}
            prompt = data.get("system_prompt", "")
            if prompt:
                return prompt
        return behavior["system_prompt"]

    return behavior["system_prompt"]


def load_checkpoint() -> int:
    """Return last completed row index (0-based), or -1 if no checkpoint."""
    if CHECKPOINT_PATH.exists():
        try:
            return int(CHECKPOINT_PATH.read_text().strip())
        except ValueError:
            pass
    return -1


def save_checkpoint(idx: int):
    CHECKPOINT_PATH.write_text(str(idx))


def build_evaluator(cfg: dict):
    """Build the Evaluator using the judge model from config."""
    from src.ollama_client import OllamaClient
    from src.evaluator import Evaluator

    eval_cfg = cfg.get("evaluation", {})
    judge_model = eval_cfg.get("judge_model", "llama3.2:3b-instruct-q4_0")
    ollama_host = cfg.get("api", {}).get("ollama_host", "http://localhost:11434")
    timeout = cfg.get("api", {}).get("timeout_seconds", 120)

    judge_client = OllamaClient(
        model_id=judge_model,
        host=ollama_host,
        timeout=timeout,
    )

    return Evaluator(
        judge_client=judge_client,
        judge_temperature=float(eval_cfg.get("judge_temperature", 0.0)),
        judge_max_tokens=int(eval_cfg.get("judge_max_tokens", 256)),
        confidence_threshold=float(eval_cfg.get("confidence_threshold", 0.7)),
        essay_min_words=int(eval_cfg.get("essay_min_words", 100)),
    )


def rescore(args):
    print(f"Loading {RUNS_PATH} ...")
    df = pd.read_csv(RUNS_PATH, on_bad_lines="warn")
    total = len(df)
    print(f"Loaded {total} rows.")

    behaviors = load_behaviors()
    cfg = load_config()

    if args.dry_run:
        print("[DRY RUN] Would rescore all eligible rows. No writes will happen.")

    evaluator = None if args.dry_run else build_evaluator(cfg)

    start_idx = 0
    if args.resume:
        last = load_checkpoint()
        if last >= 0:
            start_idx = last + 1
            print(f"Resuming from row {start_idx} (checkpoint: row {last} completed).")

    skipped_status = 0
    rescored = 0
    errors = 0

    pbar = tqdm(range(start_idx, total), total=total - start_idx, unit="row",
                desc="Rescoring", dynamic_ncols=True)
    for i in pbar:
        row = df.iloc[i]

        # Skip non-success rows (no valid response to evaluate)
        if row.get("status") != "success":
            skipped_status += 1
            continue

        model_response = str(row.get("model_response", ""))
        if not model_response or model_response == "nan":
            skipped_status += 1
            continue

        behavior_id = row["behavior_id"]
        if behavior_id not in behaviors:
            print(f"  Warning: unknown behavior_id '{behavior_id}' at row {i}, skipping.")
            errors += 1
            continue

        behavior = behaviors[behavior_id]
        prompt_variant = str(row.get("prompt_variant", "baseline"))
        language_code = str(row.get("language_code", "en"))

        system_prompt = get_system_prompt(behavior, prompt_variant, language_code)
        judge_criteria = behavior["judge_criteria"]
        user_message = str(row.get("translated_prompt", ""))
        refusal_keywords = behavior.get("refusal_keywords") or []
        bypass_indicators = behavior.get("bypass_indicators") or []
        special_eval = behavior.get("special_eval")

        if args.dry_run:
            rescored += 1
            if i < 5:
                print(f"  [DRY RUN] row {i}: behavior={behavior_id}, variant={prompt_variant}, "
                      f"lang={language_code}, special_eval={special_eval}")
            continue

        try:
            result = evaluator.evaluate(
                behavior_id=behavior_id,
                system_prompt=system_prompt,
                judge_criteria=judge_criteria,
                user_message=user_message,
                model_response=model_response,
                refusal_keywords=refusal_keywords,
                bypass_indicators=bypass_indicators,
                special_eval=special_eval,
            )
            df.at[i, "asr"] = result.asr
            df.at[i, "eval_method"] = result.method
            df.at[i, "eval_confidence"] = result.confidence
            df.at[i, "eval_reason"] = result.reason
            rescored += 1
        except Exception as e:
            print(f"  Error at row {i}: {e}")
            errors += 1

        pbar.set_postfix(rescored=rescored, skipped=skipped_status, errors=errors, refresh=False)

        # Save in batches
        if rescored > 0 and rescored % args.batch_size == 0:
            df.to_csv(RUNS_PATH, index=False)
            save_checkpoint(i)

    if not args.dry_run:
        df.to_csv(RUNS_PATH, index=False)
        if CHECKPOINT_PATH.exists():
            CHECKPOINT_PATH.unlink()

    print(f"\nDone. Rescored: {rescored}, skipped (non-success): {skipped_status}, errors: {errors}")
    if args.dry_run:
        print("[DRY RUN] No files were written.")


def main():
    parser = argparse.ArgumentParser(description="Re-score runs.csv with updated evaluator.")
    parser.add_argument("--resume", action="store_true",
                        help="Resume from last checkpoint")
    parser.add_argument("--dry-run", action="store_true",
                        help="Show what would be done without writing")
    parser.add_argument("--batch-size", type=int, default=500,
                        help="Save progress every N rows (default: 500)")
    args = parser.parse_args()
    rescore(args)


if __name__ == "__main__":
    main()
