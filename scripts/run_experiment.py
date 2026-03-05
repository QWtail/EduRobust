#!/usr/bin/env python3
"""
run_experiment.py
Main CLI entry point for the EduRobust experiment.

Usage:
  # Full experiment (resume from checkpoint if runs.csv exists)
  python scripts/run_experiment.py --resume

  # Dry run — print plan without API calls
  python scripts/run_experiment.py --dry-run

  # Limit scope for testing
  python scripts/run_experiment.py --models llama31_8b --behaviors no_homework --languages en fr zh

  # Fresh start (ignore existing results)
  python scripts/run_experiment.py
"""

import argparse
import logging
import sys
from pathlib import Path

# Allow running from project root
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config_loader import load_config
from src.experiment_runner import ExperimentRunner


def setup_logging(log_dir: Path, level: str = "INFO") -> None:
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / "experiment.log"

    # Log to file only — keeps stdout clean for the tqdm progress bar
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        handlers=[
            logging.FileHandler(log_file, encoding="utf-8"),
        ],
    )

    # Suppress noisy per-request HTTP logs from httpx/ollama
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)
    logging.getLogger("ollama").setLevel(logging.WARNING)


def main():
    parser = argparse.ArgumentParser(
        description="EduRobust: Multilingual LLM system prompt robustness experiment"
    )
    parser.add_argument(
        "--resume", action="store_true",
        help="Resume from existing results/raw/runs.csv (skip completed cells)"
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Print the experiment plan without making API calls"
    )
    parser.add_argument(
        "--models", nargs="+", metavar="MODEL_NAME",
        help="Limit to specific model names (e.g. llama31_8b qwen25_7b)"
    )
    parser.add_argument(
        "--behaviors", nargs="+", metavar="BEHAVIOR_ID",
        help="Limit to specific behavior IDs (e.g. no_homework math_only)"
    )
    parser.add_argument(
        "--languages", nargs="+", metavar="LANG_CODE",
        help="Limit to specific language codes (e.g. en fr zh)"
    )
    parser.add_argument(
        "--config-dir", type=Path, default=None,
        help="Path to config/ directory (default: auto-detect)"
    )
    args = parser.parse_args()

    # Load config
    cfg = load_config(args.config_dir)
    setup_logging(cfg.log_dir, cfg.master.output.log_level)
    logger = logging.getLogger(__name__)

    banner = "\n".join([
        "=" * 60,
        "EduRobust Experiment Starting",
        f"  Models:    {[m.name for m in cfg.enabled_models]}",
        f"  Behaviors: {[b.id for b in cfg.behaviors]}",
        f"  Languages: {[l.code for l in cfg.languages]}",
        f"  Runs/cell: {cfg.master.experiment.runs_per_cell}",
        f"  Resume:    {args.resume}",
        f"  Dry run:   {args.dry_run}",
        "=" * 60,
    ])
    print(banner)
    logger.info(banner)

    runner = ExperimentRunner(cfg)
    runner.run_all(
        resume=args.resume,
        models_filter=args.models,
        behaviors_filter=args.behaviors,
        languages_filter=args.languages,
        dry_run=args.dry_run,
    )

    print("Experiment complete.")
    logger.info("Experiment complete.")


if __name__ == "__main__":
    main()
