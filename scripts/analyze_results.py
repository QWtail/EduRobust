#!/usr/bin/env python3
"""
analyze_results.py
Standalone script to generate all analysis outputs from results/raw/runs.csv.

Usage:
  python scripts/analyze_results.py
  python scripts/analyze_results.py --results results/raw/runs.csv --output results/analysis/
  python scripts/analyze_results.py --model llama31_8b
"""

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.analyzer import ResultAnalyzer

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Generate EduRobust analysis outputs")
    parser.add_argument(
        "--results", type=Path,
        default=Path("results/raw/runs.csv"),
        help="Path to results CSV file"
    )
    parser.add_argument(
        "--output", type=Path,
        default=Path("results/analysis"),
        help="Directory for output plots and stats"
    )
    parser.add_argument(
        "--model", type=str, default=None,
        help="Also generate a per-model heatmap for this target model"
    )
    parser.add_argument(
        "--judge", type=str, default=None,
        help=(
            "Filter analysis to rows evaluated by this judge model "
            "(e.g. 'llama3.2:3b-instruct-q4_0'). "
            "Useful when comparing multiple judge configurations."
        )
    )
    args = parser.parse_args()

    if not args.results.exists():
        logger.error(f"Results file not found: {args.results}")
        sys.exit(1)

    analyzer = ResultAnalyzer(
        results_path=args.results,
        output_dir=args.output,
        judge_model=args.judge,
    )

    analyzer.run_all()

    if args.model:
        logger.info(f"Generating additional heatmap for target model: {args.model}")
        analyzer.plot_heatmap(model=args.model)

    print(f"\nAnalysis complete. Outputs saved to: {args.output}")


if __name__ == "__main__":
    main()
