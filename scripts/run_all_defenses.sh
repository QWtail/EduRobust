#!/bin/bash
# run_all_defenses.sh — Run all defense experiments sequentially overnight

set -e  # Stop on first error

cd "$(dirname "$0")/.."

echo "=== Step 1: Generate Defense B translations ==="
python scripts/translate_system_prompts.py

echo "=== Step 2: Defense A — strategy_aware (18,000 runs) ==="
python scripts/run_experiment.py --resume --variant strategy_aware

echo "=== Step 3: Defense B — multilingual (18,000 runs) ==="
python scripts/run_experiment.py --resume --variant multilingual

echo "=== Step 4: Defense C — composite (18,000 runs) ==="
python scripts/run_experiment.py --resume --variant composite

echo "=== Step 5: Analyze all results ==="
python3 scripts/analyze_results.py

echo "=== ALL DONE ==="
