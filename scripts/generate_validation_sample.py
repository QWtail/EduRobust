#!/usr/bin/env python3
"""Generate a stratified sample of 200 baseline responses for human annotation."""

import pandas as pd
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
INPUT_CSV = ROOT / "results" / "raw" / "runs.csv"
OUTPUT_CSV = ROOT / "results" / "validation_sample.csv"
SAMPLE_SIZE = 200
RANDOM_STATE = 42

EXPORT_COLS = [
    "behavior_id",
    "language_code",
    "language_name",
    "resource_tier",
    "model",
    "eval_method",
    "attack_template",
    "translated_prompt",
    "model_response",
    "human_violated",
    "human_confidence",
    "human_reason",
]


def main():
    # ── Load and filter ──────────────────────────────────────────────
    df = pd.read_csv(INPUT_CSV, on_bad_lines="warn")
    baseline = df[df["prompt_variant"] == "baseline"].copy()
    print(f"Loaded {len(df):,} total rows, {len(baseline):,} baseline rows.")

    if len(baseline) < SAMPLE_SIZE:
        print(f"ERROR: only {len(baseline)} baseline rows, need {SAMPLE_SIZE}.")
        sys.exit(1)

    sampled_idx = set()

    # ── Stratum 1: 20 per behavior (5 x 20 = 100) ───────────────────
    stratum1 = baseline.groupby("behavior_id", group_keys=False).apply(
        lambda g: g.sample(n=min(20, len(g)), random_state=RANDOM_STATE),
        include_groups=False,
    )
    sampled_idx.update(stratum1.index)
    print(f"Stratum 1 (behavior): {len(stratum1)} rows")

    # ── Stratum 2: 15 per resource tier from remaining (3 x 15 = 45)─
    remaining = baseline.loc[~baseline.index.isin(sampled_idx)]
    stratum2 = remaining.groupby("resource_tier", group_keys=False).apply(
        lambda g: g.sample(n=min(15, len(g)), random_state=RANDOM_STATE),
        include_groups=False,
    )
    sampled_idx.update(stratum2.index)
    print(f"Stratum 2 (resource tier): {len(stratum2)} rows")

    # ── Stratum 3: 15 ambiguous cases (asr == 0.5) from remaining ───
    remaining = baseline.loc[~baseline.index.isin(sampled_idx)]
    ambiguous = remaining[remaining["asr"] == 0.5]
    n_ambiguous = min(15, len(ambiguous))
    if n_ambiguous > 0:
        stratum3 = ambiguous.sample(n=n_ambiguous, random_state=RANDOM_STATE)
        sampled_idx.update(stratum3.index)
    else:
        stratum3 = pd.DataFrame()
    print(f"Stratum 3 (ambiguous asr=0.5): {len(stratum3)} rows")

    # ── Stratum 4: fill to 200 with random draws ────────────────────
    n_fill = SAMPLE_SIZE - len(sampled_idx)
    remaining = baseline.loc[~baseline.index.isin(sampled_idx)]
    if n_fill > 0:
        stratum4 = remaining.sample(n=min(n_fill, len(remaining)), random_state=RANDOM_STATE)
        sampled_idx.update(stratum4.index)
    else:
        stratum4 = pd.DataFrame()
    print(f"Stratum 4 (random fill): {len(stratum4)} rows")

    # ── Combine and export ───────────────────────────────────────────
    sample = baseline.loc[list(sampled_idx)].copy()
    sample["human_violated"] = ""
    sample["human_confidence"] = ""
    sample["human_reason"] = ""
    sample = sample[EXPORT_COLS]

    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    sample.to_csv(OUTPUT_CSV, index=False)
    print(f"\nSaved {len(sample)} rows to {OUTPUT_CSV}")

    # ── Summary ──────────────────────────────────────────────────────
    print("\n── Sample Composition ──")
    print(f"\nBy behavior_id:")
    print(sample["behavior_id"].value_counts().sort_index().to_string())
    print(f"\nBy resource_tier:")
    print(sample["resource_tier"].value_counts().sort_index().to_string())
    print(f"\nBy model:")
    print(sample["model"].value_counts().sort_index().to_string())
    print(f"\nBy eval_method:")
    print(sample["eval_method"].value_counts().sort_index().to_string())


if __name__ == "__main__":
    main()
