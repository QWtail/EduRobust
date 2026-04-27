#!/usr/bin/env python3
"""Compute inter-rater agreement between human annotations and automated ASR."""

import os
import sys
import pandas as pd
import numpy as np
from sklearn.metrics import cohen_kappa_score

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
VALIDATION_PATH = os.path.join(PROJECT_ROOT, "results", "validation_sample.csv")
RUNS_PATH = os.path.join(PROJECT_ROOT, "results", "raw", "runs.csv")
OUTPUT_PATH = os.path.join(PROJECT_ROOT, "results", "analysis", "human_validation.csv")

SUMMARY_PATH = os.path.join(PROJECT_ROOT, "results", "analysis", "agreement_summary.csv")

HUMAN_LABEL_MAP = {"yes": 1.0, "partial": 0.5, "no": 0.0}


def load_validation_sample():
    """Load validation_sample.csv and check that human labels are filled in."""
    if not os.path.exists(VALIDATION_PATH):
        print(f"ERROR: {VALIDATION_PATH} not found.")
        print("Please create the validation sample first.")
        sys.exit(0)

    df = pd.read_csv(VALIDATION_PATH)

    if "human_violated" not in df.columns:
        print("ERROR: 'human_violated' column not found in validation_sample.csv.")
        print("Please add the human_violated column (yes / partial / no).")
        sys.exit(0)

    # Drop rows where human label is missing or empty
    df["human_violated"] = df["human_violated"].astype(str)
    filled = df[~df["human_violated"].isin(["", "nan", "NaN", "None"])].copy()
    filled["human_violated"] = filled["human_violated"].str.strip().str.lower()

    if len(filled) == 0:
        print("No human annotations found in validation_sample.csv.")
        print("Please fill in the 'human_violated' column (yes / partial / no) and re-run.")
        sys.exit(0)

    n_missing = len(df) - len(filled)
    if n_missing > 0:
        print(f"Warning: {n_missing} rows have missing human annotations and will be skipped.")

    return filled


def load_runs():
    """Load automated results from runs.csv (baseline only)."""
    df = pd.read_csv(RUNS_PATH, on_bad_lines="warn")
    return df[df["prompt_variant"] == "baseline"].copy()


def merge_data(val_df, runs_df):
    """Merge validation sample with automated ASR on shared keys."""
    merge_keys = ["behavior_id", "language_code", "model", "attack_template", "model_response"]

    # Ensure merge keys exist in both DataFrames
    for key in merge_keys:
        if key not in val_df.columns:
            print(f"ERROR: '{key}' column missing from validation_sample.csv")
            sys.exit(1)
        if key not in runs_df.columns:
            print(f"ERROR: '{key}' column missing from runs.csv")
            sys.exit(1)

    # Get automated ASR and eval_method from runs
    runs_subset = runs_df[merge_keys + ["asr", "eval_method", "resource_tier"]].copy()

    # Use suffixes to distinguish columns
    merged = val_df.merge(
        runs_subset,
        on=merge_keys,
        how="left",
        suffixes=("", "_auto"),
    )

    # Use automated asr from runs.csv
    if "asr_auto" in merged.columns:
        merged["auto_asr"] = merged["asr_auto"]
    elif "asr" in merged.columns:
        merged["auto_asr"] = merged["asr"]

    # Deduplicate: if model_response matched multiple runs (identical responses),
    # keep only the first match per validation row
    merged = merged.drop_duplicates(subset=merge_keys, keep="first")

    # Use resource_tier from runs if not already present
    if "resource_tier_auto" in merged.columns and "resource_tier" not in val_df.columns:
        merged["resource_tier"] = merged["resource_tier_auto"]
    elif "resource_tier" not in merged.columns and "resource_tier_auto" in merged.columns:
        merged["resource_tier"] = merged["resource_tier_auto"]

    # Use eval_method from runs if not already present
    if "eval_method_auto" in merged.columns and "eval_method" not in val_df.columns:
        merged["eval_method"] = merged["eval_method_auto"]
    elif "eval_method" not in merged.columns and "eval_method_auto" in merged.columns:
        merged["eval_method"] = merged["eval_method_auto"]

    # Drop rows without a matched automated ASR
    n_before = len(merged)
    merged = merged.dropna(subset=["auto_asr"])
    n_dropped = n_before - len(merged)
    if n_dropped > 0:
        print(f"Warning: {n_dropped} rows could not be matched to runs.csv and will be skipped.")

    if len(merged) == 0:
        print("ERROR: No rows could be matched between validation_sample.csv and runs.csv.")
        sys.exit(1)

    return merged


def compute_metrics(merged):
    """Compute agreement metrics."""
    # Map human labels to numeric
    merged["human_score"] = merged["human_violated"].map(HUMAN_LABEL_MAP)
    merged["auto_score"] = merged["auto_asr"].astype(float)

    # Drop any rows with unmapped human labels
    unmapped = merged["human_score"].isna()
    if unmapped.any():
        bad_labels = merged.loc[unmapped, "human_violated"].unique()
        print(f"Warning: unrecognized human labels {bad_labels} will be skipped.")
        merged = merged[~unmapped].copy()

    # Cast to string labels to avoid sklearn continuous/binary type mismatch
    label_map_str = {0.0: "0.0", 0.5: "0.5", 1.0: "1.0"}
    human = merged["human_score"].map(label_map_str).values
    auto = merged["auto_score"].map(label_map_str).values

    # Weighted Cohen's kappa (quadratic weights)
    # Both raters use {0.0, 0.5, 1.0} scale
    labels = ["0.0", "0.5", "1.0"]
    kappa = cohen_kappa_score(human, auto, weights="quadratic", labels=labels)

    # Exact agreement
    exact_agree = np.mean(human == auto)

    return merged, kappa, exact_agree


def breakdown_table(merged, group_col, group_label):
    """Compute per-group agreement breakdown."""
    rows = []
    label_map_str = {0.0: "0.0", 0.5: "0.5", 1.0: "1.0"}
    labels = ["0.0", "0.5", "1.0"]
    for name, grp in merged.groupby(group_col):
        n = len(grp)
        human = grp["human_score"].map(label_map_str).values
        auto = grp["auto_score"].map(label_map_str).values
        exact = np.mean(human == auto)
        # Need at least 2 samples and some variation for kappa
        if n >= 2 and len(set(human)) > 1:
            try:
                k = cohen_kappa_score(human, auto, weights="quadratic", labels=labels)
            except Exception:
                k = float("nan")
        else:
            k = float("nan")
        rows.append({group_label: name, "n": n, "exact_agree": exact, "kappa": k})
    return pd.DataFrame(rows)


def print_summary(merged, kappa, exact_agree):
    """Print formatted summary."""
    n = len(merged)
    print("\n" + "=" * 60)
    print("INTER-RATER AGREEMENT: Human vs Automated ASR")
    print("=" * 60)
    print(f"  Samples evaluated:      {n}")
    print(f"  Weighted Cohen's kappa: {kappa:.3f}")
    print(f"  Exact agreement:        {exact_agree:.1%}")

    # Label distribution
    print("\n  Human label distribution:")
    dist = merged["human_violated"].str.strip().str.lower().value_counts()
    for label, count in dist.items():
        print(f"    {label:>8s}: {count:4d}  ({count/n:.1%})")

    # Breakdown by eval_method
    if "eval_method" in merged.columns and merged["eval_method"].notna().any():
        print("\n  Breakdown by eval_method:")
        tbl = breakdown_table(merged, "eval_method", "eval_method")
        print(f"    {'eval_method':<20s} {'n':>5s} {'agree':>8s} {'kappa':>8s}")
        print(f"    {'-'*20} {'-'*5} {'-'*8} {'-'*8}")
        for _, row in tbl.iterrows():
            k_str = f"{row['kappa']:.3f}" if not np.isnan(row["kappa"]) else "  n/a"
            print(
                f"    {row['eval_method']:<20s} {row['n']:5d} "
                f"{row['exact_agree']:7.1%} {k_str:>8s}"
            )

    # Breakdown by behavior_id
    print("\n  Breakdown by behavior_id:")
    tbl = breakdown_table(merged, "behavior_id", "behavior_id")
    print(f"    {'behavior_id':<20s} {'n':>5s} {'agree':>8s} {'kappa':>8s}")
    print(f"    {'-'*20} {'-'*5} {'-'*8} {'-'*8}")
    for _, row in tbl.iterrows():
        k_str = f"{row['kappa']:.3f}" if not np.isnan(row["kappa"]) else "  n/a"
        print(
            f"    {row['behavior_id']:<20s} {row['n']:5d} "
            f"{row['exact_agree']:7.1%} {k_str:>8s}"
        )

    # Breakdown by resource_tier
    if "resource_tier" in merged.columns and merged["resource_tier"].notna().any():
        print("\n  Breakdown by resource_tier:")
        tbl = breakdown_table(merged, "resource_tier", "resource_tier")
        tier_order = {"high": 0, "medium": 1, "low": 2}
        tbl["_order"] = tbl["resource_tier"].map(tier_order).fillna(9)
        tbl = tbl.sort_values("_order").drop(columns="_order")
        print(f"    {'resource_tier':<20s} {'n':>5s} {'agree':>8s} {'kappa':>8s}")
        print(f"    {'-'*20} {'-'*5} {'-'*8} {'-'*8}")
        for _, row in tbl.iterrows():
            k_str = f"{row['kappa']:.3f}" if not np.isnan(row["kappa"]) else "  n/a"
            print(
                f"    {row['resource_tier']:<20s} {row['n']:5d} "
                f"{row['exact_agree']:7.1%} {k_str:>8s}"
            )

    print("=" * 60)


def save_results(merged, kappa, exact_agree):
    """Save detailed results to CSV."""
    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)

    # Per-row results
    out_cols = [
        "behavior_id", "language_code", "model", "attack_template",
        "human_violated", "human_score", "auto_score", "eval_method",
        "resource_tier",
    ]
    out_cols = [c for c in out_cols if c in merged.columns]
    out = merged[out_cols].copy()
    out["agree"] = (merged["human_score"] == merged["auto_score"]).astype(int)

    out.to_csv(OUTPUT_PATH, index=False)
    print(f"\nResults saved to {OUTPUT_PATH}")


def save_summary(merged, kappa, exact_agree):
    """Save per-group kappa summary to agreement_summary.csv."""
    rows = [{
        "group_type": "overall", "group_name": "overall",
        "n": len(merged),
        "exact_agree": round(exact_agree, 4),
        "kappa": round(kappa, 4),
    }]
    for group_col, group_type in [
        ("behavior_id", "behavior"),
        ("eval_method", "eval_method"),
        ("resource_tier", "resource_tier"),
    ]:
        if group_col not in merged.columns:
            continue
        tbl = breakdown_table(merged, group_col, group_col)
        for _, r in tbl.iterrows():
            rows.append({
                "group_type": group_type,
                "group_name": r[group_col],
                "n": int(r["n"]),
                "exact_agree": round(r["exact_agree"], 4),
                "kappa": round(r["kappa"], 4) if not pd.isna(r["kappa"]) else None,
            })
    out = pd.DataFrame(rows)
    os.makedirs(os.path.dirname(SUMMARY_PATH), exist_ok=True)
    out.to_csv(SUMMARY_PATH, index=False)
    print(f"Agreement summary saved to {SUMMARY_PATH}")


def compute_pairwise(merged, col_a, col_b, name_a, name_b):
    """Compute pairwise agreement between two score columns."""
    label_map_str = {0.0: "0.0", 0.5: "0.5", 1.0: "1.0"}
    labels = ["0.0", "0.5", "1.0"]
    a = merged[col_a].map(label_map_str).values
    b = merged[col_b].map(label_map_str).values
    try:
        kappa = cohen_kappa_score(a, b, weights="quadratic", labels=labels)
    except Exception:
        kappa = float("nan")
    exact = np.mean(a == b)
    return kappa, exact


def breakdown_table_pair(merged, group_col, group_label, col_a, col_b):
    """Compute per-group agreement between two score columns."""
    rows = []
    label_map_str = {0.0: "0.0", 0.5: "0.5", 1.0: "1.0"}
    labels = ["0.0", "0.5", "1.0"]
    for name, grp in merged.groupby(group_col):
        n = len(grp)
        a = grp[col_a].map(label_map_str).values
        b = grp[col_b].map(label_map_str).values
        exact = np.mean(a == b)
        if n >= 2 and len(set(a)) > 1:
            try:
                k = cohen_kappa_score(a, b, weights="quadratic", labels=labels)
            except Exception:
                k = float("nan")
        else:
            k = float("nan")
        rows.append({group_label: name, "n": n, "exact_agree": exact, "kappa": k})
    return pd.DataFrame(rows)


def print_three_way(merged):
    """Print three-way agreement: Human vs Auto, Human vs Claude, Claude vs Auto."""
    has_claude = "claude_score" in merged.columns and merged["claude_score"].notna().any()
    if not has_claude:
        return

    pairs = [
        ("human_score", "auto_score", "Human vs Auto"),
        ("human_score", "claude_score", "Human vs Claude"),
        ("claude_score", "auto_score", "Claude vs Auto"),
    ]

    print("\n" + "=" * 70)
    print("THREE-WAY INTER-RATER AGREEMENT")
    print("=" * 70)

    # Overall
    print(f"\n  {'Pair':<25s} {'κ (weighted)':>14s} {'Exact agree':>14s}")
    print(f"  {'-'*25} {'-'*14} {'-'*14}")
    for col_a, col_b, label in pairs:
        k, e = compute_pairwise(merged, col_a, col_b, "", "")
        k_str = f"{k:.3f}" if not np.isnan(k) else "n/a"
        print(f"  {label:<25s} {k_str:>14s} {e:>13.1%}")

    # Breakdown by behavior
    print(f"\n  Per-behavior breakdown:")
    for col_a, col_b, label in pairs:
        print(f"\n    {label}:")
        tbl = breakdown_table_pair(merged, "behavior_id", "behavior_id", col_a, col_b)
        print(f"    {'behavior_id':<20s} {'n':>5s} {'agree':>8s} {'kappa':>8s}")
        print(f"    {'-'*20} {'-'*5} {'-'*8} {'-'*8}")
        for _, row in tbl.iterrows():
            k_str = f"{row['kappa']:.3f}" if not np.isnan(row["kappa"]) else "  n/a"
            print(
                f"    {row['behavior_id']:<20s} {row['n']:5d} "
                f"{row['exact_agree']:7.1%} {k_str:>8s}"
            )

    print("=" * 70)


def main():
    val_df = load_validation_sample()
    runs_df = load_runs()
    merged = merge_data(val_df, runs_df)
    merged, kappa, exact_agree = compute_metrics(merged)

    # Add Claude labels if present
    if "claude_violated" in merged.columns:
        merged["claude_violated_clean"] = merged["claude_violated"].astype(str).str.strip().str.lower()
        merged["claude_score"] = merged["claude_violated_clean"].map(HUMAN_LABEL_MAP)
        n_claude = merged["claude_score"].notna().sum()
        print(f"Claude labels found: {n_claude}")

    print_summary(merged, kappa, exact_agree)
    print_three_way(merged)
    save_results(merged, kappa, exact_agree)
    save_summary(merged, kappa, exact_agree)


if __name__ == "__main__":
    main()
