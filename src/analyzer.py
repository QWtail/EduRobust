"""
analyzer.py
Loads results/raw/runs.csv and produces:
  - Heatmaps (language × behavior)
  - Bar charts (resource tier, model comparison, language ranking)
  - Summary statistics CSV
  - Kruskal-Wallis + Mann-Whitney U tests across resource tiers

Filtering:
  - Pass judge_model= to __init__ to restrict analysis to runs evaluated
    by a specific judge model (useful when comparing judge models).
  - Pass model= to individual plot methods to restrict to a target model.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Style
# ---------------------------------------------------------------------------

sns.set_theme(style="whitegrid", font_scale=1.1)
plt.rcParams["figure.dpi"] = 150
plt.rcParams["savefig.bbox"] = "tight"


class ResultAnalyzer:
    """
    Loads the master CSV and generates all analysis outputs.
    """

    def __init__(
        self,
        results_path: Path,
        output_dir: Path,
        judge_model: Optional[str] = None,
    ):
        self._results_path = results_path
        self._output_dir = output_dir
        self._judge_model = judge_model
        self._heatmaps_dir = output_dir / "heatmaps"
        self._bar_dir = output_dir / "bar_charts"

        self._heatmaps_dir.mkdir(parents=True, exist_ok=True)
        self._bar_dir.mkdir(parents=True, exist_ok=True)

        self._df = self._load()

    def _load(self) -> pd.DataFrame:
        # on_bad_lines="warn" skips rows whose field count doesn't match the
        # header and prints a warning rather than raising a ParserError.
        # ResultStore._migrate_if_needed() should unify format before analysis,
        # but this is a safety net for direct CSV reads.
        df = pd.read_csv(self._results_path, on_bad_lines="warn")
        # Keep only successful runs for analysis
        self._df_all = df
        df = df[df["status"] == "success"].copy()

        # Dedup safety-net: if the same (model, behavior_id, language_code, run_index)
        # appears more than once (e.g. from a failed --resume that re-ran completed cells),
        # keep only the latest row per cell so ASR aggregations are not inflated.
        key_cols = ["model", "behavior_id", "language_code", "run_index"]
        if df.duplicated(subset=key_cols).any():
            before = len(df)
            df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
            df = (
                df.sort_values("timestamp", ascending=False)
                  .drop_duplicates(subset=key_cols, keep="first")
            )
            logger.warning(
                f"Duplicate cell keys detected — kept latest row per cell "
                f"({before} → {len(df)} rows). Consider running: "
                f"python src/result_store.py results/raw/runs.csv to dedup the CSV."
            )

        # Optional: filter to a specific judge model
        if self._judge_model and "judge_model" in df.columns:
            before = len(df)
            df = df[df["judge_model"] == self._judge_model]
            logger.info(
                f"Filtered to judge_model='{self._judge_model}': "
                f"{len(df)}/{before} rows retained."
            )
        elif "judge_model" not in df.columns:
            logger.warning(
                "Column 'judge_model' not found in CSV "
                "(old results file). judge_model filtering unavailable."
            )

        return df

    # -----------------------------------------------------------------------
    # Core aggregation
    # -----------------------------------------------------------------------

    def asr_matrix(
        self, model: Optional[str] = None
    ) -> pd.DataFrame:
        """
        Pivot table: rows=language_name, cols=behavior_id, values=mean ASR.
        """
        df = self._df if model is None else self._df[self._df["model"] == model]
        pivot = (
            df.groupby(["language_name", "behavior_id"])["asr"]
            .mean()
            .unstack(fill_value=0.0)
        )
        return pivot

    # -----------------------------------------------------------------------
    # Heatmaps
    # -----------------------------------------------------------------------

    def plot_heatmap(self, model: Optional[str] = None) -> None:
        """Heatmap: language × behavior, colored by mean ASR."""
        pivot = self.asr_matrix(model)
        if pivot.empty:
            logger.warning("No data to plot heatmap.")
            return

        # Sort languages by overall mean ASR (descending)
        pivot["_mean"] = pivot.mean(axis=1)
        pivot = pivot.sort_values("_mean", ascending=False).drop(columns="_mean")

        fig, ax = plt.subplots(figsize=(max(8, len(pivot.columns) * 2), max(8, len(pivot) * 0.5)))
        sns.heatmap(
            pivot,
            annot=True,
            fmt=".2f",
            cmap="Reds",
            vmin=0,
            vmax=1,
            linewidths=0.4,
            cbar_kws={"label": "Mean ASR"},
            ax=ax,
        )
        title = "Attack Success Rate: Language × Behavior"
        if model:
            title += f"\nModel: {model}"
        ax.set_title(title, fontsize=13, fontweight="bold", pad=12)
        ax.set_xlabel("Forbidden Behavior", fontsize=11)
        ax.set_ylabel("Language", fontsize=11)
        plt.xticks(rotation=30, ha="right")
        plt.yticks(rotation=0)

        suffix = f"_{model}" if model else "_all"
        fname = self._heatmaps_dir / f"asr_lang_behavior{suffix}.png"
        fig.savefig(fname, dpi=150)
        plt.close(fig)
        logger.info(f"Saved heatmap: {fname}")

    def plot_all_heatmaps(self) -> None:
        """Plot one heatmap for all models combined, plus one per model."""
        self.plot_heatmap(model=None)
        for model in self._df["model"].unique():
            self.plot_heatmap(model=model)

    # -----------------------------------------------------------------------
    # Bar charts
    # -----------------------------------------------------------------------

    def plot_asr_by_resource_tier(self) -> None:
        """Grouped bar: ASR per resource tier per behavior."""
        agg = (
            self._df.groupby(["resource_tier", "behavior_id"])["asr"]
            .mean()
            .reset_index()
        )
        if agg.empty:
            return

        fig, ax = plt.subplots(figsize=(10, 6))
        sns.barplot(
            data=agg,
            x="behavior_id",
            y="asr",
            hue="resource_tier",
            hue_order=["high", "medium", "low"],
            palette={"high": "#2196F3", "medium": "#FF9800", "low": "#F44336"},
            ax=ax,
        )
        ax.set_ylim(0, 1.05)
        ax.set_title("Mean ASR by Resource Tier and Behavior", fontsize=13, fontweight="bold")
        ax.set_xlabel("Forbidden Behavior")
        ax.set_ylabel("Mean Attack Success Rate")
        ax.legend(title="Resource Tier")
        plt.xticks(rotation=20, ha="right")

        fname = self._bar_dir / "asr_by_tier.png"
        fig.savefig(fname, dpi=150)
        plt.close(fig)
        logger.info(f"Saved: {fname}")

    def plot_model_comparison(self) -> None:
        """Grouped bar: ASR per model per behavior."""
        agg = (
            self._df.groupby(["model", "behavior_id"])["asr"]
            .mean()
            .reset_index()
        )
        if agg.empty:
            return

        fig, ax = plt.subplots(figsize=(11, 6))
        sns.barplot(data=agg, x="behavior_id", y="asr", hue="model",
                    palette="Set2", ax=ax)
        ax.set_ylim(0, 1.05)
        ax.set_title("Mean ASR by Model and Behavior", fontsize=13, fontweight="bold")
        ax.set_xlabel("Forbidden Behavior")
        ax.set_ylabel("Mean Attack Success Rate")
        plt.xticks(rotation=20, ha="right")

        fname = self._bar_dir / "model_comparison.png"
        fig.savefig(fname, dpi=150)
        plt.close(fig)
        logger.info(f"Saved: {fname}")

    def plot_language_ranked(self) -> None:
        """Horizontal bar: languages ranked by overall mean ASR (all behaviors)."""
        agg = (
            self._df.groupby(["language_name", "resource_tier"])["asr"]
            .mean()
            .reset_index()
            .sort_values("asr", ascending=True)
        )
        if agg.empty:
            return

        tier_colors = {"high": "#2196F3", "medium": "#FF9800", "low": "#F44336"}
        colors = [tier_colors.get(t, "gray") for t in agg["resource_tier"]]

        fig, ax = plt.subplots(figsize=(9, max(6, len(agg) * 0.4)))
        ax.barh(agg["language_name"], agg["asr"], color=colors)
        ax.set_xlim(0, 1.05)
        ax.set_xlabel("Mean Attack Success Rate")
        ax.set_title("Languages Ranked by Overall ASR", fontsize=13, fontweight="bold")

        # Legend for resource tiers
        from matplotlib.patches import Patch
        legend_elements = [Patch(facecolor=c, label=t)
                           for t, c in tier_colors.items()]
        ax.legend(handles=legend_elements, title="Resource Tier", loc="lower right")

        fname = self._bar_dir / "language_ranked.png"
        fig.savefig(fname, dpi=150)
        plt.close(fig)
        logger.info(f"Saved: {fname}")

    def plot_eval_method_usage(self) -> None:
        """Pie chart: proportion of runs evaluated by each method."""
        counts = self._df["eval_method"].value_counts()
        if counts.empty:
            return

        fig, ax = plt.subplots(figsize=(7, 7))
        ax.pie(counts.values, labels=counts.index, autopct="%1.1f%%",
               startangle=140, colors=sns.color_palette("Set3", len(counts)))
        ax.set_title("Evaluation Method Distribution", fontsize=13, fontweight="bold")

        fname = self._bar_dir / "eval_method_usage.png"
        fig.savefig(fname, dpi=150)
        plt.close(fig)
        logger.info(f"Saved: {fname}")

    # -----------------------------------------------------------------------
    # Statistical tests
    # -----------------------------------------------------------------------

    def statistical_tests(self) -> pd.DataFrame:
        """
        Kruskal-Wallis test across resource tiers (per behavior).
        If significant, pairwise Mann-Whitney U tests.
        Returns a DataFrame of results.
        """
        rows = []
        for behavior_id in self._df["behavior_id"].unique():
            sub = self._df[self._df["behavior_id"] == behavior_id]
            groups = {
                tier: sub[sub["resource_tier"] == tier]["asr"].dropna().values
                for tier in ["high", "medium", "low"]
            }
            valid_groups = [g for g in groups.values() if len(g) > 1]

            if len(valid_groups) < 2:
                continue

            stat, p = stats.kruskal(*valid_groups)
            rows.append({
                "behavior_id": behavior_id,
                "test": "kruskal_wallis",
                "groups": "high_vs_medium_vs_low",
                "statistic": stat,
                "p_value": p,
                "significant": p < 0.05,
            })

            # Pairwise Mann-Whitney U
            tier_pairs = [("high", "medium"), ("high", "low"), ("medium", "low")]
            for t1, t2 in tier_pairs:
                g1, g2 = groups.get(t1, np.array([])), groups.get(t2, np.array([]))
                if len(g1) > 1 and len(g2) > 1:
                    u_stat, u_p = stats.mannwhitneyu(g1, g2, alternative="two-sided")
                    rows.append({
                        "behavior_id": behavior_id,
                        "test": "mann_whitney_u",
                        "groups": f"{t1}_vs_{t2}",
                        "statistic": u_stat,
                        "p_value": u_p,
                        "significant": u_p < 0.05,
                    })

        return pd.DataFrame(rows)

    # -----------------------------------------------------------------------
    # Summary CSV
    # -----------------------------------------------------------------------

    def export_summary_csv(self) -> Path:
        """Export aggregated per-cell statistics to summary_stats.csv.

        Groups by (model, judge_model, behavior_id, language_code, language_name,
        resource_tier) so each row records both the target model and the judge
        model used. Falls back gracefully if judge_model column is absent
        (old CSV files).
        """
        group_cols = ["model", "behavior_id", "language_code", "language_name", "resource_tier"]
        if "judge_model" in self._df.columns:
            group_cols.insert(1, "judge_model")  # model, judge_model, behavior_id, ...

        summary = (
            self._df.groupby(group_cols)
            .agg(
                mean_asr=("asr", "mean"),
                std_asr=("asr", "std"),
                n_runs=("asr", "count"),
                n_bypassed=("asr", lambda x: (x == 1.0).sum()),
                n_partial=("asr", lambda x: (x == 0.5).sum()),
            )
            .reset_index()
        )
        summary["bypass_rate"] = summary["n_bypassed"] / summary["n_runs"]

        out_path = self._output_dir / "summary_stats.csv"
        summary.to_csv(out_path, index=False)
        logger.info(f"Saved summary: {out_path}")
        return out_path

    # -----------------------------------------------------------------------
    # Load summary
    # -----------------------------------------------------------------------

    def print_load_summary(self) -> None:
        """Print a quick sanity-check table of the loaded data to stdout."""
        df = self._df
        df_all = self._df_all

        n_total_raw   = len(df_all)
        n_success     = len(df)
        n_other       = n_total_raw - n_success

        models     = sorted(df["model"].unique())
        behaviors  = sorted(df["behavior_id"].unique())
        languages  = sorted(df["language_code"].unique())

        n_models    = len(models)
        n_behaviors = len(behaviors)
        n_langs     = len(languages)

        # Runs per cell — should all equal the same number (e.g. 20)
        key_cols   = ["model", "behavior_id", "language_code", "run_index"]
        runs_per_cell = df.groupby(["model", "behavior_id", "language_code"]).size()
        min_runs   = int(runs_per_cell.min()) if not runs_per_cell.empty else 0
        max_runs   = int(runs_per_cell.max()) if not runs_per_cell.empty else 0
        expected   = n_models * n_behaviors * n_langs * max_runs

        # ASR distribution
        asr_counts = df["asr"].value_counts().sort_index()

        # Eval method distribution
        eval_counts = df["eval_method"].value_counts()

        sep = "─" * 56
        print(f"\n{sep}")
        print(f"  EduRobust — Data Load Summary")
        print(sep)
        print(f"  CSV file   : {self._results_path}")
        print(f"  Total rows : {n_total_raw:,}  "
              f"(success={n_success:,}, other={n_other:,})")
        if n_other > 0:
            status_counts = df_all[df_all["status"] != "success"]["status"].value_counts()
            for s, c in status_counts.items():
                print(f"               └─ {s}: {c:,}")
        print(sep)
        print(f"  Models     : {n_models}  → {', '.join(models)}")
        print(f"  Behaviors  : {n_behaviors}  → {', '.join(behaviors)}")
        print(f"  Languages  : {n_langs}")
        print(f"  Runs/cell  : min={min_runs}  max={max_runs}")
        print(f"  Expected   : {n_models} × {n_behaviors} × {n_langs} × {max_runs}"
              f" = {expected:,}")

        # Flag if actual != expected
        if n_success != expected:
            gap = expected - n_success
            flag = "⚠  MISMATCH" if gap != 0 else "✓ OK"
            print(f"  Actual     : {n_success:,}   {flag}"
                  + (f"  ({abs(gap):,} {'missing' if gap > 0 else 'extra'})" if gap != 0 else ""))
        else:
            print(f"  Actual     : {n_success:,}   ✓ OK")

        print(sep)
        print(f"  ASR distribution:")
        for asr_val, cnt in asr_counts.items():
            pct = 100 * cnt / n_success if n_success else 0
            label = {0.0: "held (0.0)", 0.5: "ambiguous (0.5)", 1.0: "bypassed (1.0)"}.get(
                asr_val, str(asr_val)
            )
            print(f"    {label:<20} {cnt:>6,}  ({pct:5.1f}%)")

        print(f"  Eval method breakdown:")
        for method, cnt in eval_counts.items():
            pct = 100 * cnt / n_success if n_success else 0
            print(f"    {method:<24} {cnt:>6,}  ({pct:5.1f}%)")

        print(f"  Rows per model:")
        for m, cnt in df["model"].value_counts().sort_index().items():
            expected_m = n_behaviors * n_langs * max_runs
            flag = "✓" if cnt == expected_m else "⚠"
            print(f"    {flag} {m:<24} {cnt:>6,}  (expected {expected_m:,})")

        print(sep + "\n")

    # -----------------------------------------------------------------------
    # Run all outputs
    # -----------------------------------------------------------------------

    def run_all(self) -> None:
        """Generate all plots, stats, and summary CSV."""
        self.print_load_summary()
        logger.info("Generating all analysis outputs...")
        self.plot_all_heatmaps()
        self.plot_asr_by_resource_tier()
        self.plot_model_comparison()
        self.plot_language_ranked()
        self.plot_eval_method_usage()
        self.export_summary_csv()

        tests = self.statistical_tests()
        if not tests.empty:
            tests_path = self._output_dir / "statistical_tests.csv"
            tests.to_csv(tests_path, index=False)
            logger.info(f"Saved statistical tests: {tests_path}")

        logger.info("All analysis outputs generated.")
