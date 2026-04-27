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

sns.set_theme(style="whitegrid", font_scale=1.4)
plt.rcParams["figure.dpi"] = 150
plt.rcParams["savefig.bbox"] = "tight"

# Canonical B1–B5 behavior ordering (matches paper tables)
BEHAVIOR_ORDER = ["english_only", "no_essay", "no_homework", "hints_only", "math_only"]
BEHAVIOR_LABELS = {
    "english_only": "B1 english_only",
    "no_essay": "B2 no_essay",
    "no_homework": "B3 no_homework",
    "hints_only": "B4 hints_only",
    "math_only": "B5 math_only",
}

# All defense variant labels (including composite)
VARIANT_LABELS = {
    "baseline": "Baseline",
    "strategy_aware": "Defense A (Strategy-Aware)",
    "multilingual": "Defense B (Multilingual)",
    "composite": "Defense C (Composite)",
}


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

        # Backfill prompt_variant for old CSVs
        if "prompt_variant" not in df.columns:
            df["prompt_variant"] = "baseline"
        else:
            df["prompt_variant"] = df["prompt_variant"].fillna("baseline")

        # Dedup safety-net: if the same (model, behavior_id, prompt_variant,
        # language_code, run_index) appears more than once, keep only the latest
        # row per cell so ASR aggregations are not inflated.
        key_cols = ["model", "behavior_id", "prompt_variant", "language_code", "run_index"]
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

        # Sort languages by overall mean ASR (descending), then transpose
        pivot["_mean"] = pivot.mean(axis=1)
        pivot = pivot.sort_values("_mean", ascending=False).drop(columns="_mean")
        pivot = pivot.T  # rows=behaviors (5), cols=languages (sorted by mean ASR desc)

        # Reorder rows to B1–B5 and apply labels
        ordered_behaviors = [b for b in BEHAVIOR_ORDER if b in pivot.index]
        pivot = pivot.reindex(ordered_behaviors)
        pivot.index = [BEHAVIOR_LABELS.get(b, b) for b in pivot.index]

        # fig, ax = plt.subplots(figsize=(max(14, len(pivot.columns) * 0.7), max(3, len(pivot) * 1.0)))
        fig, ax = plt.subplots(figsize=(max(14, len(pivot.columns) * 0.7), 2.5))
        sns.heatmap(
            pivot,
            annot=True,
            fmt=".2f",
            cmap="Reds",
            vmin=0,
            vmax=1,
            linewidths=0.4,
            annot_kws={"size": 12},
            cbar_kws={"label": "Mean ASR"},
            ax=ax,
        )
        title = "Attack Success Rate: Behavior × Language"
        if model:
            title += f"\nModel: {model}"
        ax.set_title(title, fontsize=20, fontweight="bold", pad=12)
        ax.set_xlabel("Language", fontsize=16)
        ax.set_ylabel("Forbidden Behavior", fontsize=16)
        plt.xticks(rotation=45, ha="right", fontsize=12)
        plt.yticks(rotation=0, fontsize=12)

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

        agg["behavior_label"] = agg["behavior_id"].map(BEHAVIOR_LABELS)
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.barplot(
            data=agg,
            x="behavior_label",
            y="asr",
            hue="resource_tier",
            hue_order=["high", "medium", "low"],
            order=[BEHAVIOR_LABELS[b] for b in BEHAVIOR_ORDER if b in agg["behavior_id"].values],
            palette={"high": "#2196F3", "medium": "#FF9800", "low": "#F44336"},
            ax=ax,
        )
        ax.set_ylim(0, agg["asr"].max() * 1.15)
        ax.set_title("Mean ASR by Resource Tier and Behavior", fontsize=16, fontweight="bold")
        ax.set_xlabel("Forbidden Behavior", fontsize=14)
        ax.set_ylabel("Mean Attack Success Rate", fontsize=14)
        ax.legend(title="Resource Tier", fontsize=12, title_fontsize=13, loc="upper left")
        plt.xticks(rotation=20, ha="right", fontsize=12)
        plt.yticks(fontsize=12)

        fname = self._bar_dir / "asr_by_tier.png"
        fig.savefig(fname, dpi=150)
        plt.close(fig)
        logger.info(f"Saved: {fname}")

    def plot_model_comparison(self) -> None:
        """Grouped bar: ASR per model per behavior, plus overall mean (baseline only)."""
        df_base = self._df[self._df["prompt_variant"] == "baseline"]
        agg = (
            df_base.groupby(["model", "behavior_id"])["asr"]
            .mean()
            .reset_index()
        )
        if agg.empty:
            return

        # Add overall mean per model as a virtual behavior
        mean_agg = df_base.groupby("model")["asr"].mean().reset_index()
        mean_agg["behavior_id"] = "_mean"
        agg = pd.concat([agg, mean_agg], ignore_index=True)

        labels = {**BEHAVIOR_LABELS, "_mean": "Mean"}
        agg["behavior_label"] = agg["behavior_id"].map(labels)
        order = [BEHAVIOR_LABELS[b] for b in BEHAVIOR_ORDER if b in agg["behavior_id"].values] + ["Mean"]

        fig, ax = plt.subplots(figsize=(11, 6))
        sns.barplot(data=agg, x="behavior_label", y="asr", hue="model",
                    order=order,
                    palette="Set2", ax=ax)
        ax.set_ylim(0, agg["asr"].max() * 1.15)
        ax.set_title("Mean ASR by Model and Behavior", fontsize=16, fontweight="bold")
        ax.set_xlabel("Forbidden Behavior", fontsize=14)
        ax.set_ylabel("Mean Attack Success Rate", fontsize=14)
        ax.legend(fontsize=12, title_fontsize=13, loc="upper left")
        plt.xticks(rotation=20, ha="right", fontsize=12)
        plt.yticks(fontsize=12)

        fname = self._bar_dir / "model_comparison.png"
        fig.savefig(fname, dpi=600, bbox_inches="tight")
        plt.close(fig)
        logger.info(f"Saved: {fname}")

    def plot_behavior_boxplot(self) -> None:
        """Box plot: cell-level ASR distribution per forbidden behavior (baseline only)."""
        from matplotlib.patches import Patch

        df_base = self._df[self._df["prompt_variant"] == "baseline"]
        if df_base.empty:
            logger.warning("No baseline data for behavior box plot.")
            return

        # Cell-level ASR: mean per (behavior, model, language)
        cell_asr = (
            df_base.groupby(["behavior_id", "model", "language_code"])["asr"]
            .mean()
            .reset_index()
        )

        # Order by descending mean ASR (computed from data)
        behavior_ids = ["math_only", "no_essay", "no_homework", "english_only", "hints_only"]
        means_sort = {b: cell_asr[cell_asr["behavior_id"] == b]["asr"].mean() for b in behavior_ids}
        behavior_order_desc = sorted(behavior_ids, key=lambda b: means_sort[b], reverse=True)
        bid_to_label = {
            "math_only":    "B5\nmath_only",
            "no_essay":     "B2\nno_essay",
            "no_homework":  "B3\nno_homework",
            "english_only": "B1\nenglish_only",
            "hints_only":   "B4\nhints_only",
        }
        behavior_labels_desc = [bid_to_label[b] for b in behavior_order_desc]
        colors_map = {
            "math_only": "#e67300",       # dark orange — domain restriction
            "no_essay": "#1f77b4",        # blue — role restriction
            "no_homework": "#1f77b4",     # blue — role restriction
            "english_only": "#9467bd",    # purple — language restriction
            "hints_only": "#1f77b4",      # blue — role restriction
        }

        data = [
            cell_asr[cell_asr["behavior_id"] == b]["asr"].values
            for b in behavior_order_desc
        ]
        data = [d for d in data if len(d) > 0]
        if not data:
            return

        fig, ax = plt.subplots(figsize=(11, 6))
        bp = ax.boxplot(
            data, tick_labels=behavior_labels_desc, patch_artist=True, widths=0.5,
            medianprops=dict(color="black", linewidth=2),
            flierprops=dict(marker="o", markersize=5, alpha=0.5),
        )
        for patch, b in zip(bp["boxes"], behavior_order_desc):
            patch.set_facecolor(colors_map[b])
            patch.set_alpha(0.85)

        # Overlay mean markers
        means = [d.mean() for d in data]
        ax.scatter(range(1, len(means) + 1), means, marker="D", color="white",
                   edgecolors="black", s=60, zorder=5, label="Mean")

        ax.set_ylabel("Cell-level ASR", fontsize=16)
        ax.set_xlabel("Forbidden Behavior", fontsize=16)
        ax.set_title("Cell-level ASR Distribution per Forbidden Behavior",
                      fontsize=17, fontweight="bold")
        ax.tick_params(axis="both", labelsize=14)
        max_val = max(d.max() for d in data)
        ax.set_ylim(-0.02, max_val * 1.15)

        from matplotlib.lines import Line2D
        legend_elements = [
            Patch(facecolor="#e67300", edgecolor="black", label="Domain restriction"),
            Patch(facecolor="#1f77b4", edgecolor="black", label="Role restriction"),
            Patch(facecolor="#9467bd", edgecolor="black", label="Language restriction"),
            Line2D([0], [0], marker="D", color="w", markerfacecolor="white",
                   markeredgecolor="black", markersize=9, label="Mean"),
        ]
        ax.legend(handles=legend_elements, fontsize=14, loc="upper right")

        fname = self._bar_dir / "behavior_asr_boxplot.png"
        fig.savefig(fname, dpi=600)
        plt.close(fig)
        logger.info(f"Saved: {fname}")

    def plot_language_ranked(self) -> None:
        """Vertical bar: languages ranked by overall mean ASR (all behaviors)."""
        agg = (
            self._df.groupby(["language_name", "resource_tier"])["asr"]
            .mean()
            .reset_index()
            .sort_values("asr", ascending=False)
        )
        if agg.empty:
            return

        tier_colors = {"high": "#2196F3", "medium": "#FF9800", "low": "#F44336"}
        colors = [tier_colors.get(t, "gray") for t in agg["resource_tier"]]

        fig, ax = plt.subplots(figsize=(max(10, len(agg) * 0.5), 4))
        ax.bar(agg["language_name"], agg["asr"], color=colors)
        ax.set_ylim(0, agg["asr"].max() * 1.15)
        ax.set_ylabel("Mean Attack Success Rate", fontsize=14)
        ax.set_title("Languages Ranked by Overall ASR", fontsize=16, fontweight="bold")
        plt.xticks(rotation=45, ha="right", fontsize=11)
        plt.yticks(fontsize=12)

        # Legend for resource tiers
        from matplotlib.patches import Patch
        legend_elements = [Patch(facecolor=c, label=t)
                           for t, c in tier_colors.items()]
        ax.legend(handles=legend_elements, title="Resource Tier", loc="upper right",
                  fontsize=12, title_fontsize=13, bbox_to_anchor=(1.0, 1.0))

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
               startangle=140, colors=sns.color_palette("Set3", len(counts)),
               textprops={"fontsize": 13})
        ax.set_title("Evaluation Method Distribution", fontsize=16, fontweight="bold")

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

    def model_statistical_tests(self) -> pd.DataFrame:
        """
        Kruskal-Wallis test across models (per behavior).
        If significant, pairwise Mann-Whitney U tests.
        Returns a DataFrame of results and saves model_statistical_tests.csv.
        """
        models = sorted(self._df["model"].unique())
        rows = []
        for behavior_id in self._df["behavior_id"].unique():
            sub = self._df[self._df["behavior_id"] == behavior_id]
            groups = {
                m: sub[sub["model"] == m]["asr"].dropna().values
                for m in models
            }
            valid_groups = [g for g in groups.values() if len(g) > 1]

            if len(valid_groups) < 2:
                continue

            stat, p = stats.kruskal(*valid_groups)
            rows.append({
                "behavior_id": behavior_id,
                "test": "kruskal_wallis",
                "groups": "_vs_".join(models),
                "statistic": stat,
                "p_value": p,
                "significant": p < 0.05,
            })

            # Pairwise Mann-Whitney U
            for i, m1 in enumerate(models):
                for m2 in models[i + 1:]:
                    g1, g2 = groups.get(m1, np.array([])), groups.get(m2, np.array([]))
                    if len(g1) > 1 and len(g2) > 1:
                        u_stat, u_p = stats.mannwhitneyu(g1, g2, alternative="two-sided")
                        rows.append({
                            "behavior_id": behavior_id,
                            "test": "mann_whitney_u",
                            "groups": f"{m1}_vs_{m2}",
                            "statistic": u_stat,
                            "p_value": u_p,
                            "significant": u_p < 0.05,
                        })

        result = pd.DataFrame(rows)
        if not result.empty:
            out_path = self._output_dir / "model_statistical_tests.csv"
            result.to_csv(out_path, index=False)
            logger.info(f"Saved model statistical tests: {out_path}")
        return result

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
        group_cols = ["model", "behavior_id", "prompt_variant", "language_code", "language_name", "resource_tier"]
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
        key_cols   = ["model", "behavior_id", "prompt_variant", "language_code", "run_index"]
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
    # Per-template analysis
    # -----------------------------------------------------------------------

    @staticmethod
    def _yaml_template_order() -> dict[str, dict[str, int]]:
        """Load canonical T0–T4 order from prompts/attack_templates.yaml."""
        import yaml as _yaml
        yaml_path = Path(__file__).parent.parent / "prompts" / "attack_templates.yaml"
        with open(yaml_path) as _f:
            data = _yaml.safe_load(_f)["attack_prompts"]
        return {beh: {tmpl: idx for idx, tmpl in enumerate(tmpls)}
                for beh, tmpls in data.items()}

    def template_asr_analysis(self) -> pd.DataFrame:
        """
        Compute per-template bypass rate for every (behavior, language) cell.

        Groups by (behavior_id, language_code, language_name, attack_template)
        — averaged over models and repeated runs of that template.
        T-indices follow the canonical YAML order (T0=Direct … T4=Override).

        Returns the aggregated DataFrame and saves template_asr.csv.
        """
        df = self._df.copy()

        # Use YAML-canonical order so T-indices match tab:templates in the paper
        template_order = self._yaml_template_order()

        def _make_label(row: pd.Series) -> str:
            idx = template_order.get(row["behavior_id"], {}).get(row["attack_template"], 0)
            return f"T{idx}"

        df["template_label"] = df.apply(_make_label, axis=1)
        df["template_idx"] = df.apply(
            lambda r: template_order.get(r["behavior_id"], {}).get(r["attack_template"], 0),
            axis=1,
        )

        group_cols = ["behavior_id", "language_code", "language_name",
                      "resource_tier", "template_idx", "template_label",
                      "attack_template"]

        agg = (
            df.groupby(group_cols)
            .agg(
                mean_asr=("asr", "mean"),
                std_asr=("asr", "std"),
                n_runs=("asr", "count"),
                bypass_count=("asr", lambda x: (x == 1.0).sum()),
                bypass_rate=("asr", lambda x: (x == 1.0).mean()),
            )
            .reset_index()
            .sort_values(["behavior_id", "language_code", "template_idx"])
        )

        out_path = self._output_dir / "template_asr.csv"
        agg.to_csv(out_path, index=False)
        logger.info(f"Saved template ASR: {out_path}")
        return agg

    def plot_template_heatmaps(self) -> None:
        """
        One heatmap per behavior: rows=template (T0–T4), cols=languages,
        values=mean ASR (averaged over models and run repeats).
        Saved as template_heatmap_<behavior>.png in heatmaps/.
        """
        df = self._df.copy()

        # Use YAML-canonical order so T-indices match tab:templates in the paper
        template_order = self._yaml_template_order()
        for behavior_id in df["behavior_id"].unique():
            order = template_order.get(behavior_id, {})
            mask = df["behavior_id"] == behavior_id
            df.loc[mask, "template_label"] = df.loc[mask, "attack_template"].map(
                lambda t, o=order: f"T{o.get(t, 0)}"
            )
            df.loc[mask, "template_idx"] = df.loc[mask, "attack_template"].map(
                lambda t, o=order: o.get(t, 0)
            )

        for behavior_id in sorted(df["behavior_id"].unique()):
            sub = df[df["behavior_id"] == behavior_id]

            pivot = (
                sub.groupby(["language_name", "template_label"])["asr"]
                .mean()
                .unstack(fill_value=0.0)
            )
            if pivot.empty:
                continue

            # Sort columns by template index (T0, T1, …)
            col_order = (
                sub[["template_label", "template_idx"]]
                .drop_duplicates()
                .sort_values("template_idx")["template_label"]
                .tolist()
            )
            pivot = pivot.reindex(columns=[c for c in col_order if c in pivot.columns])

            # Sort rows by mean ASR descending, then transpose
            pivot["_mean"] = pivot.mean(axis=1)
            pivot = pivot.sort_values("_mean", ascending=False).drop(columns="_mean")

            n_langs = len(pivot)
            n_templates = len(pivot.columns)
            pivot = pivot.T  # rows=templates (5), cols=languages (20, sorted by mean ASR desc)

            fig_w = max(14, n_langs * 0.7)
            # fig_h = max(4, n_templates * 1.0)
            fig_h = 2.5
            fig, ax = plt.subplots(figsize=(fig_w, fig_h))
            sns.heatmap(
                pivot,
                annot=True,
                fmt=".2f",
                cmap="Reds",
                vmin=0,
                vmax=1,
                linewidths=0.4,
                annot_kws={"size": 12},
                cbar_kws={"label": "Mean ASR"},
                ax=ax,
            )
            b_label = BEHAVIOR_LABELS.get(behavior_id, behavior_id)
            ax.set_title(
                f"Per-Template ASR — {b_label}\n"
                f"(rows = attack templates, cols = languages; "
                f"averaged over models & run repeats)",
                fontsize=15,
                fontweight="bold",
                pad=12,
            )
            ax.set_xlabel("Language", fontsize=14)
            ax.set_ylabel("Attack Template", fontsize=14)
            plt.xticks(rotation=45, ha="right", fontsize=11)
            plt.yticks(rotation=0, fontsize=11)

            fname = self._heatmaps_dir / f"template_heatmap_{behavior_id}.png"
            fig.savefig(fname, dpi=150)
            plt.close(fig)
            logger.info(f"Saved template heatmap: {fname}")

    # -----------------------------------------------------------------------
    # Phase 2: Defense analysis
    # -----------------------------------------------------------------------

    def _has_defense_data(self) -> bool:
        """Check if any non-baseline prompt_variant data exists."""
        return (self._df["prompt_variant"] != "baseline").any()

    def plot_defense_comparison(self) -> None:
        """
        Grouped bar chart: baseline vs Defense A vs Defense B vs Defense C per behavior.
        Shows mean ASR for each variant.
        """
        if not self._has_defense_data():
            logger.info("No defense data found — skipping defense comparison plot.")
            return

        agg = (
            self._df.groupby(["prompt_variant", "behavior_id"])["asr"]
            .mean()
            .reset_index()
        )
        if agg.empty:
            return

        agg["variant_label"] = agg["prompt_variant"].map(
            lambda v: VARIANT_LABELS.get(v, v)
        )
        agg["behavior_label"] = agg["behavior_id"].map(BEHAVIOR_LABELS)
        hue_order = [v for v in VARIANT_LABELS.values() if v in agg["variant_label"].values]

        fig, ax = plt.subplots(figsize=(13, 6))
        sns.barplot(
            data=agg,
            x="behavior_label",
            y="asr",
            hue="variant_label",
            hue_order=hue_order,
            order=[BEHAVIOR_LABELS[b] for b in BEHAVIOR_ORDER if b in agg["behavior_id"].values],
            palette="Set2",
            ax=ax,
        )
        ax.set_ylim(0, agg["asr"].max() * 1.15)
        ax.set_title(
            "Defense Comparison: Mean ASR by Behavior and Prompt Variant",
            fontsize=16, fontweight="bold",
        )
        ax.set_xlabel("Forbidden Behavior", fontsize=14)
        ax.set_ylabel("Mean Attack Success Rate", fontsize=14)
        ax.legend(title="Prompt Variant", fontsize=11, title_fontsize=12, loc="upper left")
        plt.xticks(rotation=20, ha="right", fontsize=12)
        plt.yticks(fontsize=12)

        fname = self._bar_dir / "defense_comparison.png"
        fig.savefig(fname, dpi=150)
        plt.close(fig)
        logger.info(f"Saved: {fname}")

    def plot_defense_gap_reduction(self) -> None:
        """
        Bar chart: cross-language ASR range (max − min across languages)
        under each prompt variant per behavior.
        Shows whether defenses reduce the cross-language vulnerability gap.
        """
        if not self._has_defense_data():
            logger.info("No defense data found — skipping gap reduction plot.")
            return

        rows = []
        for variant in self._df["prompt_variant"].unique():
            sub = self._df[self._df["prompt_variant"] == variant]
            for behavior_id in sub["behavior_id"].unique():
                bsub = sub[sub["behavior_id"] == behavior_id]
                lang_asr = bsub.groupby("language_code")["asr"].mean()
                if len(lang_asr) < 2:
                    continue
                rows.append({
                    "prompt_variant": variant,
                    "behavior_id": behavior_id,
                    "asr_range": lang_asr.max() - lang_asr.min(),
                    "asr_max": lang_asr.max(),
                    "asr_min": lang_asr.min(),
                })

        if not rows:
            return

        gap_df = pd.DataFrame(rows)

        gap_df["variant_label"] = gap_df["prompt_variant"].map(
            lambda v: VARIANT_LABELS.get(v, v)
        )
        gap_df["behavior_label"] = gap_df["behavior_id"].map(BEHAVIOR_LABELS)
        hue_order = [v for v in VARIANT_LABELS.values() if v in gap_df["variant_label"].values]

        fig, ax = plt.subplots(figsize=(13, 6))
        sns.barplot(
            data=gap_df,
            x="behavior_label",
            y="asr_range",
            hue="variant_label",
            hue_order=hue_order,
            order=[BEHAVIOR_LABELS[b] for b in BEHAVIOR_ORDER if b in gap_df["behavior_id"].values],
            palette="Set1",
            ax=ax,
        )
        ax.set_ylim(0, gap_df["asr_range"].max() * 1.15)
        ax.set_title(
            "Cross-Language ASR Gap (max − min) by Variant",
            fontsize=16, fontweight="bold",
        )
        ax.set_xlabel("Forbidden Behavior", fontsize=14)
        ax.set_ylabel("ASR Range (max − min across languages)", fontsize=14)
        ax.legend(title="Prompt Variant", fontsize=11, title_fontsize=12, loc="upper left")
        plt.xticks(rotation=20, ha="right", fontsize=12)
        plt.yticks(fontsize=12)

        fname = self._bar_dir / "defense_gap_reduction.png"
        fig.savefig(fname, dpi=150)
        plt.close(fig)
        logger.info(f"Saved: {fname}")

    def plot_defense_tier_gradient(self) -> None:
        """
        Grouped bar chart: mean ASR per resource tier under each defense variant.
        Shows whether the inverted resource-tier gradient persists under defenses.
        """
        if not self._has_defense_data():
            logger.info("No defense data found — skipping tier gradient plot.")
            return

        rows = []
        tier_order = ["high", "medium", "low"]
        for variant in self._df["prompt_variant"].unique():
            sub = self._df[self._df["prompt_variant"] == variant]
            for tier in tier_order:
                tsub = sub[sub["resource_tier"] == tier]
                if tsub.empty:
                    continue
                rows.append({
                    "prompt_variant": variant,
                    "resource_tier": tier,
                    "mean_asr": tsub["asr"].mean(),
                })

        if not rows:
            return

        tier_df = pd.DataFrame(rows)
        tier_df["variant_label"] = tier_df["prompt_variant"].map(
            lambda v: VARIANT_LABELS.get(v, v)
        )

        hue_order = [v for v in VARIANT_LABELS.values()
                     if v in tier_df["variant_label"].values]
        tier_label_order = ["high", "medium", "low"]

        fig, ax = plt.subplots(figsize=(10, 6.5))
        sns.barplot(
            data=tier_df,
            x="resource_tier",
            y="mean_asr",
            hue="variant_label",
            hue_order=hue_order,
            order=tier_label_order,
            palette="Set2",
            ax=ax,
        )
        ax.set_ylim(0, tier_df["mean_asr"].max() * 1.15)
        ax.set_title(
            "Resource-Tier ASR Gradient Under Each Defense Variant",
            fontsize=16, fontweight="bold",
        )
        ax.set_xlabel("Resource Tier", fontsize=14)
        ax.set_ylabel("Mean ASR", fontsize=14)
        ax.legend(title="Prompt Variant", fontsize=11, title_fontsize=12, loc="upper right")
        plt.xticks(fontsize=12)
        plt.yticks(fontsize=12)

        # Add value labels on bars
        for container in ax.containers:
            ax.bar_label(container, fmt="%.3f", fontsize=9, padding=2)

        fname = self._bar_dir / "defense_tier_gradient.png"
        fig.savefig(fname, dpi=600)
        plt.close(fig)
        logger.info(f"Saved: {fname}")

    def defense_statistical_tests(self) -> pd.DataFrame:
        """
        Paired Wilcoxon signed-rank tests: baseline vs each defense variant
        per behavior. Tests whether the defense significantly reduces ASR.
        Returns a DataFrame of results and saves defense_statistical_tests.csv.
        """
        if not self._has_defense_data():
            logger.info("No defense data found — skipping defense statistical tests.")
            return pd.DataFrame()

        rows = []
        defense_variants = [
            v for v in self._df["prompt_variant"].unique()
            if v != "baseline"
        ]

        for variant in defense_variants:
            for behavior_id in self._df["behavior_id"].unique():
                # Get per-cell (model, language, run_index) ASR for baseline
                baseline = self._df[
                    (self._df["prompt_variant"] == "baseline")
                    & (self._df["behavior_id"] == behavior_id)
                ]
                defense = self._df[
                    (self._df["prompt_variant"] == variant)
                    & (self._df["behavior_id"] == behavior_id)
                ]

                if baseline.empty or defense.empty:
                    continue

                # Aggregate per (model, language_code) cell
                base_agg = (
                    baseline.groupby(["model", "language_code"])["asr"]
                    .mean()
                    .reset_index()
                    .rename(columns={"asr": "asr_baseline"})
                )
                def_agg = (
                    defense.groupby(["model", "language_code"])["asr"]
                    .mean()
                    .reset_index()
                    .rename(columns={"asr": "asr_defense"})
                )

                merged = base_agg.merge(
                    def_agg, on=["model", "language_code"], how="inner"
                )
                if len(merged) < 5:
                    continue

                try:
                    stat, p = stats.wilcoxon(
                        merged["asr_baseline"],
                        merged["asr_defense"],
                        alternative="two-sided",
                    )
                    rows.append({
                        "behavior_id": behavior_id,
                        "comparison": f"baseline_vs_{variant}",
                        "test": "wilcoxon_signed_rank",
                        "n_pairs": len(merged),
                        "baseline_mean": merged["asr_baseline"].mean(),
                        "defense_mean": merged["asr_defense"].mean(),
                        "reduction": merged["asr_baseline"].mean() - merged["asr_defense"].mean(),
                        "statistic": stat,
                        "p_value": p,
                        "significant": p < 0.05,
                    })
                except Exception as e:
                    logger.warning(
                        f"Wilcoxon test failed for {behavior_id} "
                        f"baseline vs {variant}: {e}"
                    )

        result = pd.DataFrame(rows)
        if not result.empty:
            out_path = self._output_dir / "defense_statistical_tests.csv"
            result.to_csv(out_path, index=False)
            logger.info(f"Saved defense statistical tests: {out_path}")
        return result

    # -----------------------------------------------------------------------
    # Run all outputs
    # -----------------------------------------------------------------------

    # T-index → strategy name (canonical for redesigned templates)
    _STRATEGY_NAMES: dict[int, str] = {
        0: "Direct",
        1: "Urgency",
        2: "Social",
        3: "Persona",
        4: "Override",
    }

    def template_strategy_table(self) -> pd.DataFrame:
        """
        Pivot table: rows = attack strategy (Direct/Urgency/Social/Persona/Override),
        cols = behavior + Mean column.  Values = mean ASR across all languages and models.

        Requires T-indices to follow the canonical YAML order (T0=Direct … T4=Override),
        which is enforced by _yaml_template_order() used in template_asr_analysis().

        Saves results/analysis/template_strategy.csv.
        """
        df = self._df.copy()

        # Assign template_idx via YAML canonical order
        template_order = self._yaml_template_order()
        df["template_idx"] = df.apply(
            lambda r: template_order.get(r["behavior_id"], {}).get(r["attack_template"], 0),
            axis=1,
        )
        df["strategy"] = df["template_idx"].map(self._STRATEGY_NAMES)

        pivot = (
            df.groupby(["strategy", "behavior_id"])["asr"]
            .mean()
            .unstack("behavior_id")
        )
        pivot["Mean"] = pivot.mean(axis=1)

        # Canonical row order
        row_order = [self._STRATEGY_NAMES[i] for i in sorted(self._STRATEGY_NAMES)]
        pivot = pivot.reindex([r for r in row_order if r in pivot.index])

        out_path = self._output_dir / "template_strategy.csv"
        pivot.to_csv(out_path)
        logger.info(f"Saved template strategy table: {out_path}")
        return pivot

    def run_all(self) -> None:
        """Generate all plots, stats, and summary CSV."""
        self.print_load_summary()
        logger.info("Generating all analysis outputs...")
        self.plot_all_heatmaps()
        self.plot_asr_by_resource_tier()
        self.plot_model_comparison()
        self.plot_behavior_boxplot()
        self.plot_language_ranked()
        self.plot_eval_method_usage()
        self.export_summary_csv()
        self.template_asr_analysis()
        self.plot_template_heatmaps()
        self.template_strategy_table()

        tests = self.statistical_tests()
        if not tests.empty:
            tests_path = self._output_dir / "statistical_tests.csv"
            tests.to_csv(tests_path, index=False)
            logger.info(f"Saved statistical tests: {tests_path}")

        self.model_statistical_tests()

        # Phase 2 defense analysis (runs only if defense data exists)
        self.plot_defense_comparison()
        self.plot_defense_gap_reduction()
        self.plot_defense_tier_gradient()
        self.defense_statistical_tests()

        logger.info("All analysis outputs generated.")
