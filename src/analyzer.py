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

        # Sort languages by overall mean ASR (descending), then transpose
        pivot["_mean"] = pivot.mean(axis=1)
        pivot = pivot.sort_values("_mean", ascending=False).drop(columns="_mean")
        pivot = pivot.T  # rows=behaviors (5), cols=languages (20, sorted by mean ASR desc)

        fig, ax = plt.subplots(figsize=(max(14, len(pivot.columns) * 0.7), max(4, len(pivot) * 1.0)))
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
        title = "Attack Success Rate: Behavior × Language"
        if model:
            title += f"\nModel: {model}"
        ax.set_title(title, fontsize=18, fontweight="bold", pad=12)
        ax.set_xlabel("Language", fontsize=15)
        ax.set_ylabel("Forbidden Behavior", fontsize=15)
        plt.xticks(rotation=45, ha="right")
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
        ax.set_ylim(0, 1.05)
        ax.set_ylabel("Mean Attack Success Rate")
        ax.set_title("Languages Ranked by Overall ASR", fontsize=13, fontweight="bold")
        plt.xticks(rotation=45, ha="right")

        # Legend for resource tiers
        from matplotlib.patches import Patch
        legend_elements = [Patch(facecolor=c, label=t)
                           for t, c in tier_colors.items()]
        ax.legend(handles=legend_elements, title="Resource Tier", loc="upper right")

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
            fig_h = max(4, n_templates * 1.0)

            fig, ax = plt.subplots(figsize=(fig_w, fig_h))
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
            ax.set_title(
                f"Per-Template ASR — Behavior: {behavior_id}\n"
                f"(rows = attack templates, cols = languages; "
                f"averaged over models & run repeats)",
                fontsize=12,
                fontweight="bold",
                pad=12,
            )
            ax.set_xlabel("Language", fontsize=11)
            ax.set_ylabel("Attack Template", fontsize=11)
            plt.xticks(rotation=45, ha="right", fontsize=9)
            plt.yticks(rotation=0, fontsize=9)

            fname = self._heatmaps_dir / f"template_heatmap_{behavior_id}.png"
            fig.savefig(fname, dpi=150)
            plt.close(fig)
            logger.info(f"Saved template heatmap: {fname}")

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

        logger.info("All analysis outputs generated.")
