"""
result_store.py
Thread-safe, append-only CSV writer for experiment results.
Supports resume by tracking completed (model, behavior_id, lang_code, run_index) keys.
"""

from __future__ import annotations

import csv
import dataclasses
import logging
import threading
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import pandas as pd

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Run record dataclass — one row in results/raw/runs.csv
# ---------------------------------------------------------------------------

CSV_COLUMNS = [
    "timestamp",
    "model",
    "judge_model",
    "behavior_id",
    "prompt_variant",
    "language_code",
    "language_name",
    "resource_tier",
    "run_index",
    "template_index",
    "attack_template",
    "translated_prompt",
    "model_response",
    "asr",
    "eval_method",
    "eval_confidence",
    "eval_reason",
    "status",
]


@dataclass
class RunRecord:
    timestamp: str
    model: str
    judge_model: str
    behavior_id: str
    prompt_variant: str
    language_code: str
    language_name: str
    resource_tier: str
    run_index: int
    template_index: int
    attack_template: str
    translated_prompt: str
    model_response: str
    asr: float          # 0.0, 0.5, 1.0, or float("nan") for errors
    eval_method: str
    eval_confidence: float
    eval_reason: str
    status: str         # "success", "api_error", "rate_limit_error", "translation_fallback", etc.

    @staticmethod
    def now() -> str:
        return datetime.now(timezone.utc).isoformat()


# ---------------------------------------------------------------------------
# ResultStore
# ---------------------------------------------------------------------------

class ResultStore:
    """
    Manages the master results CSV file.
    - Thread-safe appends via a threading.Lock
    - Resume support: reads completed keys on init
    """

    def __init__(self, csv_path: Path):
        self._path = csv_path
        self._lock = threading.Lock()
        self._init_file()

    def _init_file(self) -> None:
        """Create CSV file with header if it does not exist, or migrate an old-format file."""
        if not self._path.exists():
            self._path.parent.mkdir(parents=True, exist_ok=True)
            with open(self._path, "w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
                writer.writeheader()
        else:
            self._migrate_if_needed()

    def _migrate_if_needed(self) -> None:
        """
        Detect and repair schema mismatches in an existing CSV.

        Handles the case where columns were added to CSV_COLUMNS after some
        rows were already written (e.g. adding 'judge_model' mid-experiment).
        Uses csv.reader directly so it tolerates rows with different field
        counts without raising a ParserError.

        Missing columns are backfilled with empty strings for old rows.
        Rows that already have the new column count are kept as-is.
        Rows with an unexpected column count are preserved verbatim under
        'unknown' so no data is silently lost.
        """
        # Read the current header
        with open(self._path, newline="", encoding="utf-8") as f:
            reader = csv.reader(f)
            try:
                old_header = next(reader)
            except StopIteration:
                return  # empty file — nothing to migrate

        if old_header == CSV_COLUMNS:
            return  # already up to date

        missing = [c for c in CSV_COLUMNS if c not in old_header]
        if not missing:
            # Columns exist but order may differ — reorder header only
            logger.info(f"CSV column order mismatch — rewriting header: {self._path.name}")

        logger.info(
            f"Migrating {self._path.name}: adding missing columns {missing}"
        )

        # Build a position map for old columns
        old_idx: dict[str, int] = {col: i for i, col in enumerate(old_header)}

        # Read ALL data rows as raw lists (tolerates mixed row widths)
        raw_rows: list[list[str]] = []
        with open(self._path, newline="", encoding="utf-8") as f:
            reader = csv.reader(f)
            next(reader)  # skip header
            for row in reader:
                raw_rows.append(row)

        # Rewrite the file atomically: write to a temp path, then rename
        tmp_path = self._path.with_suffix(".csv.migrating")
        try:
            with open(tmp_path, "w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow(CSV_COLUMNS)
                for row in raw_rows:
                    if len(row) == len(old_header):
                        # Map by column name, fill missing with ""
                        new_row = [
                            row[old_idx[col]] if col in old_idx else ""
                            for col in CSV_COLUMNS
                        ]
                        writer.writerow(new_row)
                    elif len(row) == 0:
                        pass  # skip blank lines
                    else:
                        # Unknown format — preserve raw so nothing is lost
                        writer.writerow(row)
                        logger.warning(
                            f"Row with unexpected field count {len(row)} preserved verbatim."
                        )
            tmp_path.replace(self._path)
            logger.info(f"Migration complete: {self._path.name}")
        except Exception as exc:
            tmp_path.unlink(missing_ok=True)
            raise RuntimeError(f"CSV migration failed: {exc}") from exc

    def append(self, record: RunRecord) -> None:
        """Append a single run record to the CSV file (thread-safe)."""
        row = dataclasses.asdict(record)
        with self._lock:
            with open(self._path, "a", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
                writer.writerow(row)

    # Minimum number of characters a model_response must have to be considered valid.
    # Responses shorter than this are treated as invalid and will be regenerated on resume.
    MIN_RESPONSE_CHARS: int = 20

    def get_completed_keys(self) -> set[tuple[str, str, str, str, int]]:
        """
        Return set of (model, behavior_id, prompt_variant, language_code, run_index)
        tuples that have a valid completed run in the CSV.

        A run is considered valid (and therefore skipped on resume) only when ALL of:
          1. status == 'success'
          2. model_response is non-empty and has >= MIN_RESPONSE_CHARS characters
          3. asr is not NaN (evaluation was not skipped)

        Any run that fails these checks is excluded from the completed set,
        causing it to be regenerated on the next --resume run.
        """
        if not self._path.exists():
            return set()

        try:
            df = pd.read_csv(
                self._path,
                usecols=["model", "behavior_id", "prompt_variant",
                          "language_code",
                          "run_index", "status", "model_response", "asr"],
                on_bad_lines="warn",
            )

            # Backfill missing prompt_variant for old CSVs
            if "prompt_variant" not in df.columns:
                df["prompt_variant"] = "baseline"
            else:
                df["prompt_variant"] = df["prompt_variant"].fillna("baseline")

            # Filter 1: successful API call
            mask = df["status"] == "success"

            # Filter 2: non-empty response with sufficient length
            responses = df["model_response"].fillna("").astype(str)
            mask &= responses.str.strip().str.len() >= self.MIN_RESPONSE_CHARS

            # Filter 3: ASR was actually computed (not NaN / skipped)
            mask &= df["asr"].notna()

            completed = df[mask]
            return set(
                zip(
                    completed["model"],
                    completed["behavior_id"],
                    completed["prompt_variant"],
                    completed["language_code"],
                    completed["run_index"].astype(int),
                )
            )
        except Exception:
            return set()

    def count_completed(self) -> int:
        """Return number of successfully completed runs."""
        return len(self.get_completed_keys())

    def dedup(self) -> int:
        """
        Remove duplicate rows from the CSV, keeping the latest successful row
        per (model, behavior_id, language_code, run_index) cell.

        Duplicates arise when --resume fails to detect completed cells (e.g.
        because the CSV was in a mixed-format state) and re-runs them.

        Returns the number of duplicate rows removed.
        """
        if not self._path.exists():
            return 0

        df = pd.read_csv(self._path, on_bad_lines="warn")
        if "prompt_variant" not in df.columns:
            df["prompt_variant"] = "baseline"
        else:
            df["prompt_variant"] = df["prompt_variant"].fillna("baseline")
        before = len(df)

        key_cols = ["model", "behavior_id", "prompt_variant", "language_code", "run_index"]
        if not df.duplicated(subset=key_cols).any():
            logger.info("No duplicates found.")
            return 0

        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
        df = (
            df.sort_values("timestamp", ascending=False)
              .drop_duplicates(subset=key_cols, keep="first")
              .sort_values(["model", "behavior_id", "language_code", "run_index"])
              .reset_index(drop=True)
        )

        tmp_path = self._path.with_suffix(".csv.deduping")
        try:
            df.to_csv(tmp_path, index=False)
            tmp_path.replace(self._path)
        except Exception as exc:
            tmp_path.unlink(missing_ok=True)
            raise RuntimeError(f"Dedup write failed: {exc}") from exc

        removed = before - len(df)
        logger.info(f"Dedup complete: removed {removed} duplicate rows ({before} → {len(df)}).")
        return removed

    def load_dataframe(self) -> pd.DataFrame:
        """Load all results as a pandas DataFrame."""
        if not self._path.exists():
            return pd.DataFrame(columns=CSV_COLUMNS)
        return pd.read_csv(self._path, on_bad_lines="warn")


# ---------------------------------------------------------------------------
# Standalone migration CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    path = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("results/raw/runs.csv")
    print(f"Repairing: {path}")
    store = ResultStore(path)   # _init_file -> _migrate_if_needed runs here
    removed = store.dedup()
    if removed:
        print(f"Removed {removed} duplicate rows.")
    print("Done.")
