"""
result_store.py
Thread-safe, append-only CSV writer for experiment results.
Supports resume by tracking completed (model, behavior_id, lang_code, run_index) keys.
"""

from __future__ import annotations

import csv
import dataclasses
import threading
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import pandas as pd

# ---------------------------------------------------------------------------
# Run record dataclass — one row in results/raw/runs.csv
# ---------------------------------------------------------------------------

CSV_COLUMNS = [
    "timestamp",
    "model",
    "behavior_id",
    "language_code",
    "language_name",
    "resource_tier",
    "run_index",
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
    behavior_id: str
    language_code: str
    language_name: str
    resource_tier: str
    run_index: int
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
        """Create CSV file with header if it does not exist."""
        if not self._path.exists():
            self._path.parent.mkdir(parents=True, exist_ok=True)
            with open(self._path, "w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
                writer.writeheader()

    def append(self, record: RunRecord) -> None:
        """Append a single run record to the CSV file (thread-safe)."""
        row = dataclasses.asdict(record)
        with self._lock:
            with open(self._path, "a", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
                writer.writerow(row)

    def get_completed_keys(self) -> set[tuple[str, str, str, int]]:
        """
        Return set of (model, behavior_id, language_code, run_index) tuples
        that already have status='success' in the CSV.
        Used to skip already-done cells when resuming.
        """
        if not self._path.exists():
            return set()

        try:
            df = pd.read_csv(self._path, usecols=["model", "behavior_id",
                                                    "language_code", "run_index",
                                                    "status"])
            completed = df[df["status"] == "success"]
            return set(
                zip(
                    completed["model"],
                    completed["behavior_id"],
                    completed["language_code"],
                    completed["run_index"].astype(int),
                )
            )
        except Exception:
            return set()

    def count_completed(self) -> int:
        """Return number of successfully completed runs."""
        return len(self.get_completed_keys())

    def load_dataframe(self) -> pd.DataFrame:
        """Load all results as a pandas DataFrame."""
        if not self._path.exists():
            return pd.DataFrame(columns=CSV_COLUMNS)
        return pd.read_csv(self._path)
