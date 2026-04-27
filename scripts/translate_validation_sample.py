#!/usr/bin/env python3
"""Translate model_response in validation sample to English for human annotation.

- B1, B3, B4, B5: translate model_response to English (skip if already English)
- B2 (english_only): keep original (judgment is about response language, not content)
"""

import pandas as pd
import time
from pathlib import Path
from deep_translator import GoogleTranslator

ROOT = Path(__file__).resolve().parent.parent
INPUT_CSV = ROOT / "results" / "validation_sample.csv"
OUTPUT_CSV = ROOT / "results" / "validation_sample.csv"  # overwrite in-place

MAX_CHARS = 4900  # Google Translate limit ~5000 chars


def translate_to_english(text: str) -> str:
    """Translate text to English via Google Translate. Returns original if already English or on failure."""
    if not isinstance(text, str) or len(text.strip()) == 0:
        return text
    # Truncate if too long
    truncated = text[:MAX_CHARS] if len(text) > MAX_CHARS else text
    try:
        result = GoogleTranslator(source="auto", target="en").translate(truncated)
        return result if result else text
    except Exception as e:
        print(f"  Translation failed: {e}")
        return text


def main():
    df = pd.read_csv(INPUT_CSV)
    print(f"Loaded {len(df)} rows from {INPUT_CSV}")

    # Add column if not exists
    if "model_response_en" not in df.columns:
        df["model_response_en"] = ""

    # Split: B2 keeps original, others get translated
    b2_mask = df["behavior_id"] == "english_only"
    to_translate = df[~b2_mask]

    print(f"\nB2 english_only: {b2_mask.sum()} rows (keeping original)")
    print(f"B1/B3/B4/B5: {len(to_translate)} rows to translate")

    # For B2, model_response_en = original (annotator judges language directly)
    df.loc[b2_mask, "model_response_en"] = df.loc[b2_mask, "model_response"]

    # Translate B1/B3/B4/B5
    translated = 0
    skipped_en = 0
    for idx in to_translate.index:
        row = df.loc[idx]
        response = row["model_response"]

        # Skip if already English
        if row["language_code"] == "en":
            df.at[idx, "model_response_en"] = response
            skipped_en += 1
            continue

        # Skip if already translated
        if isinstance(row.get("model_response_en"), str) and len(row["model_response_en"].strip()) > 0:
            skipped_en += 1
            continue

        print(f"  [{translated+1}/{len(to_translate)-skipped_en}] {row['behavior_id']} / {row['language_code']} / {row['model'][:10]}...", end="")
        result = translate_to_english(response)
        df.at[idx, "model_response_en"] = result
        translated += 1
        print(" done")

        # Rate limit: ~1 req/sec to avoid Google throttling
        time.sleep(0.5)

    print(f"\nTranslated: {translated}")
    print(f"Skipped (English or already done): {skipped_en}")

    # Reorder columns: put model_response_en right after model_response
    cols = list(df.columns)
    resp_idx = cols.index("model_response")
    if "model_response_en" in cols:
        cols.remove("model_response_en")
    cols.insert(resp_idx + 1, "model_response_en")
    df = df[cols]

    df.to_csv(OUTPUT_CSV, index=False)
    print(f"\nSaved to {OUTPUT_CSV}")


if __name__ == "__main__":
    main()
