#!/usr/bin/env python3
"""
translate_system_prompts.py
One-time script to translate each behavior's strategy-aware defense system prompt
into all 24 languages for Defense B (multilingual system prompts).

Output: prompts/defense_system_prompts/{behavior_id}/{lang_code}.yaml
Each YAML file contains:
  system_prompt: "<translated hardened prompt>"

Usage:
  python scripts/translate_system_prompts.py
  python scripts/translate_system_prompts.py --behaviors no_homework math_only
  python scripts/translate_system_prompts.py --languages fr de zh
  python scripts/translate_system_prompts.py --dry-run
"""

import argparse
import logging
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import yaml
from src.config_loader import load_config

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

# Delay between Google Translate calls to avoid rate limits
_TRANSLATE_DELAY = 1.0


def translate_text(text: str, target_lang: str) -> str:
    """Translate text from English to target_lang using Google Translate."""
    from deep_translator import GoogleTranslator
    time.sleep(_TRANSLATE_DELAY)
    translated = GoogleTranslator(source="en", target=target_lang).translate(text)
    return translated or text


def build_bilingual_prompt(english_prompt: str, translated_prompt: str) -> str:
    """
    Build a bilingual system prompt: English original + translated version.
    This is the Defense B format: the model sees the restriction in both
    English and the attack language.
    """
    return (
        f"{english_prompt}\n\n"
        f"[The above instructions also apply in your language:]\n"
        f"{translated_prompt}"
    )


def main():
    parser = argparse.ArgumentParser(
        description="Translate defense system prompts for Defense B (multilingual)"
    )
    parser.add_argument(
        "--behaviors", nargs="+",
        help="Behavior IDs to translate (default: all)"
    )
    parser.add_argument(
        "--languages", nargs="+",
        help="Language codes to translate to (default: all non-English)"
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Print what would be generated without doing it"
    )
    parser.add_argument(
        "--config-dir", type=Path, default=None,
        help="Path to config/ directory (default: auto-detect)"
    )
    args = parser.parse_args()

    cfg = load_config(args.config_dir)
    output_dir = cfg.project_root / "prompts" / "defense_system_prompts"

    behaviors = [
        b for b in cfg.behaviors
        if args.behaviors is None or b.id in args.behaviors
    ]
    languages = [
        l for l in cfg.languages
        if args.languages is None or l.code in args.languages
    ]

    total = 0
    skipped = 0

    for behavior in behaviors:
        # Use the strategy_aware defense prompt as the source for translation
        source_prompt = behavior.defense_prompts.get("strategy_aware")
        if not source_prompt:
            logger.warning(
                f"No strategy_aware defense prompt for '{behavior.id}'. "
                f"Using baseline system_prompt."
            )
            source_prompt = behavior.system_prompt

        for lang in languages:
            out_path = output_dir / behavior.id / f"{lang.code}.yaml"

            if lang.code == "en":
                # English: just use the English prompt directly
                if args.dry_run:
                    print(f"[DRY RUN] [{behavior.id}][en] Write English-only prompt")
                else:
                    out_path.parent.mkdir(parents=True, exist_ok=True)
                    data = {"system_prompt": source_prompt}
                    with open(out_path, "w", encoding="utf-8") as f:
                        yaml.dump(data, f, allow_unicode=True, default_flow_style=False)
                    print(f"[{behavior.id}][en] Saved English prompt")
                total += 1
                continue

            # Check if already exists
            if out_path.exists():
                skipped += 1
                print(f"[{behavior.id}][{lang.code}] Already exists, skipping")
                continue

            if args.dry_run:
                print(
                    f"[DRY RUN] [{behavior.id}][{lang.code}] "
                    f"Would translate and save bilingual prompt"
                )
            else:
                try:
                    translated = translate_text(
                        source_prompt, lang.deep_translator_code
                    )
                    bilingual = build_bilingual_prompt(source_prompt, translated)

                    out_path.parent.mkdir(parents=True, exist_ok=True)
                    data = {"system_prompt": bilingual}
                    with open(out_path, "w", encoding="utf-8") as f:
                        yaml.dump(
                            data, f, allow_unicode=True, default_flow_style=False
                        )
                    print(
                        f"[{behavior.id}][{lang.code}] "
                        f"Saved bilingual prompt ({len(bilingual)} chars)"
                    )
                except Exception as e:
                    logger.error(
                        f"Failed to translate [{behavior.id}][{lang.code}]: {e}"
                    )
            total += 1

    action = "Would process" if args.dry_run else "Processed"
    print(f"\n{action} {total} prompts ({skipped} already existed).")
    print(f"Output directory: {output_dir}")


if __name__ == "__main__":
    main()
