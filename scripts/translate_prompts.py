#!/usr/bin/env python3
"""
translate_prompts.py
One-time script to pre-generate all attack prompt translations.
Run this before the main experiment to populate prompts/translations/.

Usage:
  python scripts/translate_prompts.py
  python scripts/translate_prompts.py --behaviors no_homework math_only
  python scripts/translate_prompts.py --languages fr de zh
  python scripts/translate_prompts.py --dry-run
"""

import argparse
import logging
import sys
from pathlib import Path

# Allow running from project root
sys.path.insert(0, str(Path(__file__).parent.parent))

import yaml
from src.config_loader import load_config
from src.translator import TranslationCache

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Pre-generate attack prompt translations")
    parser.add_argument("--behaviors", nargs="+", help="Behavior IDs to translate (default: all)")
    parser.add_argument("--languages", nargs="+", help="Language codes to translate to (default: all)")
    parser.add_argument("--dry-run", action="store_true", help="Print what would be translated without doing it")
    args = parser.parse_args()

    cfg = load_config()

    # Load attack templates
    with open(cfg.attack_templates_path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    attack_templates: dict[str, list[str]] = data.get("attack_prompts", {})

    # Filter behaviors and languages
    behaviors = [
        b for b in cfg.behaviors
        if args.behaviors is None or b.id in args.behaviors
    ]
    languages = [
        l for l in cfg.languages
        if args.languages is None or l.code in args.languages
    ]

    cache = TranslationCache(cfg.translations_dir, fallback=True)

    total = 0
    for behavior in behaviors:
        templates = attack_templates.get(behavior.id, [])
        if not templates:
            logger.warning(f"No templates found for behavior '{behavior.id}'")
            continue

        for lang in languages:
            if lang.code == "en":
                continue  # Skip English (identity translation)

            for template in templates:
                if args.dry_run:
                    print(
                        f"[DRY RUN] [{behavior.id}][{lang.code}] "
                        f"{template[:60]}..."
                    )
                else:
                    translated = cache.get(
                        template, lang.deep_translator_code, behavior.id
                    )
                    print(
                        f"[{behavior.id}][{lang.code}] "
                        f"{template[:35]!r} -> {translated[:35]!r}"
                    )
                total += 1

    print(f"\n{'Would process' if args.dry_run else 'Processed'} {total} translations.")


if __name__ == "__main__":
    main()
