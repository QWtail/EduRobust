"""
translator.py
Translation cache: loads pre-defined YAML translations, falls back to
deep_translator.GoogleTranslator and persists new translations to disk.
"""

from __future__ import annotations

import logging
import time
from pathlib import Path

import yaml

logger = logging.getLogger(__name__)

# Delay between Google Translate calls to avoid rate limits (seconds)
_TRANSLATE_DELAY = 1.0


class TranslationCache:
    """
    Manages translated attack prompts.

    Priority:
      1. In-memory cache (after first load)
      2. On-disk YAML file: translations/{behavior_id}/{lang_code}.yaml
      3. deep_translator.GoogleTranslator (fallback) -> persisted to disk

    File format (YAML):
      "English source text": "Translated text"
    """

    def __init__(self, translations_dir: Path, fallback: bool = True):
        self._translations_dir = translations_dir
        self._fallback = fallback
        # Cache: (source_text, lang_code) -> translated_text
        self._cache: dict[tuple[str, str], str] = {}
        # Track which (behavior_id, lang_code) files have been loaded
        self._loaded: set[tuple[str, str]] = set()

    def get(self, text: str, lang_code: str, behavior_id: str) -> str:
        """
        Return the translation of `text` into `lang_code` for the given behavior.
        English source returns unchanged.
        """
        if lang_code == "en":
            return text

        # Ensure the translation file for this behavior+lang is loaded
        file_key = (behavior_id, lang_code)
        if file_key not in self._loaded:
            self._load_file(behavior_id, lang_code)

        cache_key = (text, lang_code)
        if cache_key in self._cache:
            return self._cache[cache_key]

        # Fallback: translate on-the-fly
        if self._fallback:
            return self._translate_and_cache(text, lang_code, behavior_id)

        logger.warning(
            f"No translation found for lang={lang_code}, behavior={behavior_id}. "
            f"Using English original."
        )
        return text

    def _load_file(self, behavior_id: str, lang_code: str) -> None:
        """Load a translation YAML file into the cache."""
        path = self._translations_dir / behavior_id / f"{lang_code}.yaml"
        self._loaded.add((behavior_id, lang_code))

        if not path.exists():
            logger.debug(
                f"Translation file not found: {path}. Will use fallback translator."
            )
            return

        try:
            with open(path, "r", encoding="utf-8") as f:
                data = yaml.safe_load(f) or {}
            for src, tgt in data.items():
                self._cache[(str(src), lang_code)] = str(tgt)
            logger.debug(
                f"Loaded {len(data)} translations from {path}"
            )
        except Exception as e:
            logger.warning(f"Failed to load translation file {path}: {e}")

    # Marker that survives Google Translate without being translated
    _PLACEHOLDER_MARKER = "EDUROBUST99PROBLEM99"

    def _protect_placeholders(self, text: str) -> str:
        return text.replace("{problem}", self._PLACEHOLDER_MARKER)

    def _restore_placeholders(self, text: str) -> str:
        return text.replace(self._PLACEHOLDER_MARKER, "{problem}")

    def _translate_and_cache(
        self, text: str, lang_code: str, behavior_id: str
    ) -> str:
        """Translate using Google Translate and persist to disk."""
        try:
            from deep_translator import GoogleTranslator
            time.sleep(_TRANSLATE_DELAY)
            protected = self._protect_placeholders(text)
            translated = GoogleTranslator(source="en", target=lang_code).translate(protected)
            if not translated:
                logger.warning(
                    f"Empty translation returned for lang={lang_code}. "
                    f"Using English."
                )
                return text
            translated = self._restore_placeholders(translated)

            # Cache in memory
            self._cache[(text, lang_code)] = translated

            # Persist to disk
            self._persist(text, translated, lang_code, behavior_id)

            return translated

        except Exception as e:
            logger.warning(
                f"Translation failed for lang={lang_code}: {e}. Using English."
            )
            return text

    def _persist(
        self, source: str, translated: str, lang_code: str, behavior_id: str
    ) -> None:
        """Append a new translation to the on-disk YAML file."""
        path = self._translations_dir / behavior_id / f"{lang_code}.yaml"
        path.parent.mkdir(parents=True, exist_ok=True)

        # Load existing content
        existing: dict = {}
        if path.exists():
            try:
                with open(path, "r", encoding="utf-8") as f:
                    existing = yaml.safe_load(f) or {}
            except Exception:
                pass

        existing[source] = translated

        with open(path, "w", encoding="utf-8") as f:
            yaml.dump(existing, f, allow_unicode=True, default_flow_style=False)


def generate_all_translations(
    attack_templates: dict[str, list[str]],
    languages: list,  # list of LanguageConfig
    translations_dir: Path,
    dry_run: bool = False,
    behaviors_filter: list[str] | None = None,
    languages_filter: list[str] | None = None,
) -> None:
    """
    One-time translation generation for all behaviors and languages.
    Used by scripts/translate_prompts.py.
    """
    cache = TranslationCache(translations_dir, fallback=True)

    for behavior_id, templates in attack_templates.items():
        if behaviors_filter and behavior_id not in behaviors_filter:
            continue

        # english_only behavior: prompts are foreign-language prompts themselves
        # We translate "please respond in my language" style prompts
        for lang in languages:
            if languages_filter and lang.code not in languages_filter:
                continue
            if lang.code == "en":
                continue

            for template in templates:
                if dry_run:
                    print(
                        f"[DRY RUN] Would translate [{behavior_id}] [{lang.code}]: "
                        f"{template[:60]}..."
                    )
                else:
                    translated = cache.get(template, lang.deep_translator_code, behavior_id)
                    print(
                        f"[{behavior_id}][{lang.code}] {template[:40]!r} -> "
                        f"{translated[:40]!r}"
                    )
