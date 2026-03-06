"""
config_loader.py
Loads and validates all YAML configuration files using Pydantic v2.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Literal, Optional

import yaml
from dotenv import load_dotenv
from pydantic import BaseModel, Field, field_validator

# ---------------------------------------------------------------------------
# Pydantic schemas
# ---------------------------------------------------------------------------

class ExperimentSettings(BaseModel):
    runs_per_cell: int = 20
    temperature: float = 0.7
    max_new_tokens: int = 512
    seed: int = 42
    checkpoint_every: int = 50


class APISettings(BaseModel):
    model_config = {"protected_namespaces": ()}

    provider: str = "ollama"
    ollama_host: str = "http://localhost:11434"
    max_consecutive_errors: int = 10
    max_retries: int = 5
    base_backoff_seconds: float = 2.0
    timeout_seconds: int = 120
    rate_limit_pause_seconds: int = 30
    model_loading_wait_seconds: int = 30


class EvaluationSettings(BaseModel):
    judge_model: str = "llama3.1:8b"
    judge_provider: str = "ollama"
    judge_temperature: float = 0.0
    judge_max_tokens: int = 256
    confidence_threshold: float = 0.7
    use_keyword_fallback: bool = True
    essay_min_words: int = 100


class OutputSettings(BaseModel):
    results_dir: str = "results/raw"
    analysis_dir: str = "results/analysis"
    log_dir: str = "logs"
    log_level: str = "INFO"


class MasterConfig(BaseModel):
    experiment: ExperimentSettings
    api: APISettings
    evaluation: EvaluationSettings
    output: OutputSettings


class ModelConfig(BaseModel):
    id: str
    name: str
    provider: str          # "ollama" | "huggingface_local" | "huggingface"
    enabled: bool = True
    max_new_tokens: int = 512
    # huggingface_local options (ignored for other providers)
    torch_dtype: str = "auto"       # "auto" | "float16" | "bfloat16" | "float32"
    load_in_8bit: bool = False      # 8-bit quantization (needs bitsandbytes)
    load_in_4bit: bool = False      # 4-bit quantization (needs bitsandbytes)


class LanguageConfig(BaseModel):
    code: str
    name: str
    resource_tier: Literal["high", "medium", "low"]
    script: str
    deep_translator_code: str


class BehaviorConfig(BaseModel):
    id: str
    name: str
    system_prompt: str
    attack_strategy: str
    judge_criteria: str
    refusal_keywords: list[str] = Field(default_factory=list)
    bypass_indicators: list[str] = Field(default_factory=list)
    special_eval: Optional[str] = None  # "langdetect", "essay_length", or None

    @field_validator("system_prompt", mode="before")
    @classmethod
    def strip_prompt(cls, v: str) -> str:
        return v.strip()


# ---------------------------------------------------------------------------
# Top-level config object
# ---------------------------------------------------------------------------

class EduRobustConfig:
    """Aggregates all configuration from YAML files."""

    def __init__(
        self,
        master: MasterConfig,
        models: list[ModelConfig],
        languages: list[LanguageConfig],
        behaviors: list[BehaviorConfig],
        hf_token: str,
        project_root: Path,
    ):
        self.master = master
        self.models = models
        self.languages = languages
        self.behaviors = behaviors
        self.hf_token = hf_token
        self.project_root = project_root

    @property
    def enabled_models(self) -> list[ModelConfig]:
        return [m for m in self.models if m.enabled]

    @property
    def results_dir(self) -> Path:
        return self.project_root / self.master.output.results_dir

    @property
    def analysis_dir(self) -> Path:
        return self.project_root / self.master.output.analysis_dir

    @property
    def log_dir(self) -> Path:
        return self.project_root / self.master.output.log_dir

    @property
    def translations_dir(self) -> Path:
        return self.project_root / "prompts" / "translations"

    @property
    def attack_templates_path(self) -> Path:
        return self.project_root / "prompts" / "attack_templates.yaml"


# ---------------------------------------------------------------------------
# Loader function
# ---------------------------------------------------------------------------

def _load_yaml(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_config(config_dir: Path | None = None) -> EduRobustConfig:
    """
    Load all config from the config/ directory.
    If config_dir is None, auto-detect from this file's location.
    """
    load_dotenv()

    if config_dir is None:
        # Assume this file is in src/, config/ is a sibling directory
        config_dir = Path(__file__).parent.parent / "config"

    project_root = config_dir.parent

    # Load master config
    master_raw = _load_yaml(config_dir / "config.yaml")
    master = MasterConfig(**master_raw)

    # Load models
    models_raw = _load_yaml(config_dir / "models.yaml")
    models = [ModelConfig(**m) for m in models_raw["models"]]

    # Load languages
    langs_raw = _load_yaml(config_dir / "languages.yaml")
    languages = [LanguageConfig(**l) for l in langs_raw["languages"]]

    # Load behaviors
    behaviors_raw = _load_yaml(config_dir / "behaviors.yaml")
    behaviors = [BehaviorConfig(**b) for b in behaviors_raw["behaviors"]]

    # HuggingFace token: only required when HF-hosted models are enabled
    hf_token = os.environ.get("HF_TOKEN", "")
    needs_hf = any(
        m.provider.startswith("huggingface")
        for m in models
        if m.enabled
    )
    if not hf_token and needs_hf:
        import getpass
        hf_token = getpass.getpass("Enter your HuggingFace token (input hidden): ").strip()
    if not hf_token and needs_hf:
        raise EnvironmentError("No HuggingFace token provided.")

    return EduRobustConfig(
        master=master,
        models=models,
        languages=languages,
        behaviors=behaviors,
        hf_token=hf_token,
        project_root=project_root,
    )
