# EduRobust

A research framework for evaluating the robustness of LLM system prompt restrictions in educational contexts against multilingual prompt injection attacks.

## Overview

EduRobust tests whether language models can be prompted — in 24 languages across three resource tiers — to bypass behavioral restrictions defined in their system prompts (e.g., "do not do homework for students", "respond in English only"). It measures Attack Success Rate (ASR) across models, behaviors, languages, and attack strategies.

The framework supports a two-phase workflow:
- **Phase 1 (Baseline):** Measures baseline vulnerability across the full experiment grid
- **Phase 2 (Defenses):** Evaluates three defense variants informed by Phase 1 findings

**Models tested:**

| Model | Provider | Language Profile |
|---|---|---|
| Llama 3.1 8B (Q4) | Meta | English-dominant, 8 languages |
| Qwen 2.5 7B | Alibaba | Chinese+English bilingual, 29+ languages |
| Mistral 7B | Mistral AI | Primarily English |

**Behaviors tested:**
| ID | Description |
|---|---|
| `no_homework` | Model should not solve homework directly |
| `math_only` | Model should only answer math questions |
| `hints_only` | Model should give hints, not full answers |
| `no_essay` | Model should not write essays for students |
| `english_only` | Model should only respond in English |

**Attack strategies** (5 templates per behavior, applied in round-robin):

| Index | Strategy | Description |
|---|---|---|
| T0 | Direct | Straightforward request that violates the restriction |
| T1 | Urgency | Appeals to time pressure or panic |
| T2 | Social | Invokes authority or social pressure (teacher, peer) |
| T3 | Persona | Instructs the model to adopt a different persona |
| T4 | Override | Explicitly tells the model to ignore its restrictions |

**Defense variants** (Phase 2):

| Variant | Flag | Description |
|---|---|---|
| Baseline | `--variant baseline` | Original system prompts (Phase 1, default) |
| Strategy-Aware (A) | `--variant strategy_aware` | Hardened prompts with anti-jailbreak clauses targeting the most effective Phase 1 attack strategies |
| Multilingual (B) | `--variant multilingual` | Bilingual system prompts presented in both English and the attack language |
| Composite (C) | `--variant composite` | Appends an English-only response constraint to each behavior's system prompt (4 behaviors; `english_only` excluded as redundant) |

**Languages tested** (24 languages across 3 resource tiers):

| Tier | Languages |
|---|---|
| High (12) | English, French, Spanish, German, Chinese, Japanese, Korean, Arabic, Russian, Portuguese, Italian, Dutch |
| Medium (7) | Hindi, Indonesian, Turkish, Polish, Vietnamese, Bengali, Thai |
| Low (5) | Swahili, Amharic, Yoruba, Hausa, Burmese |

## Experiment Grid

Full grid: **3 models × 5 behaviors × 24 languages × 50 runs/cell = 18,000 runs per variant**

| Variant | Runs | Notes |
|---|---|---|
| Baseline | 18,000 | Phase 1 — original system prompts |
| Strategy-Aware (A) | 18,000 | Anti-jailbreak clauses per behavior |
| Multilingual (B) | 18,000 | Bilingual system prompts |
| Composite (C) | 14,400 | English-only anchor (4 behaviors; `english_only` excluded) |
| **Total** | **68,400** | |

Resume key: 5-tuple `(model, behavior_id, prompt_variant, language_code, run_index)` — each variant's data is tracked independently in `runs.csv`.

## Providers

EduRobust supports three inference backends. Choose based on your setup:

| Provider | How it works | Rate limits | Requires |
|---|---|---|---|
| `ollama` | Model runs locally via the Ollama daemon | None | Ollama installed + `ollama pull` |
| `huggingface_local` | Model weights downloaded and run via `transformers` | None | `transformers`, `torch`, `accelerate` |
| `huggingface` | Model called via HuggingFace Inference API (remote) | Yes (free tier) | `HF_TOKEN` in `.env` |

**When to use which:**
- **Ollama** — easiest local setup on a laptop or desktop
- **huggingface_local** — GPU server or research cluster; avoids Ollama dependency; fully offline after first download
- **huggingface** — want to try a model without downloading it, and rate limits are acceptable

## Requirements

- Python 3.10+

Depending on provider:
- **Ollama:** [Ollama](https://ollama.com) installed and running
- **huggingface_local:** `pip install transformers torch accelerate` (+ `bitsandbytes` for 8-bit/4-bit quantization)
- **huggingface:** `HF_TOKEN` environment variable

## Setup

**1. Install Python dependencies**
```bash
pip install -r requirements.txt
```

**2a. Ollama setup** (provider: `ollama`)
```bash
# Install from https://ollama.com, then:
ollama serve

# Pull the required models (resumes automatically if interrupted)
ollama pull llama3.1:8b-instruct-q4_0
ollama pull mistral:7b
ollama pull qwen2.5:7b
ollama pull llama3.2:3b-instruct-q4_0   # judge model
```

**2b. HuggingFace local setup** (provider: `huggingface_local`)
```bash
pip install transformers torch accelerate

# For gated models (e.g. Llama), log in once:
huggingface-cli login

# No manual download needed — weights are fetched automatically on first run
# and cached in ~/.cache/huggingface/hub
```

**2c. HuggingFace API setup** (provider: `huggingface`)
```bash
cp .env.example .env
# Edit .env and set: HF_TOKEN=hf_your_token_here
```

## Running

**Choose provider at runtime with `--provider`** (overrides `models.yaml` for all models):
```bash
# Use Ollama (default if not specified)
python scripts/run_experiment.py

# Use HuggingFace local (no rate limits, runs fully offline after first download)
python scripts/run_experiment.py --provider huggingface_local

# Use HuggingFace remote API
python scripts/run_experiment.py --provider huggingface
```

**Run defense variants (Phase 2):**
```bash
# Defense A — strategy-aware prompt hardening
python scripts/run_experiment.py --resume --variant strategy_aware

# Defense B — multilingual system prompts
python scripts/run_experiment.py --resume --variant multilingual

# Defense C — composite English-only anchoring
python scripts/run_experiment.py --resume --variant composite

# Run all defenses sequentially (overnight)
bash scripts/run_all_defenses.sh
```

**Other options:**
```bash
# Dry run — see the experiment plan without making API calls
python scripts/run_experiment.py --dry-run

# Run one model across all behaviors and languages
python scripts/run_experiment.py --models llama31_8b

# Limit scope
python scripts/run_experiment.py --models llama31_8b --behaviors no_homework --languages en fr zh

# Resume after interruption (picks up from where it stopped)
python scripts/run_experiment.py --resume

# Combine flags
python scripts/run_experiment.py --provider huggingface_local --models llama31_8b --dry-run

# Analyze results
python scripts/analyze_results.py
```

The banner printed at startup shows the effective provider for each model:
```
============================================================
EduRobust Experiment Starting
  Models:    ['llama31_8b', 'mistral_7b']
  Provider:  huggingface_local
  Effective: {'llama31_8b': 'huggingface_local', 'mistral_7b': 'huggingface_local'}
  ...
============================================================
```

## Configuration

| File | Purpose |
|---|---|
| `config/config.yaml` | Experiment settings, API provider, evaluation config |
| `config/models.yaml` | Model definitions, enable/disable flags, per-model provider |
| `config/behaviors.yaml` | System prompts and evaluation criteria per behavior |
| `config/languages.yaml` | Languages to test (24 languages across 3 resource tiers) |
| `prompts/attack_templates.yaml` | Attack prompt templates per behavior |

To set a default provider per model, edit the `provider` field in `config/models.yaml`. The `--provider` CLI flag overrides this at runtime without editing any files.

For `huggingface_local` models, additional memory options are available in `models.yaml`:
```yaml
- id: "meta-llama/Llama-3.1-8B-Instruct"
  name: "llama31_8b_hf"
  provider: huggingface_local
  enabled: true
  max_new_tokens: 512
  torch_dtype: "float16"    # "auto" | "float16" | "bfloat16" | "float32"
  load_in_8bit: false       # halves VRAM usage (needs bitsandbytes)
  load_in_4bit: false       # quarters VRAM usage (needs bitsandbytes)
```

## Output

### Raw results

Results are saved incrementally to `results/raw/runs.csv` with columns:

| Column | Description |
|---|---|
| `model` | Target model name (e.g. `llama31_8b`) |
| `judge_model` | Judge model used for evaluation (e.g. `llama3.2:3b-instruct-q4_0`) |
| `behavior_id` | Behavior being tested |
| `prompt_variant` | Defense variant (`baseline`, `strategy_aware`, `multilingual`, `composite`) |
| `language_code` | Language of the attack prompt |
| `language_name` | Full language name |
| `resource_tier` | Language resource tier (`high`, `medium`, `low`) |
| `run_index` | Run index within the cell (0–49) |
| `template_index` | Attack template index (0–4, maps to T0–T4) |
| `attack_template` | English seed template used for this run |
| `translated_prompt` | Final prompt sent to the model (translated if non-English) |
| `model_response` | Raw model response text |
| `asr` | Attack Success Rate: `1.0` = bypass, `0.0` = held, `0.5` = ambiguous |
| `eval_method` | How ASR was determined (`llm_judge`, `keyword`, `langdetect`, etc.) |
| `eval_confidence` | Confidence score from the evaluator |
| `eval_reason` | Explanation from the evaluator |
| `status` | API call outcome (`success`, `api_error`, etc.) |

### Analysis outputs

Running `python scripts/analyze_results.py` generates:

| Output | Description |
|---|---|
| `heatmaps/asr_lang_behavior_all.png` | Heatmap of mean ASR by language and behavior (all models) |
| `heatmaps/asr_lang_behavior_<model>.png` | Per-model heatmap |
| `heatmaps/template_heatmap_<behavior>.png` | Per-behavior heatmap: attack template (T0–T4) × language |
| `bar_charts/asr_by_tier.png` | Mean ASR by resource tier and behavior |
| `bar_charts/model_comparison.png` | Mean ASR by model and behavior |
| `bar_charts/language_ranked.png` | Languages ranked by overall ASR, colored by resource tier |
| `bar_charts/eval_method_usage.png` | Pie chart of evaluation method distribution |
| `summary_stats.csv` | Per-cell aggregated statistics |
| `template_asr.csv` | Per-template bypass rate for every (behavior, language) cell |
| `template_strategy.csv` | Mean ASR per attack strategy (Direct/Urgency/Social/Persona/Override) × behavior |
| `statistical_tests.csv` | Kruskal-Wallis and pairwise Mann-Whitney U tests across resource tiers |

## Project Structure

```
.
├── config/
│   ├── config.yaml           # Experiment settings, API provider, evaluation config
│   ├── models.yaml           # Model definitions, per-model provider
│   ├── behaviors.yaml        # System prompts, defense prompts, and evaluation criteria
│   └── languages.yaml        # 24 languages across 3 resource tiers
├── prompts/
│   ├── attack_templates.yaml # 5 attack templates per behavior (T0–T4)
│   ├── translations/         # Cached translated attack prompts
│   └── defense_system_prompts/  # Translated system prompts for multilingual defense
├── results/
│   ├── raw/runs.csv          # One row per run (68,400 rows across 4 variants)
│   └── analysis/             # Generated charts, stats, and heatmaps
├── logs/                     # Run logs
├── scripts/
│   ├── run_experiment.py     # Main CLI entry point
│   ├── analyze_results.py    # Results analysis and visualization
│   ├── run_all_defenses.sh   # Run all defense variants sequentially
│   ├── translate_prompts.py  # Pre-translate attack prompts
│   └── translate_system_prompts.py  # Translate system prompts for Defense B
└── src/
    ├── experiment_runner.py  # Main orchestration loop (supports 4 variants)
    ├── ollama_client.py      # Ollama local inference client
    ├── hf_client.py          # HuggingFace remote API client
    ├── hf_local_client.py    # HuggingFace local inference client (transformers)
    ├── evaluator.py          # ASR evaluation (LLM judge + keyword fallback)
    ├── analyzer.py           # Results analysis, plots, and statistical tests
    ├── prompt_builder.py     # Prompt construction with {problem} placeholder
    ├── translator.py         # Translation cache (YAML + Google Translate fallback)
    ├── result_store.py       # CSV result persistence with dedup and migration
    └── config_loader.py      # Config loading and validation
```

