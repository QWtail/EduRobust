# EduRobust

A research framework for evaluating the robustness of LLM system prompt restrictions in educational contexts against multilingual prompt injection attacks.

## Overview

EduRobust tests whether language models can be prompted — in 20 languages — to bypass behavioral restrictions defined in their system prompts (e.g., "do not do homework for students", "respond in English only"). It measures Attack Success Rate (ASR) across models, behaviors, and languages.

**Behaviors tested:**
| ID | Description |
|---|---|
| `no_homework` | Model should not solve homework directly |
| `math_only` | Model should only answer math questions |
| `hints_only` | Model should give hints, not full answers |
| `no_essay` | Model should not write essays for students |
| `english_only` | Model should only respond in English |

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
| `config/languages.yaml` | Languages to test (20 languages across resource tiers) |
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

Results are saved incrementally to `results/raw/runs.csv` with columns:

| Column | Description |
|---|---|
| `model` | Model name |
| `behavior_id` | Behavior being tested |
| `language_code` | Language of the attack prompt |
| `asr` | Attack Success Rate: `1.0` = bypass, `0.0` = held, `0.5` = ambiguous |
| `eval_method` | How ASR was determined (`llm_judge`, `keyword`, `langdetect`, etc.) |
| `status` | API call outcome (`success`, `api_error`, etc.) |

## Project Structure

```
.
├── config/             # YAML configuration files
├── prompts/            # Attack templates and translated prompts
├── results/            # Experiment outputs (gitignored)
├── logs/               # Run logs (gitignored)
├── scripts/
│   ├── run_experiment.py   # Main entry point
│   ├── analyze_results.py  # Results analysis
│   └── translate_prompts.py
└── src/
    ├── experiment_runner.py  # Main orchestration loop
    ├── ollama_client.py      # Ollama local inference client
    ├── hf_client.py          # HuggingFace remote API client
    ├── hf_local_client.py    # HuggingFace local inference client (transformers)
    ├── evaluator.py          # ASR evaluation (LLM judge + keyword fallback)
    ├── prompt_builder.py     # Prompt construction
    ├── translator.py         # Translation cache
    ├── result_store.py       # CSV result persistence
    └── config_loader.py      # Config loading and validation
```
