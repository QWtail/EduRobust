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

**Models supported:** Any model available via [Ollama](https://ollama.com) (default: Llama 3.1 8B, Mistral 7B, Qwen 2.5 7B) or HuggingFace Inference API.

## Requirements

- Python 3.10+
- [Ollama](https://ollama.com) (for local inference)

## Setup

**1. Install Python dependencies**
```bash
pip install -r requirements.txt
```

**2. Install and start Ollama**
```bash
# Install from https://ollama.com, then:
ollama serve

# Pull the models
ollama pull llama3.1:8b-instruct-q4_0
ollama pull mistral:7b
ollama pull qwen2.5:7b
```

**3. (Optional) HuggingFace models**

If using HuggingFace-hosted models instead of Ollama, set your token:
```bash
export HF_TOKEN=hf_your_token_here
# or copy .env.example to .env and fill in your token
```

## Running

```bash
# Dry run — see the experiment plan without making API calls
python scripts/run_experiment.py --dry-run

# Run one model across all behaviors and languages
python scripts/run_experiment.py --models llama31_8b

# Run multiple models
python scripts/run_experiment.py --models llama31_8b mistral_7b

# Limit scope
python scripts/run_experiment.py --models llama31_8b --behaviors no_homework --languages en fr zh

# Resume after interruption (picks up from where it stopped)
python scripts/run_experiment.py --models llama31_8b --resume

# Restart from scratch (ignores existing results)
python scripts/run_experiment.py --models llama31_8b

# Analyze results
python scripts/analyze_results.py
```

## Configuration

| File | Purpose |
|---|---|
| `config/config.yaml` | Experiment settings, API provider, evaluation config |
| `config/models.yaml` | Model definitions and enable/disable flags |
| `config/behaviors.yaml` | System prompts and evaluation criteria per behavior |
| `config/languages.yaml` | Languages to test (20 languages across resource tiers) |
| `prompts/attack_templates.yaml` | Attack prompt templates per behavior |

To switch between Ollama and HuggingFace, change `provider` in `config/config.yaml` and update model IDs in `config/models.yaml`.

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
    ├── hf_client.py          # HuggingFace inference client
    ├── evaluator.py          # ASR evaluation (LLM judge + keyword fallback)
    ├── prompt_builder.py     # Prompt construction
    ├── translator.py         # Translation cache
    ├── result_store.py       # CSV result persistence
    └── config_loader.py      # Config loading and validation
```
