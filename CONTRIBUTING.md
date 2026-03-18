# Contributing to EduRobust

Thank you for your interest in contributing! EduRobust is a research framework for evaluating multilingual robustness of LLM system prompt restrictions.

## Bug Reports & Feature Requests

Open an issue at https://github.com/QWtail/EduRobust/issues. Please include:
- Python version and OS
- Provider used (ollama / huggingface_local / huggingface)
- Minimal reproduction steps and error output

## Adding Languages

1. Edit `config/languages.yaml` — add a new entry with `code`, `name`, `resource_tier`, `script`, and `deep_translator_code` (must be a valid [Google Translate language code](https://cloud.google.com/translate/docs/languages))
2. Verify with a dry run: `python scripts/run_experiment.py --languages <new_code> --dry-run`
3. Run translation pre-cache: `python scripts/run_experiment.py --languages <new_code> --resume`

## Adding Behaviors

1. Add a new entry to `config/behaviors.yaml` with:
   - `id`, `name`, `system_prompt`, `refusal_keywords`, `bypass_indicators`
2. Add exactly 5 attack templates to `prompts/attack_templates.yaml` under the new behavior ID, following the T0–T4 strategy convention (Direct / Urgency / Social / Persona / Override)

## Code Style

- Python 3.10+, type hints used throughout
- Run `pip install ruff && ruff check src/` before submitting a pull request
- Keep docstrings on all public classes and methods

## Pull Request Process

1. Fork the repo and create a feature branch
2. Make your changes with clear commit messages
3. Run `python scripts/run_experiment.py --dry-run` to confirm the framework still loads cleanly
4. Open a pull request against `main` with a description of your changes

## License

By contributing, you agree that your contributions will be licensed under the [MIT License](LICENSE).
