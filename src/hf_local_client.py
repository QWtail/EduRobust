"""
hf_local_client.py
HuggingFace local inference client — loads model weights directly onto this machine.
Uses the transformers library. Same .chat() interface as OllamaClient / RobustHFClient.

When to use:
  provider: huggingface_local   — model is downloaded and run locally (no API limits)
  provider: huggingface         — model is called via HF Inference API (rate-limited)
  provider: ollama              — model is served by a local Ollama daemon
"""

from __future__ import annotations

import logging
from typing import Optional

logger = logging.getLogger(__name__)


def _detect_device() -> str:
    """Return the best available device: cuda > mps > cpu."""
    try:
        import torch
        if torch.cuda.is_available():
            return "cuda"
        if torch.backends.mps.is_available():
            return "mps"
    except ImportError:
        pass
    return "cpu"


class HFLocalClient:
    """
    Loads a HuggingFace model locally using the transformers library.
    Model weights are downloaded to ~/.cache/huggingface/hub on first use
    and reused on subsequent runs — no API calls, no rate limits.

    Requirements:
        pip install transformers torch accelerate

    For gated models (e.g. Llama):
        huggingface-cli login   (once, stores token in ~/.cache/huggingface/token)

    Config example (models.yaml):
        - id: "meta-llama/Llama-3.1-8B-Instruct"
          name: "llama31_8b_local"
          provider: huggingface_local
          enabled: true
          max_new_tokens: 512
    """

    def __init__(
        self,
        model_id: str,
        device: Optional[str] = None,
        torch_dtype: str = "auto",
        load_in_8bit: bool = False,
        load_in_4bit: bool = False,
        hf_token: Optional[str] = None,
    ):
        self._model_id = model_id

        try:
            import torch
            from transformers import AutoTokenizer, AutoModelForCausalLM
        except ImportError as e:
            raise ImportError(
                "transformers and torch are required for huggingface_local provider. "
                "Install with: pip install transformers torch accelerate"
            ) from e

        resolved_device = device or _detect_device()
        logger.info(f"[{model_id}] Loading local model on device={resolved_device} ...")

        # Resolve dtype
        dtype_map = {
            "auto": "auto",
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
            "float32": torch.float32,
        }
        resolved_dtype = dtype_map.get(torch_dtype, "auto")

        # Build quantization kwargs
        model_kwargs: dict = {"torch_dtype": resolved_dtype}
        if load_in_4bit:
            model_kwargs["load_in_4bit"] = True
            model_kwargs["device_map"] = "auto"
        elif load_in_8bit:
            model_kwargs["load_in_8bit"] = True
            model_kwargs["device_map"] = "auto"
        else:
            # device_map="auto" handles multi-GPU or single-device placement
            model_kwargs["device_map"] = "auto"

        token_kwargs = {"token": hf_token} if hf_token else {}

        self._tokenizer = AutoTokenizer.from_pretrained(
            model_id, **token_kwargs
        )
        self._model = AutoModelForCausalLM.from_pretrained(
            model_id, **model_kwargs, **token_kwargs
        )
        self._model.eval()

        self._device = resolved_device
        logger.info(f"[{model_id}] Model loaded successfully.")

    def chat(
        self,
        system_prompt: str,
        user_message: str,
        temperature: float = 0.7,
        max_new_tokens: int = 512,
    ) -> str:
        """
        Run a chat completion locally.
        Uses the tokenizer's chat template if available, otherwise falls back
        to a simple system/user text format.
        """
        import torch

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_message},
        ]

        # Apply chat template if the tokenizer supports it
        if hasattr(self._tokenizer, "apply_chat_template") and self._tokenizer.chat_template:
            input_text = self._tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
        else:
            # Simple fallback format
            input_text = (
                f"System: {system_prompt}\n"
                f"User: {user_message}\n"
                f"Assistant:"
            )

        inputs = self._tokenizer(input_text, return_tensors="pt")
        # Move inputs to same device as model
        inputs = {k: v.to(self._model.device) for k, v in inputs.items()}

        gen_kwargs: dict = {
            "max_new_tokens": max_new_tokens,
            "pad_token_id": self._tokenizer.eos_token_id,
        }
        if temperature > 0:
            gen_kwargs["do_sample"] = True
            gen_kwargs["temperature"] = temperature
        else:
            gen_kwargs["do_sample"] = False

        with torch.no_grad():
            output_ids = self._model.generate(**inputs, **gen_kwargs)

        # Decode only the newly generated tokens (exclude the prompt)
        prompt_len = inputs["input_ids"].shape[1]
        new_tokens = output_ids[0][prompt_len:]
        response = self._tokenizer.decode(new_tokens, skip_special_tokens=True)
        return response.strip()
