"""
ollama_client.py
Ollama local inference client — same .chat() interface as RobustHFClient.
"""

from __future__ import annotations

import logging

import ollama as _ollama

from .hf_client import HFClientError

logger = logging.getLogger(__name__)


class OllamaClient:
    """
    Wraps the Ollama Python client to match RobustHFClient's interface.
    Requires Ollama to be running locally: `ollama serve`
    """

    def __init__(self, model_id: str, host: str = "http://localhost:11434", timeout: int = 120):
        self._model_id = model_id
        self._timeout = timeout
        self._client = _ollama.Client(host=host, timeout=timeout)

    def chat(
        self,
        system_prompt: str,
        user_message: str,
        temperature: float = 0.7,
        max_new_tokens: int = 512,
    ) -> str:
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_message},
        ]
        try:
            response = self._client.chat(
                model=self._model_id,
                messages=messages,
                options={
                    "temperature": temperature,
                    "num_predict": max_new_tokens,
                },
            )
            return response["message"]["content"] or ""
        except Exception as e:
            err_str = str(e).lower()
            if "connection" in err_str or "refused" in err_str:
                raise HFClientError(
                    f"Cannot connect to Ollama. Is it running? Try: ollama serve"
                ) from e
            if "not found" in err_str or "does not exist" in err_str:
                raise HFClientError(
                    f"Model '{self._model_id}' not found in Ollama. "
                    f"Pull it first: ollama pull {self._model_id}"
                ) from e
            raise HFClientError(f"Ollama error for {self._model_id}: {e}") from e
