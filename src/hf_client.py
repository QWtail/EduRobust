"""
hf_client.py
HuggingFace InferenceClient wrapper with exponential backoff and error handling.
"""

from __future__ import annotations

import logging
import time
from typing import Optional

from huggingface_hub import InferenceClient
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
    before_sleep_log,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Custom exceptions
# ---------------------------------------------------------------------------

class HFClientError(Exception):
    """Base exception for HuggingFace client errors."""


class HFRateLimitError(HFClientError):
    """Raised when the API returns a 429 rate limit response."""


class HFModelUnavailableError(HFClientError):
    """Raised when the model is loading (503) or temporarily unavailable."""


# ---------------------------------------------------------------------------
# Client
# ---------------------------------------------------------------------------

class RobustHFClient:
    """
    Wraps HuggingFace InferenceClient with:
    - Automatic exponential backoff on rate limits (tenacity)
    - Fixed-wait retry on model loading (503)
    - Structured error classification
    """

    def __init__(
        self,
        token: str,
        model_id: str,
        timeout: int = 60,
        max_retries: int = 5,
        base_backoff: float = 2.0,
        model_loading_wait: int = 30,
    ):
        self._client = InferenceClient(model=model_id, token=token, timeout=timeout)
        self._model_id = model_id
        self._max_retries = max_retries
        self._base_backoff = base_backoff
        self._model_loading_wait = model_loading_wait

    def chat(
        self,
        system_prompt: str,
        user_message: str,
        temperature: float = 0.7,
        max_new_tokens: int = 512,
    ) -> str:
        """
        Send a chat completion request with automatic retry on rate limits.
        Raises HFClientError subclasses on non-recoverable errors.
        """
        return self._chat_with_retry(
            system_prompt=system_prompt,
            user_message=user_message,
            temperature=temperature,
            max_new_tokens=max_new_tokens,
        )

    @retry(
        stop=stop_after_attempt(5),
        wait=wait_exponential(multiplier=2, min=2, max=64),
        retry=retry_if_exception_type(HFRateLimitError),
        before_sleep=before_sleep_log(logger, logging.WARNING),
        reraise=True,
    )
    def _chat_with_retry(
        self,
        system_prompt: str,
        user_message: str,
        temperature: float,
        max_new_tokens: int,
    ) -> str:
        """Internal method with tenacity retry decorator for rate limits."""
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_message},
        ]

        loading_retries = 0
        max_loading_retries = 3

        while True:
            try:
                response = self._client.chat_completion(
                    messages=messages,
                    temperature=temperature,
                    max_tokens=max_new_tokens,
                )
                return response.choices[0].message.content or ""

            except Exception as e:
                err_str = str(e).lower()

                if "rate limit" in err_str or "429" in err_str:
                    logger.warning(
                        f"[{self._model_id}] Rate limit hit. Triggering backoff..."
                    )
                    raise HFRateLimitError(str(e))

                elif "503" in err_str or "loading" in err_str or "currently loading" in err_str:
                    loading_retries += 1
                    if loading_retries > max_loading_retries:
                        raise HFModelUnavailableError(
                            f"Model {self._model_id} still loading after "
                            f"{max_loading_retries} retries."
                        )
                    logger.warning(
                        f"[{self._model_id}] Model loading (503). "
                        f"Waiting {self._model_loading_wait}s "
                        f"(attempt {loading_retries}/{max_loading_retries})..."
                    )
                    time.sleep(self._model_loading_wait)
                    # Continue the while loop to retry

                else:
                    raise HFClientError(
                        f"Unexpected error from {self._model_id}: {e}"
                    ) from e


def build_clients(
    model_ids: list[str],
    token: str,
    timeout: int = 60,
    max_retries: int = 5,
    base_backoff: float = 2.0,
    model_loading_wait: int = 30,
) -> dict[str, RobustHFClient]:
    """Build a dict of model_id -> RobustHFClient for all given model IDs."""
    return {
        model_id: RobustHFClient(
            token=token,
            model_id=model_id,
            timeout=timeout,
            max_retries=max_retries,
            base_backoff=base_backoff,
            model_loading_wait=model_loading_wait,
        )
        for model_id in model_ids
    }
