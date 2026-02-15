"""Abstract base class for LLM clients."""

from __future__ import annotations

from abc import ABC, abstractmethod

from pipeline.models import LLMRequest, LLMResponse


class BaseLLMClient(ABC):
    """Abstract base class defining the interface for LLM clients."""

    @abstractmethod
    def get_llm_responses(self, requests: list[LLMRequest]) -> list[LLMResponse]:
        """Send multiple LLM requests and return responses in the same order.

        Args:
            requests: The LLM requests to send.

        Returns:
            A list of LLMResponses in the same order as the requests.
        """

    @abstractmethod
    def get_token_usage(self) -> int:
        """Return the total number of tokens used across all requests.

        Returns:
            Total token count (input + output).
        """
