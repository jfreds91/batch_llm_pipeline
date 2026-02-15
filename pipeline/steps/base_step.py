"""BaseStep abstract base class for pipeline steps."""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from typing import Any

from pipeline.llm_client import LLMClient
from pipeline.models import ITEM_DATA, METADATA, StepOutput, LLMRequest, LLMResponse

logger = logging.getLogger(__name__)


class BaseStep(ABC):
    """Abstract base class for a pipeline step.

    Each step reads items from state, processes them through
    ingest -> transform -> request -> validate -> end, and writes
    results back to state.
    """

    name: str
    llm_client: LLMClient

    def __init__(self, name: str, llm_client: LLMClient) -> None:
        self.name = name
        self.llm_client = llm_client

    def ingest(self, state: dict[str, Any]) -> list[dict[str, Any]]:
        """Deserialize items from disk if needed. Default is passthrough.
        passes through the payload of ITEM_DATA, but does not expose metadata to transform/validate for now

        Args:
            state: The pipeline state dict.

        Returns:
            A list of item dicts for this step.
        """
        return list(state[ITEM_DATA])

    @abstractmethod
    def transform(self, items: list[dict[str, Any]]) -> list[LLMRequest]:
        """Convert input items into LLM requests. 1:1 with items.

        Each item is a cumulative dict carrying outputs from all prior
        steps, keyed by each output model's ``get_statedict_name()``.
        Use ``from_state_dict()`` to get typed access::

            for item in items:
                room = Room.from_state_dict(item)
                # room.name, room.reasoning, etc. are fully typed

        Args:
            items: Cumulative item dicts. Each dict contains the original
                seed data (e.g. ``{"sku": "..."}``), plus a key per
                upstream step output (e.g. ``"Room": {...}``).

        Returns:
            A list of LLM requests to send.
        """

    def request(self, requests: list[LLMRequest]) -> list[LLMResponse]:
        """Send requests through the LLM client.

        Args:
            requests: The LLM requests to send.

        Returns:
            A list of LLM responses in the same order.
        """
        return self.llm_client.send(requests)

    @abstractmethod
    def validate(
        self, items: list[dict[str, Any]], responses: list[LLMResponse]
    ) -> list[tuple[dict[str, Any], StepOutput]]:
        """Validate responses and produce step outputs.

        Each element is a (source_item, output) pair. The source item
        carries the cumulative state up to this point. Fan-out happens
        here: one source item can produce multiple outputs.

        Invalid results should be discarded with a logging.warning,
        not raised as exceptions. Users can choose to instead log them
        somewhere for a retry later, but not in this pipeline

        Args:
            items: The original input items (parallel with responses).
            responses: The LLM responses (parallel with items).

        Returns:
            A list of (source_item, step_output) pairs.
        """

    def end(
        self,
        validated: list[tuple[dict[str, Any], StepOutput]],
    ) -> dict[str, Any]:
        """Merge each output onto its source item under get_statedict_name().

        Args:
            validated: The (source_item, output) pairs from validate.

        Returns:
            A partial state dict update.
        """
        items: list[dict[str, Any]] = []
        for source_item, output in validated:
            new_item = {
                **source_item,
                **output.to_state_dict(),
            }
            items.append(new_item)
        output_keys = sorted(items[0].keys()) if items else []
        logger.info(
            "[%s] end: %d items, keys: %s",
            self.name, len(items), output_keys,
        )
        return {ITEM_DATA: items}

    def __call__(self, state: dict[str, Any]) -> dict[str, Any]:
        """Execute the full step pipeline.

        Args:
            state: The pipeline state dict.

        Returns:
            A partial state dict update.
        """
        logger.info("[%s] === starting ===", self.name)

        items = self.ingest(state)
        input_keys = sorted(items[0].keys()) if items else []
        logger.info("[%s] ingest: %d items, keys: %s", self.name, len(items), input_keys)

        requests = self.transform(items)
        if len(requests) != len(items):
            raise ValueError(
                f"[{self.name}] transform must return 1:1 with items: "
                f"got {len(requests)} requests for {len(items)} items"
            )
        logger.info("[%s] transform: %d requests", self.name, len(requests))

        responses = self.request(requests)
        logger.info("[%s] request: %d responses", self.name, len(responses))

        validated = self.validate(items, responses)
        logger.info("[%s] validate: %d outputs", self.name, len(validated))

        result = self.end(validated)
        result[METADATA] = state[METADATA]
        logger.info("[%s] === done ===", self.name)
        return result
