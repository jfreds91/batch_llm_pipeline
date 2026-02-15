"""Step 1: Room recommendation step."""

from __future__ import annotations

import logging
from typing import Any

from pipeline.models import (
    LLMRequest,
    LLMResponse,
    Product,
    Room,
    RoomRecommendationResponse,
)
from pipeline.steps.base_step import BaseStep

logger = logging.getLogger(__name__)


class RoomRecommendationStep(BaseStep):
    """Recommend suitable rooms for each product SKU."""

    def transform(self, items: list[dict[str, Any]]) -> list[LLMRequest]:
        """Ask which rooms suit each product."""
        requests: list[LLMRequest] = []
        for item in items:
            product = Product.from_state_dict(item)
            prompt = (
                f"You are an interior design assistant.\n\n"
                f"Product: {product.name}\n"
                f"Category: {product.category}\n"
                f"Material: {product.material}\n"
                f"Price: ${product.price:.2f}\n\n"
                f"Which rooms in a home would this product be best suited for? "
                f"List 1-3 rooms and explain your reasoning."
            )
            requests.append(
                LLMRequest(
                    prompt=prompt,
                    response_model=RoomRecommendationResponse,
                )
            )
        return requests

    def validate(
        self, items: list[dict[str, Any]], responses: list[LLMResponse]
    ) -> list[tuple[dict[str, Any], Room]]:
        """Fan out into one Room per (sku, room_name) pair."""
        outputs: list[tuple[dict[str, Any], Room]] = []
        for item, response in zip(items, responses):
            product = Product.from_state_dict(item)
            if response.parsed is None:
                logger.warning(
                    "[%s] Skipping %s: LLM error: %s",
                    self.name, product.sku, response.error,
                )
                continue
            rec: RoomRecommendationResponse = response.parsed  # type: ignore[assignment]
            if not rec.rooms:
                logger.warning("[%s] Skipping %s: no rooms returned", self.name, product.sku)
                continue
            for room in rec.rooms:
                outputs.append((item, room))
        return outputs
