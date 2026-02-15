"""Step 2: Style recommendation step."""

from __future__ import annotations

import logging
from typing import Any

from pipeline.models import (
    LLMRequest,
    LLMResponse,
    Product,
    Room,
    Style,
    StyleRecommendationResponse,
)
from pipeline.steps.base_step import BaseStep

logger = logging.getLogger(__name__)


class StyleRecommendationStep(BaseStep):
    """Recommend interior design styles for each product-room pair."""

    def transform(self, items: list[dict[str, Any]]) -> list[LLMRequest]:
        """Ask which design styles suit each product-room pair."""
        requests: list[LLMRequest] = []
        for item in items:
            product = Product.from_state_dict(item)
            room = Room.from_state_dict(item)
            prompt = (
                f"You are an interior design assistant.\n\n"
                f"Product: {product.name}\n"
                f"Material: {product.material}\n"
                f"Room: {room.name}\n\n"
                f"Suggest 1-3 interior design styles that would complement "
                f"this {product.name} ({product.material}) in a {room.name}. "
                f"Explain your reasoning."
            )
            requests.append(
                LLMRequest(
                    prompt=prompt,
                    response_model=StyleRecommendationResponse,
                )
            )
        return requests

    def validate(
        self, items: list[dict[str, Any]], responses: list[LLMResponse]
    ) -> list[tuple[dict[str, Any], Style]]:
        """Fan out into one Style per (sku, room, style_name) triple."""
        outputs: list[tuple[dict[str, Any], Style]] = []
        for item, response in zip(items, responses):
            product = Product.from_state_dict(item)
            room = Room.from_state_dict(item)
            if response.parsed is None:
                logger.warning(
                    "[%s] Skipping %s/%s: LLM error: %s",
                    self.name, product.sku, room.name, response.error,
                )
                continue
            rec: StyleRecommendationResponse = response.parsed  # type: ignore[assignment]
            if not rec.styles:
                logger.warning(
                    "[%s] Skipping %s/%s: no styles returned",
                    self.name, product.sku, room.name,
                )
                continue
            for style in rec.styles:
                outputs.append((item, style))
        return outputs
