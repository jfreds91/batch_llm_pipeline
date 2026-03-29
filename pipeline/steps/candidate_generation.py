"""Step 3: Candidate generation step using a neural network model."""

from __future__ import annotations

import logging
import random
from typing import Any

from pipeline.models import (
    Candidate,
    CandidateGenerationResponse,
    LLMRequest,
    LLMResponse,
    Product,
    Room,
    Style,
)
from pipeline.steps.base_step import BaseStep

logger = logging.getLogger(__name__)

_TOP_K = 3


class NeuralCandidateModel:
    """Stub neural-network candidate-generation model.

    Internally holds a hardcoded pool of accessory items and returns a
    random top-K selection with mock confidence scores, simulating the
    output of a real embedding-based retrieval model.
    """

    _ITEM_POOL: list[str] = [
        "Velvet Throw Pillow",
        "Brass Candle Holders",
        "Woven Jute Rug",
        "Abstract Wall Art Print",
        "Geometric Terracotta Pot",
        "Rattan Pendant Light",
        "Linen Curtain Panel",
        "Hammered Copper Bowl",
        "Bamboo Tray Set",
        "Slate Coaster Set",
        "Macramé Wall Hanging",
        "Driftwood Sculpture",
        "Pressed Botanical Print",
        "Hand-Thrown Ceramic Mug",
        "Sheepskin Throw Blanket",
    ]

    def predict(self, context_key: str, top_k: int = _TOP_K) -> list[Candidate]:
        """Return top-K candidate items with mock confidence scores.

        Uses a seeded RNG so results are deterministic for a given
        context, as a real model would be.

        Args:
            context_key: A string key identifying the input context
                (e.g. the full prompt describing product/room/style),
                used to seed the RNG.
            top_k: Number of candidates to return.

        Returns:
            A list of Candidates sorted by score descending.
        """
        rng = random.Random(hash(context_key))
        selected = rng.sample(self._ITEM_POOL, min(top_k, len(self._ITEM_POOL)))
        scores = sorted((rng.uniform(0.70, 1.0) for _ in selected), reverse=True)
        return [
            Candidate(name=name, score=round(score, 4))
            for name, score in zip(selected, scores)
        ]


class CandidateGenerationStep(BaseStep):
    """Generate complementary product candidates for each product-room-style triple."""

    def __init__(self, name: str, llm_client, top_k: int = _TOP_K) -> None:
        super().__init__(name, llm_client)
        self._model = NeuralCandidateModel()
        self._top_k = top_k

    def transform(self, items: list[dict[str, Any]]) -> list[LLMRequest]:
        """Build a context descriptor for each product-room-style triple."""
        requests: list[LLMRequest] = []
        for item in items:
            product = Product.from_state_dict(item)
            room = Room.from_state_dict(item)
            style = Style.from_state_dict(item)
            prompt = (
                f"Product: {product.name}\n"
                f"Room: {room.name}\n"
                f"Style: {style.name}\n"
            )
            requests.append(
                LLMRequest(
                    prompt=prompt,
                    response_model=CandidateGenerationResponse,
                )
            )
        return requests

    def request(self, requests: list[LLMRequest]) -> list[LLMResponse]:
        """Score candidates through the NN model instead of an LLM."""
        responses: list[LLMResponse] = []
        for req in requests:
            candidates = self._model.predict(
                context_key=req.prompt,
                top_k=self._top_k,
            )
            responses.append(
                LLMResponse(
                    request=req,
                    parsed=CandidateGenerationResponse(candidates=candidates),
                )
            )
        return responses

    def validate(
        self, items: list[dict[str, Any]], responses: list[LLMResponse]
    ) -> list[tuple[dict[str, Any], Candidate]]:
        """Fan out into one Candidate per (sku, room, style, candidate_name) quad."""
        outputs: list[tuple[dict[str, Any], Candidate]] = []
        for item, response in zip(items, responses):
            product = Product.from_state_dict(item)
            room = Room.from_state_dict(item)
            style = Style.from_state_dict(item)
            if response.parsed is None:
                logger.warning(
                    "[%s] Skipping %s/%s/%s: model error: %s",
                    self.name, product.sku, room.name, style.name, response.error,
                )
                continue
            rec: CandidateGenerationResponse = response.parsed  # type: ignore[assignment]
            if not rec.candidates:
                logger.warning(
                    "[%s] Skipping %s/%s/%s: no candidates returned",
                    self.name, product.sku, room.name, style.name,
                )
                continue
            for candidate in rec.candidates:
                outputs.append((item, candidate))
        return outputs
