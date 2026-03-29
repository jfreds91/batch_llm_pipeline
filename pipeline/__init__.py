"""CStep pipeline package — public API."""

from pipeline.catalog import CATALOG
from pipeline.llm_client import BaseLLMClient, InteractiveAnthropicClient
from pipeline.models import (
    ITEM_DATA,
    METADATA,
    StepOutput,
    Candidate,
    CandidateGenerationResponse,
    LLMRequest,
    LLMResponse,
    Product,
    Room,
    RoomRecommendationResponse,
    Style,
    StyleRecommendationResponse,
)
from pipeline.steps import (
    BaseStep,
    CandidateGenerationStep,
    NeuralCandidateModel,
    RoomRecommendationStep,
    StyleRecommendationStep,
    load_products,
)

__all__ = [
    "BaseStep",
    "CATALOG",
    "Candidate",
    "CandidateGenerationResponse",
    "CandidateGenerationStep",
    "NeuralCandidateModel",
    "StepOutput",
    "ITEM_DATA",
    "METADATA",
    "BaseLLMClient",
    "InteractiveAnthropicClient",
    "LLMRequest",
    "LLMResponse",
    "Product",
    "Room",
    "RoomRecommendationResponse",
    "RoomRecommendationStep",
    "Style",
    "StyleRecommendationResponse",
    "StyleRecommendationStep",
    "load_products",
]
