"""CStep pipeline package — public API."""

from pipeline.catalog import CATALOG
from pipeline.llm_client import LLMClient
from pipeline.models import (
    ITEM_DATA,
    METADATA,
    StepOutput,
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
    RoomRecommendationStep,
    StyleRecommendationStep,
    load_products,
)

__all__ = [
    "BaseStep",
    "CATALOG",
    "StepOutput",
    "ITEM_DATA",
    "METADATA",
    "LLMClient",
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
