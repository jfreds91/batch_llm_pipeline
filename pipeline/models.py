"""Pydantic domain models for the CStep pipeline."""

from __future__ import annotations

from typing import Any, Self

from pydantic import BaseModel

ITEM_DATA = "item_data"
METADATA = "metadata"


class StepOutput(BaseModel):
    """Base class for step outputs that flow through the pipeline."""

    @classmethod
    def get_statedict_name(cls) -> str:
        return cls.__name__

    @classmethod
    def from_state_dict(cls, item: dict[str, Any]) -> Self:
        """Deserialize this output from a cumulative item dict."""
        return cls.model_validate(item[cls.get_statedict_name()])

    def to_state_dict(self) -> dict[str, Any]:
        """Serialize this output into a dict keyed by get_statedict_name()."""
        return {self.get_statedict_name(): self.model_dump()}


class Product(StepOutput):
    """Product information container"""

    sku: str
    name: str
    category: str
    material: str
    price: float


class Room(StepOutput):
    """Room information container"""

    name: str
    reasoning: str


class RoomRecommendationResponse(BaseModel):
    """LLM response with one or more Rooms."""
    rooms: list[Room]


class Style(StepOutput):
    """Style information container"""

    name: str
    color_palette: list[str]
    reasoning: str


class StyleRecommendationResponse(BaseModel):
    """LLM response with one or more Styles."""
    styles: list[Style]


class LLMRequest(BaseModel, arbitrary_types_allowed=True):
    """A request to be sent to the LLM."""

    # TODO: instead of a prompt string, we should be using list[types.Part]
    prompt: str
    response_model: type[BaseModel]


class LLMResponse(BaseModel, arbitrary_types_allowed=True):
    """The LLM's response, including token usage."""

    request: LLMRequest
    parsed: BaseModel | None = None
    raw_text: str = ""
    input_tokens: int = 0
    output_tokens: int = 0
    error: str | None = None
