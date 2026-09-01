from typing import Literal

from pydantic import BaseModel, Field


class Prediction(BaseModel):
    label: Literal["negative", "neutral", "positive"]
    confidence: float = Field(ge=0.0, le=1.0)
    reason: str = Field(max_length=200)

    model_config = {"extra": "forbid"}  


JSON_SCHEMA = {
    "type": "object",
    "properties": {
        "label": {"type": "string", "enum": ["negative", "neutral", "positive"]},
        "confidence": {"type": "number", "minimum": 0, "maximum": 1},
        "reason": {"type": "string", "maxLength": 200},
    },
    "required": ["label", "confidence", "reason"],
    "additionalProperties": False,
}