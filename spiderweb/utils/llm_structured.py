"""Unwrap Pydantic models from ``structured_complete`` results (superglue-compatible shape)."""

import json
from typing import Any, TypeVar

from pydantic import BaseModel

TModel = TypeVar("TModel", bound=BaseModel)


def model_from_structured_complete(result: Any, model_cls: type[TModel]) -> TModel:
    """Return ``model_cls`` instance from a ``structured_complete`` result.

    Expects ``structured_output`` (parsed model or dict) and/or ``final_response``
    (raw JSON string). Falls back to parsing ``final_response`` as JSON
    (including fenced blocks).
    """
    structured = getattr(result, "structured_output", None)
    if isinstance(structured, model_cls):
        return structured
    if isinstance(structured, dict):
        return model_cls(**structured)
    text = getattr(result, "final_response", None)
    text = (str(text) if text is not None else "").strip()
    if not text:
        raise ValueError(f"No structured_output or final_response for {model_cls.__name__}")
    if "```json" in text:
        text = text.split("```json")[1].split("```")[0].strip()
    elif "```" in text:
        text = text.split("```")[1].split("```")[0].strip()
    return model_cls(**json.loads(text))
