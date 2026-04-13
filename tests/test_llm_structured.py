"""Tests for structured_complete result unwrapping (superglue-compatible shape)."""

from types import SimpleNamespace

import pytest
from pydantic import BaseModel, Field

from spiderweb.utils.llm_structured import model_from_structured_complete


class _Sample(BaseModel):
    x: int = Field(default=0)


def test_model_from_structured_complete_uses_structured_output_instance():
    plan = _Sample(x=42)
    result = SimpleNamespace(structured_output=plan, final_response="")
    assert model_from_structured_complete(result, _Sample).x == 42


def test_model_from_structured_complete_parses_structured_output_dict():
    result = SimpleNamespace(structured_output={"x": 7}, final_response="")
    assert model_from_structured_complete(result, _Sample).x == 7


def test_model_from_structured_complete_parses_final_response_json():
    result = SimpleNamespace(structured_output=None, final_response='{"x": 99}')
    assert model_from_structured_complete(result, _Sample).x == 99


def test_model_from_structured_complete_raises_when_empty():
    result = SimpleNamespace(structured_output=None, final_response="")
    with pytest.raises(ValueError, match="No structured_output"):
        model_from_structured_complete(result, _Sample)
