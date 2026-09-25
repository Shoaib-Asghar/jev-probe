from dataclasses import FrozenInstanceError

import pytest

from src.models import (
    ChoiceResponse,
    NoulResponse,
    QuestionSpec,
    RunResult,
    ScoreResponse,
)


def test_question_spec_instantiation():
    """Verify QuestionSpec correctly stores attributes."""
    q = QuestionSpec(
        key="is_spam",
        type="noul",
        instructions="Is this spam?",
    )
    assert q.key == "is_spam"
    assert q.type == "noul"
    assert q.instructions == "Is this spam?"
    assert q.criteria is None


def test_noul_response_no_confidence():
    """Verify NoulResponse strictly stores probability and judgment, not confidence."""
    noul = NoulResponse(probability=0.95, judgment=True)
    assert noul.probability == 0.95
    assert noul.judgment is True
    assert not hasattr(noul, "confidence")


def test_choice_and_score_responses():
    """Verify ChoiceResponse and ScoreResponse store distributions and confidences."""
    choice = ChoiceResponse(
        selected="spam",
        probabilities={"spam": 0.8, "ham": 0.2},
        confidence=0.6,
    )
    assert choice.selected == "spam"
    assert choice.confidence == 0.6
    assert choice.probabilities["spam"] == 0.8

    score = ScoreResponse(
        score=4.5,
        distribution={"1-3": 0.1, "4-5": 0.9},
        confidence=0.8,
    )
    assert score.score == 4.5
    assert score.confidence == 0.8


def test_run_result_immutability():
    """Verify RunResult is completely immutable (frozen)."""
    result = RunResult(
        run_id="run_123",
        batch_id="batch_456",
        provider="jev",
        model_version="v1",
        use_case="spam",
        question_key="is_spam",
        original_state="Buy now!",
        perturbed_state="BUY NOW!",
        perturbation_category="A",
        transform_name="uppercase",
        expectation="invariant",
        raw_api_response={"raw": "data"},
        parsed_response=NoulResponse(0.9, True),
        latency_ms=100.0,
        cost_usd=0.001,
        http_status=200,
        retry_count=0,
    )

    with pytest.raises(FrozenInstanceError):
        # Should raise an error because the dataclass is frozen
        result.latency_ms = 200.0  # type: ignore
