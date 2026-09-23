"""Schemas layer: Boundary validation and serialization.

Architectural boundary:
- Uses Pydantic to validate external API inputs and outputs at the system boundary.
- Converts validated raw API payloads into internal domain dataclasses via `to_dataclass()`.
- Catches breaking vendor schema changes early with clear ValidationErrors.
"""
from typing import Annotated, Any, Literal

from pydantic import BaseModel, Field

from src.models import ChoiceResponse, JevResponse, NoulResponse, ScoreResponse


class JevUsageSchema(BaseModel):
    """Token usage metadata returned by Jev API."""

    input_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0


class JevNoulAnswerSchema(BaseModel):
    """Schema for binary/boolean 'noul' question responses."""

    type: Literal["noul"] = "noul"
    probability: float = Field(ge=0.0, le=1.0)
    judgment: bool

    def to_dataclass(self) -> NoulResponse:
        """Convert validated schema into internal domain dataclass."""
        return NoulResponse(probability=self.probability, judgment=self.judgment)


class JevChoiceAnswerSchema(BaseModel):
    """Schema for categorical 'choice' question responses."""

    type: Literal["choice"] = "choice"
    selected: str
    probabilities: dict[str, float] = Field(default_factory=dict)
    confidence: float = Field(default=1.0, ge=0.0, le=1.0)

    def to_dataclass(self) -> ChoiceResponse:
        """Convert validated schema into internal domain dataclass."""
        return ChoiceResponse(
            selected=self.selected,
            probabilities=self.probabilities,
            confidence=self.confidence,
        )


class JevScoreAnswerSchema(BaseModel):
    """Schema for numeric/continuous 'score' question responses."""

    type: Literal["score"] = "score"
    score: float
    distribution: dict[str, float] = Field(default_factory=dict)
    confidence: float = Field(default=1.0, ge=0.0, le=1.0)

    def to_dataclass(self) -> ScoreResponse:
        """Convert validated schema into internal domain dataclass."""
        return ScoreResponse(
            score=self.score,
            distribution=self.distribution,
            confidence=self.confidence,
        )


JevAnswerSchema = Annotated[
    JevNoulAnswerSchema | JevChoiceAnswerSchema | JevScoreAnswerSchema,
    Field(discriminator="type"),
]


class JevResponseSchema(BaseModel):
    """Top-level boundary validation schema for Jev API responses."""

    model: str = "jev-1"
    answer: JevAnswerSchema | None = None
    answers: dict[str, JevAnswerSchema] = Field(default_factory=dict)
    usage: JevUsageSchema = Field(default_factory=JevUsageSchema)
    raw_payload: dict[str, Any] = Field(default_factory=dict)

    def to_dataclass(self, question_key: str | None = None) -> JevResponse:
        """Convert response to internal domain dataclass for a target question."""
        if question_key and question_key in self.answers:
            return self.answers[question_key].to_dataclass()
        if self.answer is not None:
            return self.answer.to_dataclass()
        if self.answers:
            return next(iter(self.answers.values())).to_dataclass()
        raise ValueError("No answers found in Jev response payload")
