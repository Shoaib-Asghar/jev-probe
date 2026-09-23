"""Domain data models (internal representation).

Architectural boundary:
- Plain Python dataclasses representing the core domain concepts:
  QuestionSpec, PerturbedCase, NoulResponse, ChoiceResponse, ScoreResponse,
  RunRecord, and MetricSummary.
- Optimized for internal execution speed and clean static type checking using slotted dataclasses.
- Standard library only (no external dependencies).
"""

from dataclasses import dataclass, field
from typing import Any, Literal

QuestionType = Literal["noul", "choice", "score"]


@dataclass(slots=True)
class QuestionSpec:
    """Specification of an evaluation question submitted to Jev or baseline models.

    Attributes:
        key: Unique identifier for this question within a scenario (e.g., 'is_spam').
        type: Output paradigm:
            - 'noul': Binary judgment with true posterior probability.
            - 'choice': Multi-class categorical selection with probability distribution.
            - 'score': Continuous/numeric evaluation along defined criteria.
        instructions: Natural language prompt or task directive for the question.
        criteria: Optional list or mapping of categorical options or scoring criteria.
    """

    key: str
    type: QuestionType
    instructions: str
    criteria: list[str] | dict[str, str] | None = None


@dataclass(slots=True)
class NoulResponse:
    """Parsed response for a binary 'noul' evaluation from Jev.

    Note: In Noul questions, the calibrated probability IS the confidence.
    There is no separate synthetic confidence field.
    """

    probability: float
    judgment: bool


@dataclass(slots=True)
class ChoiceResponse:
    """Parsed response for a categorical 'choice' evaluation from Jev."""

    selected: str
    probabilities: dict[str, float]
    confidence: float


@dataclass(slots=True)
class ScoreResponse:
    """Parsed response for a numeric/continuous 'score' evaluation from Jev."""

    score: float
    distribution: dict[str, float]
    confidence: float


JevResponse = NoulResponse | ChoiceResponse | ScoreResponse

TestExpectation = Literal[
    "invariant",
    "bounded_drift",
    "graceful_degradation",
    "directional_flip",
]


@dataclass(slots=True)
class PerturbedCase:
    """A test case containing original input, perturbed variation, and expected behavior.

    Expectation-driven testing declares what the model is expected to do BEFORE execution:
    - 'invariant': Classification/judgment mustn't change (e.g., case changes, criteria reordering).
    - 'bounded_drift': Confidence/probability may drift slightly, but core judgment remains stable.
    - 'graceful_degradation': Under heavy noise or corruption, certainty decays toward 0.5.
    - 'directional_flip': Adversarial or boundary stress intentionally targeting a judgment flip.
    """

    original_state: str
    perturbed_state: str
    perturbation_category: str
    transform_name: str
    expectation: TestExpectation
    metadata: dict[str, Any] = field(default_factory=dict)

