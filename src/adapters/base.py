"""Base adapter abstract interface.

Architectural boundary:
- Defines the abstract contract that all model adapters (Jev, LiteLLM, mocks) must implement.
- Enforces Dependency Inversion: the evaluation engine depends on this abstraction,
  never directly on third-party SDKs.
"""

from abc import ABC, abstractmethod

from src.models import PerturbedCase, QuestionSpec, RunResult


class BaseAdapter(ABC):
    """Abstract base class defining the contract for all evaluation model adapters."""

    @property
    @abstractmethod
    def provider_name(self) -> str:
        """Name of the model provider (e.g., 'jev', 'google', 'openai', 'groq')."""
        ...

    @property
    @abstractmethod
    def model_version(self) -> str:
        """Specific model identifier or version string (e.g., 'jev-1', 'gemini-1.5-flash')."""
        ...

    @abstractmethod
    def evaluate(
        self,
        case: PerturbedCase,
        question: QuestionSpec,
        run_id: str,
        batch_id: str,
        use_case: str,
    ) -> RunResult:
        """Evaluate a single perturbed case against this model and return an immutable RunResult.

        Must:
        1. Measure wall-clock latency with high-resolution timer (`time.perf_counter`).
        2. Capture and retain the exact unparsed API response in `raw_api_response`.
        3. Parse the output into a validated domain response (`parsed_response`).
        4. Calculate financial cost via `src.config.settings.get_pricing`.
        5. Return a fully populated, immutable `RunResult`.
        """
        ...
