"""Perturbation interface and registry.

Architectural boundary:
- Defines the `Perturbation` Protocol (Strategy pattern) for input transformation functions.
- Provides `PerturbationRegistry` to decouple perturbation generators from the runner.
- Zero network calls or external state.
"""

from collections.abc import Callable
from typing import Any, Protocol, runtime_checkable

from src.models import PerturbedCase


@runtime_checkable
class Perturbation(Protocol):
    """Protocol defining the callable contract for all perturbation transform functions.

    Every perturbation transform function must accept a text string and optional keyword
    arguments (e.g., seed, rate) and return a list of typed PerturbedCase objects.
    """

    def __call__(self, text: str, **kwargs: Any) -> list[PerturbedCase]:
        """Generate a list of perturbed test cases from input text."""
        ...


PerturbationCallable = Perturbation | Callable[..., list[PerturbedCase]]


class PerturbationRegistry:
    """Registry mapping category identifiers to perturbation transform strategies."""

    def __init__(self) -> None:
        self._registry: dict[str, list[PerturbationCallable]] = {}

    def register(self, category: str, perturbation: PerturbationCallable) -> None:
        """Register a perturbation strategy under a category identifier."""
        cat_key = category.strip().upper()
        if cat_key not in self._registry:
            self._registry[cat_key] = []
        if perturbation not in self._registry[cat_key]:
            self._registry[cat_key].append(perturbation)

    def get(self, category: str) -> list[PerturbationCallable]:
        """Retrieve all perturbation strategies registered under a category identifier."""
        return self._registry.get(category.strip().upper(), [])

    def categories(self) -> list[str]:
        """Return all registered category identifiers."""
        return sorted(self._registry.keys())

    def generate(self, category: str, text: str, **kwargs: Any) -> list[PerturbedCase]:
        """Execute all registered perturbations for a category and return aggregated cases."""
        cases: list[PerturbedCase] = []
        for transform in self.get(category):
            cases.extend(transform(text, **kwargs))
        return cases



# Global singleton registry instance
registry = PerturbationRegistry()
