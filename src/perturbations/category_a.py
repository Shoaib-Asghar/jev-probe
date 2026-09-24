"""Category A: Case variations and formatting perturbations.

Architectural boundary:
- Pure functions applying deterministic casing transformations to text.
- Every perturbation generated expects invariance: case changes must not alter classification.
- No network calls or external state.
"""

import random

from src.models import PerturbedCase


def to_uppercase(text: str) -> str:
    """Convert input text to uppercase."""
    return text.upper()


def to_lowercase(text: str) -> str:
    """Convert input text to lowercase."""
    return text.lower()


def to_titlecase(text: str) -> str:
    """Convert input text to title case."""
    return text.title()


def to_random_case(text: str, seed: int = 42) -> str:
    """Randomly alternate casing of alphabetic characters using a deterministic seed."""
    rng = random.Random(seed)
    return "".join(c.upper() if rng.random() > 0.5 else c.lower() for c in text)


def generate_case_perturbations(text: str, seed: int = 42) -> list[PerturbedCase]:
    """Generate all Category A case perturbation variations for a given input text.

    All generated variations declare `expectation='invariant'` because case changes
    preserve semantic meaning.
    """
    transforms = [
        ("uppercase", to_uppercase(text), {}),
        ("lowercase", to_lowercase(text), {}),
        ("titlecase", to_titlecase(text), {}),
        ("random_case", to_random_case(text, seed=seed), {"seed": seed}),
    ]

    cases: list[PerturbedCase] = []
    for name, transformed_text, metadata in transforms:
        cases.append(
            PerturbedCase(
                original_state=text,
                perturbed_state=transformed_text,
                perturbation_category="A",
                transform_name=name,
                expectation="invariant",
                metadata=metadata,
            )
        )
    return cases
