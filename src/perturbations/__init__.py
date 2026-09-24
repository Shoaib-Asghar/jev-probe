"""Perturbations layer: Input transformation engine.

Architectural boundary:
- Pure transformation functions that take an input string or QuestionSpec and generate
  perturbed variations (casing, typo noise, order shuffling, paraphrasing, adversarial text).
- Every perturbation is tagged with an explicit perturbation category (A through F)
  and metadata detailing the exact transform applied.
"""

from src.perturbations.category_a import (
    generate_case_perturbations,
    to_lowercase,
    to_random_case,
    to_titlecase,
    to_uppercase,
)

__all__ = [
    "generate_case_perturbations",
    "to_lowercase",
    "to_random_case",
    "to_titlecase",
    "to_uppercase",
]

