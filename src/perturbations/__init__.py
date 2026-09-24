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
from src.perturbations.category_b import (
    drop_random_chars,
    duplicate_chars,
    generate_typo_perturbations,
    keyboard_typos,
    swap_adjacent_chars,
)

__all__ = [
    "drop_random_chars",
    "duplicate_chars",
    "generate_case_perturbations",
    "generate_typo_perturbations",
    "keyboard_typos",
    "swap_adjacent_chars",
    "to_lowercase",
    "to_random_case",
    "to_titlecase",
    "to_uppercase",
]


