"""Perturbations layer: Input transformation engine.

Architectural boundary:
- Pure transformation functions that take an input string or QuestionSpec and generate
  perturbed variations (casing, typo noise, order shuffling, paraphrasing, adversarial text).
- Every perturbation is tagged with an explicit perturbation category (A through G)
  and metadata detailing the exact transform applied.
"""

from src.perturbations.base import (
    Perturbation,
    PerturbationCallable,
    PerturbationRegistry,
    registry,
)
from src.perturbations.ordering import (
    generate_option_order_cases,
    generate_option_permutations,
    shuffle_options,
)
from src.perturbations.surface_noise import (
    drop_random_chars,
    duplicate_chars,
    generate_case_perturbations,
    generate_typo_perturbations,
    keyboard_typos,
    swap_adjacent_chars,
    to_lowercase,
    to_random_case,
    to_titlecase,
    to_uppercase,
    typo_heavy,
    typo_light,
    typo_medium,
)

__all__ = [
    "Perturbation",
    "PerturbationCallable",
    "PerturbationRegistry",
    "drop_random_chars",
    "duplicate_chars",
    "generate_case_perturbations",
    "generate_option_order_cases",
    "generate_option_permutations",
    "generate_typo_perturbations",
    "keyboard_typos",
    "registry",
    "shuffle_options",
    "swap_adjacent_chars",
    "to_lowercase",
    "to_random_case",
    "to_titlecase",
    "to_uppercase",
    "typo_heavy",
    "typo_light",
    "typo_medium",
]

# Register Category A (Surface Noise) perturbation strategies
registry.register("A", generate_case_perturbations)
registry.register("A", generate_typo_perturbations)

# Register Category B (Option Ordering) perturbation strategies
registry.register("B", generate_option_order_cases)
