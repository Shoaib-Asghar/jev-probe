"""Perturbations layer: Input transformation engine.

Architectural boundary:
- Pure transformation functions that take an input string or QuestionSpec and generate
  perturbed variations (casing, typo noise, order shuffling, paraphrasing, adversarial text).
- Every perturbation is tagged with an explicit perturbation category (A through F)
  and metadata detailing the exact transform applied.
"""
