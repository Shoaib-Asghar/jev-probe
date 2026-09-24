"""Category B: Option order shuffle perturbations (Position bias testing).

Architectural boundary:
- Pure transformation functions generating deterministic permutations of categorical options.
- Tests whether model decisions or confidence shift depending on the position of an option
  (e.g., whether the first or last option receives disproportionate probability mass).
- Declares `expectation='invariant'` because option ordering has no semantic
  impact on classification.
- Zero network calls or external state; deterministic seeding via `random.Random`.
"""

import itertools
import random
from typing import Any

from src.models import PerturbedCase, QuestionSpec


def shuffle_options(question: QuestionSpec, seed: int = 42) -> QuestionSpec:
    """Return a new QuestionSpec with its choice options permuted deterministically.

    Args:
        question: A Choice QuestionSpec containing options in its `criteria` field.
        seed: Random seed for deterministic shuffling.

    Returns:
        A new QuestionSpec with permuted criteria.

    Raises:
        ValueError: If question.type is not 'choice' or criteria is empty.
    """
    if question.type != "choice":
        msg = f"Option order shuffle only applies to 'choice' questions, got '{question.type}'"
        raise ValueError(msg)

    if not question.criteria:
        raise ValueError(f"Question '{question.key}' has no criteria/options to shuffle")

    rng = random.Random(seed)
    shuffled_criteria: dict[str, str] | list[str]

    if isinstance(question.criteria, dict):
        keys = list(question.criteria.keys())
        shuffled_keys = rng.sample(keys, len(keys))
        shuffled_criteria = {k: question.criteria[k] for k in shuffled_keys}
    elif isinstance(question.criteria, list):
        items = list(question.criteria)
        shuffled_criteria = rng.sample(items, len(items))
    else:
        raise ValueError(f"Unsupported criteria type: {type(question.criteria)}")

    return QuestionSpec(
        key=question.key,
        type=question.type,
        instructions=question.instructions,
        criteria=shuffled_criteria,
    )


def generate_option_permutations(
    question: QuestionSpec, n: int = 5, seed: int = 42
) -> list[QuestionSpec]:
    """Generate up to N unique permutations of option ordering for a Choice question.

    Args:
        question: A Choice QuestionSpec.
        n: Number of unique permutations to generate.
        seed: Random seed for deterministic selection.

    Returns:
        A list of QuestionSpec objects, each with a unique option permutation.
    """
    if question.type != "choice":
        msg = f"Option order shuffle only applies to 'choice' questions, got '{question.type}'"
        raise ValueError(msg)

    if not question.criteria:
        return []

    is_dict = isinstance(question.criteria, dict)
    original_keys = list(question.criteria.keys()) if is_dict else list(question.criteria)

    if len(original_keys) <= 1:
        return [question]

    # Generate all possible permutations or sample deterministically
    all_perms = list(itertools.permutations(original_keys))
    # Exclude identity ordering if we have alternatives, then sample
    non_identity = [p for p in all_perms if list(p) != original_keys]
    pool = non_identity if non_identity else all_perms

    rng = random.Random(seed)
    chosen_perms = rng.sample(pool, min(n, len(pool)))

    permuted_questions: list[QuestionSpec] = []
    for perm in chosen_perms:
        new_criteria: dict[str, str] | list[str]
        if is_dict and isinstance(question.criteria, dict):
            new_criteria = {k: question.criteria[k] for k in perm}
        else:
            new_criteria = list(perm)

        permuted_questions.append(
            QuestionSpec(
                key=question.key,
                type=question.type,
                instructions=question.instructions,
                criteria=new_criteria,
            )
        )

    return permuted_questions


def generate_option_order_cases(
    text: str,
    question: QuestionSpec,
    n: int = 5,
    seed: int = 42,
    **kwargs: Any,
) -> list[PerturbedCase]:
    """Generate Category B PerturbedCase representations for option order variations.

    Args:
        text: Input text/state to be evaluated.
        question: The target Choice QuestionSpec.
        n: Number of order permutations to generate.
        seed: Random seed for deterministic permutation generation.

    Returns:
        List of PerturbedCase objects with expectation='invariant'.
    """
    permuted_questions = generate_option_permutations(question, n=n, seed=seed)
    cases: list[PerturbedCase] = []

    for i, p_q in enumerate(permuted_questions):
        order_list: list[str]
        if isinstance(p_q.criteria, dict):
            order_list = list(p_q.criteria.keys())
        elif isinstance(p_q.criteria, list):
            order_list = list(p_q.criteria)
        else:
            order_list = []

        cases.append(
            PerturbedCase(
                original_state=text,
                perturbed_state=text,
                perturbation_category="B",
                transform_name=f"option_order_permutation_{i + 1}",
                expectation="invariant",
                metadata={"option_order": order_list, "question_key": question.key},
            )
        )
    return cases
