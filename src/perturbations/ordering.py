"""Category B: Option order shuffle perturbations (Position bias testing).

Architectural boundary:
- Pure transformation functions generating deterministic permutations of categorical options.
- Tests whether model decisions or confidence shift depending on the position of an option
  (e.g., whether the first or last option receives disproportionate probability mass).
- Declares `expectation='invariant'` because option ordering has no semantic
  impact on classification.
- Zero network calls or external state; deterministic seeding via `random.Random`.
"""

import random
from typing import Any

from src.models import PerturbedCase, QuestionSpec


def shuffle_options(question: QuestionSpec, seed: int = 42) -> QuestionSpec:
    """Return a new QuestionSpec with its choice options permuted deterministically.

    Guarantees a non-identity permutation when more than 1 option is present.

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
        if len(keys) > 1 and shuffled_keys == keys:
            shuffled_keys[0], shuffled_keys[1] = shuffled_keys[1], shuffled_keys[0]
        shuffled_criteria = {k: question.criteria[k] for k in shuffled_keys}
    elif isinstance(question.criteria, list):
        items = list(question.criteria)
        shuffled_items = rng.sample(items, len(items))
        if len(items) > 1 and shuffled_items == items:
            shuffled_items[0], shuffled_items[1] = shuffled_items[1], shuffled_items[0]
        shuffled_criteria = shuffled_items
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

    Uses rejection sampling rather than generating all M! permutations in memory,
    guaranteeing O(N) memory complexity and preventing MemoryErrors on questions
    with large option spaces.

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

    original_keys: list[str]
    criteria_dict: dict[str, str] | None = None

    if isinstance(question.criteria, dict):
        criteria_dict = question.criteria
        original_keys = list(question.criteria.keys())
    elif isinstance(question.criteria, list):
        original_keys = list(question.criteria)
    else:
        return []

    if len(original_keys) <= 1:
        return [question]

    rng = random.Random(seed)
    seen: set[tuple[str, ...]] = {tuple(original_keys)}
    permuted_questions: list[QuestionSpec] = []

    # Cap attempts to prevent an infinite loop when total permutations < n (e.g. binary choice)
    max_attempts = n * 50
    attempts = 0

    while len(permuted_questions) < n and attempts < max_attempts:
        attempts += 1
        perm = tuple(rng.sample(original_keys, len(original_keys)))
        if perm in seen:
            continue
        seen.add(perm)

        new_criteria: dict[str, str] | list[str]
        if criteria_dict is not None:
            new_criteria = {k: criteria_dict[k] for k in perm}
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
    question: QuestionSpec | None = None,
    n: int = 5,
    seed: int = 42,
    **kwargs: Any,
) -> list[PerturbedCase]:
    """Generate Category B PerturbedCase representations for option order variations.

    Gracefully returns an empty list if the question is not a Choice question or
    has no criteria options to permute, allowing Category B to run smoothly in
    heterogeneous evaluation runs.

    Args:
        text: Input text/state to be evaluated.
        question: Optional target QuestionSpec. Only Choice questions are permuted.
        n: Number of order permutations to generate.
        seed: Random seed for deterministic permutation generation.

    Returns:
        List of PerturbedCase objects with expectation='invariant'.
    """
    if question is None or question.type != "choice" or not question.criteria:
        return []

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
