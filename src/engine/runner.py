"""Evaluation execution runner.

Architectural boundary:
- Orchestrates evaluation pipelines across UseCase, adapters, and perturbation categories.
- Yields RunResult records as a generator for streaming consumption and persistence.
- Strictly computational orchestration; contains no metric calculation or UI logic.
"""

import time
import uuid
from collections.abc import Generator

from src.adapters.base import BaseAdapter
from src.models import PerturbedCase, RunResult, UseCase
from src.perturbations import generate_case_perturbations, generate_typo_perturbations


def get_perturbation_cases(category: str, text: str, seed: int = 42) -> list[PerturbedCase]:
    """Generate perturbed cases for a specific category code."""
    cat = category.strip().upper()
    if cat == "A":
        return generate_case_perturbations(text, seed=seed)
    if cat == "B":
        return generate_typo_perturbations(text, seed=seed)
    return []


def run_evaluation(
    use_case: UseCase,
    adapter: BaseAdapter,
    categories: list[str] | None = None,
    max_seeds: int | None = None,
    seed: int = 42,
) -> Generator[RunResult, None, None]:
    """Execute an evaluation run across a UseCase and adapter, yielding RunResult records.

    Execution flow for each seed and each question:
    1. Execute the baseline (unperturbed) input first.
    2. Generate and execute perturbed cases across selected categories.
    3. Yield each immutable RunResult immediately upon completion.
    """
    batch_id = f"batch_{int(time.time())}_{uuid.uuid4().hex[:8]}"
    active_categories = categories or use_case.applicable_perturbations or ["A", "B"]
    seeds = use_case.seed_states[:max_seeds] if max_seeds else use_case.seed_states

    for seed_text in seeds:
        for question in use_case.questions:
            # 1. Baseline unperturbed call
            baseline_case = PerturbedCase(
                original_state=seed_text,
                perturbed_state=seed_text,
                perturbation_category="baseline",
                transform_name="identity",
                expectation="invariant",
            )
            baseline_run_id = f"run_{uuid.uuid4().hex[:12]}"
            yield adapter.evaluate(
                case=baseline_case,
                question=question,
                run_id=baseline_run_id,
                batch_id=batch_id,
                use_case=use_case.name,
            )

            # 2. Perturbed calls
            for cat in active_categories:
                perturbed_cases = get_perturbation_cases(cat, seed_text, seed=seed)
                for p_case in perturbed_cases:
                    run_id = f"run_{uuid.uuid4().hex[:12]}"
                    yield adapter.evaluate(
                        case=p_case,
                        question=question,
                        run_id=run_id,
                        batch_id=batch_id,
                        use_case=use_case.name,
                    )
