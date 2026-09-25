"""Evaluation execution runner.

Architectural boundary:
- Orchestrates evaluation pipelines across UseCase, adapters, and perturbation categories.
- Yields RunResult records as a generator for streaming consumption and persistence.
- Strictly computational orchestration; contains no metric calculation or UI logic.
"""

import asyncio
import time
import uuid
from collections.abc import AsyncGenerator

from src.adapters.base import BaseAdapter
from src.models import PerturbedCase, QuestionSpec, RunResult, UseCase
from src.perturbations import registry


def get_perturbation_cases(
    category: str,
    text: str,
    question: QuestionSpec | None = None,
    seed: int = 42,
) -> list[PerturbedCase]:
    """Generate perturbed cases dynamically via the PerturbationRegistry (Strategy Pattern)."""
    return registry.generate(category, text, question=question, seed=seed)


async def run_concurrent(
    use_case: UseCase,
    adapter: BaseAdapter,
    categories: list[str] | None = None,
    max_seeds: int | None = None,
    seed: int = 42,
    concurrency: int = 5,
) -> AsyncGenerator[RunResult, None]:
    """Execute an evaluation run concurrently using an asyncio Semaphore."""
    batch_id = f"batch_{int(time.time())}_{uuid.uuid4().hex[:8]}"
    active_categories = categories or use_case.applicable_perturbations or ["A", "B"]
    seeds = use_case.seed_states[:max_seeds] if max_seeds else use_case.seed_states

    import logging
    from dataclasses import replace

    logger = logging.getLogger(__name__)
    semaphore = asyncio.Semaphore(concurrency)
    max_retries = 3

    async def sem_evaluate(case: PerturbedCase, q: QuestionSpec, r_id: str) -> RunResult:
        async with semaphore:
            for attempt in range(max_retries + 1):
                try:
                    res = await adapter.evaluate(
                        case=case,
                        question=q,
                        run_id=r_id,
                        batch_id=batch_id,
                        use_case=use_case.name,
                    )
                    if attempt > 0:
                        return replace(res, retry_count=attempt)
                    return res
                except Exception as e:
                    if attempt == max_retries:
                        logger.error(f"Failed after {max_retries} retries: {e}")
                        raise
                    wait_time = 2 ** attempt
                    logger.warning(
                        f"Evaluation failed (attempt {attempt + 1}/{max_retries}): {e}. "
                        f"Retrying in {wait_time}s..."
                    )
                    await asyncio.sleep(wait_time)

            raise RuntimeError("Unreachable code path in sem_evaluate")

    tasks = []

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
            tasks.append(
                asyncio.create_task(
                    sem_evaluate(baseline_case, question, baseline_run_id)
                )
            )

            # 2. Perturbed calls
            for cat in active_categories:
                perturbed_cases = get_perturbation_cases(
                    cat, seed_text, question=question, seed=seed
                )
                for p_case in perturbed_cases:
                    run_id = f"run_{uuid.uuid4().hex[:12]}"
                    tasks.append(
                        asyncio.create_task(sem_evaluate(p_case, question, run_id))
                    )

    for coro in asyncio.as_completed(tasks):
        yield await coro


def run_sequential(
    use_case: UseCase,
    adapter: BaseAdapter,
    categories: list[str] | None = None,
    max_seeds: int | None = None,
    seed: int = 42,
) -> list[RunResult]:
    """Execute evaluation run blocking (used for CLI tools & compatibility)."""

    from src.config import settings
    async def _run() -> list[RunResult]:
        provider_config = settings.get_provider_config(adapter.provider_name)
        return [
            res
            async for res in run_concurrent(
                use_case=use_case,
                adapter=adapter,
                categories=categories,
                max_seeds=max_seeds,
                seed=seed,
                concurrency=provider_config.concurrency_limit,
            )
        ]

    return asyncio.run(_run())
