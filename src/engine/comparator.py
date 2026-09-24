"""Comparator engine: empirical metrics calculation.

Architectural boundary:
- Calculates quantitative consistency metrics (flip rate, probability deltas, latency drift).
- Compares baseline RunResults against perturbed RunResults.
- Pure analytical computation; does not invoke network APIs or write to DuckDB directly.
"""

from dataclasses import dataclass, field
from typing import Any

from src.models import ChoiceResponse, NoulResponse, RunResult, ScoreResponse


@dataclass(slots=True)
class CaseComparison:
    """Detailed comparison between a single perturbed result and its baseline."""

    case_run_id: str
    baseline_run_id: str
    perturbation_category: str
    transform_name: str
    expectation: str
    baseline_judgment: Any
    perturbed_judgment: Any
    is_flipped: bool
    prob_delta: float
    baseline_latency_ms: float
    perturbed_latency_ms: float
    cost_usd: float


@dataclass(slots=True)
class ComparisonReport:
    """Aggregated empirical consistency and performance report."""

    question_key: str
    total_cases: int
    total_flips: int
    flip_rate: float
    mean_prob_delta: float
    max_prob_delta: float
    mean_latency_ms: float
    total_cost_usd: float
    cases: list[CaseComparison] = field(default_factory=list)
    category_breakdown: dict[str, dict[str, float]] = field(default_factory=dict)


def extract_judgment_and_probability(response: Any) -> tuple[Any, float]:
    """Extract (judgment, probability_or_confidence) from any typed response."""
    if isinstance(response, NoulResponse):
        return response.judgment, response.probability
    if isinstance(response, ChoiceResponse):
        return response.selected, response.confidence
    if isinstance(response, ScoreResponse):
        return response.score, response.score
    if isinstance(response, dict):
        judgment = response.get("judgment", response.get("selected", response.get("score")))
        prob = float(response.get("probability", response.get("confidence", 0.0)))
        return judgment, prob
    return str(response), 0.0


def compare_results(baseline: RunResult, perturbed: list[RunResult]) -> ComparisonReport:
    """Compare a list of perturbed results against a single baseline result."""
    base_judgment, base_prob = extract_judgment_and_probability(baseline.parsed_response)

    cases: list[CaseComparison] = []
    total_flips = 0
    prob_deltas: list[float] = []
    latencies: list[float] = [baseline.latency_ms]
    total_cost = baseline.cost_usd

    # Tracking by category
    cat_counts: dict[str, int] = {}
    cat_flips: dict[str, int] = {}
    cat_deltas: dict[str, list[float]] = {}

    for r in perturbed:
        p_judgment, p_prob = extract_judgment_and_probability(r.parsed_response)
        is_flipped = p_judgment != base_judgment
        delta = round(abs(p_prob - base_prob), 6)

        if is_flipped:
            total_flips += 1

        prob_deltas.append(delta)
        latencies.append(r.latency_ms)
        total_cost += r.cost_usd

        cat = r.perturbation_category
        cat_counts[cat] = cat_counts.get(cat, 0) + 1
        cat_flips[cat] = cat_flips.get(cat, 0) + (1 if is_flipped else 0)
        cat_deltas.setdefault(cat, []).append(delta)

        cases.append(
            CaseComparison(
                case_run_id=r.run_id,
                baseline_run_id=baseline.run_id,
                perturbation_category=r.perturbation_category,
                transform_name=r.transform_name,
                expectation=r.expectation,
                baseline_judgment=base_judgment,
                perturbed_judgment=p_judgment,
                is_flipped=is_flipped,
                prob_delta=delta,
                baseline_latency_ms=baseline.latency_ms,
                perturbed_latency_ms=r.latency_ms,
                cost_usd=r.cost_usd,
            )
        )

    n = len(perturbed)
    flip_rate = round(total_flips / n, 4) if n > 0 else 0.0
    mean_prob_delta = round(sum(prob_deltas) / n, 6) if n > 0 else 0.0
    max_prob_delta = max(prob_deltas) if prob_deltas else 0.0
    mean_latency = round(sum(latencies) / len(latencies), 2) if latencies else 0.0

    category_breakdown: dict[str, dict[str, float]] = {}
    for cat, count in cat_counts.items():
        flips = cat_flips.get(cat, 0)
        deltas = cat_deltas.get(cat, [])
        category_breakdown[cat] = {
            "total_cases": count,
            "flips": flips,
            "flip_rate": round(flips / count, 4) if count > 0 else 0.0,
            "mean_prob_delta": round(sum(deltas) / len(deltas), 6) if deltas else 0.0,
            "max_prob_delta": max(deltas) if deltas else 0.0,
        }

    return ComparisonReport(
        question_key=baseline.question_key,
        total_cases=n,
        total_flips=total_flips,
        flip_rate=flip_rate,
        mean_prob_delta=mean_prob_delta,
        max_prob_delta=max_prob_delta,
        mean_latency_ms=mean_latency,
        total_cost_usd=round(total_cost, 8),
        cases=cases,
        category_breakdown=category_breakdown,
    )
