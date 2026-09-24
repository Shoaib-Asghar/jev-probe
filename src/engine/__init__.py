"""Engine layer: Evaluation orchestration, analytical comparisons, and persistence.

Architectural boundary:
- Coordinates the execution pipeline (runner): sending inputs across selected adapters.
- Computes empirical metrics (comparator): flip rates, probability delta distributions,
  latency differences, and token/financial cost differences.
- Persists run metadata and raw analytical outputs into DuckDB for high-throughput querying.
"""

from src.engine.comparator import (
    DEFAULT_THRESHOLDS,
    CategoryThresholds,
    CategoryVerdict,
    ComparisonResult,
    compare_runs,
    compute_confidence_drift,
    compute_distribution_delta,
    compute_flip_rate,
    compute_margin,
    detect_flip,
    judge_category,
)
from src.engine.runner import run_evaluation, run_sequential

__all__ = [
    "CategoryThresholds",
    "CategoryVerdict",
    "ComparisonResult",
    "DEFAULT_THRESHOLDS",
    "compare_runs",
    "compute_confidence_drift",
    "compute_distribution_delta",
    "compute_flip_rate",
    "compute_margin",
    "detect_flip",
    "judge_category",
    "run_evaluation",
    "run_sequential",
]

