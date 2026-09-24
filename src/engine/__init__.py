"""Engine layer: Evaluation orchestration, analytical comparisons, and persistence.

Architectural boundary:
- Coordinates the execution pipeline (runner): sending inputs across selected adapters.
- Computes empirical metrics (comparator): flip rates, probability delta distributions,
  latency differences, and token/financial cost differences.
- Persists run metadata and raw analytical outputs into DuckDB for high-throughput querying.
"""

from src.engine.comparator import compute_distribution_delta, compute_flip_rate, detect_flip
from src.engine.runner import run_evaluation, run_sequential

__all__ = [
    "compute_distribution_delta",
    "compute_flip_rate",
    "detect_flip",
    "run_evaluation",
    "run_sequential",
]

