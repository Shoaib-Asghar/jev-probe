"""Engine layer: Evaluation orchestration, analytical comparisons, and persistence.

Architectural boundary:
- Coordinates the execution pipeline (runner): sending inputs across selected adapters.
- Computes empirical metrics (comparator): flip rates, probability delta distributions,
  latency differences, and token/financial cost differences.
- Persists run metadata and raw analytical outputs into DuckDB for high-throughput querying.
"""

from src.engine.comparator import CaseComparison, ComparisonReport, compare_results
from src.engine.runner import run_evaluation

__all__ = ["CaseComparison", "ComparisonReport", "compare_results", "run_evaluation"]


