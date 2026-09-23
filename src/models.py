"""Domain data models (internal representation).

Architectural boundary:
- Plain Python dataclasses representing the core domain concepts:
  QuestionSpec, PerturbedCase, NoulResponse, ChoiceResponse, ScoreResponse,
  RunRecord, and MetricSummary.
- Optimized for internal execution speed and clean static type checking.
"""
