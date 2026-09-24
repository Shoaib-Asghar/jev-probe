"""Use cases layer: Domain scenarios and seed datasets.

Architectural boundary:
- Pre-configured domain templates (e.g., spam filtering, customer intent, sentiment/escalation).
- Defines baseline inputs, gold labels (if available), and relevant QuestionSpecs tailored
  for empirical stress-testing.
"""

from src.use_cases.spam import get_spam_use_case

__all__ = ["get_spam_use_case"]

