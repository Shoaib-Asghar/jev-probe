# jev-probe
> Empirical consistency & robustness evaluation harness comparing **Jev** against baseline LLMs under real-world input perturbations.

---

## What is this?
`jev-probe` is an interactive, empirical testbench designed to test the real-world production viability of **Jev** (a purpose-built classification model with calibrated probability outputs) against general-purpose LLMs across real-world edge cases:
- Case variations & typo noise
- Input/criteria order swaps
- Semantic paraphrasing
- Semantic boundary stress & adversarial perturbations
- Token-level and financial cost/latency differences

## Quickstart

### Prerequisites
- Python 3.11+
- [`uv`](https://github.com/astral-sh/uv) package manager

### Environment Setup
```bash
# Sync dependencies and virtual environment
uv sync

# Run code linter
uv run ruff check .
```
