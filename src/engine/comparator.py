"""Analytical comparator for empirical consistency and robustness metrics.

Architectural boundary:
- Computes empirical deltas and metrics between baseline RunResults and perturbed RunResults.
- Zero network calls, zero UI rendering logic, standard library only.
- Compares domain objects: NoulResponse, ChoiceResponse, ScoreResponse.
"""

from src.models import ChoiceResponse, NoulResponse, RunResult, ScoreResponse


def detect_flip(original: RunResult, perturbed: RunResult) -> bool:
    """Determine whether the model's core decision flipped between original and perturbed runs.

    Decision boundaries:
    - Noul: Did judgment flip (i.e. did posterior probability cross the 0.5 threshold)?
    - Choice: Did the selected categorical option change?
    - Score: Did the integer-rounded score change?

    Args:
        original: The baseline (unperturbed) RunResult audit record.
        perturbed: The perturbed RunResult audit record.

    Returns:
        True if the classification or discrete score flipped; False otherwise.

    Raises:
        ValueError: If the compared responses belong to incompatible response types.
    """
    orig_resp = original.parsed_response
    pert_resp = perturbed.parsed_response

    if isinstance(orig_resp, NoulResponse) and isinstance(pert_resp, NoulResponse):
        return orig_resp.judgment != pert_resp.judgment

    if isinstance(orig_resp, ChoiceResponse) and isinstance(pert_resp, ChoiceResponse):
        return orig_resp.selected != pert_resp.selected

    if isinstance(orig_resp, ScoreResponse) and isinstance(pert_resp, ScoreResponse):
        return round(orig_resp.score) != round(pert_resp.score)

    if isinstance(orig_resp, dict) and isinstance(pert_resp, dict):
        if "judgment" in orig_resp and "judgment" in pert_resp:
            return bool(orig_resp["judgment"]) != bool(pert_resp["judgment"])
        if "selected" in orig_resp and "selected" in pert_resp:
            return str(orig_resp["selected"]) != str(pert_resp["selected"])
        if "score" in orig_resp and "score" in pert_resp:
            return round(float(orig_resp["score"])) != round(float(pert_resp["score"]))

    msg = (
        f"Incompatible response types for flip detection: "
        f"{type(orig_resp).__name__} vs {type(pert_resp).__name__}"
    )
    raise ValueError(msg)


def compute_flip_rate(pairs: list[tuple[RunResult, RunResult]]) -> float:
    """Compute the empirical flip rate across a list of (original, perturbed) RunResult pairs.

    Args:
        pairs: List of (baseline_run, perturbed_run) tuples.

    Returns:
        A float in the range [0.0, 1.0] representing the proportion of runs that flipped.
        Returns 0.0 if the input list is empty.
    """
    if not pairs:
        return 0.0

    flip_count = sum(1 for original, perturbed in pairs if detect_flip(original, perturbed))
    return round(flip_count / len(pairs), 4)
