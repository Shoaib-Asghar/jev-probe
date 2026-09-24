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


DistributionInput = RunResult | ChoiceResponse | ScoreResponse | NoulResponse | dict[str, float]


def _extract_distribution(obj: DistributionInput) -> dict[str, float]:
    """Extract probability distribution mapping from a supported model response or container."""
    if isinstance(obj, RunResult):
        return _extract_distribution(obj.parsed_response)

    if isinstance(obj, ChoiceResponse):
        return dict(obj.probabilities)

    if isinstance(obj, ScoreResponse):
        return dict(obj.distribution)

    if isinstance(obj, NoulResponse):
        return {"true": obj.probability, "false": round(1.0 - obj.probability, 6)}

    if isinstance(obj, dict):
        if "probabilities" in obj and isinstance(obj["probabilities"], dict):
            return {str(k): float(v) for k, v in obj["probabilities"].items()}
        if "distribution" in obj and isinstance(obj["distribution"], dict):
            return {str(k): float(v) for k, v in obj["distribution"].items()}
        return {str(k): float(v) for k, v in obj.items() if isinstance(v, (int, float))}

    msg = f"Cannot extract distribution from object of type: {type(obj).__name__}"
    raise ValueError(msg)


def compute_distribution_delta(
    original: DistributionInput,
    perturbed: DistributionInput,
) -> dict[str, float]:
    """Compute per-option signed probability delta between baseline and perturbed distributions.

    For each option:
        delta = perturbed_probability - original_probability

    Positive delta indicates the option gained probability mass under perturbation;
    negative delta indicates probability mass was lost.
    Because probability distributions sum to 1.0, the sum of signed deltas equals 0.0.

    Args:
        original: Baseline RunResult, ChoiceResponse, ScoreResponse, NoulResponse, or prob dict.
        perturbed: Perturbed RunResult, ChoiceResponse, ScoreResponse, NoulResponse, or prob dict.

    Returns:
        Mapping of {option: signed_probability_delta}, rounded to 6 decimal places.
    """
    orig_dist = _extract_distribution(original)
    pert_dist = _extract_distribution(perturbed)

    all_keys = sorted(set(orig_dist.keys()) | set(pert_dist.keys()))
    deltas: dict[str, float] = {}

    for k in all_keys:
        orig_val = orig_dist.get(k, 0.0)
        pert_val = pert_dist.get(k, 0.0)
        delta = round(pert_val - orig_val, 6)
        deltas[k] = 0.0 if delta == 0.0 else delta

    return deltas
