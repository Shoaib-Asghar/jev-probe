"""Analytical comparator for empirical consistency and robustness metrics.

Architectural boundary:
- Computes empirical deltas and metrics between baseline RunResults and perturbed RunResults.
- Zero network calls, zero UI rendering logic, standard library only.
- Compares domain objects: NoulResponse, ChoiceResponse, ScoreResponse.
"""

from dataclasses import dataclass

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


def compute_margin(distribution: dict[str, float]) -> float:
    """Compute the probability margin between the top-1 and top-2 candidate options.

    Args:
        distribution: Dictionary mapping options/categories to probabilities.

    Returns:
        The difference P(top-1) - P(top-2), rounded to 6 decimal places.
        Returns the top-1 probability if only 1 option exists, or 0.0 if empty.
    """
    if not distribution:
        return 0.0

    sorted_probs = sorted(distribution.values(), reverse=True)
    top1 = sorted_probs[0]
    top2 = sorted_probs[1] if len(sorted_probs) > 1 else 0.0
    return round(top1 - top2, 6)


def _extract_confidence(obj: DistributionInput) -> float:
    """Extract scalar confidence or max posterior probability from a model response."""
    if isinstance(obj, RunResult):
        return _extract_confidence(obj.parsed_response)

    if isinstance(obj, ChoiceResponse):
        return obj.confidence

    if isinstance(obj, ScoreResponse):
        return obj.confidence

    if isinstance(obj, NoulResponse):
        return max(obj.probability, round(1.0 - obj.probability, 6))

    if isinstance(obj, dict):
        if "confidence" in obj and isinstance(obj["confidence"], (int, float)):
            return float(obj["confidence"])
        dist = _extract_distribution(obj)
        return max(dist.values()) if dist else 1.0

    msg = f"Cannot extract confidence from object of type: {type(obj).__name__}"
    raise ValueError(msg)


def compute_confidence_drift(
    original: DistributionInput,
    perturbed: DistributionInput,
) -> float:
    """Compute scalar confidence delta (perturbed_confidence - original_confidence).

    A negative drift indicates that the model's certainty decayed under perturbation.

    Args:
        original: Baseline RunResult, ChoiceResponse, ScoreResponse, or probability container.
        perturbed: Perturbed RunResult, ChoiceResponse, ScoreResponse, or probability container.

    Returns:
        Signed confidence drift rounded to 6 decimal places.
    """
    orig_conf = _extract_confidence(original)
    pert_conf = _extract_confidence(perturbed)
    drift = round(pert_conf - orig_conf, 6)
    return 0.0 if drift == 0.0 else drift


@dataclass(slots=True)
class ComparisonResult:
    """Bundled analytical metrics comparing a baseline run against a perturbed run.

    Attributes:
        flipped: True if the model's categorical judgment, binary label, or rounded score changed.
        distribution_delta: Per-option signed probability difference (perturbed - original).
        confidence_delta: Scalar shift in top confidence/certainty (perturbed - original).
        margin_original: Gap between top-1 and top-2 probabilities in baseline distribution.
        margin_perturbed: Gap between top-1 and top-2 probabilities in perturbed distribution.
        margin_collapse: Margin reduction (margin_original - margin_perturbed).
    """

    flipped: bool
    distribution_delta: dict[str, float]
    confidence_delta: float
    margin_original: float
    margin_perturbed: float
    margin_collapse: float = 0.0


def compare_runs(original: RunResult, perturbed: RunResult) -> ComparisonResult:
    """Perform a comprehensive level 1-3 metric comparison between baseline and perturbed runs."""
    flipped = detect_flip(original, perturbed)
    dist_delta = compute_distribution_delta(original, perturbed)
    conf_delta = compute_confidence_drift(original, perturbed)

    orig_dist = _extract_distribution(original)
    pert_dist = _extract_distribution(perturbed)
    margin_orig = compute_margin(orig_dist)
    margin_pert = compute_margin(pert_dist)
    margin_collapse = round(margin_orig - margin_pert, 6)

    return ComparisonResult(
        flipped=flipped,
        distribution_delta=dist_delta,
        confidence_delta=conf_delta,
        margin_original=margin_orig,
        margin_perturbed=margin_pert,
        margin_collapse=margin_collapse,
    )
