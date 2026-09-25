
from src.models import QuestionSpec
from src.perturbations import registry
from src.perturbations.ordering import generate_option_order_cases
from src.perturbations.surface_noise import (
    generate_case_perturbations,
    generate_typo_perturbations,
)


def test_registry_resolution():
    """Verify registry resolves registered functions properly by category."""
    # The registry should yield multiple lists of PerturbedCases
    result_a = registry.generate("A", "Test string")
    assert len(result_a) > 0

    result_b = registry.generate("B", "Test string")
    # Without a question, category B (ordering) usually yields empty list or handles gracefully
    # Let's ensure it doesn't crash at least.
    assert isinstance(result_b, list)


def test_case_perturbations():
    """Verify case change transforms produce expected cases."""
    cases = generate_case_perturbations("Hello World")

    assert len(cases) == 4

    transforms = {c.transform_name for c in cases}
    assert transforms == {"uppercase", "lowercase", "titlecase", "random_case"}

    for case in cases:
        assert case.original_state == "Hello World"
        assert case.perturbation_category == "A"
        assert case.expectation == "invariant"
        assert isinstance(case.perturbed_state, str)
        assert case.perturbed_state != ""


def test_typo_perturbations():
    """Verify typo injections generate perturbed strings with appropriate length constraints."""
    text = "The quick brown fox jumps over the lazy dog."
    cases = generate_typo_perturbations(text)

    assert len(cases) == 7
    transforms = {c.transform_name for c in cases}
    assert transforms == {
        "adjacent_swap",
        "dropped_chars",
        "keyboard_typos",
        "duplicate_chars",
        "typo_light",
        "typo_medium",
        "typo_heavy",
    }

    for case in cases:
        assert case.original_state == text
        assert case.expectation == "invariant"
        # Typoes shouldn't completely obliterate the string
        assert len(case.perturbed_state) > 0


def test_option_order_perturbations():
    """Verify option shuffling works for Choice questions."""
    q = QuestionSpec(
        key="test_q",
        type="choice",
        instructions="Pick one",
        criteria=["apple", "banana", "cherry"],
    )

    cases = generate_option_order_cases("Original text", question=q, seed=42)

    # We should get a few combinations based on max attempts
    assert len(cases) > 0
    for case in cases:
        assert case.perturbation_category == "B"
        assert case.expectation == "invariant"

        # Verify metadata holds the option order
        assert "option_order" in case.metadata

        perm = case.metadata["option_order"]
        assert len(perm) == 3
        # Ensure all original criteria are present
        assert set(perm) == {"apple", "banana", "cherry"}
