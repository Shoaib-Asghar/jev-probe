"""Category A: Surface noise perturbations (casing variations and typographical noise).

Architectural boundary:
- Pure transformation functions generating deterministic casing and typo noise.
- Every perturbation generated declares `expectation='invariant'` because surface noise
  preserves the underlying semantic truth.
- Zero network calls or external state; deterministic seeding via `random.Random`.
"""

import random
from typing import Any

from src.models import PerturbedCase

# QWERTY keyboard adjacency mapping for realistic keyboard typo simulation
QWERTY_NEIGHBORS: dict[str, list[str]] = {
    "q": ["w", "a", "s"],
    "w": ["q", "e", "a", "s", "d"],
    "e": ["w", "r", "s", "d", "f"],
    "r": ["e", "t", "d", "f", "g"],
    "t": ["r", "y", "f", "g", "h"],
    "y": ["t", "u", "g", "h", "j"],
    "u": ["y", "i", "h", "j", "k"],
    "i": ["u", "o", "j", "k", "l"],
    "o": ["i", "p", "k", "l"],
    "p": ["o", "l"],
    "a": ["q", "w", "s", "z"],
    "s": ["w", "e", "a", "d", "z", "x"],
    "d": ["e", "r", "s", "f", "x", "c"],
    "f": ["r", "t", "d", "g", "c", "v"],
    "g": ["t", "y", "f", "h", "v", "b"],
    "h": ["y", "u", "g", "j", "b", "n"],
    "j": ["u", "i", "h", "k", "n", "m"],
    "k": ["i", "o", "j", "l", "m"],
    "l": ["o", "p", "k"],
    "z": ["a", "s", "x"],
    "x": ["z", "s", "d", "c"],
    "c": ["x", "d", "f", "v"],
    "v": ["c", "f", "g", "b"],
    "b": ["v", "g", "h", "n"],
    "n": ["b", "h", "j", "m"],
    "m": ["n", "j", "k"],
}


# --- Case Variation Transforms (Step 0.15) ---


def to_uppercase(text: str) -> str:
    """Convert input text to uppercase."""
    return text.upper()


def to_lowercase(text: str) -> str:
    """Convert input text to lowercase."""
    return text.lower()


def to_titlecase(text: str) -> str:
    """Convert input text to title case."""
    return text.title()


def to_random_case(text: str, seed: int = 42) -> str:
    """Randomly alternate casing of alphabetic characters using a deterministic seed."""
    rng = random.Random(seed)
    return "".join(c.upper() if rng.random() > 0.5 else c.lower() for c in text)


def generate_case_perturbations(
    text: str, seed: int = 42, **kwargs: Any
) -> list[PerturbedCase]:
    """Generate all Category A case perturbation variations for a given input text."""
    transforms = [
        ("uppercase", to_uppercase(text), {}),
        ("lowercase", to_lowercase(text), {}),
        ("titlecase", to_titlecase(text), {}),
        ("random_case", to_random_case(text, seed=seed), {"seed": seed}),
    ]

    cases: list[PerturbedCase] = []
    for name, transformed_text, metadata in transforms:
        cases.append(
            PerturbedCase(
                original_state=text,
                perturbed_state=transformed_text,
                perturbation_category="A",
                transform_name=name,
                expectation="invariant",
                metadata=metadata,
            )
        )
    return cases


# --- Typographical Noise Transforms (Step 0.16) ---


def swap_adjacent_chars(text: str, rate: float = 0.05, seed: int = 42) -> str:
    """Swap adjacent characters in words to simulate transposition typos (e.g., 'teh')."""
    if len(text) < 2:
        return text

    rng = random.Random(seed)
    chars = list(text)
    valid_indices = [
        i for i in range(len(chars) - 1) if chars[i].isalnum() and chars[i + 1].isalnum()
    ]
    if not valid_indices:
        return text

    count = max(1, int(len(valid_indices) * rate))
    swap_points = sorted(rng.sample(valid_indices, min(count, len(valid_indices))))
    for i in swap_points:
        chars[i], chars[i + 1] = chars[i + 1], chars[i]
    return "".join(chars)


def drop_random_chars(text: str, rate: float = 0.05, seed: int = 42) -> str:
    """Drop random alphanumeric characters to simulate omitted keystrokes."""
    if len(text) < 1:
        return text

    rng = random.Random(seed)
    valid_indices = [i for i, c in enumerate(text) if c.isalnum()]
    if not valid_indices:
        return text

    count = max(1, int(len(valid_indices) * rate))
    indices_to_drop = set(rng.sample(valid_indices, min(count, len(valid_indices))))
    return "".join(c for i, c in enumerate(text) if i not in indices_to_drop)


def duplicate_chars(text: str, rate: float = 0.05, seed: int = 42) -> str:
    """Duplicate characters to simulate key chatter or bouncing."""
    if len(text) < 1:
        return text

    rng = random.Random(seed)
    valid_indices = [i for i, c in enumerate(text) if c.isalnum()]
    if not valid_indices:
        return text

    count = max(1, int(len(valid_indices) * rate))
    indices_to_dup = set(rng.sample(valid_indices, min(count, len(valid_indices))))
    result: list[str] = []
    for i, c in enumerate(text):
        result.append(c)
        if i in indices_to_dup:
            result.append(c)
    return "".join(result)


def keyboard_typos(text: str, rate: float = 0.05, seed: int = 42) -> str:
    """Replace characters with adjacent keys on a standard QWERTY keyboard layout."""
    if len(text) < 1:
        return text

    rng = random.Random(seed)
    valid_indices = [i for i, c in enumerate(text) if c.lower() in QWERTY_NEIGHBORS]
    if not valid_indices:
        return text

    count = max(1, int(len(valid_indices) * rate))
    targets = rng.sample(valid_indices, min(count, len(valid_indices)))

    chars = list(text)
    for i in targets:
        char = chars[i]
        neighbors = QWERTY_NEIGHBORS.get(char.lower(), [])
        if neighbors:
            replacement = rng.choice(neighbors)
            chars[i] = replacement.upper() if char.isupper() else replacement
    return "".join(chars)


# Intensity presets specified in Step 0.16
def typo_light(text: str, seed: int = 42) -> str:
    """Light typo noise (1-2 typos per sentence, ~3% rate)."""
    return keyboard_typos(text, rate=0.03, seed=seed)


def typo_medium(text: str, seed: int = 42) -> str:
    """Medium typo noise (3-5 typos per sentence, ~7% rate)."""
    return keyboard_typos(text, rate=0.07, seed=seed)


def typo_heavy(text: str, seed: int = 42) -> str:
    """Heavy typo noise (multiple typos per word, ~15% rate)."""
    return keyboard_typos(text, rate=0.15, seed=seed)


def generate_typo_perturbations(
    text: str, rate: float = 0.05, seed: int = 42, **kwargs: Any
) -> list[PerturbedCase]:
    """Generate all Category A typo noise perturbation variations for a given input text."""
    transforms = [
        ("adjacent_swap", swap_adjacent_chars(text, rate=rate, seed=seed), {"rate": rate}),
        ("dropped_chars", drop_random_chars(text, rate=rate, seed=seed), {"rate": rate}),
        ("keyboard_typos", keyboard_typos(text, rate=rate, seed=seed), {"rate": rate}),
        ("duplicate_chars", duplicate_chars(text, rate=rate, seed=seed), {"rate": rate}),
        ("typo_light", typo_light(text, seed=seed), {"preset": "light"}),
        ("typo_medium", typo_medium(text, seed=seed), {"preset": "medium"}),
        ("typo_heavy", typo_heavy(text, seed=seed), {"preset": "heavy"}),
    ]

    cases: list[PerturbedCase] = []
    for name, transformed_text, metadata in transforms:
        meta = {"seed": seed, **metadata}
        cases.append(
            PerturbedCase(
                original_state=text,
                perturbed_state=transformed_text,
                perturbation_category="A",
                transform_name=name,
                expectation="invariant",
                metadata=meta,
            )
        )
    return cases
