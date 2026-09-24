"""Category B: Typographical noise and keyboard perturbation transforms.

Architectural boundary:
- Pure functions simulating real-world human typing errors (adjacent swaps, QWERTY slips, drops).
- Every perturbation generated expects bounded drift: small probability shift without judgment flip.
- No network calls or external state; deterministic seeding via `random.Random`.
"""

import random

from src.models import PerturbedCase

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


def swap_adjacent_chars(text: str, rate: float = 0.05, seed: int = 42) -> str:
    """Swap adjacent characters in words to simulate transposition typos (e.g., 'teh')."""
    if len(text) < 4:
        return text

    rng = random.Random(seed)
    chars = list(text)
    # Valid indices where both i and i+1 are alphanumeric characters
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
    if len(text) < 4:
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
    if len(text) < 4:
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
    if len(text) < 4:
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


def generate_typo_perturbations(
    text: str, rate: float = 0.05, seed: int = 42
) -> list[PerturbedCase]:
    """Generate all Category B typo perturbation variations for a given input text.

    All generated variations declare `expectation='bounded_drift'` because realistic
    character typos should not flip core classification judgments.
    """
    transforms = [
        ("adjacent_swap", swap_adjacent_chars(text, rate=rate, seed=seed)),
        ("dropped_chars", drop_random_chars(text, rate=rate, seed=seed)),
        ("keyboard_typos", keyboard_typos(text, rate=rate, seed=seed)),
        ("duplicate_chars", duplicate_chars(text, rate=rate, seed=seed)),
    ]

    cases: list[PerturbedCase] = []
    for name, transformed_text in transforms:
        cases.append(
            PerturbedCase(
                original_state=text,
                perturbed_state=transformed_text,
                perturbation_category="B",
                transform_name=name,
                expectation="bounded_drift",
                metadata={"seed": seed, "error_rate": rate},
            )
        )
    return cases
