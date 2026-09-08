"""Readable RNA helix selection and dot-bracket conversion.

This module intentionally uses a transparent stacking-score proxy.  It is a
reference implementation for representation and legality; an INN-HB energy
table can replace ``helix_score`` without changing the subset decoder.
"""

from __future__ import annotations

from dataclasses import dataclass


PAIR_SCORES = {
    "CG": -3.0,
    "GC": -3.0,
    "AU": -2.0,
    "UA": -2.0,
    "GU": -1.0,
    "UG": -1.0,
}


@dataclass(frozen=True, slots=True)
class Helix:
    id: int
    start: int
    end: int
    length: int
    pairs: tuple[tuple[int, int], ...]
    score: float

    @property
    def occupied(self) -> frozenset[int]:
        return frozenset(position for pair in self.pairs for position in pair)


def normalize_sequence(sequence: str) -> str:
    normalized = "".join(sequence.upper().split()).replace("T", "U")
    if not normalized:
        raise ValueError("RNA sequence is empty")
    invalid = sorted(set(normalized) - set("ACGU"))
    if invalid:
        raise ValueError(f"RNA sequence contains invalid symbols: {''.join(invalid)}")
    return normalized


def helix_score(sequence: str, pairs: tuple[tuple[int, int], ...]) -> float:
    return float(sum(PAIR_SCORES[sequence[left] + sequence[right]] for left, right in pairs))


def enumerate_helices(
    sequence: str,
    *,
    min_stem_length: int = 2,
    min_loop_length: int = 3,
    max_helices: int = 5000,
) -> list[Helix]:
    """Enumerate maximal canonical stems from every possible outer pair."""
    sequence = normalize_sequence(sequence)
    if min_stem_length < 1:
        raise ValueError("min_stem_length must be positive")
    if min_loop_length < 0:
        raise ValueError("min_loop_length cannot be negative")
    helices: list[Helix] = []
    seen: set[tuple[tuple[int, int], ...]] = set()
    n = len(sequence)
    for start in range(n):
        for end in range(start + min_loop_length + 1, n):
            pairs: list[tuple[int, int]] = []
            offset = 0
            while start + offset < end - offset:
                left, right = start + offset, end - offset
                if sequence[left] + sequence[right] not in PAIR_SCORES:
                    break
                pairs.append((left, right))
                offset += 1
            if len(pairs) < min_stem_length:
                continue
            encoded = tuple(pairs)
            if encoded in seen:
                continue
            seen.add(encoded)
            helices.append(
                Helix(len(helices), start, end, len(encoded), encoded, helix_score(sequence, encoded))
            )
    if len(helices) <= max_helices:
        return helices
    # Deterministic candidate beam: retain the strongest stems while reserving
    # coverage for distinct 5' and 3' endpoints.  This avoids an arbitrary
    # scan-order cutoff and keeps permutation models tractable on long RNA.
    ranked = sorted(helices, key=lambda item: (item.score, -item.length, item.start, item.end))
    selected: list[Helix] = []
    selected_pairs: set[tuple[tuple[int, int], ...]] = set()
    coverage_budget = max(1, max_helices // 3)
    seen_start: set[int] = set()
    seen_end: set[int] = set()
    for item in ranked:
        if len(selected) >= coverage_budget:
            break
        if item.start not in seen_start or item.end not in seen_end:
            selected.append(item); selected_pairs.add(item.pairs)
            seen_start.add(item.start); seen_end.add(item.end)
    for item in ranked:
        if len(selected) >= max_helices:
            break
        if item.pairs not in selected_pairs:
            selected.append(item); selected_pairs.add(item.pairs)
    return [Helix(index, item.start, item.end, item.length, item.pairs, item.score)
            for index, item in enumerate(selected)]


def helices_compatible(first: Helix, second: Helix, *, allow_pseudoknots: bool = False) -> bool:
    if first.occupied & second.occupied:
        return False
    if allow_pseudoknots:
        return True
    for a, b in first.pairs:
        for c, d in second.pairs:
            if a < c < b < d or c < a < d < b:
                return False
    return True


def validate_subset(helices: list[Helix], *, allow_pseudoknots: bool = False) -> None:
    for index, first in enumerate(helices):
        for second in helices[index + 1 :]:
            if not helices_compatible(first, second, allow_pseudoknots=allow_pseudoknots):
                raise ValueError(f"helices {first.id} and {second.id} are incompatible")


def greedy_compatible_subset(
    helices: list[Helix], *, allow_pseudoknots: bool = False
) -> list[Helix]:
    """A deterministic baseline: strongest/longest compatible stems first."""
    selected: list[Helix] = []
    for helix in sorted(helices, key=lambda item: (item.score, -item.length, item.start, item.end)):
        if all(helices_compatible(helix, other, allow_pseudoknots=allow_pseudoknots) for other in selected):
            selected.append(helix)
    return sorted(selected, key=lambda item: (item.start, -item.end))


def subset_to_dot_bracket(
    sequence_length: int,
    helices: list[Helix],
    *,
    allow_pseudoknots: bool = False,
) -> str:
    validate_subset(helices, allow_pseudoknots=allow_pseudoknots)
    if allow_pseudoknots:
        raise ValueError("single-alphabet dot-bracket output cannot encode pseudoknots")
    structure = ["."] * sequence_length
    for helix in helices:
        for left, right in helix.pairs:
            if not (0 <= left < right < sequence_length):
                raise ValueError("helix pair is outside the RNA sequence")
            if structure[left] != "." or structure[right] != ".":
                raise ValueError("a nucleotide was paired more than once")
            structure[left], structure[right] = "(", ")"
    return "".join(structure)


def structure_pairs(dot_bracket: str) -> list[tuple[int, int]]:
    stack: list[int] = []
    pairs: list[tuple[int, int]] = []
    for index, symbol in enumerate(dot_bracket):
        if symbol == "(":
            stack.append(index)
        elif symbol == ")":
            if not stack:
                raise ValueError("unbalanced dot-bracket structure")
            pairs.append((stack.pop(), index))
        elif symbol != ".":
            raise ValueError(f"unsupported dot-bracket symbol: {symbol}")
    if stack:
        raise ValueError("unbalanced dot-bracket structure")
    return sorted(pairs)
