"""Fixed-array, batch RNA permutation decoding with optional Numba acceleration."""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import numpy as np

from coin.problems.rna import Helix, helices_compatible

try:
    from numba import njit, prange
except ImportError:  # pragma: no cover
    def njit(*args, **kwargs):
        def decorate(function): return function
        return decorate
    prange = range


@dataclass(slots=True)
class BatchDecode:
    selected: np.ndarray
    proxy_energy: np.ndarray
    pair_count: np.ndarray


def fixed_helix_arrays(helices: list[Helix], sequence_length: int):
    count = len(helices)
    conflicts = np.zeros((count, count), dtype=np.uint8)
    proxy = np.asarray([item.score for item in helices], dtype=np.float64)
    lengths = np.asarray([item.length for item in helices], dtype=np.int16)
    max_pairs = max((item.length for item in helices), default=0)
    left = np.full((count, max_pairs), -1, dtype=np.int16)
    right = np.full((count, max_pairs), -1, dtype=np.int16)
    for i, helix in enumerate(helices):
        for k, (a, b) in enumerate(helix.pairs): left[i, k], right[i, k] = a, b
        for j in range(i):
            if not helices_compatible(helix, helices[j]): conflicts[i, j] = conflicts[j, i] = 1
    return conflicts, proxy, lengths, left, right


@njit(cache=False, parallel=True)
def decode_population_numba(permutations, conflicts, proxy, lengths):
    population_size, helix_count = permutations.shape
    selected = np.zeros((population_size, helix_count), dtype=np.uint8)
    energies = np.zeros(population_size, dtype=np.float64)
    pair_counts = np.zeros(population_size, dtype=np.int32)
    for row in prange(population_size):
        accepted = np.empty(helix_count, dtype=np.int32)
        accepted_count = 0
        for position in range(helix_count):
            candidate = int(permutations[row, position]); legal = True
            for old_position in range(accepted_count):
                if conflicts[candidate, accepted[old_position]]:
                    legal = False; break
            if legal:
                selected[row, candidate] = 1
                accepted[accepted_count] = candidate; accepted_count += 1
                energies[row] += proxy[candidate]
                pair_counts[row] += lengths[candidate]
    return selected, energies, pair_counts


def pair_table_from_mask(mask, lengths, left, right, sequence_length):
    table = np.full(sequence_length, -1, dtype=np.int16)
    for helix_index in np.flatnonzero(mask):
        for k in range(int(lengths[helix_index])):
            a, b = int(left[helix_index, k]), int(right[helix_index, k])
            table[a], table[b] = b, a
    return table


def pair_table_to_dot_bracket(table):
    structure = np.full(len(table), ".", dtype="<U1")
    for index, partner in enumerate(table):
        if partner > index: structure[index], structure[int(partner)] = "(", ")"
    return "".join(structure.tolist())


def pair_table_key(table):
    return hashlib.blake2b(np.asarray(table, dtype=np.int16).tobytes(), digest_size=16).digest()


class NumbaBatchRnaDecoder:
    def __init__(self, helices: list[Helix], sequence_length: int):
        self.sequence_length = sequence_length
        self.conflicts, self.proxy, self.lengths, self.left, self.right = fixed_helix_arrays(helices, sequence_length)

    def decode(self, population) -> BatchDecode:
        permutations = np.ascontiguousarray(population, dtype=np.int16)
        selected, energy, pairs = decode_population_numba(permutations, self.conflicts, self.proxy, self.lengths)
        return BatchDecode(selected, energy, pairs)

    def pair_table(self, mask):
        return pair_table_from_mask(mask, self.lengths, self.left, self.right, self.sequence_length)

    def structure(self, mask):
        return pair_table_to_dot_bracket(self.pair_table(mask))
