import pytest
import numpy as np

from coin.problems.rna import (
    Helix,
    enumerate_helices,
    greedy_compatible_subset,
    normalize_sequence,
    structure_pairs,
    subset_to_dot_bracket,
)


def test_normalizes_dna_and_whitespace_to_rna():
    assert normalize_sequence(" aug t\nca ") == "AUGUCA"


def test_selected_subset_becomes_dot_bracket():
    helix = Helix(0, 5, 14, 3, ((5, 14), (6, 13), (7, 12)), -7.0)
    assert subset_to_dot_bracket(18, [helix]) == ".....(((....)))..."


def test_dot_bracket_pairs_round_trip():
    assert structure_pairs(".....(((....)))...") == [(5, 14), (6, 13), (7, 12)]


def test_rejects_overlapping_selected_helices():
    first = Helix(0, 0, 8, 2, ((0, 8), (1, 7)), -4.0)
    second = Helix(1, 1, 10, 2, ((1, 10), (2, 9)), -4.0)
    with pytest.raises(ValueError, match="incompatible"):
        subset_to_dot_bracket(12, [first, second])


def test_enumerator_and_greedy_decoder_produce_legal_structure():
    sequence = "AUGGCCAUGGCGCCCAGAACUGAGUAACCAUGA"
    candidates = enumerate_helices(sequence)
    selected = greedy_compatible_subset(candidates)
    structure = subset_to_dot_bracket(len(sequence), selected)
    assert candidates
    assert len(structure) == len(sequence)
    assert structure.count("(") == structure.count(")")


def test_long_sequence_candidate_beam_prunes_instead_of_failing():
    sequence = "GCGCAU" * 80
    candidates = enumerate_helices(sequence, max_helices=120)
    assert len(candidates) == 120
    assert [item.id for item in candidates] == list(range(120))


def test_rna_permutation_operators_preserve_all_helix_ids():
    from coin.experiments.rna_prediction import _ox, _erx
    rng = np.random.default_rng(7)
    a = np.arange(12, dtype=np.int16)
    b = a[::-1].copy()
    assert sorted(_ox(a, b, rng)) == list(range(12))
    assert sorted(_erx(a, b, rng)) == list(range(12))


def test_numba_batch_decoder_matches_readable_reference_for_random_permutations():
    from coin.experiments.rna_prediction import RnaPermutationEvaluator
    sequence = "AUGGCCAUGGCGCCCAGAACUGAGUAACCAUGA"
    helices = enumerate_helices(sequence, max_helices=80)
    evaluator = RnaPermutationEvaluator(sequence, helices)
    rng = np.random.default_rng(19)
    population = np.asarray([rng.permutation(len(helices)) for _ in range(24)], dtype=np.int16)
    decoded = evaluator._decoder.decode(population)
    for row, mask, proxy, pair_count in zip(population, decoded.selected, decoded.proxy_energy, decoded.pair_count):
        readable = evaluator.decode(row)
        assert evaluator._decoder.structure(mask) == subset_to_dot_bracket(len(sequence), readable)
        assert proxy == pytest.approx(sum(item.score for item in readable))
        assert pair_count == sum(item.length for item in readable)


def test_exact_energy_cache_hashes_pair_table_and_avoids_duplicate_vienna_calls():
    from coin.experiments.rna_prediction import RnaPermutationEvaluator
    sequence = "AUGGCCAUGGCGCCCAGAACUGAGUAACCAUGA"
    helices = enumerate_helices(sequence, max_helices=40)
    evaluator = RnaPermutationEvaluator(sequence, helices)
    row = np.arange(len(helices), dtype=np.int16)
    energies, results = evaluator.population(np.asarray([row, row, row]))
    assert energies[0] == energies[1] == energies[2]
    assert results[0].dot_bracket == results[1].dot_bracket == results[2].dot_bracket
    assert evaluator.proxy_evaluations == 3
    assert evaluator.exact_evaluations == 1
    assert evaluator.cache_hits == 2


def test_screened_population_exact_scores_only_top_k_unique_candidates():
    from coin.experiments.rna_prediction import RnaPermutationEvaluator
    sequence = "AUGGCCAUGGCGCCCAGAACUGAGUAACCAUGA"
    helices = enumerate_helices(sequence, max_helices=50)
    evaluator = RnaPermutationEvaluator(sequence, helices)
    rng = np.random.default_rng(23)
    population = np.asarray([rng.permutation(len(helices)) for _ in range(20)], dtype=np.int16)
    proxy, exact = evaluator.screened_population(population, top_k=5)
    assert proxy.shape == (20,)
    assert len(exact) == 5
    assert evaluator.proxy_evaluations == 20
    assert evaluator.exact_evaluations <= 5


def test_all_rna_algorithms_share_one_objective_and_exact_budget():
    from coin.experiments.rna_prediction import compare
    result = compare(
        "AUGGCCAUGGCGCCCAGAACUGAGUAACCAUGA",
        algorithms=["coin", "ehbsa", "nhbsa", "ga_ox", "ga_er", "sarna"],
        population_size=10, generations=2, seed=1,
    )
    assert result["objective"]["count"] == 1
    assert {x["algorithm"] for x in result["results"]} == {
        "coin", "ehbsa", "nhbsa", "ga_ox", "ga_er", "sarna"
    }
    assert all(x["evaluations"] == 20 for x in result["results"])


def test_published_benchmarks_have_known_structures_and_expected_counts():
    from coin.problems.rna_benchmark_catalog import list_benchmarks
    items = {x.id: x for x in list_benchmarks()}
    assert (len(items["X67579"].sequence), items["X67579"].pair_count) == (118, 37)
    assert (len(items["AF034620"].sequence), items["AF034620"].pair_count) == (122, 38)
    assert (len(items["X05914"].sequence), items["X05914"].pair_count) == (784, 233)
    assert all(len(x.dot_bracket) == len(x.sequence) for x in items.values())


def test_benchmark_catalog_has_three_examples_in_every_size_band():
    from collections import Counter
    from coin.problems.rna_benchmark_catalog import list_benchmarks
    items = list_benchmarks()
    assert Counter(x.size for x in items) == {
        "tiny": 3, "easy": 3, "teaching": 3, "medium": 3, "large": 3,
    }
    assert all(x.dot_bracket.count("(") == x.pair_count for x in items)


def test_known_structure_metrics_are_reported_without_becoming_objective():
    from coin.experiments.rna_prediction import compare
    from coin.problems.rna_benchmark_catalog import get_benchmark
    item = get_benchmark("X67579")
    result = compare(item.sequence, algorithms=["ga_ox"], population_size=10,
                     generations=1, seed=1, known_structure=item.dot_bracket)
    assert result["objective"]["count"] == 1
    metrics = result["results"][0]["known_structure_metrics"]
    assert set(metrics) == {"tp", "fp", "fn", "pair_distance", "sensitivity", "ppv", "f1", "exact_pair_match"}
