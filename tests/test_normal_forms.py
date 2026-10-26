"""The computer-assisted parts of the proof of the sphere-size recurrences of SL_2(Z)."""

from godsnumber.integer_growth import sl2z_layers
from godsnumber.normal_forms import (
    all_syllable_strings,
    directed_length_formula,
    directed_sphere_sizes_from_formula,
    imul,
    is_pure_t_power,
    length_formula,
    lifted_word,
    local_cost,
    matrix_of,
    normal_forms_in_ball,
    reduce_s_runs,
    sphere_sizes_from_formula,
    verify_lipschitz,
    verify_lipschitz_directed,
    verify_pure_family,
    word_matrix,
)

S, Si, T, Ti = (0, -1, 1, 0), (0, 1, -1, 0), (1, 1, 0, 1), (1, -1, 0, 1)


def bfs_distances(gens, radius):
    dist = {(1, 0, 0, 1): 0}
    layer = [(1, 0, 0, 1)]
    for n in range(1, radius + 1):
        nxt = []
        for m in layer:
            for g in gens:
                y = imul(m, g)
                if y not in dist:
                    dist[y] = n
                    nxt.append(y)
        layer = nxt
    return dist


def test_normal_forms_are_consistent_and_injective():
    nf = normal_forms_in_ball(12)
    seen = set()
    for m, (sg, sy) in nf.items():
        assert matrix_of(sg, sy) == m
        assert (sg, sy) not in seen
        seen.add((sg, sy))


def test_undirected_formula_equals_word_length_on_a_ball():
    nf = normal_forms_in_ball(12)
    dist = bfs_distances([S, Si, T, Ti], 12)
    for m, (sg, sy) in nf.items():
        L, w = length_formula(sg, sy)
        assert word_matrix(w) == m
        assert L == dist[m]


def test_directed_formula_equals_word_length_on_a_ball():
    nf = normal_forms_in_ball(12)
    dist = bfs_distances([S, T], 18)
    for m, (sg, sy) in nf.items():
        if m in dist and dist[m] <= 16:
            L, w = directed_length_formula(sg, sy)
            assert word_matrix(w) == m
            assert L == dist[m]


def test_lipschitz_lemmas():
    r = verify_lipschitz(7)
    assert r["max_increase_short_elements"] == 1 and r["max_increase_tails"] == 1
    r = verify_lipschitz_directed(7)
    assert r["max_increase_short_elements"] == 1 and r["max_increase_tails"] == 1
    assert verify_pure_family(10)["violations"] == []


def test_auxiliary_claims():
    # the reduced undirected word has no S-letter exactly for the pure T-power patterns
    for s in all_syllable_strings(10):
        s_free = not any(ch in "Ss" for ch in reduce_s_runs(lifted_word(s))[0])
        assert s_free == is_pure_t_power(s)
    # two consecutive syllables raise L_0 by at least one (used to bound the transfer-matrix search)
    assert min(local_cost(s) - local_cost(s[:-2]) for s in all_syllable_strings(8) if len(s) >= 2) >= 1


def test_counting_reproduces_enumeration_and_recurrences():
    F = [0, 1]
    while len(F) < 50:
        F.append(F[-1] + F[-2])
    und = sphere_sizes_from_formula(40)
    assert und[:19] == sl2z_layers(18, "undirected")
    assert all(und[n] == 4 * F[n + 2] for n in range(3, 41))
    d = directed_sphere_sizes_from_formula(40)
    assert d[:23] == sl2z_layers(22, "directed")
    assert all(d[n] == d[n - 1] + d[n - 3] for n in range(8, 41))
