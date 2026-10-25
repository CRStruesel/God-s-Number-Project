import math

import numpy as np

from godsnumber import bfs, group_order
from godsnumber.integer_growth import (
    entry_bound,
    growth_lower_bound,
    injective_radius,
    layers_extended,
    sl2z_layers,
)
from godsnumber.spectral import adjacency, spectrum


def test_sl2z_layers_match_mod_p_layers_in_the_injective_range():
    p = 101
    n = injective_radius(p)
    assert n >= 5
    for conv in ["directed", "undirected"]:
        z = sl2z_layers(n, conv)
        modp = bfs(p, conv).layer_sizes[: n + 1]
        assert z == modp


def test_undirected_layers_are_four_times_fibonacci():
    L = sl2z_layers(14, "undirected")
    F = [0, 1]
    while len(F) < 20:
        F.append(F[-1] + F[-2])
    assert L[:3] == [1, 4, 11]
    assert all(L[n] == 4 * F[n + 2] for n in range(3, 15))


def test_directed_recurrence():
    L = sl2z_layers(16, "directed")
    assert L[:8] == [1, 2, 4, 7, 11, 17, 25, 37]
    assert all(L[n] == L[n - 1] + L[n - 3] for n in range(8, 17))


def test_layers_extended_is_consistent_with_exact_enumeration():
    for conv in ["directed", "undirected"]:
        ext = layers_extended(18, conv)
        assert ext == sl2z_layers(18, conv)


def test_entry_bound_is_a_real_bound():
    # the maximal entry in each layer of SL_2(Z) (undirected) is below F_{n+2}
    true_max = [1, 1, 2, 3, 4, 5, 7, 10, 13, 17, 24, 33, 43, 58, 79]
    assert all(true_max[n] <= entry_bound(n) for n in range(len(true_max)))
    assert entry_bound(0) == 1 and entry_bound(1) == 2 and entry_bound(2) == 3


def test_growth_lower_bound_is_respected_and_beats_counting():
    from godsnumber import counting_lower_bound

    for p in [11, 31, 61, 101]:
        for conv in ["directed", "undirected"]:
            g = growth_lower_bound(p, conv)
            assert counting_lower_bound(p, conv) <= g <= bfs(p, conv).diameter


def test_adjacency_is_regular_symmetric_and_connected():
    p = 7
    A = adjacency(p)
    N = group_order(p)
    assert A.shape == (N, N)
    assert np.all(np.asarray(A.sum(axis=1)).ravel() == 4)
    assert (A - A.T).nnz == 0
    s = spectrum(p)
    assert abs(s.lambda2) < 4 - 1e-6  # connected (and the second eigenvalue is a real gap)
    assert s.lambda_min > -4 + 1e-6  # not bipartite for p >= 5 (SL_2(F_p) is perfect)
    assert math.isclose(s.ramanujan_threshold, 2 * math.sqrt(3))
    assert s.diameter_bound >= bfs(p, "undirected").diameter
