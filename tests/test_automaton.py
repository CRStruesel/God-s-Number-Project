import math

import numpy as np

from godsnumber.automaton import build_geodesic_automaton, verify
from godsnumber.transfer import plancherel_check, representation_blocks, sphere_operators, transfer_matrix


def test_geodesic_automaton_is_a_bijection_with_the_group():
    r = verify(12, 30)
    assert r["counts_match_theorem_1"] and r["words_match_ball"] and r["unambiguous"]
    assert abs(r["perron_root"] - (1 + 5**0.5) / 2) < 1e-9


def test_trivial_block_counts_spheres():
    aut = build_geodesic_automaton()
    counts = aut.path_counts(10)
    assert counts[:6] == [1, 4, 11, 20, 32, 52]


def test_sphere_operators_on_the_trivial_representation_count():
    p = 7
    aut = build_geodesic_automaton()
    blk = next(b for b in representation_blocks(p) if b.label.startswith("Ind(1)"))
    ops = sphere_operators(blk, 8, aut)
    ones = np.ones(p + 1)
    for n, A in enumerate(ops):
        assert abs((ones @ A @ ones) / (p + 1) - aut.path_counts(8)[n]) < 1e-8


def test_plancherel_identity_small_p():
    assert plancherel_check(7, 8) < 1e-8
    assert plancherel_check(11, 8) < 1e-8
