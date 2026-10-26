import math

import numpy as np
import pytest

from godsnumber import bfs, det, group_order, inverse
from godsnumber.lps import lps_generators, psl_distances


@pytest.mark.parametrize("p", [29, 41, 61])
def test_lps_generators_are_a_symmetric_generating_set_of_sl2(p):
    g = lps_generators(p)
    assert len(g) == 6 and len(set(g)) == 6
    assert all(det(m, p) == 1 for m in g)
    assert all(inverse(m, p) in g for m in g)
    r = bfs(p, gens=g, keep_dist=True)
    assert r.order == group_order(p)  # connected: the six elements generate SL_2(F_p)
    dpsl = psl_distances(r.dist, p)
    assert dpsl.max() <= r.diameter
    # every PSL class has exactly two preimages, so the folded distribution has N/2 "pairs"
    assert int((dpsl == 0).sum()) == 2  # only ±I are at PSL-distance 0


def test_lps_requires_square_q():
    with pytest.raises(ValueError):
        lps_generators(37)  # 5 is not a square mod 37 (37 ≡ 2 mod 5)


def test_known_psl_diameter():
    # computed value, p = 29: the 6-regular LPS graph on PSL_2(F_29) (12180 vertices) has diameter 8,
    # between the Sardari lower bound (4/3) log_5 N - O(1) and the LPS upper bound 2 log_5 N + O(1)
    r = bfs(29, gens=lps_generators(29), keep_dist=True)
    G = int(psl_distances(r.dist, 29).max())
    assert G == 8
    assert G <= 2 * math.log(12180, 5) + 2 * math.log(2, 5) + 1
