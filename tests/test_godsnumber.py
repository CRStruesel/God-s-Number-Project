"""Tests: run with ``python -m pytest`` (or ``python -m unittest`` style via pytest discovery)."""

import itertools

import numpy as np
import pytest

from godsnumber import (
    S,
    T,
    U,
    bfs,
    bfs_reference,
    counting_lower_bound,
    decode,
    decode_scalar,
    det,
    encode,
    encode_scalar,
    generators,
    group_order,
    inverse,
    inverse_table,
    mat_pow,
    matmul,
    order,
    reduce_mod,
)

SMALL_PRIMES = [2, 3, 5, 7, 11, 13]


def all_elements(p):
    """Every matrix over Z/pZ with determinant 1, by brute force."""
    return [m for m in itertools.product(range(p), repeat=4) if det(m, p) == 1 % p]


@pytest.mark.parametrize("p", SMALL_PRIMES)
def test_group_order_matches_brute_force(p):
    assert len(all_elements(p)) == group_order(p) == p * (p - 1) * (p + 1)


@pytest.mark.parametrize("p", SMALL_PRIMES)
def test_encode_decode_is_a_bijection(p):
    elems = all_elements(p)
    codes = sorted(encode_scalar(m, p) for m in elems)
    assert codes == list(range(group_order(p)))  # injective and onto [0, N)
    for m in elems:
        assert decode_scalar(encode_scalar(m, p), p) == m


@pytest.mark.parametrize("p", SMALL_PRIMES)
def test_vectorised_encode_decode_agree_with_scalar(p):
    idx = np.arange(group_order(p), dtype=np.int64)
    a, b, c, d = decode(idx, p, inverse_table(p))
    back = encode(a, b, c, d, p)
    assert np.array_equal(back, idx)
    for i in [0, 1, group_order(p) - 1, group_order(p) // 2]:
        assert (a[i], b[i], c[i], d[i]) == decode_scalar(i, p)
    assert np.all((a * d - b * c) % p == 1 % p)


@pytest.mark.parametrize("p", [3, 5, 7, 11])
def test_generator_relations(p):
    s, t = reduce_mod(S, p), reduce_mod(T, p)
    assert mat_pow(s, 2, p) == reduce_mod((-1, 0, 0, -1), p)  # S^2 = -I
    assert order(s, p) == 4
    assert order(t, p) == p  # T^k = [[1, k], [0, 1]]
    assert order(matmul(s, t, p), p) == 6  # (ST)^3 = -I, so ST has order 6
    assert matmul(matmul(s, inverse(t, p), p), inverse(s, p), p) == reduce_mod(U, p)  # S T^-1 S^-1 = U
    assert matmul(s, inverse(s, p), p) == (1, 0, 0, 1)


@pytest.mark.parametrize("p", SMALL_PRIMES)
@pytest.mark.parametrize("convention", ["directed", "undirected"])
def test_fast_bfs_matches_reference(p, convention):
    ref = bfs_reference(p, convention)
    fast = bfs(p, convention, keep_dist=True)
    assert fast.layer_sizes == ref.layer_sizes
    assert fast.order == group_order(p)
    assert fast.diameter == ref.diameter
    # the stored distance array must agree with the layer sizes
    counts = np.bincount(fast.dist.astype(np.int64), minlength=fast.diameter + 1)
    assert counts.tolist() == fast.layer_sizes
    # and the reported farthest element really sits in the last layer
    assert fast.dist[encode_scalar(fast.farthest, p)] == fast.diameter


def test_known_small_values():
    # p = 2: SL_2(F_2) is the symmetric group S_3 and S, T are two transpositions;
    # the Cayley graph is a hexagon, so the diameter is 3 in both conventions.
    assert bfs_reference(2, "directed").layer_sizes == [1, 2, 2, 1]
    assert bfs_reference(2, "undirected").layer_sizes == [1, 2, 2, 1]
    # undirected length is symmetric under inversion: ell(g) = ell(g^-1)
    p = 7
    r = bfs(p, "undirected", keep_dist=True)
    idx = np.arange(group_order(p))
    a, b, c, d = decode(idx, p, inverse_table(p))
    inv_idx = encode(d, (-b) % p, (-c) % p, a, p)
    assert np.array_equal(r.dist, r.dist[inv_idx])


@pytest.mark.parametrize("p", [3, 5, 7, 11, 13])
def test_lower_bound_is_respected(p):
    for conv in ["directed", "undirected"]:
        assert bfs(p, conv).diameter >= counting_lower_bound(p, conv)
    # the undirected graph has more edges, so its distances are never larger
    assert bfs(p, "undirected").diameter <= bfs(p, "directed").diameter


def test_other_generating_set():
    p = 11
    r = bfs(p, "undirected", gens=generators(p, "undirected", base=(T, U)))
    assert r.order == group_order(p)


def test_custom_convention_rejected():
    with pytest.raises(ValueError):
        generators(5, "sideways")
