import numpy as np
import pytest

from godsnumber.reps import calibrate_weil, check_against_bruteforce, full_spectrum, second_eigenvalue


@pytest.mark.parametrize("p", [5, 7, 11])
def test_assembled_spectrum_equals_direct_diagonalisation(p):
    assert calibrate_weil(p) == (-1, 1)  # the Weil index of the anisotropic plane is -1
    assert check_against_bruteforce(p) < 1e-9


def test_second_eigenvalue_matches_arpack_values():
    # values computed independently with scipy.sparse.linalg.eigsh (data/spectrum.csv)
    assert abs(second_eigenvalue(17)["lambda2"] - 3.867088) < 1e-5
    assert abs(second_eigenvalue(23)["lambda_min"] - (-3.918412)) < 1e-5


def test_spectrum_size_and_trivial_eigenvalue():
    ev = full_spectrum(7)
    assert ev.size == 7**3 - 7
    assert abs(ev[-1] - 4) < 1e-9 and abs(ev[-2] - 4) > 1e-6  # the trivial eigenvalue is simple
    assert np.all(ev >= -4 - 1e-9)
