import numpy as np
import pytest

from syndiffix.blob import _shrink_entropy_1dim, _shrink_matrix


def test_shrink_matrix() -> None:
    N = 5
    matrix = np.array([np.array([i * N + j for j in range(N)]) for i in range(N)], dtype=object)
    comb = (1, 3, 4)
    expected_matrix = np.array([[6, 8, 9], [16, 18, 19], [21, 23, 24]])
    shrunk_matrix = _shrink_matrix(matrix, comb)
    assert np.array_equal(shrunk_matrix, expected_matrix), f"Expected {expected_matrix}, but got {shrunk_matrix}"


def test_shrink_entropy_1dim() -> None:
    entropy_1dim = np.array([0.1, 0.2, 0.3, 0.4, 0.5], dtype=np.float64)
    comb = (1, 3, 4)
    expected_entropy_1dim = np.array([0.2, 0.4, 0.5], dtype=np.float64)
    new_entropy_1dim = _shrink_entropy_1dim(entropy_1dim, comb)
    assert np.array_equal(
        new_entropy_1dim, expected_entropy_1dim
    ), f"Expected {expected_entropy_1dim}, but got {new_entropy_1dim}"
