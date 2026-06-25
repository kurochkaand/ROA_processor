import numpy as np

from roa_processor.processing.isolate_blocks import _diff_cumulative_matrix, _diff_cumulative_vector


def test_diff_cumulative_matrix():
    cumulative = np.array([
        [10, 20, 30],
        [15, 25, 50],
        [18, 40, 80],
    ], dtype=float)
    isolated = _diff_cumulative_matrix(cumulative)
    expected = np.array([
        [10, 20, 30],
        [5, 5, 20],
        [3, 15, 30],
    ], dtype=float)
    np.testing.assert_allclose(isolated, expected)


def test_diff_cumulative_vector():
    cumulative = np.array([20, 40, 60], dtype=float)
    isolated = _diff_cumulative_vector(cumulative)
    expected = np.array([20, 20, 20], dtype=float)
    np.testing.assert_allclose(isolated, expected)
