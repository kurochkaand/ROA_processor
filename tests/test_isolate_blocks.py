import numpy as np
import pytest

from roa_processor.models import IsolatedExperiment
from roa_processor.processing.isolate_blocks import (
    _diff_cumulative_matrix,
    _diff_cumulative_vector,
    filter_isolated_by_block_range,
)


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


def test_filter_isolated_by_block_range_keeps_true_isolated_blocks():
    isolated = IsolatedExperiment(
        wavenumber=np.array([100.0, 200.0]),
        raman_raw=np.array([
            [10.0, 20.0],
            [5.0, 5.0],
            [3.0, 15.0],
        ]),
        roa_raw=np.array([
            [1.0, 2.0],
            [0.5, 0.5],
            [0.3, 1.5],
        ]),
        raman_norm=np.array([
            [10.0, 20.0],
            [5.0, 5.0],
            [3.0, 15.0],
        ]),
        roa_norm=np.array([
            [1.0, 2.0],
            [0.5, 0.5],
            [0.3, 1.5],
        ]),
        delta_times_s=np.array([10.0, 10.0, 10.0]),
        delta_cycles=np.array([10.0, 10.0, 10.0]),
        power_at_sample_mw=np.array([2.0, 2.0, 2.0]),
        block_indices=np.array([0, 1, 2]),
    )

    filtered = filter_isolated_by_block_range(isolated, (1, 2))

    np.testing.assert_array_equal(filtered.block_indices, [1, 2])
    np.testing.assert_allclose(filtered.raman_raw, [[5.0, 5.0], [3.0, 15.0]])
    np.testing.assert_allclose(filtered.roa_raw, [[0.5, 0.5], [0.3, 1.5]])


def test_filter_isolated_by_block_range_rejects_empty_selection():
    isolated = IsolatedExperiment(
        wavenumber=np.array([100.0]),
        raman_raw=np.array([[1.0]]),
        roa_raw=np.array([[1.0]]),
        raman_norm=np.array([[1.0]]),
        roa_norm=np.array([[1.0]]),
        delta_times_s=np.array([1.0]),
        delta_cycles=np.array([1.0]),
        power_at_sample_mw=np.array([1.0]),
        block_indices=np.array([0]),
    )

    with pytest.raises(ValueError, match="did not match any loaded blocks"):
        filter_isolated_by_block_range(isolated, (1, 2))
