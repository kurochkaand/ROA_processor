from __future__ import annotations

import numpy as np

from roa_processor.models import IsolatedExperiment, LoadedExperiment


def _diff_cumulative_matrix(matrix: np.ndarray) -> np.ndarray:
    """
    Convert cumulative block matrix into isolated blocks.

    Input shape:
    n_blocks x n_points
    """
    isolated = np.empty_like(matrix, dtype=float)
    isolated[0, :] = matrix[0, :]
    isolated[1:, :] = matrix[1:, :] - matrix[:-1, :]
    return isolated


def _diff_cumulative_vector(values: np.ndarray) -> np.ndarray:
    isolated = np.empty_like(values, dtype=float)
    isolated[0] = values[0]
    isolated[1:] = values[1:] - values[:-1]
    return isolated


def cumulative_to_isolated(
    experiment: LoadedExperiment,
    normalize_time: bool = True,
    normalize_power: bool = True,
) -> IsolatedExperiment:
    blocks = experiment.blocks
    wavenumber = experiment.wavenumber.copy()

    raman_cum = np.vstack([b.raman for b in blocks]).astype(float)
    roa_cum = np.vstack([b.roa for b in blocks]).astype(float)

    raman_iso = _diff_cumulative_matrix(raman_cum)
    roa_iso = _diff_cumulative_matrix(roa_cum)

    block_indices = np.array([b.metadata.block_index for b in blocks], dtype=int)

    cumulative_times = np.array(
        [
            b.metadata.header.total_time_scp_s
            if b.metadata.header.total_time_scp_s is not None
            else np.nan
            for b in blocks
        ],
        dtype=float,
    )

    cumulative_cycles = np.array(
        [
            b.metadata.header.cumulative_cycles
            if b.metadata.header.cumulative_cycles is not None
            else np.nan
            for b in blocks
        ],
        dtype=float,
    )

    powers = np.array(
        [
            b.metadata.header.power_at_sample_mw
            if b.metadata.header.power_at_sample_mw is not None
            else np.nan
            for b in blocks
        ],
        dtype=float,
    )

    if np.isnan(cumulative_times).any() and normalize_time:
        raise ValueError("Cannot normalize by time because some total times are missing.")
    if np.isnan(powers).any() and normalize_power:
        raise ValueError("Cannot normalize by power because some powers are missing.")

    delta_times = _diff_cumulative_vector(cumulative_times) if not np.isnan(cumulative_times).any() else cumulative_times
    delta_cycles = _diff_cumulative_vector(cumulative_cycles) if not np.isnan(cumulative_cycles).any() else cumulative_cycles

    raman_norm = raman_iso.copy()
    roa_norm = roa_iso.copy()

    scale = np.ones(len(blocks), dtype=float)
    if normalize_time:
        if np.any(delta_times <= 0):
            raise ValueError(f"Non-positive delta acquisition times found: {delta_times}")
        scale *= delta_times

    if normalize_power:
        if np.any(powers <= 0):
            raise ValueError(f"Non-positive power values found: {powers}")
        scale *= powers

    raman_norm = raman_norm / scale[:, None]
    roa_norm = roa_norm / scale[:, None]

    return IsolatedExperiment(
        wavenumber=wavenumber,
        raman_raw=raman_iso,
        roa_raw=roa_iso,
        raman_norm=raman_norm,
        roa_norm=roa_norm,
        delta_times_s=delta_times,
        delta_cycles=delta_cycles,
        power_at_sample_mw=powers,
        block_indices=block_indices,
    )


def filter_isolated_by_block_range(
    isolated: IsolatedExperiment,
    block_range: tuple[int, int] | None,
) -> IsolatedExperiment:
    if block_range is None:
        return isolated

    start, end = block_range
    mask = (isolated.block_indices >= start) & (isolated.block_indices <= end)
    if not np.any(mask):
        available = block_index_range(isolated.block_indices)
        available_text = format_block_index_range(available)
        raise ValueError(
            f"Block range {start}-{end} did not match any loaded blocks. "
            f"Available block indices are {available_text}."
        )

    return IsolatedExperiment(
        wavenumber=isolated.wavenumber.copy(),
        raman_raw=isolated.raman_raw[mask, :].copy(),
        roa_raw=isolated.roa_raw[mask, :].copy(),
        raman_norm=isolated.raman_norm[mask, :].copy(),
        roa_norm=isolated.roa_norm[mask, :].copy(),
        delta_times_s=isolated.delta_times_s[mask].copy(),
        delta_cycles=isolated.delta_cycles[mask].copy(),
        power_at_sample_mw=isolated.power_at_sample_mw[mask].copy(),
        block_indices=isolated.block_indices[mask].copy(),
    )


def block_index_range(block_indices: np.ndarray) -> tuple[int, int]:
    if block_indices.size == 0:
        raise ValueError("Cannot determine range for no block indices.")
    return (int(block_indices[0]), int(block_indices[-1]))


def format_block_index_range(block_range: tuple[int, int] | None) -> str:
    if block_range is None:
        return "all"
    start, end = block_range
    return f"{start:03d}-{end:03d}"
