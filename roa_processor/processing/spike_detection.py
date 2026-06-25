from __future__ import annotations

from typing import Any

import numpy as np

from roa_processor.models import SpikeResult


def detect_and_replace_spikes_block_mad(
    roa_blocks: np.ndarray,
    block_indices: np.ndarray,
    threshold: float = 8.0,
    eps: float = 1e-12,
) -> SpikeResult:
    """
    Detect cosmic spikes in ROA isolated blocks using across-block robust statistics.

    roa_blocks shape:
    n_blocks x n_wavenumbers

    For each wavenumber, detect outlier blocks using:
    abs(value - median_across_blocks) > threshold * MAD

    Replacement rule:
    replace bad point by median of clean blocks at the same wavenumber.
    """
    if roa_blocks.ndim != 2:
        raise ValueError("roa_blocks must be a 2D matrix: n_blocks x n_points.")

    data = roa_blocks.astype(float, copy=True)
    n_blocks, n_points = data.shape

    median = np.nanmedian(data, axis=0)
    abs_dev = np.abs(data - median[None, :])
    mad = np.nanmedian(abs_dev, axis=0)

    # Robust noise estimate from MAD; if MAD is zero, use fallback local/global floor.
    # 1.4826 converts MAD to an estimate of standard deviation for normal noise.
    sigma = 1.4826 * mad

    global_sigma = float(np.nanmedian(sigma[sigma > 0])) if np.any(sigma > 0) else 1.0
    sigma_floor = max(global_sigma * 0.05, eps)
    sigma_safe = np.maximum(sigma, sigma_floor)

    spike_mask = abs_dev > (threshold * sigma_safe[None, :])

    cleaned = data.copy()

    for j in range(n_points):
        bad = spike_mask[:, j]
        if not np.any(bad):
            continue

        clean = ~bad
        if np.any(clean):
            replacement = np.nanmedian(data[clean, j])
        else:
            # Extremely unlikely, but keep program safe.
            replacement = median[j]

        cleaned[bad, j] = replacement

    spike_summary: list[dict[str, Any]] = []
    for i in range(n_blocks):
        n_spikes = int(np.sum(spike_mask[i, :]))
        spike_summary.append(
            {
                "block_index": int(block_indices[i]),
                "n_spikes": n_spikes,
                "fraction_spikes": float(n_spikes / n_points),
            }
        )

    return SpikeResult(
        roa_cleaned=cleaned,
        spike_mask=spike_mask,
        spike_summary=spike_summary,
        threshold=threshold,
    )
