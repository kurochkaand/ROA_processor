from __future__ import annotations

import numpy as np

from roa_processor.models import FinalSpectra, IsolatedExperiment, SpikeResult


def trimmed_mean(matrix: np.ndarray, proportion_to_cut: float = 0.1) -> np.ndarray:
    """
    Simple trimmed mean along axis 0 without scipy dependency.
    """
    if not 0 <= proportion_to_cut < 0.5:
        raise ValueError("proportion_to_cut must be >= 0 and < 0.5")

    sorted_data = np.sort(matrix, axis=0)
    n = sorted_data.shape[0]
    k = int(n * proportion_to_cut)

    if k == 0:
        return np.mean(sorted_data, axis=0)

    return np.mean(sorted_data[k : n - k, :], axis=0)


def make_final_spectra(
    isolated: IsolatedExperiment,
    spike_result: SpikeResult,
    qc_accepted_mask: np.ndarray | None = None,
    roa_qc_weighted_mean: np.ndarray | None = None,
    roa_qc_weighted_smoothed: np.ndarray | None = None,
    roa_qc_removed_noise: np.ndarray | None = None,
) -> FinalSpectra:
    roa_mean_after_qc_rejection = None
    if qc_accepted_mask is not None:
        qc_accepted_mask = np.asarray(qc_accepted_mask, dtype=bool)
        if qc_accepted_mask.shape != (spike_result.roa_cleaned.shape[0],):
            raise ValueError("qc_accepted_mask must have one value per ROA block.")
        if np.any(qc_accepted_mask) and not np.all(qc_accepted_mask):
            roa_mean_after_qc_rejection = np.mean(
                spike_result.roa_cleaned[qc_accepted_mask, :],
                axis=0,
            )

    return FinalSpectra(
        wavenumber=isolated.wavenumber.copy(),
        raman_mean=np.mean(isolated.raman_norm, axis=0),
        raman_median=np.median(isolated.raman_norm, axis=0),
        roa_mean_before_spike_removal=np.mean(isolated.roa_norm, axis=0),
        roa_mean_after_spike_removal=np.mean(spike_result.roa_cleaned, axis=0),
        roa_median_after_spike_removal=np.median(spike_result.roa_cleaned, axis=0),
        n_spikes_at_wavenumber=np.sum(spike_result.spike_mask, axis=0).astype(int),
        roa_mean_after_qc_rejection=roa_mean_after_qc_rejection,
        roa_qc_weighted_mean=roa_qc_weighted_mean,
        roa_qc_weighted_smoothed=roa_qc_weighted_smoothed,
        roa_qc_removed_noise=roa_qc_removed_noise,
    )
