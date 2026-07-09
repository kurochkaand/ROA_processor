from __future__ import annotations

from typing import Any

import numpy as np

from roa_processor.models import RoaQcResult


MAD_TO_SIGMA = 1.4826


def estimate_qc_noise(values: np.ndarray) -> dict[str, float]:
    finite = np.asarray(values, dtype=float)
    finite = finite[np.isfinite(finite)]
    if finite.size == 0:
        return {
            "qc_median_offset": np.nan,
            "qc_noise_mad": np.nan,
            "qc_noise_sigma": np.nan,
            "qc_rms": np.nan,
        }

    offset = float(np.median(finite))
    centered = finite - offset
    mad = float(np.median(np.abs(centered - np.median(centered))))
    sigma = float(MAD_TO_SIGMA * mad)
    rms = float(np.sqrt(np.mean(centered**2)))
    return {
        "qc_median_offset": offset,
        "qc_noise_mad": mad,
        "qc_noise_sigma": sigma,
        "qc_rms": rms,
    }


def qc_noise_weights(
    sigmas: np.ndarray,
    eps: float = 1e-12,
    max_weight_to_median: float = 10.0,
) -> np.ndarray:
    sigmas = np.asarray(sigmas, dtype=float)
    if sigmas.ndim != 1:
        raise ValueError("sigmas must be a 1D array.")
    if sigmas.size == 0:
        raise ValueError("Cannot calculate QC weights for no blocks.")
    if not np.all(np.isfinite(sigmas)):
        raise ValueError("All QC sigmas used for weighting must be finite.")

    positive = sigmas[sigmas > 0]
    sigma_floor = float(np.median(positive) * 0.05) if positive.size else eps
    sigma_floor = max(sigma_floor, eps)
    safe_sigmas = np.maximum(sigmas, sigma_floor)

    raw_weights = 1.0 / (safe_sigmas**2 + eps)
    median_raw = float(np.median(raw_weights))
    if median_raw > 0 and np.isfinite(median_raw):
        raw_weights = np.minimum(raw_weights, median_raw * max_weight_to_median)

    total = float(np.sum(raw_weights))
    if total <= 0 or not np.isfinite(total):
        return np.full(sigmas.shape, 1.0 / sigmas.size)
    return raw_weights / total


def analyze_roa_qc(
    wavenumber: np.ndarray,
    roa_cleaned: np.ndarray,
    spike_mask: np.ndarray,
    block_indices: np.ndarray,
    qc_range: tuple[float, float] = (1800.0, 2609.0),
    denoise_qc: bool = False,
    reject_blocks: bool = False,
    max_block_noise: float = 3.0,
    highest_noise_reject_blocks: int = 0,
    min_qc_points: int = 5,
    smooth_qc_weighted: bool = False,
) -> RoaQcResult:
    if roa_cleaned.ndim != 2:
        raise ValueError("roa_cleaned must be a 2D matrix.")
    if spike_mask.shape != roa_cleaned.shape:
        raise ValueError("spike_mask shape must match roa_cleaned.")
    if max_block_noise <= 0:
        raise ValueError("max_block_noise must be positive.")
    if highest_noise_reject_blocks < 0:
        raise ValueError("highest_noise_reject_blocks must be non-negative.")

    low, high = qc_range
    if low > high:
        raise ValueError("QC range lower bound cannot be greater than upper bound.")

    qc_mask = (wavenumber >= low) & (wavenumber <= high)
    if int(np.sum(qc_mask)) < min_qc_points:
        warning = (
            f"ROA QC disabled: only {int(np.sum(qc_mask))} points are present in "
            f"{low:g}-{high:g} cm^-1 after wavenumber filtering."
        )
        return _unavailable_result(
            requested_range=(float(low), float(high)),
            qc_mask=qc_mask,
            block_indices=block_indices,
            spike_mask=spike_mask,
            denoise_qc=denoise_qc,
            reject_blocks=reject_blocks or highest_noise_reject_blocks > 0,
            smooth_qc_weighted=smooth_qc_weighted,
            warning=warning,
        )

    summary: list[dict[str, Any]] = []
    sigmas = np.empty(roa_cleaned.shape[0], dtype=float)
    for i, block_index in enumerate(block_indices):
        noise = estimate_qc_noise(roa_cleaned[i, qc_mask])
        sigmas[i] = noise["qc_noise_sigma"]
        summary.append(
            {
                "block_index": int(block_index),
                "number_of_spike_points": int(np.sum(spike_mask[i, :])),
                **noise,
            }
        )

    finite_sigmas = sigmas[np.isfinite(sigmas)]
    median_sigma = float(np.median(finite_sigmas)) if finite_sigmas.size else np.nan
    accepted_mask = np.ones(roa_cleaned.shape[0], dtype=bool)

    if reject_blocks:
        accepted_mask = np.isfinite(sigmas)
        if np.isfinite(median_sigma) and median_sigma > 0:
            accepted_mask &= (sigmas / median_sigma) <= max_block_noise

    if highest_noise_reject_blocks > 0:
        accepted_mask &= np.isfinite(sigmas)
        accepted_mask &= ~_highest_noise_rejected_mask(sigmas, highest_noise_reject_blocks)

    rejected_mask = ~accepted_mask
    weights = None
    weighted_mean = None
    warning = None

    if denoise_qc:
        usable_mask = accepted_mask & np.isfinite(sigmas)
        if not np.any(usable_mask):
            warning = "ROA QC weighted averaging disabled: no accepted blocks with finite QC noise."
        else:
            selected_weights = qc_noise_weights(sigmas[usable_mask])
            weights = np.zeros(roa_cleaned.shape[0], dtype=float)
            weights[usable_mask] = selected_weights
            weighted_mean = np.sum(roa_cleaned * weights[:, None], axis=0)

    smoothed = None
    removed_noise = None
    smoothing_parameters = None
    if smooth_qc_weighted and weighted_mean is not None:
        smoothed, smoothing_parameters = smooth_qc_weighted_mean(weighted_mean, qc_mask)
        removed_noise = weighted_mean - smoothed

    for i, row in enumerate(summary):
        ratio = sigmas[i] / median_sigma if np.isfinite(median_sigma) and median_sigma > 0 else np.nan
        row["qc_noise_ratio_to_median"] = float(ratio) if np.isfinite(ratio) else np.nan
        row["qc_accepted"] = bool(accepted_mask[i])
        row["qc_weight"] = np.nan if weights is None else float(weights[i])

    return RoaQcResult(
        requested_range=(float(low), float(high)),
        used_range=(
            float(wavenumber[qc_mask][0]),
            float(wavenumber[qc_mask][-1]),
        ),
        qc_mask=qc_mask,
        block_summary=summary,
        accepted_mask=accepted_mask,
        rejected_mask=rejected_mask,
        weights=weights,
        weighted_mean=weighted_mean,
        smoothed=smoothed,
        removed_noise=removed_noise,
        denoise_enabled=denoise_qc,
        reject_blocks_enabled=reject_blocks or highest_noise_reject_blocks > 0,
        smoothing_enabled=smooth_qc_weighted,
        smoothing_parameters=smoothing_parameters,
        warning=warning,
    )


def _highest_noise_rejected_mask(sigmas: np.ndarray, n_blocks: int) -> np.ndarray:
    finite_indices = np.flatnonzero(np.isfinite(sigmas))
    if n_blocks >= finite_indices.size:
        raise ValueError(
            "highest_noise_reject_blocks must leave at least one block with finite QC noise."
        )

    rejected = np.zeros(sigmas.shape, dtype=bool)
    if n_blocks == 0:
        return rejected

    order = finite_indices[np.argsort(sigmas[finite_indices], kind="mergesort")]
    rejected[order[-n_blocks:]] = True
    return rejected


def smooth_qc_weighted_mean(
    spectrum: np.ndarray,
    qc_mask: np.ndarray,
    candidate_windows: tuple[int, ...] = (5, 7, 9, 11, 15),
) -> tuple[np.ndarray, dict[str, Any]]:
    windows = [w for w in candidate_windows if w % 2 == 1 and 1 < w <= spectrum.size]
    if not windows:
        return spectrum.copy(), {
            "method": "gaussian",
            "selected_window_points": 1,
            "selected_sigma_points": 0.0,
            "baseline_qc_rms": _centered_rms(spectrum[qc_mask]),
            "selected_qc_rms": _centered_rms(spectrum[qc_mask]),
        }

    baseline_rms = _centered_rms(spectrum[qc_mask])
    candidates = []
    for window in windows:
        smoothed = _gaussian_smooth(spectrum, window)
        qc_rms = _centered_rms(smoothed[qc_mask])
        candidates.append((window, qc_rms, smoothed))

    finite_candidates = [c for c in candidates if np.isfinite(c[1])]
    if not finite_candidates:
        selected_window, selected_rms, selected = candidates[0]
    else:
        best_rms = min(c[1] for c in finite_candidates)
        if np.isfinite(baseline_rms) and best_rms < baseline_rms:
            target_rms = baseline_rms - 0.75 * (baseline_rms - best_rms)
            selected_window, selected_rms, selected = next(
                c for c in finite_candidates if c[1] <= target_rms
            )
        else:
            selected_window, selected_rms, selected = finite_candidates[0]

    params = {
        "method": "gaussian",
        "candidate_windows_points": windows,
        "selected_window_points": int(selected_window),
        "selected_sigma_points": float(max(selected_window / 6.0, 1e-12)),
        "baseline_qc_rms": float(baseline_rms),
        "selected_qc_rms": float(selected_rms),
    }
    return selected, params


def _gaussian_smooth(values: np.ndarray, window: int) -> np.ndarray:
    sigma = max(window / 6.0, 1e-12)
    x = np.arange(window, dtype=float) - window // 2
    kernel = np.exp(-0.5 * (x / sigma) ** 2)
    kernel /= np.sum(kernel)
    pad = window // 2
    padded = np.pad(values, pad_width=pad, mode="edge")
    return np.convolve(padded, kernel, mode="valid")


def _centered_rms(values: np.ndarray) -> float:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return np.nan
    centered = values - np.median(values)
    return float(np.sqrt(np.mean(centered**2)))


def _unavailable_result(
    requested_range: tuple[float, float],
    qc_mask: np.ndarray,
    block_indices: np.ndarray,
    spike_mask: np.ndarray,
    denoise_qc: bool,
    reject_blocks: bool,
    smooth_qc_weighted: bool,
    warning: str,
) -> RoaQcResult:
    accepted_mask = np.ones(len(block_indices), dtype=bool)
    summary = [
        {
            "block_index": int(block_index),
            "number_of_spike_points": int(np.sum(spike_mask[i, :])),
            "qc_median_offset": np.nan,
            "qc_noise_mad": np.nan,
            "qc_noise_sigma": np.nan,
            "qc_rms": np.nan,
            "qc_noise_ratio_to_median": np.nan,
            "qc_accepted": True,
            "qc_weight": np.nan,
        }
        for i, block_index in enumerate(block_indices)
    ]
    return RoaQcResult(
        requested_range=requested_range,
        used_range=None,
        qc_mask=qc_mask,
        block_summary=summary,
        accepted_mask=accepted_mask,
        rejected_mask=~accepted_mask,
        weights=None,
        weighted_mean=None,
        smoothed=None,
        removed_noise=None,
        denoise_enabled=denoise_qc,
        reject_blocks_enabled=reject_blocks,
        smoothing_enabled=smooth_qc_weighted,
        smoothing_parameters=None,
        warning=warning,
    )
