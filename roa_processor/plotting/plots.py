from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from roa_processor.models import FinalSpectra, IsolatedExperiment, RoaQcResult, SpikeResult


def _save_or_show(path: Path | None) -> None:
    plt.tight_layout()
    if path is not None:
        path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(path, dpi=200)
        plt.close()
    else:
        plt.show()


def plot_isolated_roa_blocks(
    isolated: IsolatedExperiment,
    output_path: str | Path | None = None,
    cleaned_roa: np.ndarray | None = None,
    max_blocks: int | None = None,
) -> None:
    y = cleaned_roa if cleaned_roa is not None else isolated.roa_norm
    x = isolated.wavenumber

    if max_blocks is not None:
        y = y[:max_blocks, :]

    plt.figure(figsize=(10, 6))
    for row in y:
        plt.plot(x, row, linewidth=0.7, alpha=0.5)

    title = "Isolated ROA blocks"
    if cleaned_roa is not None:
        title += " after spike replacement"

    plt.title(title)
    plt.xlabel("Wavenumber / cm$^{-1}$")
    plt.ylabel("ROA intensity, normalized")
    _save_or_show(Path(output_path) if output_path else None)


def plot_isolated_roa_spike_removal_comparison(
    wavenumber: np.ndarray,
    roa_before: np.ndarray,
    roa_after: np.ndarray,
    block_indices: np.ndarray,
    output_path: str | Path | None = None,
    n_blocks: int = 10,
) -> None:
    if n_blocks <= 0:
        raise ValueError("n_blocks must be positive.")
    if roa_before.shape != roa_after.shape:
        raise ValueError("roa_before and roa_after must have the same shape.")
    if roa_before.ndim != 2:
        raise ValueError("ROA block arrays must be two-dimensional.")
    if len(block_indices) != roa_before.shape[0]:
        raise ValueError("block_indices length must match the number of ROA blocks.")
    if len(wavenumber) != roa_before.shape[1]:
        raise ValueError("wavenumber length must match the number of spectral points.")
    if roa_before.shape[0] == 0:
        raise ValueError("No ROA blocks are available to plot.")

    row_indices = _representative_row_indices(roa_before.shape[0], n_blocks)
    before = roa_before[row_indices, :]
    after = roa_after[row_indices, :]
    offsets = np.arange(len(row_indices) - 1, -1, -1) * _stack_offset_step(before, after)

    plt.figure(figsize=(10, max(8, 1.0 * len(row_indices) + 2)))
    for plot_index, row_index in enumerate(row_indices):
        offset = offsets[plot_index]
        before_label = "Before spike removal" if plot_index == 0 else "_nolegend_"
        after_label = "After spike removal" if plot_index == 0 else "_nolegend_"
        plt.plot(
            wavenumber,
            roa_before[row_index] + offset,
            color="C3",
            linewidth=0.8,
            alpha=0.7,
            label=before_label,
        )
        plt.plot(
            wavenumber,
            roa_after[row_index] + offset,
            color="C0",
            linewidth=0.9,
            alpha=0.9,
            label=after_label,
        )

    tick_labels = [f"Block {int(block_indices[row_index]):03d}" for row_index in row_indices]
    plt.yticks(offsets, tick_labels)
    plt.title("Representative isolated ROA blocks before/after spike removal")
    plt.xlabel("Wavenumber / cm$^{-1}$")
    plt.ylabel("ROA block, offset")
    plt.legend(loc="upper right")
    plt.margins(x=0)
    _save_or_show(Path(output_path) if output_path else None)


def _representative_row_indices(n_rows: int, n_blocks: int) -> np.ndarray:
    n_selected = min(n_rows, n_blocks)
    return np.linspace(0, n_rows - 1, n_selected, dtype=int)


def _stack_offset_step(before: np.ndarray, after: np.ndarray) -> float:
    finite = np.concatenate([before.ravel(), after.ravel()])
    finite = finite[np.isfinite(finite)]
    if finite.size == 0:
        return 1.0

    robust_span = float(np.percentile(finite, 95) - np.percentile(finite, 5))
    if robust_span > 0:
        return robust_span * 1.35

    max_abs = float(np.max(np.abs(finite)))
    if max_abs > 0:
        return max_abs * 1.35

    return 1.0


def plot_isolated_raman_blocks(
    isolated: IsolatedExperiment,
    output_path: str | Path | None = None,
    max_blocks: int | None = None,
) -> None:
    x = isolated.wavenumber
    y = isolated.raman_norm
    block_indices = isolated.block_indices

    if max_blocks is not None:
        y = y[:max_blocks, :]
        block_indices = block_indices[:max_blocks]

    plt.figure(figsize=(10, 6))
    if len(y) > 2:
        for row in y[1:-1]:
            plt.plot(x, row, color="0.55", linewidth=0.7, alpha=0.35)

    first_label = f"First block {int(block_indices[0]):03d}"
    plt.plot(x, y[0], label=first_label, linewidth=2.0, alpha=0.95)

    if len(y) > 1:
        last_label = f"Last block {int(block_indices[-1]):03d}"
        plt.plot(x, y[-1], label=last_label, linewidth=2.0, alpha=0.95)

    plt.title("Isolated Raman block overlap")
    plt.xlabel("Wavenumber / cm$^{-1}$")
    plt.ylabel("Raman intensity, normalized")
    plt.legend()
    _save_or_show(Path(output_path) if output_path else None)


def plot_spike_heatmap(
    isolated: IsolatedExperiment,
    spike_result: SpikeResult,
    output_path: str | Path | None = None,
) -> None:
    plt.figure(figsize=(10, 5))
    plt.imshow(
        spike_result.spike_mask.astype(int),
        aspect="auto",
        interpolation="nearest",
        extent=[
            isolated.wavenumber[0],
            isolated.wavenumber[-1],
            isolated.block_indices[-1],
            isolated.block_indices[0],
        ],
    )
    plt.title("ROA spike mask")
    plt.xlabel("Wavenumber / cm$^{-1}$")
    plt.ylabel("Block index")
    plt.colorbar(label="Spike")
    _save_or_show(Path(output_path) if output_path else None)


def plot_final_spectra(
    final: FinalSpectra,
    output_path: str | Path | None = None,
) -> None:
    x = final.wavenumber

    plt.figure(figsize=(10, 6))
    plt.plot(x, final.raman_mean, label="Raman mean")
    plt.title("Final Raman spectrum")
    plt.xlabel("Wavenumber / cm$^{-1}$")
    plt.ylabel("Raman intensity, normalized")
    plt.legend()
    _save_or_show(Path(output_path).with_name("final_raman.png") if output_path else None)

    plt.figure(figsize=(10, 6))
    plt.plot(x, final.roa_mean_before_spike_removal, label="ROA before spike removal", alpha=0.6)
    plt.plot(x, final.roa_mean_after_spike_removal, label="ROA after spike removal")
    if final.roa_mean_after_qc_rejection is not None:
        plt.plot(x, final.roa_mean_after_qc_rejection, label="ROA after QC block rejection")
    plt.title("Final ROA spectrum")
    plt.xlabel("Wavenumber / cm$^{-1}$")
    plt.ylabel("ROA intensity, normalized")
    plt.legend()
    _save_or_show(Path(output_path).with_name("final_roa.png") if output_path else None)


def plot_roa_qc_region(
    final: FinalSpectra,
    qc_result: RoaQcResult,
    output_path: str | Path | None = None,
) -> None:
    x = final.wavenumber
    plt.figure(figsize=(10, 6))
    plt.plot(x, final.roa_mean_after_spike_removal, label="ROA mean after spike removal")
    if final.roa_qc_weighted_mean is not None:
        plt.plot(x, final.roa_qc_weighted_mean, label="ROA QC-weighted mean", alpha=0.85)
    _highlight_qc_range(qc_result.requested_range, x)
    plt.title("ROA QC region")
    plt.xlabel("Wavenumber / cm$^{-1}$")
    plt.ylabel("ROA intensity, normalized")
    plt.legend()
    _save_or_show(Path(output_path) if output_path else None)


def plot_roa_qc_noise_by_block(
    qc_result: RoaQcResult,
    output_path: str | Path | None = None,
) -> None:
    df = pd.DataFrame(qc_result.block_summary)
    if df.empty:
        return

    plt.figure(figsize=(10, 5))
    colors = np.where(df["qc_accepted"].to_numpy(dtype=bool), "C0", "C3")
    plt.scatter(df["block_index"], df["qc_noise_sigma"], c=colors)
    plt.plot(df["block_index"], df["qc_noise_sigma"], color="0.6", linewidth=0.8)
    plt.title("ROA QC noise by block")
    plt.xlabel("Block index")
    plt.ylabel("QC noise sigma")
    _save_or_show(Path(output_path) if output_path else None)


def plot_final_roa_qc_comparison(
    final: FinalSpectra,
    output_path: str | Path | None = None,
) -> None:
    x = final.wavenumber
    plt.figure(figsize=(10, 6))
    plt.plot(x, final.roa_mean_after_spike_removal, label="ROA mean after spike removal")
    if final.roa_mean_after_qc_rejection is not None:
        plt.plot(x, final.roa_mean_after_qc_rejection, label="ROA mean after QC block rejection")
    if final.roa_qc_weighted_mean is not None:
        plt.plot(x, final.roa_qc_weighted_mean, label="ROA QC-weighted mean", alpha=0.85)
    if final.roa_qc_weighted_smoothed is not None:
        plt.plot(
            x,
            final.roa_qc_weighted_smoothed,
            label="ROA QC-weighted smoothed",
            linewidth=1.5,
        )
    plt.title("Final ROA QC comparison")
    plt.xlabel("Wavenumber / cm$^{-1}$")
    plt.ylabel("ROA intensity, normalized")
    plt.legend()
    _save_or_show(Path(output_path) if output_path else None)


def plot_roa_before_after_qc_rejection(
    final: FinalSpectra,
    output_path: str | Path | None = None,
) -> None:
    if final.roa_mean_after_qc_rejection is None:
        return

    x = final.wavenumber
    plt.figure(figsize=(10, 6))
    plt.plot(x, final.roa_mean_after_spike_removal, label="ROA before QC block rejection", alpha=0.7)
    plt.plot(x, final.roa_mean_after_qc_rejection, label="ROA after QC block rejection")
    plt.title("ROA before and after QC block rejection")
    plt.xlabel("Wavenumber / cm$^{-1}$")
    plt.ylabel("ROA intensity, normalized")
    plt.legend()
    _save_or_show(Path(output_path) if output_path else None)


def plot_roa_qc_removed_noise(
    final: FinalSpectra,
    output_path: str | Path | None = None,
) -> None:
    if final.roa_qc_removed_noise is None:
        return

    plt.figure(figsize=(10, 5))
    plt.plot(final.wavenumber, final.roa_qc_removed_noise)
    plt.title("ROA QC removed noise")
    plt.xlabel("Wavenumber / cm$^{-1}$")
    plt.ylabel("ROA weighted - smoothed")
    _save_or_show(Path(output_path) if output_path else None)


def _highlight_qc_range(qc_range: tuple[float, float], x: np.ndarray) -> None:
    low, high = qc_range
    span_low = max(float(x[0]), low)
    span_high = min(float(x[-1]), high)
    if span_low <= span_high:
        plt.axvspan(span_low, span_high, color="0.85", alpha=0.5, label="QC range")


def plot_final_from_csv(
    csv_path: str | Path,
    output_path: str | Path | None = None,
) -> None:
    df = pd.read_csv(csv_path)
    x = df["wavenumber_cm-1"]

    plt.figure(figsize=(10, 6))
    plt.plot(x, df["raman_mean"], label="Raman mean")
    plt.title("Final Raman spectrum")
    plt.xlabel("Wavenumber / cm$^{-1}$")
    plt.ylabel("Raman intensity, normalized")
    plt.legend()
    _save_or_show(Path(output_path).with_name("final_raman_from_csv.png") if output_path else None)

    plt.figure(figsize=(10, 6))
    plt.plot(x, df["roa_mean_before_spike_removal"], label="ROA before spike removal", alpha=0.6)
    plt.plot(x, df["roa_mean_after_spike_removal"], label="ROA after spike removal")
    if "roa_mean_after_qc_rejection" in df and not df["roa_mean_after_qc_rejection"].isna().all():
        plt.plot(x, df["roa_mean_after_qc_rejection"], label="ROA after QC block rejection")
    if "roa_qc_weighted_mean" in df and not df["roa_qc_weighted_mean"].isna().all():
        plt.plot(x, df["roa_qc_weighted_mean"], label="ROA QC-weighted mean", alpha=0.85)
    if "roa_qc_weighted_smoothed" in df and not df["roa_qc_weighted_smoothed"].isna().all():
        plt.plot(x, df["roa_qc_weighted_smoothed"], label="ROA QC-weighted smoothed")
    plt.title("Final ROA spectrum")
    plt.xlabel("Wavenumber / cm$^{-1}$")
    plt.ylabel("ROA intensity, normalized")
    plt.legend()
    _save_or_show(Path(output_path).with_name("final_roa_from_csv.png") if output_path else None)
