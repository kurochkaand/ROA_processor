from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from roa_processor.models import FinalSpectra, IsolatedExperiment, SpikeResult


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
    plt.title("Final ROA spectrum")
    plt.xlabel("Wavenumber / cm$^{-1}$")
    plt.ylabel("ROA intensity, normalized")
    plt.legend()
    _save_or_show(Path(output_path).with_name("final_roa.png") if output_path else None)


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
    plt.title("Final ROA spectrum")
    plt.xlabel("Wavenumber / cm$^{-1}$")
    plt.ylabel("ROA intensity, normalized")
    plt.legend()
    _save_or_show(Path(output_path).with_name("final_roa_from_csv.png") if output_path else None)
