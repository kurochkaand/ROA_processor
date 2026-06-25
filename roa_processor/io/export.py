from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd

from roa_processor.models import (
    FinalSpectra,
    IsolatedExperiment,
    LoadedExperiment,
    SpikeResult,
)


def ensure_output_dirs(output: str | Path) -> Path:
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    (output / "figures").mkdir(parents=True, exist_ok=True)
    return output


def save_metadata(
    output: str | Path,
    experiment: LoadedExperiment,
    extra: dict | None = None,
) -> None:
    output = ensure_output_dirs(output)
    metadata = experiment.metadata_dict()
    if extra:
        metadata["processing"] = extra

    with (output / "metadata.json").open("w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)


def save_isolated_npz(output: str | Path, isolated: IsolatedExperiment) -> None:
    output = ensure_output_dirs(output)
    np.savez_compressed(
        output / "isolated_blocks.npz",
        wavenumber=isolated.wavenumber,
        raman_raw=isolated.raman_raw,
        roa_raw=isolated.roa_raw,
        raman_norm=isolated.raman_norm,
        roa_norm=isolated.roa_norm,
        delta_times_s=isolated.delta_times_s,
        delta_cycles=isolated.delta_cycles,
        power_at_sample_mw=isolated.power_at_sample_mw,
        block_indices=isolated.block_indices,
    )


def save_spikes(output: str | Path, isolated: IsolatedExperiment, spike_result: SpikeResult) -> None:
    output = ensure_output_dirs(output)

    # Spike mask table: rows = blocks, columns = wavenumbers.
    spike_mask_df = pd.DataFrame(
        spike_result.spike_mask.astype(int),
        index=isolated.block_indices,
        columns=[f"{x:.6g}" for x in isolated.wavenumber],
    )
    spike_mask_df.index.name = "block_index"
    spike_mask_df.to_csv(output / "spike_mask.csv")

    spike_summary_df = pd.DataFrame(spike_result.spike_summary)
    spike_summary_df.to_csv(output / "spike_summary.csv", index=False)

    np.savez_compressed(
        output / "roa_spike_cleaning.npz",
        wavenumber=isolated.wavenumber,
        roa_before=isolated.roa_norm,
        roa_cleaned=spike_result.roa_cleaned,
        spike_mask=spike_result.spike_mask,
        block_indices=isolated.block_indices,
    )


def save_final_spectra(output: str | Path, final: FinalSpectra) -> None:
    output = ensure_output_dirs(output)
    df = pd.DataFrame(
        {
            "wavenumber_cm-1": final.wavenumber,
            "raman_mean": final.raman_mean,
            "raman_median": final.raman_median,
            "roa_mean_before_spike_removal": final.roa_mean_before_spike_removal,
            "roa_mean_after_spike_removal": final.roa_mean_after_spike_removal,
            "roa_median_after_spike_removal": final.roa_median_after_spike_removal,
            "number_of_spikes_at_this_wavenumber": final.n_spikes_at_wavenumber,
        }
    )
    df.to_csv(output / "final_spectra.csv", index=False)


def load_processed_npz(output: str | Path) -> dict[str, np.ndarray]:
    output = Path(output)
    path = output / "roa_spike_cleaning.npz"
    if not path.exists():
        raise FileNotFoundError(
            f"Cannot find {path}. Run 'roa process ...' first."
        )
    return dict(np.load(path))
