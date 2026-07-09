from __future__ import annotations

import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd

from roa_processor.models import (
    FinalSpectra,
    IsolatedExperiment,
    LoadedExperiment,
    RoaQcResult,
    SpikeResult,
)


def resolve_output_path(output: str | Path, base_dir: str | Path | None = None) -> Path:
    output = Path(output)
    if not output.is_absolute():
        if base_dir is not None:
            output = Path(base_dir) / output
        output = output.resolve()
    return output


def ensure_output_dirs(
    output: str | Path,
    base_dir: str | Path | None = None,
    *,
    replace_existing: bool = False,
) -> Path:
    output = resolve_output_path(output, base_dir=base_dir)
    if replace_existing and output.exists():
        _validate_replacement_target(output, base_dir=base_dir)
        shutil.rmtree(output)
    output.mkdir(parents=True, exist_ok=True)
    (output / "figures").mkdir(parents=True, exist_ok=True)
    return output


def _validate_replacement_target(
    output: Path,
    base_dir: str | Path | None = None,
) -> None:
    if output.is_symlink() or not output.is_dir():
        raise FileExistsError(f"Output path exists and is not a directory: {output}")

    if output == output.parent:
        raise ValueError(f"Refusing to replace filesystem root: {output}")

    if base_dir is not None:
        source_dir = Path(base_dir).resolve()
        if source_dir == output or source_dir.is_relative_to(output):
            raise ValueError(
                "Refusing to replace output directory because it contains the input data "
                f"directory: {output}"
            )


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


def save_processing_config(output: str | Path, config: dict) -> None:
    output = ensure_output_dirs(output)
    with (output / "processing_config.json").open("w", encoding="utf-8") as f:
        json.dump(config, f, indent=2)


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
    n_points = len(final.wavenumber)
    df = pd.DataFrame(
        {
            "wavenumber_cm-1": final.wavenumber,
            "raman_mean": final.raman_mean,
            "raman_median": final.raman_median,
            "roa_mean_before_spike_removal": final.roa_mean_before_spike_removal,
            "roa_mean_after_spike_removal": final.roa_mean_after_spike_removal,
            "roa_mean_after_qc_rejection": _optional_column(
                final.roa_mean_after_qc_rejection,
                n_points,
            ),
            "roa_median_after_spike_removal": final.roa_median_after_spike_removal,
            "roa_qc_weighted_mean": _optional_column(final.roa_qc_weighted_mean, n_points),
            "roa_qc_weighted_smoothed": _optional_column(
                final.roa_qc_weighted_smoothed,
                n_points,
            ),
            "roa_qc_removed_noise": _optional_column(final.roa_qc_removed_noise, n_points),
            "number_of_spikes_at_this_wavenumber": final.n_spikes_at_wavenumber,
        }
    )
    df.to_csv(output / "final_spectra.csv", index=False)


def _optional_column(values: np.ndarray | None, n_points: int) -> np.ndarray:
    if values is None:
        return np.full(n_points, np.nan)
    return values


def save_roa_qc(output: str | Path, qc_result: RoaQcResult) -> None:
    output = ensure_output_dirs(output)
    pd.DataFrame(qc_result.block_summary).to_csv(
        output / "roa_qc_block_summary.csv",
        index=False,
    )

    weights = (
        qc_result.weights
        if qc_result.weights is not None
        else np.full(len(qc_result.accepted_mask), np.nan)
    )
    np.savez_compressed(
        output / "roa_qc.npz",
        qc_mask=qc_result.qc_mask,
        accepted_mask=qc_result.accepted_mask,
        rejected_mask=qc_result.rejected_mask,
        weights=weights,
        weighted_mean=np.array([])
        if qc_result.weighted_mean is None
        else qc_result.weighted_mean,
        smoothed=np.array([]) if qc_result.smoothed is None else qc_result.smoothed,
        removed_noise=np.array([])
        if qc_result.removed_noise is None
        else qc_result.removed_noise,
    )


def load_processed_npz(output: str | Path) -> dict[str, np.ndarray]:
    output = Path(output)
    path = output / "roa_spike_cleaning.npz"
    if not path.exists():
        raise FileNotFoundError(
            f"Cannot find {path}. Run 'roa process ...' first."
        )
    return dict(np.load(path))


def load_isolated_npz(output: str | Path) -> dict[str, np.ndarray]:
    output = Path(output)
    path = output / "isolated_blocks.npz"
    if not path.exists():
        raise FileNotFoundError(
            f"Cannot find {path}. Run 'roa process ...' first."
        )
    return dict(np.load(path))
