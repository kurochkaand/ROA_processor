from pathlib import Path

import numpy as np
import pandas as pd

from roa_processor.io.export import resolve_output_path, save_final_spectra
from roa_processor.io.load_experiment import load_experiment
from roa_processor.models import FinalSpectra, IsolatedExperiment, SpikeResult
from roa_processor.processing.average import make_final_spectra
from roa_processor.processing.isolate_blocks import cumulative_to_isolated
from roa_processor.processing.roa_qc import (
    MAD_TO_SIGMA,
    analyze_roa_qc,
    estimate_qc_noise,
    qc_noise_weights,
)


def _write_experiment(folder: Path) -> Path:
    prefix = "sample-N10-500mW-100000s_0012026-06-21"
    info_file = folder / f"{prefix}_info.txt"
    info_file.write_text("synthetic info\n", encoding="utf-8")

    wavenumber_desc = np.array([500.0, 300.0, 100.0])
    for block_index, scale in enumerate([1.0, 3.0]):
        path = folder / f"{prefix}_A-{block_index:03d}_out.txt"
        header = (
            "#1/cm sum, dif: SCP # Gain 1 e/ADc # Power at sample 2 mW "
            f"# Cycles {(block_index + 1) * 10} "
            f"# Total times [s] {(block_index + 1) * 10} "
            f"{(block_index + 1) * 10} {(block_index + 1) * 10} {(block_index + 1) * 10}\n"
        )
        raman = np.array([50.0, 30.0, 10.0]) * scale
        roa = np.array([5.0, 3.0, 1.0]) * scale
        data = np.column_stack([wavenumber_desc, raman, roa])
        body = "\n".join(" ".join(f"{value:g}" for value in row) for row in data)
        path.write_text(header + body + "\n", encoding="utf-8")

    return info_file


def test_min_wavenumber_filters_loaded_experiment_before_isolation(tmp_path):
    info_file = _write_experiment(tmp_path)

    experiment = load_experiment(info_file, min_wavenumber=300)
    isolated = cumulative_to_isolated(experiment)

    np.testing.assert_allclose(experiment.wavenumber, [300.0, 500.0])
    assert experiment.original_wavenumber_range == (100.0, 500.0)
    assert experiment.processed_wavenumber_range == (300.0, 500.0)
    assert isolated.raman_raw.shape == (2, 2)
    np.testing.assert_allclose(isolated.wavenumber, [300.0, 500.0])


def test_min_wavenumber_filter_is_reflected_in_export(tmp_path):
    output = tmp_path / "processed"
    final = FinalSpectra(
        wavenumber=np.array([300.0, 500.0]),
        raman_mean=np.array([1.0, 2.0]),
        raman_median=np.array([1.0, 2.0]),
        roa_mean_before_spike_removal=np.array([0.1, 0.2]),
        roa_mean_after_spike_removal=np.array([0.1, 0.2]),
        roa_median_after_spike_removal=np.array([0.1, 0.2]),
        n_spikes_at_wavenumber=np.array([0, 1]),
    )

    save_final_spectra(output, final)

    df = pd.read_csv(output / "final_spectra.csv")
    assert df["wavenumber_cm-1"].min() == 300.0
    assert df["roa_mean_after_qc_rejection"].isna().all()
    assert df["roa_qc_weighted_mean"].isna().all()


def test_relative_output_resolves_against_input_data_directory(tmp_path):
    data_dir = tmp_path / "data"
    data_dir.mkdir()

    resolved = resolve_output_path("processed", base_dir=data_dir)

    assert resolved == (data_dir / "processed").resolve()


def test_absolute_output_is_preserved(tmp_path):
    absolute = tmp_path / "absolute_processed"

    resolved = resolve_output_path(absolute, base_dir=tmp_path / "data")

    assert resolved == absolute


def test_qc_range_selection_works_after_wavenumber_filtering():
    wavenumber = np.array([200.0, 1800.0, 1900.0, 2609.0, 2700.0])
    roa = np.array(
        [
            [0.0, -1.0, 0.0, 1.0, 0.0],
            [0.0, -2.0, 0.0, 2.0, 0.0],
        ]
    )
    spikes = np.zeros_like(roa, dtype=bool)

    result = analyze_roa_qc(
        wavenumber,
        roa,
        spikes,
        np.array([0, 1]),
        qc_range=(1800.0, 2609.0),
        min_qc_points=3,
    )

    np.testing.assert_array_equal(result.qc_mask, [False, True, True, True, False])
    assert result.used_range == (1800.0, 2609.0)


def test_qc_noise_calculation_uses_mad_sigma():
    noise = estimate_qc_noise(np.array([9.0, 10.0, 11.0]))

    assert noise["qc_median_offset"] == 10.0
    assert noise["qc_noise_mad"] == 1.0
    assert noise["qc_noise_sigma"] == MAD_TO_SIGMA
    np.testing.assert_allclose(noise["qc_rms"], np.sqrt(2.0 / 3.0))


def test_qc_weighted_average_gives_lower_weight_to_noisier_blocks():
    weights = qc_noise_weights(np.array([1.0, 2.0, 4.0]))

    assert weights[0] > weights[1] > weights[2]
    np.testing.assert_allclose(weights.sum(), 1.0)


def test_highest_noise_rejection_rejects_requested_noisiest_blocks():
    wavenumber = np.array([1800.0, 1900.0, 2609.0])
    roa = np.array(
        [
            [-1.0, 0.0, 1.0],
            [-2.0, 0.0, 2.0],
            [-5.0, 0.0, 5.0],
            [-10.0, 0.0, 10.0],
        ]
    )
    spikes = np.zeros_like(roa, dtype=bool)

    result = analyze_roa_qc(
        wavenumber,
        roa,
        spikes,
        np.array([1, 2, 3, 4]),
        qc_range=(1800.0, 2609.0),
        highest_noise_reject_blocks=2,
        min_qc_points=3,
    )

    np.testing.assert_array_equal(result.rejected_mask, [False, False, True, True])
    assert result.n_rejected == 2
    assert result.reject_blocks_enabled
    assert [row["qc_accepted"] for row in result.block_summary] == [True, True, False, False]


def test_final_spectra_includes_mean_after_qc_rejection():
    isolated = IsolatedExperiment(
        wavenumber=np.array([100.0, 200.0]),
        raman_raw=np.zeros((3, 2)),
        roa_raw=np.zeros((3, 2)),
        raman_norm=np.zeros((3, 2)),
        roa_norm=np.array([[1.0, 10.0], [3.0, 30.0], [100.0, 1000.0]]),
        delta_times_s=np.ones(3),
        delta_cycles=np.ones(3),
        power_at_sample_mw=np.ones(3),
        block_indices=np.array([1, 2, 3]),
    )
    spike_result = SpikeResult(
        roa_cleaned=isolated.roa_norm.copy(),
        spike_mask=np.zeros((3, 2), dtype=bool),
        spike_summary=[],
        threshold=8.0,
    )

    final = make_final_spectra(
        isolated,
        spike_result,
        qc_accepted_mask=np.array([True, True, False]),
    )

    np.testing.assert_allclose(final.roa_mean_after_qc_rejection, [2.0, 20.0])


def test_default_loading_keeps_full_wavenumber_range(tmp_path):
    info_file = _write_experiment(tmp_path)

    experiment = load_experiment(info_file)

    np.testing.assert_allclose(experiment.wavenumber, [100.0, 300.0, 500.0])
    assert experiment.original_wavenumber_range == experiment.processed_wavenumber_range
