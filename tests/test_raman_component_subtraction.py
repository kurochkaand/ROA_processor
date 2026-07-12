from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
import pytest

matplotlib.use("Agg")

from roa_processor.cli import build_parser
from roa_processor.components.models import RamanComponent
from roa_processor.processing.raman_component_subtraction import (
    manual_raman_component_subtraction,
)


def _component(name: str, intensity: list[float]) -> RamanComponent:
    return RamanComponent(
        name=name,
        source_file=Path(f"{name}.prn"),
        wavenumber=np.array([100.0, 200.0, 300.0]),
        intensity=np.array(intensity, dtype=float),
    )


def _manual_components() -> list[RamanComponent]:
    return [
        _component("water", [1.0, 1.0, 1.0]),
        _component("quartz", [2.0, 2.0, 2.0]),
        _component("air", [3.0, 3.0, 3.0]),
    ]


def _write_prn(path: Path, intensity: list[float]) -> None:
    wavenumber = [100.0, 200.0, 300.0]
    path.write_text(
        "\n".join(f"{x:g} {y:g}" for x, y in zip(wavenumber, intensity, strict=True))
        + "\n",
        encoding="utf-8",
    )


def _write_experiment(folder: Path) -> Path:
    prefix = "sample-N10-500mW-100000s_0012026-06-21"
    info_file = folder / f"{prefix}_info.txt"
    info_file.write_text("synthetic info\n", encoding="utf-8")

    wavenumber_desc = np.array([300.0, 200.0, 100.0])
    for block_index, scale in enumerate([1.0, 2.0]):
        path = folder / f"{prefix}_A-{block_index:03d}_out.txt"
        header = (
            "#1/cm sum, dif: SCP # Gain 1 e/ADc # Power at sample 2 mW "
            f"# Cycles {(block_index + 1) * 10} "
            f"# Total times [s] {(block_index + 1) * 10} "
            f"{(block_index + 1) * 10} {(block_index + 1) * 10} "
            f"{(block_index + 1) * 10}\n"
        )
        data = np.column_stack(
            [
                wavenumber_desc,
                np.array([30.0, 20.0, 10.0]) * scale,
                np.array([3.0, 2.0, 1.0]) * scale,
            ]
        )
        body = "\n".join(" ".join(f"{value:g}" for value in row) for row in data)
        path.write_text(header + body + "\n", encoding="utf-8")

    components_dir = folder / "components"
    components_dir.mkdir()
    _write_prn(components_dir / "water.prn", [1.0, 1.0, 1.0])
    _write_prn(components_dir / "quartz.prn", [2.0, 2.0, 2.0])
    _write_prn(components_dir / "air.prn", [3.0, 3.0, 3.0])
    return info_file


def test_manual_subtraction_with_all_coefficients_one():
    result = manual_raman_component_subtraction(
        np.array([100.0, 200.0, 300.0]),
        np.array([20.0, 30.0, 40.0]),
        _manual_components(),
        water_scale=1.0,
        quartz_scale=1.0,
        air_scale=1.0,
    )

    np.testing.assert_allclose(result.raman_after, [14.0, 24.0, 34.0])


def test_manual_subtraction_allows_quartz_coefficient_greater_than_one():
    result = manual_raman_component_subtraction(
        np.array([100.0, 200.0, 300.0]),
        np.array([20.0, 30.0, 40.0]),
        _manual_components(),
        water_scale=1.0,
        quartz_scale=1.5,
        air_scale=1.0,
    )

    np.testing.assert_allclose(result.raman_after, [13.0, 23.0, 33.0])


def test_manual_subtraction_allows_zero_coefficient():
    result = manual_raman_component_subtraction(
        np.array([100.0, 200.0, 300.0]),
        np.array([20.0, 30.0, 40.0]),
        _manual_components(),
        water_scale=1.0,
        quartz_scale=1.0,
        air_scale=0.0,
    )

    np.testing.assert_allclose(result.raman_after, [17.0, 27.0, 37.0])


def test_manual_subtraction_rejects_negative_coefficients():
    with pytest.raises(ValueError, match="non-negative"):
        manual_raman_component_subtraction(
            np.array([100.0, 200.0, 300.0]),
            np.array([20.0, 30.0, 40.0]),
            _manual_components(),
            water_scale=-1.0,
            quartz_scale=1.0,
            air_scale=1.0,
        )


def test_roa_is_unchanged_by_manual_raman_component_subtraction(tmp_path):
    info_file = _write_experiment(tmp_path)
    none_output = tmp_path / "processed_none"
    manual_output = tmp_path / "processed_manual"

    none_args = build_parser().parse_args(
        [
            "process",
            str(info_file),
            "--output",
            str(none_output),
            "--no-normalize-time",
            "--no-normalize-power",
        ]
    )
    manual_args = build_parser().parse_args(
        [
            "process",
            str(info_file),
            "--output",
            str(manual_output),
            "--no-normalize-time",
            "--no-normalize-power",
            "--raman-component-subtraction",
            "manual",
            "--water-scale",
            "1",
            "--quartz-scale",
            "1",
            "--air-scale",
            "1",
        ]
    )

    none_args.func(none_args)
    manual_args.func(manual_args)

    none_final = pd.read_csv(none_output / "final_spectra.csv")
    manual_final = pd.read_csv(manual_output / "final_spectra.csv")
    np.testing.assert_allclose(
        manual_final["roa_mean_after_spike_removal"],
        none_final["roa_mean_after_spike_removal"],
    )
    np.testing.assert_allclose(manual_final["raman_mean"], [10.0, 20.0, 30.0])
    np.testing.assert_allclose(
        manual_final["raman_component_corrected_manual"],
        [4.0, 14.0, 24.0],
    )

    correction_dir = manual_output / "raman_correction"
    assert (correction_dir / "raman_before_component_subtraction.csv").exists()
    assert (correction_dir / "raman_after_manual_component_subtraction.csv").exists()
    assert (correction_dir / "manual_component_coefficients.csv").exists()
    assert (correction_dir / "manual_component_residual.csv").exists()
    assert (correction_dir / "manual_negative_check.csv").exists()
    assert (manual_output / "figures" / "raman_manual_component_subtraction.png").exists()
    assert (manual_output / "figures" / "raman_manual_components_scaled.png").exists()
    assert (manual_output / "figures" / "raman_manual_before_after.png").exists()
