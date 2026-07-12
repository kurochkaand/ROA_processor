from argparse import Namespace
from pathlib import Path

import numpy as np
import pytest

from roa_processor.cli import cmd_inspect
from roa_processor.components.discover import load_components_for_experiment
from roa_processor.components.read_prn import read_prn_component


def _write_prn(path: Path, wavenumber: list[float], intensity: list[float]) -> None:
    rows = zip(wavenumber, intensity, strict=True)
    path.write_text(
        "\n".join(f"{x:g} {y:g}" for x, y in rows) + "\n",
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

    return info_file


def test_read_prn_component_loads_name_source_and_xy_data(tmp_path):
    path = tmp_path / "water.prn"
    _write_prn(path, [100.0, 200.0], [1.0, 2.0])

    component = read_prn_component(path)

    assert component.name == "water"
    assert component.source_file == path
    np.testing.assert_allclose(component.wavenumber, [100.0, 200.0])
    np.testing.assert_allclose(component.intensity, [1.0, 2.0])


def test_read_prn_component_rejects_non_two_column_files(tmp_path):
    path = tmp_path / "water.prn"
    path.write_text("100 1 99\n200 2 99\n", encoding="utf-8")

    with pytest.raises(ValueError, match="exactly two numeric columns"):
        read_prn_component(path)


def test_missing_components_folder_disables_component_loading(tmp_path):
    result = load_components_for_experiment(
        tmp_path / "sample_info.txt",
        processed_wavenumber_axis=np.array([100.0, 200.0]),
    )

    assert not result.folder_found
    assert result.components == []


def test_components_folder_loads_all_prn_files_sorted_by_filename(tmp_path):
    components_dir = tmp_path / "components"
    components_dir.mkdir()
    _write_prn(components_dir / "water.prn", [100.0, 200.0], [1.0, 2.0])
    _write_prn(components_dir / "air.prn", [100.0, 200.0], [3.0, 4.0])
    (components_dir / "notes.txt").write_text("ignored\n", encoding="utf-8")

    result = load_components_for_experiment(
        tmp_path / "sample_info.txt",
        processed_wavenumber_axis=np.array([100.0, 200.0]),
    )

    assert result.folder_found
    assert result.names == ["air", "water"]


def test_component_files_must_share_wavenumber_axis(tmp_path):
    components_dir = tmp_path / "components"
    components_dir.mkdir()
    _write_prn(components_dir / "air.prn", [100.0, 200.0], [1.0, 2.0])
    _write_prn(components_dir / "water.prn", [100.0, 250.0], [1.0, 2.0])

    with pytest.raises(ValueError, match="Component wavenumber axes do not match"):
        load_components_for_experiment(tmp_path / "sample_info.txt")


def test_component_axis_must_match_processed_raman_axis(tmp_path):
    components_dir = tmp_path / "components"
    components_dir.mkdir()
    _write_prn(components_dir / "water.prn", [100.0, 250.0], [1.0, 2.0])

    with pytest.raises(ValueError, match="processed Raman wavenumber axis"):
        load_components_for_experiment(
            tmp_path / "sample_info.txt",
            processed_wavenumber_axis=np.array([100.0, 200.0]),
        )


def test_inspect_reports_loaded_components(tmp_path, capsys):
    info_file = _write_experiment(tmp_path)
    components_dir = tmp_path / "components"
    components_dir.mkdir()
    _write_prn(components_dir / "water.prn", [100.0, 200.0, 300.0], [1.0, 2.0, 3.0])

    cmd_inspect(
        Namespace(
            info_file=info_file,
            camera="A",
            min_wavenumber=None,
            max_wavenumber=None,
        )
    )

    output = capsys.readouterr().out
    assert "Components folder: found" in output
    assert "Components loaded:" in output
    assert "- water" in output


def test_inspect_reports_missing_components_folder(tmp_path, capsys):
    info_file = _write_experiment(tmp_path)

    cmd_inspect(
        Namespace(
            info_file=info_file,
            camera="A",
            min_wavenumber=None,
            max_wavenumber=None,
        )
    )

    output = capsys.readouterr().out
    assert "Components folder: not found" in output
    assert "Raman component correction: disabled" in output

