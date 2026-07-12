from __future__ import annotations

from pathlib import Path

import numpy as np

from roa_processor.components.models import ComponentLoadResult, RamanComponent
from roa_processor.components.read_prn import read_prn_component


AXIS_ATOL = 1e-8


def load_components_for_experiment(
    info_file: str | Path,
    processed_wavenumber_axis: np.ndarray | None = None,
) -> ComponentLoadResult:
    experiment_root = Path(info_file).parent
    components_dir = experiment_root / "components"

    if not components_dir.exists():
        return ComponentLoadResult(
            components_dir=components_dir,
            folder_found=False,
            components=[],
        )
    if not components_dir.is_dir():
        raise NotADirectoryError(
            f"Components path exists but is not a folder: {components_dir}"
        )

    components = [
        read_prn_component(path)
        for path in discover_component_files(components_dir)
    ]
    validate_component_axes(components)
    if processed_wavenumber_axis is not None:
        validate_components_match_processed_axis(
            components,
            processed_wavenumber_axis,
        )

    return ComponentLoadResult(
        components_dir=components_dir,
        folder_found=True,
        components=components,
    )


def discover_component_files(components_dir: str | Path) -> list[Path]:
    components_dir = Path(components_dir)
    return sorted(
        path
        for path in components_dir.iterdir()
        if path.is_file() and path.suffix.lower() == ".prn"
    )


def validate_component_axes(components: list[RamanComponent]) -> None:
    if not components:
        return

    reference = components[0]
    for component in components[1:]:
        if not _axes_match(reference.wavenumber, component.wavenumber):
            raise ValueError(
                "Component wavenumber axes do not match: "
                f"{reference.source_file} and {component.source_file}."
            )


def validate_components_match_processed_axis(
    components: list[RamanComponent],
    processed_wavenumber_axis: np.ndarray,
) -> None:
    if not components:
        return

    reference = components[0]
    if not _axes_match(reference.wavenumber, processed_wavenumber_axis):
        raise ValueError(
            "Component wavenumber axis does not match the processed Raman "
            f"wavenumber axis: {reference.source_file}. Interpolation is not "
            "implemented yet, so component axes must match exactly."
        )


def _axes_match(left: np.ndarray, right: np.ndarray) -> bool:
    return left.shape == right.shape and bool(
        np.allclose(left, right, rtol=0, atol=AXIS_ATOL)
    )

