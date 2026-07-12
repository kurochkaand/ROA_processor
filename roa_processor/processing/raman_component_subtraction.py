from __future__ import annotations

import numpy as np

from roa_processor.components.discover import AXIS_ATOL
from roa_processor.components.models import (
    ManualRamanCorrectionResult,
    RamanComponent,
)


MANUAL_COMPONENT_NAMES = ("water", "quartz", "air")


def manual_raman_component_subtraction(
    wavenumber: np.ndarray,
    raman_before: np.ndarray,
    components: list[RamanComponent],
    *,
    water_scale: float,
    quartz_scale: float,
    air_scale: float,
) -> ManualRamanCorrectionResult:
    wavenumber = np.asarray(wavenumber, dtype=float)
    raman_before = np.asarray(raman_before, dtype=float)
    if wavenumber.shape != raman_before.shape:
        raise ValueError("wavenumber and Raman spectrum must have the same shape.")

    coefficients = {
        "water": float(water_scale),
        "quartz": float(quartz_scale),
        "air": float(air_scale),
    }
    _validate_non_negative_coefficients(coefficients)

    component_by_name = {component.name: component for component in components}
    missing = [
        name
        for name in MANUAL_COMPONENT_NAMES
        if name not in component_by_name
    ]
    if missing:
        missing_files = ", ".join(f"{name}.prn" for name in missing)
        required_files = ", ".join(f"{name}.prn" for name in MANUAL_COMPONENT_NAMES)
        raise ValueError(
            "Manual Raman component subtraction requires component files "
            f"{required_files}. Missing: {missing_files}."
        )

    selected_components = {
        name: component_by_name[name]
        for name in MANUAL_COMPONENT_NAMES
    }
    _validate_component_axes(wavenumber, selected_components)

    scaled_components = {
        name: coefficients[name] * component.intensity
        for name, component in selected_components.items()
    }
    total_component = np.sum(
        np.vstack([scaled_components[name] for name in MANUAL_COMPONENT_NAMES]),
        axis=0,
    )
    corrected = raman_before - total_component

    return ManualRamanCorrectionResult(
        wavenumber=wavenumber.copy(),
        raman_before=raman_before.copy(),
        raman_after=corrected,
        components=selected_components,
        coefficients=coefficients,
        scaled_components=scaled_components,
        total_component=total_component,
        negative_check=negative_value_diagnostic(wavenumber, corrected),
    )


def negative_value_diagnostic(
    wavenumber: np.ndarray,
    intensity: np.ndarray,
) -> dict[str, float | int]:
    intensity = np.asarray(intensity, dtype=float)
    wavenumber = np.asarray(wavenumber, dtype=float)
    if intensity.size == 0:
        raise ValueError("Cannot run negative-value diagnostic on an empty spectrum.")
    if wavenumber.shape != intensity.shape:
        raise ValueError("wavenumber and intensity must have the same shape.")

    negative_mask = intensity < 0
    negative_depth = np.where(negative_mask, -intensity, 0.0)
    if intensity.size == 1:
        negative_area = float(negative_depth[0])
    else:
        dx = np.diff(wavenumber)
        negative_area = float(
            np.sum((negative_depth[:-1] + negative_depth[1:]) * 0.5 * dx)
        )

    return {
        "minimum_intensity": float(np.min(intensity)),
        "number_of_negative_points": int(np.sum(negative_mask)),
        "negative_area": negative_area,
        "percentage_negative_points": float(100.0 * np.mean(negative_mask)),
    }


def _validate_non_negative_coefficients(coefficients: dict[str, float]) -> None:
    negative = [
        name
        for name, value in coefficients.items()
        if value < 0
    ]
    if negative:
        names = ", ".join(negative)
        raise ValueError(f"Manual Raman component coefficients must be non-negative: {names}.")


def _validate_component_axes(
    wavenumber: np.ndarray,
    components: dict[str, RamanComponent],
) -> None:
    wavenumber = np.asarray(wavenumber, dtype=float)
    for name, component in components.items():
        if component.wavenumber.shape != wavenumber.shape or not np.allclose(
            component.wavenumber,
            wavenumber,
            rtol=0,
            atol=AXIS_ATOL,
        ):
            raise ValueError(
                "Component wavenumber axis does not match the processed Raman "
                f"wavenumber axis for {name}: {component.source_file}."
            )
