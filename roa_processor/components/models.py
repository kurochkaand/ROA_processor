from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class RamanComponent:
    name: str
    source_file: Path
    wavenumber: np.ndarray
    intensity: np.ndarray


@dataclass(frozen=True)
class ComponentLoadResult:
    components_dir: Path
    folder_found: bool
    components: list[RamanComponent]

    @property
    def names(self) -> list[str]:
        return [component.name for component in self.components]


@dataclass(frozen=True)
class ManualRamanCorrectionResult:
    wavenumber: np.ndarray
    raman_before: np.ndarray
    raman_after: np.ndarray
    components: dict[str, RamanComponent]
    coefficients: dict[str, float]
    scaled_components: dict[str, np.ndarray]
    total_component: np.ndarray
    negative_check: dict[str, float | int]

    @property
    def has_negative_warning(self) -> bool:
        return self.negative_check["percentage_negative_points"] > 5.0
