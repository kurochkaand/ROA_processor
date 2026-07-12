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

