from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from roa_processor.components.models import RamanComponent


def read_prn_component(path: str | Path) -> RamanComponent:
    path = Path(path)
    data = _read_two_column_table(path)

    try:
        wavenumber = data.iloc[:, 0].to_numpy(dtype=float)
        intensity = data.iloc[:, 1].to_numpy(dtype=float)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"Component file {path} must contain exactly two numeric columns."
        ) from exc

    if wavenumber.size == 0:
        raise ValueError(f"Component file {path} has no data rows.")
    if not np.isfinite(wavenumber).all() or not np.isfinite(intensity).all():
        raise ValueError(f"Component file {path} contains non-finite numeric values.")

    return RamanComponent(
        name=path.stem,
        source_file=path,
        wavenumber=wavenumber,
        intensity=intensity,
    )


def _read_two_column_table(path: Path) -> pd.DataFrame:
    try:
        data = pd.read_csv(
            path,
            sep=r"\s+",
            comment="#",
            header=None,
            engine="python",
        )
    except pd.errors.EmptyDataError as exc:
        raise ValueError(f"Component file {path} has no data rows.") from exc
    except pd.errors.ParserError as exc:
        raise ValueError(
            f"Component file {path} must contain exactly two numeric columns."
        ) from exc

    if data.shape[1] != 2:
        raise ValueError(
            f"Component file {path} must contain exactly two numeric columns; "
            f"found {data.shape[1]}."
        )

    return data

