from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from roa_processor.io.parse_filename import parse_filename
from roa_processor.io.parse_header import parse_header
from roa_processor.models import BlockMetadata, SpectrumBlock


def read_out_file(path: str | Path, reverse_axis: bool = True) -> SpectrumBlock:
    """
    Read one *_out.txt file.

    For MVP we use only SCP:
    column 1 = wavenumber
    column 2 = Raman sum
    column 3 = ROA difference
    """
    path = Path(path)

    with path.open("r", encoding="utf-8", errors="replace") as f:
        header = f.readline().strip()

    header_meta = parse_header(header)
    filename_meta = parse_filename(path)

    # Comment/header line starts with '#', so pandas skips it.
    data = pd.read_csv(
        path,
        sep=r"\s+",
        comment="#",
        header=None,
        engine="python",
    )

    if data.shape[1] < 3:
        raise ValueError(
            f"Expected at least 3 numeric columns in {path.name}, got {data.shape[1]}"
        )

    wavenumber = data.iloc[:, 0].to_numpy(dtype=float)
    raman = data.iloc[:, 1].to_numpy(dtype=float)
    roa = data.iloc[:, 2].to_numpy(dtype=float)

    if reverse_axis and wavenumber[0] > wavenumber[-1]:
        wavenumber = wavenumber[::-1].copy()
        raman = raman[::-1].copy()
        roa = roa[::-1].copy()

    return SpectrumBlock(
        wavenumber=wavenumber,
        raman=raman,
        roa=roa,
        metadata=BlockMetadata(filename=filename_meta, header=header_meta),
    )
