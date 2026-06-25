from __future__ import annotations

from pathlib import Path

import numpy as np

from roa_processor.io.discover import discover_out_files
from roa_processor.io.read_out_file import read_out_file
from roa_processor.models import LoadedExperiment, SpectrumBlock


def load_experiment(
    info_file: str | Path,
    camera: str = "A",
    reverse_axis: bool = True,
) -> LoadedExperiment:
    prefix, paths = discover_out_files(info_file, camera=camera)
    blocks = [read_out_file(p, reverse_axis=reverse_axis) for p in paths]
    blocks = sorted(blocks, key=lambda b: b.metadata.block_index)
    validate_blocks(blocks)
    return LoadedExperiment(info_file=Path(info_file), prefix=prefix, blocks=blocks)


def validate_blocks(blocks: list[SpectrumBlock]) -> None:
    if not blocks:
        raise ValueError("No blocks loaded.")

    block_indices = [b.metadata.block_index for b in blocks]
    expected = list(range(min(block_indices), max(block_indices) + 1))
    if block_indices != expected:
        raise ValueError(
            f"Block indices are not continuous. Found {block_indices[:5]}...{block_indices[-5:]}, "
            f"expected continuous range {expected[0]}..{expected[-1]}"
        )

    ref_axis = blocks[0].wavenumber
    for block in blocks[1:]:
        if block.wavenumber.shape != ref_axis.shape:
            raise ValueError(
                f"Wavenumber axis length differs in block {block.metadata.block_index}."
            )
        if not np.allclose(block.wavenumber, ref_axis, rtol=0, atol=1e-8):
            raise ValueError(
                f"Wavenumber axis differs in block {block.metadata.block_index}."
            )

    cycles = [b.metadata.header.cumulative_cycles for b in blocks]
    if all(c is not None for c in cycles):
        if any(cycles[i] <= cycles[i - 1] for i in range(1, len(cycles))):
            raise ValueError(f"Cumulative cycles are not strictly increasing: {cycles}")

    times = [b.metadata.header.total_time_scp_s for b in blocks]
    if all(t is not None for t in times):
        if any(times[i] <= times[i - 1] for i in range(1, len(times))):
            raise ValueError(f"Cumulative total times are not strictly increasing: {times}")

    cameras = {b.metadata.filename.camera for b in blocks}
    if len(cameras) != 1:
        raise ValueError(f"More than one camera found in blocks: {sorted(cameras)}")
