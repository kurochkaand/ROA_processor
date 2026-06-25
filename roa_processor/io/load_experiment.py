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
    min_wavenumber: float | None = None,
    max_wavenumber: float | None = None,
) -> LoadedExperiment:
    prefix, paths = discover_out_files(info_file, camera=camera)
    blocks = [read_out_file(p, reverse_axis=reverse_axis) for p in paths]
    blocks = sorted(blocks, key=lambda b: b.metadata.block_index)
    validate_blocks(blocks)

    original_range = wavenumber_range(blocks[0].wavenumber)
    original_points = len(blocks[0].wavenumber)
    blocks = filter_blocks_by_wavenumber(
        blocks,
        min_wavenumber=min_wavenumber,
        max_wavenumber=max_wavenumber,
    )
    processed_range = wavenumber_range(blocks[0].wavenumber)

    return LoadedExperiment(
        info_file=Path(info_file),
        prefix=prefix,
        blocks=blocks,
        original_wavenumber_range=original_range,
        processed_wavenumber_range=processed_range,
        original_spectral_points=original_points,
        processed_spectral_points=len(blocks[0].wavenumber),
    )


def wavenumber_range(wavenumber: np.ndarray) -> tuple[float, float]:
    if wavenumber.size == 0:
        raise ValueError("Cannot determine range for an empty wavenumber axis.")
    return (float(wavenumber[0]), float(wavenumber[-1]))


def wavenumber_filter_mask(
    wavenumber: np.ndarray,
    min_wavenumber: float | None = None,
    max_wavenumber: float | None = None,
) -> np.ndarray:
    if min_wavenumber is not None and max_wavenumber is not None:
        if min_wavenumber > max_wavenumber:
            raise ValueError(
                f"min_wavenumber ({min_wavenumber}) cannot be greater than "
                f"max_wavenumber ({max_wavenumber})."
            )

    mask = np.ones(wavenumber.shape, dtype=bool)
    if min_wavenumber is not None:
        mask &= wavenumber >= min_wavenumber
    if max_wavenumber is not None:
        mask &= wavenumber <= max_wavenumber

    if not np.any(mask):
        raise ValueError(
            "Wavenumber filtering removed every spectral point. "
            f"Available range is {wavenumber[0]:.6g} to {wavenumber[-1]:.6g} cm^-1."
        )

    return mask


def filter_blocks_by_wavenumber(
    blocks: list[SpectrumBlock],
    min_wavenumber: float | None = None,
    max_wavenumber: float | None = None,
) -> list[SpectrumBlock]:
    if min_wavenumber is None and max_wavenumber is None:
        return blocks

    mask = wavenumber_filter_mask(
        blocks[0].wavenumber,
        min_wavenumber=min_wavenumber,
        max_wavenumber=max_wavenumber,
    )

    return [
        SpectrumBlock(
            wavenumber=block.wavenumber[mask].copy(),
            raman=block.raman[mask].copy(),
            roa=block.roa[mask].copy(),
            metadata=block.metadata,
        )
        for block in blocks
    ]


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
