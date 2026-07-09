from __future__ import annotations

from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any

import numpy as np


@dataclass(frozen=True)
class FilenameMetadata:
    source_file: str
    prefix: str
    sample_name: str
    save_interval_cycles: int | None
    nominal_laser_power_mw: float | None
    camera_exposure_s: float | None
    experiment_number: str | None
    date: str | None
    camera: str
    block_index: int


@dataclass(frozen=True)
class HeaderMetadata:
    gain_e_per_adc: float | None
    power_at_sample_mw: float | None
    cumulative_cycles: int | None
    total_times_s: tuple[float, ...]
    raw_header: str

    @property
    def total_time_scp_s(self) -> float | None:
        if not self.total_times_s:
            return None
        return self.total_times_s[0]


@dataclass(frozen=True)
class BlockMetadata:
    filename: FilenameMetadata
    header: HeaderMetadata

    @property
    def block_index(self) -> int:
        return self.filename.block_index

    @property
    def source_file(self) -> str:
        return self.filename.source_file


@dataclass
class SpectrumBlock:
    wavenumber: np.ndarray
    raman: np.ndarray
    roa: np.ndarray
    metadata: BlockMetadata


@dataclass
class LoadedExperiment:
    info_file: Path
    prefix: str
    blocks: list[SpectrumBlock]
    original_wavenumber_range: tuple[float, float] | None = None
    processed_wavenumber_range: tuple[float, float] | None = None
    original_spectral_points: int | None = None
    processed_spectral_points: int | None = None

    @property
    def n_blocks(self) -> int:
        return len(self.blocks)

    @property
    def wavenumber(self) -> np.ndarray:
        if not self.blocks:
            raise ValueError("Experiment has no blocks.")
        return self.blocks[0].wavenumber

    def metadata_dict(self) -> dict[str, Any]:
        return {
            "info_file": str(self.info_file),
            "prefix": self.prefix,
            "n_blocks": self.n_blocks,
            "original_wavenumber_range": self.original_wavenumber_range,
            "processed_wavenumber_range": self.processed_wavenumber_range,
            "original_spectral_points": self.original_spectral_points,
            "processed_spectral_points": self.processed_spectral_points,
            "blocks": [
                {
                    "filename": asdict(block.metadata.filename),
                    "header": asdict(block.metadata.header),
                }
                for block in self.blocks
            ],
        }


@dataclass
class IsolatedExperiment:
    wavenumber: np.ndarray
    raman_raw: np.ndarray
    roa_raw: np.ndarray
    raman_norm: np.ndarray
    roa_norm: np.ndarray
    delta_times_s: np.ndarray
    delta_cycles: np.ndarray
    power_at_sample_mw: np.ndarray
    block_indices: np.ndarray


@dataclass
class SpikeResult:
    roa_cleaned: np.ndarray
    spike_mask: np.ndarray
    spike_summary: list[dict[str, Any]]
    threshold: float


@dataclass
class FinalSpectra:
    wavenumber: np.ndarray
    raman_mean: np.ndarray
    raman_median: np.ndarray
    roa_mean_before_spike_removal: np.ndarray
    roa_mean_after_spike_removal: np.ndarray
    roa_median_after_spike_removal: np.ndarray
    n_spikes_at_wavenumber: np.ndarray
    roa_mean_after_qc_rejection: np.ndarray | None = None
    roa_qc_weighted_mean: np.ndarray | None = None
    roa_qc_weighted_smoothed: np.ndarray | None = None
    roa_qc_removed_noise: np.ndarray | None = None


@dataclass
class RoaQcResult:
    requested_range: tuple[float, float]
    used_range: tuple[float, float] | None
    qc_mask: np.ndarray
    block_summary: list[dict[str, Any]]
    accepted_mask: np.ndarray
    rejected_mask: np.ndarray
    weights: np.ndarray | None
    weighted_mean: np.ndarray | None
    smoothed: np.ndarray | None
    removed_noise: np.ndarray | None
    denoise_enabled: bool
    reject_blocks_enabled: bool
    smoothing_enabled: bool
    smoothing_parameters: dict[str, Any] | None
    warning: str | None = None

    @property
    def is_available(self) -> bool:
        return self.used_range is not None and bool(np.any(self.qc_mask))

    @property
    def n_rejected(self) -> int:
        return int(np.sum(self.rejected_mask))
