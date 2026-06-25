from __future__ import annotations

import re
from pathlib import Path

from roa_processor.models import FilenameMetadata


OUT_FILE_RE = re.compile(
    r"^(?P<full_prefix>.+)_(?P<camera>[A-Za-z])-(?P<block_index>\d+)_out\.txt$"
)

# This is intentionally permissive because the user-written sample name may contain many underscores.
# It extracts the instrument-like suffix from the full prefix.
PREFIX_RE = re.compile(
    r"^(?P<sample_name>.+)-N(?P<save_cycles>\d+)-"
    r"(?P<nominal_power_mw>[0-9.]+)mW-"
    r"(?P<camera_exposure_raw>\d+)s_"
    r"(?P<experiment_number>\d+)"
    r"(?P<date>\d{4}-\d{2}-\d{2})$"
)


def parse_camera_exposure(raw: str) -> float:
    """
    Convert camera exposure encoded as e.g. 146791s to 1.46791 s.

    The instrument filename seems to store 1.46791 s as 146791s,
    so we divide by 100000.
    """
    return int(raw) / 100000.0


def parse_filename(path: str | Path) -> FilenameMetadata:
    path = Path(path)
    match = OUT_FILE_RE.match(path.name)
    if not match:
        raise ValueError(f"Cannot parse output filename: {path.name}")

    full_prefix = match.group("full_prefix")
    camera = match.group("camera")
    block_index = int(match.group("block_index"))

    prefix_match = PREFIX_RE.match(full_prefix)

    sample_name: str
    save_cycles: int | None = None
    nominal_power_mw: float | None = None
    camera_exposure_s: float | None = None
    experiment_number: str | None = None
    date: str | None = None

    if prefix_match:
        sample_name = prefix_match.group("sample_name")
        save_cycles = int(prefix_match.group("save_cycles"))
        nominal_power_mw = float(prefix_match.group("nominal_power_mw"))
        camera_exposure_s = parse_camera_exposure(prefix_match.group("camera_exposure_raw"))
        experiment_number = prefix_match.group("experiment_number")
        date = prefix_match.group("date")
    else:
        # Fall back to using the whole prefix as the sample name.
        # This keeps the parser usable even if naming slightly changes.
        sample_name = full_prefix

    return FilenameMetadata(
        source_file=str(path),
        prefix=full_prefix,
        sample_name=sample_name,
        save_interval_cycles=save_cycles,
        nominal_laser_power_mw=nominal_power_mw,
        camera_exposure_s=camera_exposure_s,
        experiment_number=experiment_number,
        date=date,
        camera=camera,
        block_index=block_index,
    )
