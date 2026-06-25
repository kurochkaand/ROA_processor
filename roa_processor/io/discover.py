from __future__ import annotations

from pathlib import Path


def prefix_from_info_file(info_file: str | Path) -> tuple[Path, str]:
    """
    Convert path/to/sample_info.txt to:
    parent folder and sample prefix before _info.txt.
    """
    path = Path(info_file)
    if not path.name.endswith("_info.txt"):
        raise ValueError(
            f"Expected an info file ending with '_info.txt', got: {path.name}"
        )
    prefix = path.name[: -len("_info.txt")]
    return path.parent, prefix


def discover_out_files(
    info_file: str | Path,
    camera: str = "A",
) -> tuple[str, list[Path]]:
    """
    Find all matching cumulative output files for one experiment.

    Example pattern:
    prefix_A-*_out.txt
    """
    folder, prefix = prefix_from_info_file(info_file)
    pattern = f"{prefix}_{camera}-*_out.txt"
    files = sorted(folder.glob(pattern))

    if not files:
        raise FileNotFoundError(
            f"No output files found with pattern {pattern!r} in {folder}"
        )

    return prefix, files
