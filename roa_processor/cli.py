from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from roa_processor.io.export import (
    ensure_output_dirs,
    load_processed_npz,
    save_final_spectra,
    save_isolated_npz,
    save_metadata,
    save_spikes,
)
from roa_processor.io.load_experiment import load_experiment
from roa_processor.plotting.plots import (
    plot_final_from_csv,
    plot_final_spectra,
    plot_isolated_roa_blocks,
    plot_spike_heatmap,
)
from roa_processor.processing.average import make_final_spectra
from roa_processor.processing.isolate_blocks import cumulative_to_isolated
from roa_processor.processing.spike_detection import detect_and_replace_spikes_block_mad


def cmd_inspect(args: argparse.Namespace) -> None:
    experiment = load_experiment(
        args.info_file,
        camera=args.camera,
        reverse_axis=True,
    )

    blocks = experiment.blocks
    wavenumber = experiment.wavenumber

    powers = [b.metadata.header.power_at_sample_mw for b in blocks]
    cycles = [b.metadata.header.cumulative_cycles for b in blocks]
    times = [b.metadata.header.total_time_scp_s for b in blocks]

    print()
    print("ROA experiment inspection")
    print("=" * 28)
    print(f"Info file: {experiment.info_file}")
    print(f"Experiment prefix: {experiment.prefix}")
    print(f"Files found: {experiment.n_blocks}")
    print(f"Camera: {args.camera}")
    print("Mode: SCP")
    print(f"Block indices: {blocks[0].metadata.block_index:03d}–{blocks[-1].metadata.block_index:03d}")
    print(f"Wavenumber range after reversal: {wavenumber[0]:.6g} to {wavenumber[-1]:.6g} cm^-1")
    print(f"Number of spectral points: {len(wavenumber)}")

    unique_powers = sorted({p for p in powers if p is not None})
    if unique_powers:
        if len(unique_powers) == 1:
            print(f"Power at sample: {unique_powers[0]} mW")
        else:
            print(f"Power at sample: variable, {unique_powers}")

    if all(c is not None for c in cycles):
        print(f"Cycles: {cycles[0]} → {cycles[-1]}")
    if all(t is not None for t in times):
        print(f"Total time SCP: {times[0]} → {times[-1]} s")

    first_meta = blocks[0].metadata.filename
    print()
    print("Parsed filename metadata")
    print("-" * 28)
    print(f"Sample name: {first_meta.sample_name}")
    print(f"Nominal laser power from filename: {first_meta.nominal_laser_power_mw} mW")
    print(f"Camera exposure from filename: {first_meta.camera_exposure_s} s")
    print(f"Experiment number: {first_meta.experiment_number}")
    print(f"Date: {first_meta.date}")
    print()


def cmd_process(args: argparse.Namespace) -> None:
    output = ensure_output_dirs(args.output)

    experiment = load_experiment(
        args.info_file,
        camera=args.camera,
        reverse_axis=True,
    )

    isolated = cumulative_to_isolated(
        experiment,
        normalize_time=not args.no_normalize_time,
        normalize_power=not args.no_normalize_power,
    )

    spike_result = detect_and_replace_spikes_block_mad(
        isolated.roa_norm,
        block_indices=isolated.block_indices,
        threshold=args.spike_threshold,
    )

    final = make_final_spectra(isolated, spike_result)

    processing_config = {
        "camera": args.camera,
        "mode": "SCP",
        "reverse_axis_to_increasing": True,
        "normalize_time": not args.no_normalize_time,
        "normalize_power": not args.no_normalize_power,
        "spike_method": "block_mad",
        "spike_threshold": args.spike_threshold,
        "spike_replacement": "median_of_clean_blocks_at_same_wavenumber",
    }

    save_metadata(output, experiment, extra=processing_config)
    save_isolated_npz(output, isolated)
    save_spikes(output, isolated, spike_result)
    save_final_spectra(output, final)

    figures = output / "figures"
    plot_isolated_roa_blocks(
        isolated,
        output_path=figures / "isolated_roa_blocks_before_spike_removal.png",
    )
    plot_isolated_roa_blocks(
        isolated,
        cleaned_roa=spike_result.roa_cleaned,
        output_path=figures / "isolated_roa_blocks_after_spike_removal.png",
    )
    plot_spike_heatmap(
        isolated,
        spike_result,
        output_path=figures / "spike_mask_heatmap.png",
    )
    plot_final_spectra(
        final,
        output_path=figures / "final.png",
    )

    n_spikes = int(spike_result.spike_mask.sum())
    print()
    print("Processing complete")
    print("=" * 22)
    print(f"Output folder: {output}")
    print(f"Blocks processed: {experiment.n_blocks}")
    print(f"Spectral points: {len(isolated.wavenumber)}")
    print(f"ROA spike points replaced: {n_spikes}")
    print(f"Final spectra: {output / 'final_spectra.csv'}")
    print(f"Figures: {figures}")
    print()


def cmd_plot(args: argparse.Namespace) -> None:
    output = Path(args.processed_folder)
    figures = output / "figures"
    figures.mkdir(parents=True, exist_ok=True)

    if args.kind == "final":
        csv_path = output / "final_spectra.csv"
        if not csv_path.exists():
            raise FileNotFoundError(f"Cannot find {csv_path}")
        plot_final_from_csv(csv_path, output_path=figures / "final_from_csv.png")
        print(f"Saved final plots to {figures}")

    elif args.kind in {"spikes", "isolated-roa"}:
        data = load_processed_npz(output)
        # Lightweight plotting from NPZ without fully reconstructing all dataclasses.
        import matplotlib.pyplot as plt
        import numpy as np

        wavenumber = data["wavenumber"]
        block_indices = data["block_indices"]

        if args.kind == "spikes":
            spike_mask = data["spike_mask"]
            plt.figure(figsize=(10, 5))
            plt.imshow(
                spike_mask.astype(int),
                aspect="auto",
                interpolation="nearest",
                extent=[wavenumber[0], wavenumber[-1], block_indices[-1], block_indices[0]],
            )
            plt.title("ROA spike mask")
            plt.xlabel("Wavenumber / cm$^{-1}$")
            plt.ylabel("Block index")
            plt.colorbar(label="Spike")
            path = figures / "spike_mask_heatmap_from_npz.png"
            plt.tight_layout()
            plt.savefig(path, dpi=200)
            plt.close()
            print(f"Saved {path}")

        elif args.kind == "isolated-roa":
            roa_before = data["roa_before"]
            roa_cleaned = data["roa_cleaned"]

            plt.figure(figsize=(10, 6))
            for row in roa_before:
                plt.plot(wavenumber, row, linewidth=0.7, alpha=0.5)
            plt.title("Isolated ROA blocks before spike replacement")
            plt.xlabel("Wavenumber / cm$^{-1}$")
            plt.ylabel("ROA intensity, normalized")
            path = figures / "isolated_roa_blocks_before_from_npz.png"
            plt.tight_layout()
            plt.savefig(path, dpi=200)
            plt.close()

            plt.figure(figsize=(10, 6))
            for row in roa_cleaned:
                plt.plot(wavenumber, row, linewidth=0.7, alpha=0.5)
            plt.title("Isolated ROA blocks after spike replacement")
            plt.xlabel("Wavenumber / cm$^{-1}$")
            plt.ylabel("ROA intensity, normalized")
            path2 = figures / "isolated_roa_blocks_after_from_npz.png"
            plt.tight_layout()
            plt.savefig(path2, dpi=200)
            plt.close()

            print(f"Saved {path}")
            print(f"Saved {path2}")

    else:
        raise ValueError(f"Unknown plot kind: {args.kind}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="roa",
        description="MVP CLI for Raman/ROA cumulative block processing.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    inspect = subparsers.add_parser("inspect", help="Inspect one experiment without processing.")
    inspect.add_argument("info_file", help="Path to *_info.txt file.")
    inspect.add_argument("--camera", default="A", help="Camera to process. Default: A")
    inspect.set_defaults(func=cmd_inspect)

    process = subparsers.add_parser("process", help="Process one experiment.")
    process.add_argument("info_file", help="Path to *_info.txt file.")
    process.add_argument("--camera", default="A", help="Camera to process. Default: A")
    process.add_argument("--output", default="processed", help="Output folder. Default: processed")
    process.add_argument(
        "--spike-threshold",
        type=float,
        default=8.0,
        help="MAD-based spike threshold. Default: 8.0",
    )
    process.add_argument(
        "--no-normalize-time",
        action="store_true",
        help="Disable normalization by delta acquisition time.",
    )
    process.add_argument(
        "--no-normalize-power",
        action="store_true",
        help="Disable normalization by power at sample.",
    )
    process.set_defaults(func=cmd_process)

    plot = subparsers.add_parser("plot", help="Create plots from a processed folder.")
    plot.add_argument("processed_folder", help="Folder created by 'roa process'.")
    plot.add_argument(
        "--kind",
        choices=["final", "spikes", "isolated-roa"],
        default="final",
        help="Plot kind. Default: final",
    )
    plot.set_defaults(func=cmd_plot)

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
