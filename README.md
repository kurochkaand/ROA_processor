# ROA Processor MVP

A CLI program for processing Raman and ROA spectra saved as cumulative block files.

This first version does:

1. File discovery from an `*_info.txt` path
2. Filename parsing
3. Header parsing
4. Reading SCP Raman and ROA columns
5. Reversing wavenumber axis to increasing order
6. Converting cumulative blocks to isolated blocks
7. Normalizing isolated blocks by acquisition time and power at sample
8. Detecting ROA spikes across isolated blocks
9. Replacing bad ROA points by the median of clean blocks at the same wavenumber
10. Averaging cleaned blocks
11. Exporting CSV/NPZ/JSON files
12. Plotting diagnostic figures

## Install in editable mode

Open the folder in VS Code, then in the terminal:

```bash
python -m venv .venv
source .venv/Scripts/activate
python -m pip install -U pip
pip install -e .
```

For test development:

```bash
pip install -e ".[dev]"
python -m pytest
```

## Inspect an experiment

```bash
roa inspect "X:\Cations_and_PGA\ROA-data\260621_CdCl2_precipitated\prec_42uL_92mgml_PGA_6uL_1M_CdCL2_pH7-N20-500mW-146791s_0312026-06-21_info.txt"
```

## Process an experiment

```bash
roa process "X:\Cations_and_PGA\ROA-data\260621_CdCl2_precipitated\prec_42uL_92mgml_PGA_6uL_1M_CdCL2_pH7-N20-500mW-146791s_0312026-06-21_info.txt" --output processed
roa process "X:\Cations_and_PGA\ROA-data\260618_ZnCl2\260619_32uL_Zn_in_42uL_PGA_05eqv\Zn_05eqv-N20-400mW-2.93581s_0242026-06-19_info.txt" --output processed
```

Relative output folders are created beside the input `*_info.txt` file. For example,
`--output processed` writes to `X:\...\sample_folder\processed\`, not to the
directory where the command was launched.

## Useful options

```bash
roa inspect "X:\...\sample_info.txt" --min-wavenumber 200
roa process "X:\...\sample_info.txt" --output processed --min-wavenumber 200
roa process "X:\...\sample_info.txt" --output processed --min-wavenumber 200 --max-wavenumber 1800
roa process "X:\...\sample_info.txt" --output processed --spike-threshold 2
roa process "X:\...\sample_info.txt" --output processed --no-normalize-power
roa process "X:\...\sample_info.txt" --output processed --no-normalize-time
```

The wavenumber filter is applied after file reading and axis reversal, before
cumulative blocks are converted to isolated blocks. `inspect` reports both the
original and processed wavenumber ranges.

## ROA QC and conservative denoising

By default, processing calculates QC diagnostics in the ROA silent region
`1800-2609 cm^-1` when enough points are present, but it does not denoise or
reject blocks unless requested.

```bash
roa process "X:\...\sample_info.txt" --output processed --roa-qc-range 1800 2609
roa process "X:\...\sample_info.txt" --output processed --roa-denoise-qc
roa process "X:\...\sample_info.txt" --output processed --roa-denoise-qc --roa-qc-reject-blocks --roa-qc-max-block-noise 3
roa process "X:\...\sample_info.txt" --output processed --roa-qc-smooth
```

QC denoising exports comparison columns such as `roa_qc_weighted_mean`,
`roa_qc_weighted_smoothed`, and `roa_qc_removed_noise`. It never overwrites the
standard `roa_mean_after_spike_removal` column.

## Plot from processed result

```bash
roa plot processed --info-file "X:\...\sample_info.txt" --kind final
roa plot processed --info-file "X:\...\sample_info.txt" --kind spikes
roa plot processed --info-file "X:\...\sample_info.txt" --kind isolated-roa
```

`--info-file` lets the plot command resolve a relative processed folder the same
way as `process`.

## Main exports

Processing writes:

```text
metadata.json
processing_config.json
final_spectra.csv
roa_qc_block_summary.csv
isolated_blocks.npz
roa_spike_cleaning.npz
roa_qc.npz
figures/
```

## Important assumption

The program assumes that the `*_out.txt` files are cumulative sums:

```text
isolated_0 = cumulative_0
isolated_i = cumulative_i - cumulative_(i-1)
```

The wavenumber axis is reversed internally to increasing order, from low to high cm^-1.
