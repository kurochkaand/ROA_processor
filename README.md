# ROA Processor MVP

A minimal CLI program for processing Raman and ROA spectra saved as cumulative block files.

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

## Inspect an experiment

```bash
roa inspect "X:\Cations_and_PGA\ROA-data\260621_CdCl2_precipitated\prec_42uL_92mgml_PGA_6uL_1M_CdCL2_pH7-N20-500mW-146791s_0312026-06-21_info.txt"
```

## Process an experiment

```bash
roa process "X:\Cations_and_PGA\ROA-data\260621_CdCl2_precipitated\prec_42uL_92mgml_PGA_6uL_1M_CdCL2_pH7-N20-500mW-146791s_0312026-06-21_info.txt" --output processed
```

## Useful options

```bash
roa process "X:\...\sample_info.txt" --output processed --spike-threshold 8 --average mean
roa process "X:\...\sample_info.txt" --output processed --spike-threshold 10 --average median
roa process "X:\...\sample_info.txt" --output processed --no-normalize-power
roa process "X:\...\sample_info.txt" --output processed --no-normalize-time
```

## Plot from processed result

```bash
roa plot processed --kind final
roa plot processed --kind spikes
roa plot processed --kind isolated-roa
```

## Important assumption

The program assumes that the `*_out.txt` files are cumulative sums:

```text
isolated_0 = cumulative_0
isolated_i = cumulative_i - cumulative_(i-1)
```

The wavenumber axis is reversed internally to increasing order, from low to high cm^-1.
