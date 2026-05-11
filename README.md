# Powdr

Python tools for full-pattern summation analysis of powder X-ray diffraction (XRD) data. This package implements an automated full-pattern summation workflow (`afps`) for estimating phase concentrations from measured XRD patterns using reference patterns and Reference Intensity Ratio (RIR) values.

The implementation follows the workflow described by Butler and Hillier (2021), *Computers & Geosciences*, 147, 104662, and includes a validation example based on the RockJock synthetic mixture dataset.

## What this package does

This package can be used to:

- Fit measured XRD patterns using a library of reference patterns.
- Estimate phase concentrations in wt.% from fitted scaling coefficients and RIR values.
- Use an internal standard, such as corundum, when a known standard concentration is available.
- Automatically remove phases with zero, negative, or below-limit-of-detection fitted contributions.
- Harmonize sample and reference patterns onto a common 2θ grid.
- Align measured patterns to an internal standard reference pattern.
- Save fitted-vs-experimental plots for each sample.
- Validate the workflow against RockJock synthetic mixtures.

## Repository structure

```text
Powdr/
├── afps.py                         # Main automated full-pattern summation workflow
├── config.py                       # User-editable analysis settings
├── fitting.py                      # Objective functions, NNLS, optimization, concentration calculations
├── preprocessing.py                # Interpolation, harmonization, and alignment utilities
├── plotting.py                     # Plotting functions for fitted patterns
├── run_analysis.py                 # Example/legacy analysis script
├── my_reference_patterns.csv       # Example reference-pattern library
├── phases.csv                      # Example phase information table
├── my_samples/                     # Example sample XRD patterns
├── plots/                          # Example output plots
├── results_afps_batch.csv          # Example batch output table
└── validation/
    ├── validate_rockjock.py        # RockJock validation script
    ├── rockjock_patterns.csv       # RockJock reference patterns
    ├── rockjock_phases.csv         # RockJock phase/RIR table
    ├── rockjock_weights.csv        # Known mixture compositions
    ├── mixture_Mix*.csv            # Synthetic mixture XRD patterns
    └── validation_*.csv/png        # Validation outputs
```

## Installation

Clone the repository and enter the package folder:

```bash
git clone https://github.com/YOUR-USERNAME/Powdr.git
cd Powdr
```

Create and activate a Python environment:

```bash
conda create -n powdr python=3.12
conda activate powdr
```

Install the required packages:

```bash
pip install numpy pandas scipy matplotlib
```

## Input data format

### 1. Reference-pattern library

The reference-pattern CSV should have:

- First column: 2θ values.
- Remaining columns: reference intensities for each phase.
- Column names: phase IDs used in `config.py`.

Example:

```csv
tth,COR,HYM,CAL
5.00,0,12,4
5.02,0,15,5
5.04,1,20,6
```

### 2. Sample XRD files

Each sample CSV should contain two columns:

```csv
tth,counts
5.00,120
5.02,135
5.04,148
```

The default workflow reads all `*.csv` files in the folder specified by `SAMPLES_FOLDER` in `config.py`.

### 3. Phase table and RIR values

Phase IDs, phase names, and RIR values are defined in `config.py`:

```python
PHASES = pd.DataFrame({
    "phase_id":   ["COR", "HYM", "CAL"],
    "phase_name": ["Corundum", "Hydromagnesite", "Calcite"],
    "rir":        [1.00, 0.94, 2.51],
})
```

The `phase_id` values must exactly match the reference-pattern column headers.

## Basic workflow

### Step 1. Edit `config.py`

Update the file paths:

```python
REFERENCE_CSV  = "my_reference_patterns.csv"
SAMPLES_FOLDER = "my_samples/"
OUTPUT_CSV     = "results_afps_batch.csv"
PLOTS_FOLDER   = "plots/"
```

Then define the phases and RIR values:

```python
PHASES = pd.DataFrame({
    "phase_id":   ["COR", "HYM", "CAL"],
    "phase_name": ["Corundum", "Hydromagnesite", "Calcite"],
    "rir":        [1.00, 0.94, 2.51],
})
```

Set the internal standard information:

```python
STD_ID = "COR"
STD_CONC = None      # Use None if phases should be normalized to 100 wt.%
# STD_CONC = 20.0    # Use a number if a known wt.% standard was added
```

### Step 2. Run `afps` on a sample

A minimal single-sample example is:

```python
import pandas as pd
from afps import run_afps

ref_df = pd.read_csv("my_reference_patterns.csv", index_col=0)
sample = pd.read_csv("my_samples/CAL_COR.csv")

row, plot_path = run_afps(
    tth_smpl=sample["tth"].to_numpy(),
    counts_smpl=sample["counts"].to_numpy(),
    sample_name="CAL_COR",
    ref_df=ref_df,
)

print(row)
print(plot_path)
```

### Step 3. Run batch analysis

A minimal batch-analysis example is:

```python
from pathlib import Path
import pandas as pd

import config as cfg
from afps import run_afps
from plotting import open_plots

ref_df = pd.read_csv(cfg.REFERENCE_CSV, index_col=0)
rows = []
plot_paths = []

for path in sorted(Path(cfg.SAMPLES_FOLDER).glob("*.csv")):
    sample = pd.read_csv(path)
    if list(sample.columns[:2]) != ["tth", "counts"]:
        sample = sample.iloc[:, :2]
        sample.columns = ["tth", "counts"]

    row, plot_path = run_afps(
        tth_smpl=sample["tth"].to_numpy(),
        counts_smpl=sample["counts"].to_numpy(),
        sample_name=path.stem,
        ref_df=ref_df,
    )
    rows.append(row)
    if plot_path is not None:
        plot_paths.append(plot_path)

results = pd.DataFrame(rows).fillna(0)
results.to_csv(cfg.OUTPUT_CSV, index=False)
open_plots(plot_paths)
```

This will save:

- A combined concentration table, such as `results_afps_batch.csv`.
- One fitted-pattern plot per sample in the folder specified by `PLOTS_FOLDER`.

## Validation example

To run the RockJock validation example:

```bash
cd validation
python validate_rockjock.py
```

The script fits the synthetic mixture patterns and compares the calculated phase concentrations with the known mixture weights. Outputs are saved as CSV and PNG files in the `validation/` folder.

## Main settings in `config.py`

| Setting | Description |
|---|---|
| `REFERENCE_CSV` | Reference-pattern library CSV. |
| `SAMPLES_FOLDER` | Folder containing sample XRD CSV files. |
| `OUTPUT_CSV` | Output concentration table. |
| `PLOTS_FOLDER` | Folder for fitted-pattern plots. |
| `PHASES` | Phase IDs, names, and RIR values. |
| `STD_ID` | Phase ID of the internal standard. |
| `STD_CONC` | Known wt.% of internal standard, or `None` for normalization to 100 wt.%. |
| `FORCE` | Phases that should always be retained. |
| `AMORPHOUS_IDS` | Phase IDs treated as amorphous components. |
| `HARMONISE` | Whether to interpolate sample and references onto a common 2θ grid. |
| `ALIGN` | Maximum allowed 2θ alignment shift. |
| `TTH_FPS_MIN`, `TTH_FPS_MAX` | 2θ fitting range. |
| `LOD` | Limit-of-detection estimate for the standard phase. |
| `OMIT_STD` | Whether to exclude the internal standard from the output table. |

## Notes

- `run_analysis.py` is included as an example/legacy script and may require editing before use, especially if the local import path points to a different package location.
- The current workflow expects CSV input files. Other formats should be converted to CSV before analysis.
- RIR values strongly affect calculated phase concentrations. Check that the RIR values are appropriate for the reference patterns and instrument conditions used.
- The calculated wt.% values should be interpreted together with the fit quality metrics, especially `Rwp` and visual inspection of the fitted pattern.

## Citation

If this workflow is used in academic work, cite the method that the implementation follows:

Butler, B. M., & Hillier, S. (2021). *powdR: An R package for quantitative mineralogy using full pattern summation of X-ray powder diffraction data*. Computers & Geosciences, 147, 104662.
