"""
config.py
---------
All user-editable settings for the afps() batch analysis.

INSTRUCTIONS
------------
1. Set FILE PATHS to point at your data.
2. Fill in PHASES with your mineral IDs, names and RIR values.
3. Adjust ALGORITHM CONTROLS if needed (defaults are usually fine).
4. Run:  python run_analysis.py
"""

import pandas as pd


# ===========================================================================
# SECTION 1 - FILE PATHS
# ===========================================================================

# Reference CSV: first column = 2-theta, remaining columns = phase IDs
REFERENCE_CSV  = "my_reference_patterns.csv"

# Folder containing sample CSVs (one file per sample, 2 columns: tth, counts)
# All *.csv files found in this folder will be processed automatically.
SAMPLES_FOLDER = "my_samples/"

# Single combined output CSV (rows = samples, columns = phases + Rwp)
OUTPUT_CSV     = "results_afps_batch.csv"

# Folder where one PNG plot per sample will be saved and auto-opened
PLOTS_FOLDER   = "plots/"


# ===========================================================================
# SECTION 2 - YOUR MINERALS
# ===========================================================================
# phase_id   -> must exactly match the column header in REFERENCE_CSV
# phase_name -> human-readable label; duplicate names are grouped/summed
# rir        -> Reference Intensity Ratio relative to corundum = 1.00
#               (see https://rruff.geo.arizona.edu/ for values)

PHASES = pd.DataFrame({
    "phase_id":   ["COR",      "HYM",             "CAL"],
    "phase_name": ["Corundum", "Hydromagnesite",   "Calcite"],
    "rir":        [1.00,       0.94,               2.51],
})

# Phase ID of the internal standard in REFERENCE_CSV (usually corundum)
STD_ID = "COR"

# Known wt-% of the internal standard added to the sample.
# Set to None -> Eq. (1): phases sum to 100 wt-%.
# Set to a float (e.g. 20.0) -> Eq. (2): absolute concentrations.
STD_CONC = None          # e.g. 20.0

# Phases that must ALWAYS be kept regardless of LOD or negative coefficients
FORCE = []               # e.g. ["COR", "CAL"]

# Phases to treat as amorphous (their LOD is controlled separately)
AMORPHOUS_IDS = []       # e.g. ["OBS"]
AMORPHOUS_LOD = 0.0      # wt-% below which amorphous phases are removed


# ===========================================================================
# SECTION 3 - ALGORITHM CONTROLS
# ===========================================================================

HARMONISE    = True    # Harmonise sample & library to the same 2theta scale
SOLVER       = "BFGS"  # Optimiser: "BFGS" | "Nelder-Mead" | "CG"
OBJ          = "Rwp"   # Objective: "Rwp" | "R" | "Delta"
ALIGN        = 0.2     # Max +/-2theta shift for alignment (degrees); 0 = skip
MANUAL_ALIGN = False   # True -> shift exactly by ALIGN; False -> optimise
TTH_ALIGN    = None    # [min, max] 2theta range for alignment; None = full
TTH_FPS_MIN  = 5.0     # Lower 2theta limit for fitting
TTH_FPS_MAX  = 70.0    # Upper 2theta limit for fitting
SHIFT        = 0.0     # Max per-pattern 2theta shift (0 = disabled)
LOD          = 0.5     # LOD estimate (wt-%) for the std phase (Eq. 6)
OMIT_STD     = False   # Exclude the internal standard from the output table
