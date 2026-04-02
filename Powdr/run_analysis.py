"""
run_analysis.py
---------------
Tailored for your setup:
  - Reference patterns : one CSV file (columns = minerals, first col = 2-theta)
  - RIR values         : provided manually below
  - Internal standard  : Corundum (COR)
  - Sample files       : CSV files (2 columns: tth, counts)

INSTRUCTIONS
------------
1. Fill in the FILE PATHS section below.
2. Fill in YOUR_MINERALS with your phase IDs, names and RIR values.
3. Run:  python run_analysis.py
"""

import pandas as pd
import numpy as np
import sys, os

# ── Make sure the powdr package is importable ─────────────────────────────
sys.path.insert(0, r"C:\Users\yxian\Desktop\Powdr\powdR_python")
from powdr import PowdRLib, fps, afps, read_xy
from powdr.batch import summarise_batch
from powdr import batch_fps


# ═══════════════════════════════════════════════════════════════════════════
# SECTION 1 — FILE PATHS  (edit these)
# ═══════════════════════════════════════════════════════════════════════════

# Path to your big reference CSV.
# Expected layout:
#   First column  → 2-theta values
#   Other columns → one column per mineral, header = your phase ID

REFERENCE_CSV = "my_reference_patterns.csv"

# Your sample file(s).
# Each CSV should have exactly 2 columns: tth and counts.
# Either list individual files, or point to a folder (see batch section).
#   Example single file:
SAMPLE_CSV = "my_sample.csv"

#   Example folder of samples (used in batch mode at the bottom):
SAMPLES_FOLDER = "my_samples/"          # folder containing multiple CSVs

# Where to save results
OUTPUT_CSV = "results.csv"


# ═══════════════════════════════════════════════════════════════════════════
# SECTION 2 — YOUR MINERALS  (edit these)
# ═══════════════════════════════════════════════════════════════════════════
# phase_id   → must exactly match the column name in your reference CSV
# phase_name → human-readable label used in output tables
# rir        → Reference Intensity Ratio relative to Corundum = 1.00
#              Full list: https://rruff.geo.arizona.edu/

PHASES = pd.DataFrame({
    "phase_id":   [
        'COR',
        'CAL',
    ],
    "phase_name": [
        'Corundum',
        "Calcite",
    ],
    "rir": [
        1.00,
        3.36,           
    ],
})

# ID of the corundum column in your reference CSV
STD_PHASE_ID = "COR"

# Weight-% of corundum you added to your sample.
# Set to None if you did NOT add a known amount (phases will sum to 100%).
STD_CONC = 25.0        # e.g. 20 wt%


# ═══════════════════════════════════════════════════════════════════════════
# SECTION 3 — FPS SETTINGS  (usually fine to leave as defaults)
# ═══════════════════════════════════════════════════════════════════════════

ALIGN     = 1.0    # max 2θ alignment correction in degrees
TTH_MIN   = None   # lower 2θ limit for fitting (None = use full range)
TTH_MAX   = None   # upper 2θ limit for fitting (None = use full range)
OMIT_STD  = True   # exclude corundum from the output table


# ═══════════════════════════════════════════════════════════════════════════
# STEP 1 — Load the reference library
# ═══════════════════════════════════════════════════════════════════════════

print("Loading reference library ...")
ref_df = pd.read_csv(REFERENCE_CSV, index_col=0)
# ref_df now has: index = tth, columns = phase IDs

# Quick sanity check
missing = [pid for pid in PHASES["phase_id"] if pid not in ref_df.columns]
if missing:
    raise ValueError(
        f"These phase_ids are in PHASES but not found as columns in "
        f"{REFERENCE_CSV}: {missing}\n"
        f"Columns in your CSV: {list(ref_df.columns)}"
    )

lib = PowdRLib(ref_df, PHASES)
print(lib)


# ═══════════════════════════════════════════════════════════════════════════
# STEP 2 — Load a single sample and run fps()
# ═══════════════════════════════════════════════════════════════════════════

print(f"\nLoading sample: {SAMPLE_CSV}")
smpl = pd.read_csv(SAMPLE_CSV, header=0)

# Accept either named columns or positional
if list(smpl.columns[:2]) != ["tth", "counts"]:
    smpl = smpl.iloc[:, :2]
    smpl.columns = ["tth", "counts"]

print("Running fps() ...")
result = fps(
    lib      = lib,
    smpl     = smpl,
    refs     = list(PHASES["phase_id"]),
    std      = STD_PHASE_ID,
    std_conc = STD_CONC,
    align    = ALIGN,
    tth_min  = TTH_MIN,
    tth_max  = TTH_MAX,
    omit_std = OMIT_STD,
    verbose  = True,
)

# Print summary
result.summary()

# Save phase concentrations to CSV
result.phases_grouped.to_csv(OUTPUT_CSV, index=False)
print(f"\nResults saved to: {OUTPUT_CSV}")


# ═══════════════════════════════════════════════════════════════════════════
# STEP 3 — (Optional) Automated afps() — let the algorithm choose phases
# ═══════════════════════════════════════════════════════════════════════════
# Uncomment this block if you want afps() to automatically drop phases
# that are below the detection limit.

from powdr import afps
print("\nRunning afps() ...")
result_auto = afps(
     lib      = lib,
     smpl     = smpl,
     std      = STD_PHASE_ID,
     std_conc = STD_CONC,
     lod      = 1.0,        # remove phases below 1 wt-%
     align    = ALIGN,
     omit_std = OMIT_STD,
     verbose  = True,
 )
result_auto.summary()


# ═══════════════════════════════════════════════════════════════════════════
# STEP 4 — (Optional) Batch — run fps() on every CSV in a folder
# ═══════════════════════════════════════════════════════════════════════════
# Uncomment and set SAMPLES_FOLDER above to process multiple samples at once.

# from pathlib import Path
# import pandas as pd
# from powdr import batch_fps
# from powdr.batch import summarise_batch
#
# sample_files = sorted(Path(SAMPLES_FOLDER).glob("*.csv"))
# if not sample_files:
#     print(f"No CSV files found in {SAMPLES_FOLDER}")
# else:
#     samples = {}
#     for f in sample_files:
#         df = pd.read_csv(f, header=0)
#         if list(df.columns[:2]) != ["tth", "counts"]:
#             df = df.iloc[:, :2]
#             df.columns = ["tth", "counts"]
#         samples[f.stem] = df
#
#     print(f"\nRunning batch fps() on {len(samples)} samples ...")
#     batch = batch_fps(
#         lib      = lib,
#         samples  = samples,
#         refs     = list(PHASES["phase_id"]),
#         std      = STD_PHASE_ID,
#         std_conc = STD_CONC,
#         omit_std = OMIT_STD,
#         n_jobs   = -1,       # use all CPU cores
#         verbose  = True,
#     )
#
#     table = summarise_batch(batch)
#     print("\nBatch results (wt %):")
#     print(table.round(2))
#     table.to_csv("batch_results.csv")
#     print("Saved to batch_results.csv")


# ═══════════════════════════════════════════════════════════════════════════
# STEP 5 — (Optional) Plot the fit
# ═══════════════════════════════════════════════════════════════════════════
# Uncomment to plot measured vs fitted pattern + residuals.

import matplotlib.pyplot as plt

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(11, 6), sharex=True,
                                 gridspec_kw={"height_ratios": [3, 1]})

ax1.plot(result.tth, result.measured, lw=0.8, color="black",  label="Measured")
ax1.plot(result.tth, result.fitted,   lw=0.8, color="red",    label="Fitted", ls="--")
for phase_id, col in result.weighted_pure_patterns.items():
     ax1.fill_between(result.tth, 0, col.values, alpha=0.3, label=phase_id)
ax1.set_ylabel("Counts")
ax1.legend(fontsize=7)
ax1.set_title(f"Rwp = {result.obj['Rwp']:.4f}  |  R = {result.obj['R']:.4f}")

ax2.plot(result.tth, result.residuals, lw=0.6, color="gray")
ax2.axhline(0, color="k", lw=0.5)
ax2.set_ylabel("Residuals")
ax2.set_xlabel("2θ (°)")

plt.tight_layout()
plt.savefig("fit_plot.png", dpi=150)
print("Plot saved to fit_plot.png")
