"""
validate_rockjock.py
--------------------
Validates the Python afps() implementation against the RockJock
synthetic mixtures dataset, following the settings described in
Butler & Hillier (2021), Computers & Geosciences 147, 104662
Section 4 (fps/afps validation).

Setup (paper Section 4):
  - Reference library : rockjock_patterns.csv  (168 phases)
  - Phases used       : CORUNDUM, ORDERED_MICROCLINE, LABRADORITE,
                        KAOLINITE_DRY_BRANCH, MONTMORILLONITE_WYO,
                        ILLITE_1M_RM30, QUARTZ
  - Internal standard : CORUNDUM  (std_conc = None -> Eq. 1)
  - Alignment         : align = 0.3 deg
  - Samples           : mixture_Mix1.csv ... mixture_Mix8.csv
  - Known weights     : rockjock_weights.csv

USAGE
-----
    python validate_rockjock.py

Place this file in the same folder as the other modules:
    config.py, preprocessing.py, fitting.py, afps.py, plotting.py

All four CSV data files are also expected in the same folder
(or adjust DATA_DIR below).
"""

import os
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")

# Import modules from the same folder
sys.path.insert(0, str(Path(__file__).parent))
from preprocessing import harmonise_data, align_sample
from fitting import (apply_nnls, optimise_coefficients,
                     compute_concentrations, compute_lods,
                     compute_rwp, compute_r)
from plotting import save_fit_plot

# ===========================================================================
# PATHS  -  adjust if your CSVs are in a different folder
# ===========================================================================

DATA_DIR    = Path(__file__).parent          # same folder as this script
PLOTS_DIR   = DATA_DIR / "plots_validation"  # PNGs saved here
OUTPUT_CSV  = DATA_DIR / "validation_results.csv"
COMPARE_CSV = DATA_DIR / "validation_comparison.csv"

REFERENCE_CSV = DATA_DIR / "rockjock_patterns.csv"
PHASES_CSV    = DATA_DIR / "rockjock_phases.csv"
WEIGHTS_CSV   = DATA_DIR / "rockjock_weights.csv"
MIXTURE_DIR   = DATA_DIR                    # mixture_Mix1.csv ... are here

# ===========================================================================
# PAPER SETTINGS  (Section 4 of Butler & Hillier 2021)
# ===========================================================================

PHASE_IDS = [
    "CORUNDUM",
    "ORDERED_MICROCLINE",
    "LABRADORITE",
    "KAOLINITE_DRY_BRANCH",
    "MONTMORILLONITE_WYO",
    "ILLITE_1M_RM30",
    "QUARTZ",
]

STD_ID       = "CORUNDUM"
STD_CONC     = None      # phases sum to 100 wt-% (Eq. 1)
ALIGN        = 0.3       # max +/- 2theta shift (paper uses 0.3)
MANUAL_ALIGN = False
TTH_ALIGN    = None
TTH_FPS_MIN  = 5.0
TTH_FPS_MAX  = 65.0
HARMONISE    = True
SOLVER       = "BFGS"
OBJ          = "Rwp"
SHIFT        = 0.0
LOD          = 0.0       # no LOD filtering for validation (use all phases)
FORCE        = []
AMORPHOUS_IDS = []
AMORPHOUS_LOD = 0.0
OMIT_STD     = False


# ===========================================================================
# CORE FITTING  (mirrors afps.py but uses local settings, not config.py)
# ===========================================================================

def fit_sample(tth_smpl, counts_smpl, sample_name, ref_df, phases):
    """Run full pattern summation for one sample using paper settings."""

    tth_lib    = ref_df.index.to_numpy(dtype=float)
    ref_matrix = ref_df[phases["phase_id"].tolist()].to_numpy(dtype=float)

    active_ids  = phases["phase_id"].tolist()
    active_rirs = phases["rir"].to_numpy(dtype=float)
    active_ref  = ref_matrix.copy()

    std_idx = active_ids.index(STD_ID)

    # Harmonise
    if HARMONISE:
        tth_work, counts_work, ref_work = harmonise_data(
            tth_smpl, counts_smpl, tth_lib, active_ref)
    else:
        tth_work    = tth_smpl.copy()
        counts_work = counts_smpl.copy()
        ref_work    = active_ref.copy()

    # Alignment
    if ALIGN != 0:
        std_col = ref_df[STD_ID].to_numpy(dtype=float)
        tth_work, counts_work = align_sample(
            tth_work, counts_work, tth_lib, std_col,
            align=ALIGN, manual_align=MANUAL_ALIGN,
            tth_align_range=TTH_ALIGN)
        tth_work, counts_work, ref_work = harmonise_data(
            tth_work, counts_work, tth_lib, active_ref)

    # Subset 2theta range
    mask = np.ones(len(tth_work), dtype=bool)
    if TTH_FPS_MIN is not None:
        mask &= tth_work >= TTH_FPS_MIN
    if TTH_FPS_MAX is not None:
        mask &= tth_work <= TTH_FPS_MAX
    tth_fit    = tth_work[mask]
    counts_fit = counts_work[mask]
    ref_fit    = ref_work[mask, :]

    # NNLS initial estimate
    coeffs = apply_nnls(counts_fit, ref_fit)

    # Remove zero-coefficient phases
    zero_mask = (coeffs == 0) & \
                np.array([pid not in FORCE for pid in active_ids])
    if zero_mask.any():
        keep        = ~zero_mask
        active_ids  = [pid for pid, k in zip(active_ids, keep) if k]
        active_rirs = active_rirs[keep]
        ref_fit     = ref_fit[:, keep]
        coeffs      = coeffs[keep]
        std_idx     = (active_ids.index(STD_ID)
                       if STD_ID in active_ids else 0)

    # Optimise
    coeffs = optimise_coefficients(
        counts_fit, ref_fit, coeffs, SOLVER, OBJ)

    # Remove negative and reoptimise
    while True:
        le_zero = (coeffs <= 0) & \
                  np.array([pid not in FORCE for pid in active_ids])
        if not le_zero.any():
            break
        keep        = ~le_zero
        active_ids  = [pid for pid, k in zip(active_ids, keep) if k]
        active_rirs = active_rirs[keep]
        ref_fit     = ref_fit[:, keep]
        coeffs      = coeffs[keep]
        std_idx     = (active_ids.index(STD_ID)
                       if STD_ID in active_ids else 0)
        coeffs = optimise_coefficients(
            counts_fit, ref_fit, coeffs, SOLVER, OBJ)

    # Concentrations
    concentrations = compute_concentrations(
        coeffs, active_rirs, std_idx, STD_CONC)

    # Fit metrics
    fitted  = ref_fit @ coeffs
    rwp_val = compute_rwp(counts_fit, fitted)
    r_val   = compute_r(counts_fit, fitted)

    # Build results
    id_to_name = dict(zip(phases["phase_id"], phases["phase_name"]))
    results = pd.DataFrame({
        "phase_id":      active_ids,
        "phase_name":    [id_to_name.get(pid, pid) for pid in active_ids],
        "concentration": concentrations,
    })

    grouped = (results
               .groupby("phase_name", as_index=False)["concentration"]
               .sum()
               .rename(columns={"concentration": "wt_pct"}))

    # Save plot
    lods = np.zeros(len(active_ids))   # LOD filtering off for validation
    save_fit_plot(
        tth_fit        = tth_fit,
        counts_fit     = counts_fit,
        fitted         = fitted,
        ref_fit        = ref_fit,
        coeffs         = coeffs,
        concentrations = concentrations,
        active_ids     = active_ids,
        id_to_name     = id_to_name,
        rwp_val        = rwp_val,
        r_val          = r_val,
        sample_name    = sample_name,
        plots_folder   = str(PLOTS_DIR),
    )

    row = {"sample_id": sample_name, "Rwp": round(rwp_val, 4)}
    for _, r in grouped.iterrows():
        row[r["phase_name"]] = round(r["wt_pct"], 2)

    return row


# ===========================================================================
# COMPARISON AND PLOTTING
# ===========================================================================

def compute_mae(fitted_df, known_df):
    """
    Compute Mean Absolute Error per phase and overall,
    comparing fitted concentrations to known weights.
    """
    # Map known column names to phase_name used in fitting
    col_map = {
        "Quartz":                "Quartz",
        "K_feldspar":            "K-feldspar",
        "Plagioclase":           "Plagioclase",
        "Kaolinite":             "Kaolinite",
        "Dioctahedral_smectite": "Smectite (Di)",
        "Illite":                "Illite",
        "Corundum":              "Corundum",
    }

    errors = {}
    all_errors = []
    for known_col, phase_name in col_map.items():
        if known_col not in known_df.columns:
            continue
        if phase_name not in fitted_df.columns:
            continue
        known_vals  = known_df.set_index("sample_id")[known_col]
        fitted_vals = fitted_df.set_index("sample_id")[phase_name]
        common      = known_vals.index.intersection(fitted_vals.index)
        abs_errors  = (known_vals[common] - fitted_vals[common]).abs()
        mae = abs_errors.mean()
        errors[phase_name] = round(mae, 3)
        all_errors.extend(abs_errors.tolist())

    errors["OVERALL"] = round(np.mean(all_errors), 3)
    return errors


def plot_comparison(fitted_df, known_df, output_path):
    """
    Bar chart comparing known vs fitted concentrations for all 8 mixtures,
    replicating the style of Fig. 4 in the paper.
    """
    col_map = {
        "Corundum":              "Corundum",
        "K_feldspar":            "K-feldspar",
        "Plagioclase":           "Plagioclase",
        "Kaolinite":             "Kaolinite",
        "Dioctahedral_smectite": "Smectite (Di)",
        "Illite":                "Illite",
        "Quartz":                "Quartz",
    }
    phases_ordered = list(col_map.values())
    mix_names = [f"Mix{i}" for i in range(1, 9)]

    fig, axes = plt.subplots(4, 2, figsize=(14, 18))
    fig.suptitle(
        "Validation: Known vs Fitted concentrations (wt-%)\n"
        "Python afps() vs RockJock synthetic mixtures\n"
        "Butler & Hillier (2021) Fig. 4 replication",
        fontsize=13, y=1.01)

    x     = np.arange(len(phases_ordered))
    width = 0.35

    for idx, mix in enumerate(mix_names):
        ax   = axes[idx // 2][idx % 2]
        row_known  = known_df[known_df["sample_id"] == mix]
        row_fitted = fitted_df[fitted_df["sample_id"] == mix]

        known_vals  = []
        fitted_vals = []
        for known_col, phase_name in col_map.items():
            kv = float(row_known[known_col].values[0]) \
                 if (not row_known.empty and known_col in row_known.columns) \
                 else 0.0
            fv = float(row_fitted[phase_name].values[0]) \
                 if (not row_fitted.empty and phase_name in row_fitted.columns) \
                 else 0.0
            known_vals.append(kv)
            fitted_vals.append(fv)

        bars1 = ax.bar(x - width/2, known_vals,  width,
                       label="Known",  color="silver",  edgecolor="black", lw=0.5)
        bars2 = ax.bar(x + width/2, fitted_vals, width,
                       label="Fitted", color="steelblue", edgecolor="black",
                       lw=0.5, alpha=0.85)

        rwp = float(fitted_df[fitted_df["sample_id"] == mix]["Rwp"].values[0]) \
              if not fitted_df[fitted_df["sample_id"] == mix].empty else float("nan")
        ax.set_title(f"{mix}  (Rwp = {rwp:.3f})", fontsize=10)
        ax.set_xticks(x)
        ax.set_xticklabels(phases_ordered, rotation=30, ha="right", fontsize=8)
        ax.set_ylabel("Concentration (wt-%)", fontsize=8)
        ax.set_ylim(0, 50)
        ax.legend(fontsize=7)
        ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    fig.savefig(str(output_path), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[PLOT]  Comparison chart saved -> {os.path.abspath(output_path)}")


# ===========================================================================
# MAIN
# ===========================================================================

def main():
    print("=" * 65)
    print(" RockJock Validation")
    print(" Python afps() vs Butler & Hillier (2021) Section 4")
    print("=" * 65)

    # Load reference library
    print(f"\n[STEP] Loading reference library: {REFERENCE_CSV.name}")
    ref_df = pd.read_csv(str(REFERENCE_CSV), index_col=0)
    ref_df.index = ref_df.index.astype(float)
    ref_df = ref_df[ref_df.index.notna()]   # drop NaN index rows
    ref_df = ref_df.dropna()                   # drop rows with any NaN values

    # Load and subset phases table
    all_phases = pd.read_csv(str(PHASES_CSV))
    phases = all_phases[all_phases["phase_id"].isin(PHASE_IDS)].reset_index(drop=True)

    # Verify all phase IDs exist in reference library
    missing = [p for p in PHASE_IDS if p not in ref_df.columns]
    if missing:
        raise ValueError(f"Phase IDs not found in reference CSV: {missing}")

    print(f"\n[STEP] Phases used for fitting:")
    for _, row in phases.iterrows():
        print(f"       {row['phase_id']:30s}  {row['phase_name']:20s}  rir={row['rir']:.4f}")

    # Load known weights
    print(f"\n[STEP] Loading known weights: {WEIGHTS_CSV.name}")
    known_df = pd.read_csv(str(WEIGHTS_CSV))

    # Process all 8 mixtures
    PLOTS_DIR.mkdir(exist_ok=True)
    batch_rows = []
    failed     = []

    for i in range(1, 9):
        mix_name = f"Mix{i}"
        mix_file = MIXTURE_DIR / f"mixture_{mix_name}.csv"
        print(f"\n[{i}/8] Fitting {mix_name} ...")

        if not mix_file.exists():
            print(f"  [ERROR] File not found: {mix_file}")
            failed.append(mix_name)
            continue

        try:
            smpl = pd.read_csv(str(mix_file))
            if list(smpl.columns[:2]) != ["tth", "counts"]:
                smpl = smpl.iloc[:, :2]
                smpl.columns = ["tth", "counts"]
            tth    = smpl["tth"].to_numpy(dtype=float)
            counts = smpl["counts"].to_numpy(dtype=float)

            row = fit_sample(tth, counts, mix_name, ref_df, phases)
            batch_rows.append(row)
            print(f"  Rwp = {row['Rwp']:.4f}")

        except Exception as e:
            import traceback
            print(f"  [ERROR] {mix_name} failed: {e}")
            traceback.print_exc()
            failed.append(mix_name)

    if not batch_rows:
        print("\n[ERROR] No samples processed.")
        return

    # Build results table
    fitted_df  = pd.DataFrame(batch_rows)
    phase_cols = [c for c in fitted_df.columns if c not in ("sample_id", "Rwp")]
    fitted_df  = fitted_df[["sample_id"] + sorted(phase_cols) + ["Rwp"]]
    fitted_df[phase_cols] = fitted_df[phase_cols].fillna(0.0)

    # Print results table
    print("\n" + "=" * 65)
    print(" FITTED RESULTS  (wt-%)")
    print("=" * 65)
    print(fitted_df.to_string(index=False))

    # Compute MAE vs known weights
    mae = compute_mae(fitted_df, known_df)
    print("\n" + "=" * 65)
    print(" MEAN ABSOLUTE ERROR vs KNOWN WEIGHTS  (wt-%)")
    print("=" * 65)
    for phase, val in mae.items():
        marker = " <--" if phase == "OVERALL" else ""
        print(f"  {phase:25s}  MAE = {val:.3f} wt-%{marker}")

    # Build comparison table
    col_map = {
        "Quartz":                "Quartz",
        "K_feldspar":            "K-feldspar",
        "Plagioclase":           "Plagioclase",
        "Kaolinite":             "Kaolinite",
        "Dioctahedral_smectite": "Smectite (Di)",
        "Illite":                "Illite",
        "Corundum":              "Corundum",
    }
    compare_rows = []
    for _, row in known_df.iterrows():
        mix = row["sample_id"]
        fit_row = fitted_df[fitted_df["sample_id"] == mix]
        for known_col, phase_name in col_map.items():
            kv = float(row[known_col]) if known_col in known_df.columns else float("nan")
            fv = float(fit_row[phase_name].values[0]) \
                 if (not fit_row.empty and phase_name in fit_row.columns) \
                 else float("nan")
            compare_rows.append({
                "sample_id":  mix,
                "phase":      phase_name,
                "known_wt%":  kv,
                "fitted_wt%": round(fv, 2),
                "error_wt%":  round(fv - kv, 2),
            })
    compare_df = pd.DataFrame(compare_rows)

    # Save CSVs
    fitted_df.to_csv(str(OUTPUT_CSV), index=False)
    compare_df.to_csv(str(COMPARE_CSV), index=False)
    print(f"\n[OUTPUT] Fitted results  -> {os.path.abspath(OUTPUT_CSV)}")
    print(f"[OUTPUT] Comparison table -> {os.path.abspath(COMPARE_CSV)}")

    # Save comparison bar chart
    compare_plot = PLOTS_DIR / "validation_comparison.png"
    plot_comparison(fitted_df, known_df, compare_plot)

    if failed:
        print(f"\n[WARNING] Failed samples: {failed}")

    print(f"\n[DONE] Validation complete.")
    print(f"       Overall MAE = {mae['OVERALL']:.3f} wt-%")
    print(f"       Paper reports fps() MAE = ~0.9 wt-% for RockJock mixtures.")


if __name__ == "__main__":
    main()
