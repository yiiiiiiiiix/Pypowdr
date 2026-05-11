"""
afps.py
-------
Single-sample afps() implementation following Fig. 3 of
Butler & Hillier (2021), Computers & Geosciences 147, 104662.
"""

import numpy as np
import pandas as pd
from scipy.optimize import minimize as sp_minimize

import config as cfg
from preprocessing import harmonise_data, align_sample, interpolate_to_grid
from fitting import (apply_nnls, optimise_coefficients,
                     compute_concentrations, compute_lods,
                     compute_rwp, compute_r, objective)
from plotting import save_fit_plot


def run_afps(tth_smpl, counts_smpl, sample_name, ref_df):
    """
    Run the afps() flowchart for a single sample.

    Parameters
    ----------
    tth_smpl    : np.ndarray   - 2theta axis of the sample
    counts_smpl : np.ndarray   - count intensities of the sample
    sample_name : str          - label used in output and plot title
    ref_df      : pd.DataFrame - reference library
                  (index = 2theta values, columns = phase IDs)

    Returns
    -------
    row       : dict  {sample_id, phase_name: wt-%, ..., Rwp}
    plot_path : str path to the saved PNG, or None
    """

    tth_lib    = ref_df.index.to_numpy(dtype=float)
    ref_matrix = ref_df[cfg.PHASES["phase_id"].tolist()].to_numpy(dtype=float)

    active_ids  = cfg.PHASES["phase_id"].tolist()
    active_rirs = cfg.PHASES["rir"].to_numpy(dtype=float)
    active_ref  = ref_matrix.copy()

    std_idx = active_ids.index(cfg.STD_ID)

    # Harmonise
    if cfg.HARMONISE:
        tth_work, counts_work, ref_work = harmonise_data(
            tth_smpl, counts_smpl, tth_lib, active_ref)
    else:
        tth_work    = tth_smpl.copy()
        counts_work = counts_smpl.copy()
        ref_work    = active_ref.copy()

    # Alignment
    if cfg.ALIGN != 0:
        std_col = ref_df[cfg.STD_ID].to_numpy(dtype=float)
        tth_work, counts_work = align_sample(
            tth_work, counts_work, tth_lib, std_col,
            align=cfg.ALIGN, manual_align=cfg.MANUAL_ALIGN,
            tth_align_range=cfg.TTH_ALIGN)
        tth_work, counts_work, ref_work = harmonise_data(
            tth_work, counts_work, tth_lib, active_ref)

    # Subset to fitting 2theta range
    mask = np.ones(len(tth_work), dtype=bool)
    if cfg.TTH_FPS_MIN is not None:
        mask &= tth_work >= cfg.TTH_FPS_MIN
    if cfg.TTH_FPS_MAX is not None:
        mask &= tth_work <= cfg.TTH_FPS_MAX
    tth_fit    = tth_work[mask]
    counts_fit = counts_work[mask]
    ref_fit    = ref_work[mask, :]

    # NNLS initial estimate
    coeffs = apply_nnls(counts_fit, ref_fit)

    # Remove zero-coefficient phases
    zero_mask = (coeffs == 0) & \
                np.array([pid not in cfg.FORCE for pid in active_ids])
    if zero_mask.any():
        keep        = ~zero_mask
        active_ids  = [pid for pid, k in zip(active_ids, keep) if k]
        active_rirs = active_rirs[keep]
        ref_fit     = ref_fit[:, keep]
        coeffs      = coeffs[keep]
        std_idx     = (active_ids.index(cfg.STD_ID)
                       if cfg.STD_ID in active_ids else 0)

    # Optimise scaling coefficients
    coeffs = optimise_coefficients(
        counts_fit, ref_fit, coeffs, cfg.SOLVER, cfg.OBJ)

    # Remove negative coefficients and reoptimise
    while True:
        le_zero = (coeffs <= 0) & \
                  np.array([pid not in cfg.FORCE for pid in active_ids])
        if not le_zero.any():
            break
        keep        = ~le_zero
        active_ids  = [pid for pid, k in zip(active_ids, keep) if k]
        active_rirs = active_rirs[keep]
        ref_fit     = ref_fit[:, keep]
        coeffs      = coeffs[keep]
        std_idx     = (active_ids.index(cfg.STD_ID)
                       if cfg.STD_ID in active_ids else 0)
        coeffs = optimise_coefficients(
            counts_fit, ref_fit, coeffs, cfg.SOLVER, cfg.OBJ)

    # Per-pattern shift
    if cfg.SHIFT > 0:
        best_shifts = np.zeros(ref_fit.shape[1])
        for j, pid in enumerate(active_ids):
            def obj_shift(delta, j=j):
                ref_j_shifted  = interpolate_to_grid(
                    tth_fit, tth_fit + delta[0], ref_fit[:, j])
                ref_test       = ref_fit.copy()
                ref_test[:, j] = ref_j_shifted
                return objective(coeffs, counts_fit, ref_test, cfg.OBJ)
            res   = sp_minimize(obj_shift, x0=[0.0],
                                bounds=[(-cfg.SHIFT, cfg.SHIFT)],
                                method="L-BFGS-B")
            delta = float(res.x[0])
            if abs(delta) <= cfg.SHIFT:
                best_shifts[j] = delta
                ref_fit[:, j]  = interpolate_to_grid(
                    tth_fit, tth_fit + delta, ref_fit[:, j])
        coeffs = optimise_coefficients(
            counts_fit, ref_fit, coeffs, cfg.SOLVER, cfg.OBJ)

    # Concentrations + LODs
    concentrations = compute_concentrations(
        coeffs, active_rirs, std_idx, cfg.STD_CONC)
    rir_std = active_rirs[std_idx]
    lods    = compute_lods(active_rirs, cfg.LOD, rir_std)

    # Remove phases below LOD and reoptimise
    while True:
        below_lod = (concentrations < lods) & np.array([
            pid not in cfg.FORCE and pid not in cfg.AMORPHOUS_IDS
            for pid in active_ids
        ])
        if not below_lod.any():
            break
        keep           = ~below_lod
        active_ids     = [pid for pid, k in zip(active_ids, keep) if k]
        active_rirs    = active_rirs[keep]
        ref_fit        = ref_fit[:, keep]
        coeffs         = coeffs[keep]
        concentrations = concentrations[keep]
        lods           = lods[keep]
        std_idx        = (active_ids.index(cfg.STD_ID)
                          if cfg.STD_ID in active_ids else 0)
        coeffs = optimise_coefficients(
            counts_fit, ref_fit, coeffs, cfg.SOLVER, cfg.OBJ)
        concentrations = compute_concentrations(
            coeffs, active_rirs, std_idx, cfg.STD_CONC)

    # Amorphous phase check
    if cfg.AMORPHOUS_IDS and cfg.AMORPHOUS_LOD > 0:
        amorph_below = np.array([
            pid in cfg.AMORPHOUS_IDS
            and conc < cfg.AMORPHOUS_LOD
            and pid not in cfg.FORCE
            for pid, conc in zip(active_ids, concentrations)
        ])
        if amorph_below.any():
            keep           = ~amorph_below
            active_ids     = [pid for pid, k in zip(active_ids, keep) if k]
            active_rirs    = active_rirs[keep]
            ref_fit        = ref_fit[:, keep]
            coeffs         = coeffs[keep]
            concentrations = concentrations[keep]
            coeffs = optimise_coefficients(
                counts_fit, ref_fit, coeffs, cfg.SOLVER, cfg.OBJ)
            concentrations = compute_concentrations(
                coeffs, active_rirs, std_idx, cfg.STD_CONC)

    # Final fit metrics
    fitted  = ref_fit @ coeffs
    rwp_val = compute_rwp(counts_fit, fitted)
    r_val   = compute_r(counts_fit, fitted)

    # Build output table
    id_to_name = dict(zip(cfg.PHASES["phase_id"], cfg.PHASES["phase_name"]))

    results = pd.DataFrame({
        "phase_id":      active_ids,
        "phase_name":    [id_to_name.get(pid, pid) for pid in active_ids],
        "rir":           active_rirs,
        "coefficient":   coeffs,
        "concentration": concentrations,
        "lod":           lods,
    })

    if cfg.OMIT_STD and cfg.STD_ID in results["phase_id"].values:
        results = results[results["phase_id"] != cfg.STD_ID].copy()

    grouped = (results
               .groupby("phase_name", as_index=False)["concentration"]
               .sum()
               .rename(columns={"concentration": "concentration_wt_pct"}))

    # Print summary
    print(f"\n  Phases retained : {len(results)}")
    print(f"  Rwp             : {rwp_val:.6f}  |  R : {r_val:.6f}")
    print()
    print(results[["phase_id", "phase_name", "concentration", "lod"]]
          .rename(columns={"concentration": "wt_%", "lod": "LOD_wt_%"})
          .round(3)
          .to_string(index=False))

    # Save plot
    plot_path = save_fit_plot(
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
        plots_folder   = cfg.PLOTS_FOLDER,
    )

    # Return flat row for batch table
    row = {"sample_id": sample_name, "Rwp": round(rwp_val, 6)}
    for _, r in grouped.iterrows():
        row[r["phase_name"]] = round(r["concentration_wt_pct"], 3)

    print(f"\n[DONE] {sample_name} complete.\n")
    return row, plot_path
