"""
fitting.py
----------
Core numerical routines for afps():
  - Objective functions  (Delta, R, Rwp)
  - Scaling coefficient optimisation (BFGS / Nelder-Mead / CG)
  - Non-negative least squares (NNLS)
  - Phase concentration calculation (Eq. 1 and Eq. 2)
  - Limit of detection calculation (Eq. 6)
  - Goodness-of-fit metrics (Rwp, R)

All equations reference Butler & Hillier (2021),
Computers & Geosciences 147, 104662.
"""

import numpy as np
from scipy.optimize import minimize, nnls as scipy_nnls


# ───────────────────────────────────────────────────────────────────────────
# Objective functions  (Eqs. 3–5)
# ───────────────────────────────────────────────────────────────────────────

def objective(coeffs: np.ndarray,
              measured: np.ndarray,
              ref_mat: np.ndarray,
              obj: str = "Rwp") -> float:
    """
    Compute the chosen objective function between measured and fitted patterns.

    Parameters
    ----------
    coeffs   : scaling coefficients (≥ 0 enforced internally)
    measured : measured intensity array
    ref_mat  : reference pattern matrix (columns = phases)
    obj      : "Delta" | "R" | "Rwp"

    Returns
    -------
    Scalar value of the objective function.
    """
    fitted = ref_mat @ np.maximum(coeffs, 0.0)
    diff   = measured - fitted

    if obj == "Delta":                              # Eq. (3)
        return float(np.sum(np.abs(diff)))

    elif obj == "R":                                # Eq. (4)
        denom = np.sum(measured ** 2)
        return float(np.sqrt(np.sum(diff ** 2) / denom)) if denom > 0 else 1e9

    else:                                           # Eq. (5)  Rwp (default)
        w = 1.0 / np.maximum(measured, 1.0)
        return float(np.sqrt(
            np.sum(w * diff ** 2) / np.sum(w * measured ** 2)
        ))


# ───────────────────────────────────────────────────────────────────────────
# Coefficient optimisation
# ───────────────────────────────────────────────────────────────────────────

def optimise_coefficients(measured: np.ndarray,
                           ref_mat: np.ndarray,
                           init: np.ndarray,
                           solver: str,
                           obj: str) -> np.ndarray:
    """
    FLOWCHART NODE: Optimise scaling coefficients using the chosen
    solver and objective function. Coefficients are bounded ≥ 0.

    Parameters
    ----------
    measured : measured intensity array
    ref_mat  : reference pattern matrix
    init     : initial coefficient estimates
    solver   : "BFGS" | "Nelder-Mead" | "CG"
    obj      : "Delta" | "R" | "Rwp"

    Returns
    -------
    Optimised non-negative coefficient array.
    """
    bounds = [(0, None)] * len(init)
    res    = minimize(
        objective,
        x0      = init,
        args    = (measured, ref_mat, obj),
        method  = solver,
        bounds  = bounds,
        options = {"maxiter": 2000, "ftol": 1e-9, "gtol": 1e-7},
    )
    return np.maximum(res.x, 0.0)


# ───────────────────────────────────────────────────────────────────────────
# Non-negative least squares
# ───────────────────────────────────────────────────────────────────────────

def apply_nnls(measured: np.ndarray,
               ref_mat: np.ndarray) -> np.ndarray:
    """
    FLOWCHART NODE: Non-negative least squares for fast initial
    coefficient estimation. Coefficients are constrained to ≥ 0.

    Parameters
    ----------
    measured : measured intensity array
    ref_mat  : reference pattern matrix

    Returns
    -------
    Non-negative coefficient array.
    """
    coeffs, _ = scipy_nnls(ref_mat, measured)
    return coeffs


# ───────────────────────────────────────────────────────────────────────────
# Phase concentration calculation
# ───────────────────────────────────────────────────────────────────────────

def compute_concentrations(coeffs: np.ndarray,
                            rirs: np.ndarray,
                            std_idx: int,
                            std_conc) -> np.ndarray:
    """
    Compute phase concentrations from scaling coefficients and RIRs.

    No internal standard concentration supplied → Eq. (1): phases sum to 100%.
    Known internal standard concentration supplied → Eq. (2): absolute wt-%.

    Parameters
    ----------
    coeffs   : optimised scaling coefficients
    rirs     : RIR values for each active phase
    std_idx  : index of the internal standard in the active phase list
    std_conc : known wt-% of internal standard, or None

    Returns
    -------
    Array of phase concentrations in wt-%.
    """
    if std_conc is None:
        # Eq. (1) — normalised to 100 wt-%
        numerators = coeffs / rirs
        total      = numerators.sum()
        if total == 0:
            return np.zeros_like(coeffs)
        return numerators / total * 100.0
    else:
        # Eq. (2) — absolute concentrations using known std concentration
        return (std_conc / (rirs / rirs[std_idx])) * (coeffs / coeffs[std_idx])


# ───────────────────────────────────────────────────────────────────────────
# Limit of detection
# ───────────────────────────────────────────────────────────────────────────

def compute_lods(rirs: np.ndarray,
                 lod_std: float,
                 rir_std: float) -> np.ndarray:
    """
    Estimate limits of detection for all phases using Eq. (6):
        LOD_x = LOD_std × (RIR_std / RIR_x)

    Parameters
    ----------
    rirs    : RIR values for each active phase
    lod_std : LOD estimate (wt-%) for the internal standard
    rir_std : RIR of the internal standard

    Returns
    -------
    Array of LOD estimates (wt-%) for each active phase.
    """
    return lod_std * (rir_std / rirs)


# ───────────────────────────────────────────────────────────────────────────
# Goodness-of-fit metrics
# ───────────────────────────────────────────────────────────────────────────

def compute_rwp(measured: np.ndarray, fitted: np.ndarray) -> float:
    """Weighted profile R-factor (Rwp), Eq. (5)."""
    w = 1.0 / np.maximum(measured, 1.0)
    return float(np.sqrt(
        np.sum(w * (measured - fitted) ** 2) / np.sum(w * measured ** 2)
    ))


def compute_r(measured: np.ndarray, fitted: np.ndarray) -> float:
    """Unweighted R-factor, Eq. (4)."""
    denom = np.sum(measured ** 2)
    return float(np.sqrt(
        np.sum((measured - fitted) ** 2) / denom
    )) if denom > 0 else float("nan")
