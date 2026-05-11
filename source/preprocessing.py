"""
preprocessing.py
----------------
Data preprocessing for afps():
  - Cubic spline interpolation onto a common 2θ grid
  - Harmonisation of sample and reference library
  - Sample alignment to internal standard
"""

import numpy as np
from scipy.interpolate import CubicSpline
from scipy.optimize import minimize


# ───────────────────────────────────────────────────────────────────────────
# Interpolation
# ───────────────────────────────────────────────────────────────────────────

def interpolate_to_grid(tth_new: np.ndarray,
                        tth_old: np.ndarray,
                        counts: np.ndarray) -> np.ndarray:
    """
    Cubic spline interpolation of `counts` (measured on `tth_old`)
    onto the new grid `tth_new`. Values outside the original range
    are set to 0 and negative values are clipped to 0.
    """
    cs  = CubicSpline(tth_old, counts, extrapolate=False)
    out = cs(tth_new)
    out = np.where(np.isnan(out), 0.0, out)
    out = np.maximum(out, 0.0)
    return out


# ───────────────────────────────────────────────────────────────────────────
# Harmonisation
# ───────────────────────────────────────────────────────────────────────────

def harmonise_data(tth_smpl: np.ndarray,
                   counts_smpl: np.ndarray,
                   tth_lib: np.ndarray,
                   ref_matrix: np.ndarray) -> tuple:
    """
    FLOWCHART NODE: Harmonise sample and library onto the same 2θ scale.

    The common grid spans the overlapping 2θ range and uses the coarser
    of the two step sizes. Both sample and all reference patterns are
    interpolated onto this grid via cubic spline.

    Returns
    -------
    tth_common   : np.ndarray  — common 2θ axis
    counts_harm  : np.ndarray  — sample counts on the common grid
    ref_harm     : np.ndarray  — reference matrix on the common grid
    """
    tth_min = max(tth_smpl.min(), tth_lib.min())
    tth_max = min(tth_smpl.max(), tth_lib.max())

    step_smpl  = np.median(np.diff(tth_smpl))
    step_lib   = np.median(np.diff(tth_lib))
    step       = max(step_smpl, step_lib)

    tth_common = np.arange(tth_min, tth_max + step * 0.5, step)

    counts_harm = interpolate_to_grid(tth_common, tth_smpl, counts_smpl)
    ref_harm    = np.column_stack([
        interpolate_to_grid(tth_common, tth_lib, ref_matrix[:, j])
        for j in range(ref_matrix.shape[1])
    ])
    return tth_common, counts_harm, ref_harm


# ───────────────────────────────────────────────────────────────────────────
# Alignment
# ───────────────────────────────────────────────────────────────────────────

def align_sample(tth: np.ndarray,
                 counts: np.ndarray,
                 tth_std: np.ndarray,
                 counts_std: np.ndarray,
                 align: float,
                 manual_align: bool,
                 tth_align_range=None) -> tuple:
    """
    FLOWCHART NODE: Align sample 2θ axis to the internal standard pattern.

    manual_align=True  → shift the sample exactly by `align` degrees.
    manual_align=False → find the shift (within ±align) that maximises
                         the Pearson correlation between sample and standard.

    Parameters
    ----------
    tth             : sample 2θ axis
    counts          : sample counts
    tth_std         : standard reference 2θ axis
    counts_std      : standard reference counts
    align           : maximum allowed shift (degrees)
    manual_align    : if True, apply fixed shift; if False, optimise
    tth_align_range : [min, max] to restrict alignment region; None = full

    Returns
    -------
    tth_shifted, counts  (counts unchanged; only the tth axis is shifted)
    """
    if manual_align:
        return tth + align, counts

    # Restrict to the alignment 2θ range if specified
    if tth_align_range is not None:
        lo, hi   = tth_align_range
        mask_s   = (tth >= lo)     & (tth <= hi)
        mask_r   = (tth_std >= lo) & (tth_std <= hi)
    else:
        mask_s   = np.ones(len(tth),     dtype=bool)
        mask_r   = np.ones(len(tth_std), dtype=bool)

    def neg_correlation(delta):
        tth_shifted  = tth[mask_s] + delta[0]
        std_interp   = interpolate_to_grid(tth_shifted,
                                           tth_std[mask_r],
                                           counts_std[mask_r])
        corr = np.corrcoef(counts[mask_s], std_interp)[0, 1]
        return -corr if np.isfinite(corr) else 0.0

    res   = minimize(neg_correlation, x0=[0.0],
                     bounds=[(-align, align)], method="L-BFGS-B")
    delta = float(res.x[0])
    return tth + delta, counts
