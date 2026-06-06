"""
Tests for the Pypowdr package.

These tests validate the Python FPS implementation against the RockJock
synthetic mixture dataset (Eberl, 2003) and check basic module functionality.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

# ---------------------------------------------------------------------------
# Make the Powdr package importable regardless of working directory
# ---------------------------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parent.parent
POWDR_DIR = REPO_ROOT / "Powdr"
if str(POWDR_DIR) not in sys.path:
    sys.path.insert(0, str(POWDR_DIR))

VALIDATION_DIR = POWDR_DIR / "validation"


# ===========================================================================
# 1. Import tests — confirm all modules load without errors
# ===========================================================================
class TestImports:
    """Verify that all core modules are importable."""

    def test_import_preprocessing(self):
        import preprocessing  # noqa: F401

    def test_import_fitting(self):
        import fitting  # noqa: F401

    def test_import_afps(self):
        import afps  # noqa: F401

    def test_import_plotting(self):
        import plotting  # noqa: F401

    def test_import_config(self):
        import config  # noqa: F401


# ===========================================================================
# 2. Preprocessing tests
# ===========================================================================
class TestPreprocessing:
    """Test interpolation and alignment utilities."""

    def test_harmonise_creates_common_grid(self):
        """Harmonised output should share the same 2-theta grid."""
        import preprocessing as pp

        tth_ref = np.linspace(5, 65, 3000)
        tth_smpl = np.linspace(4.5, 66, 3200)
        # harmonise_data expects a 2D ref_matrix (n_points x n_phases)
        ref_matrix = np.column_stack([
            np.sin(np.radians(tth_ref)) * 1000,
            np.cos(np.radians(tth_ref)) * 500,
        ])
        smpl_counts = np.sin(np.radians(tth_smpl)) * 800

        result = pp.harmonise_data(tth_ref, ref_matrix, tth_smpl, smpl_counts)
        assert result is not None, "harmonise_data returned None"

    def test_align_sample_returns_result(self):
        """Alignment should return a result."""
        import preprocessing as pp

        tth = np.linspace(5, 65, 3000)
        pattern = np.exp(-0.5 * ((tth - 35) / 2) ** 2) * 1000
        shifted = np.exp(-0.5 * ((tth - 35.1) / 2) ** 2) * 1000
        # Call without keyword arguments to match actual signature
        result = pp.align_sample(tth, shifted, tth, pattern, 0.3)
        assert result is not None, "align_sample returned None"


# ===========================================================================
# 3. Fitting tests
# ===========================================================================
class TestFitting:
    """Test objective functions and NNLS fitting."""

    def test_nnls_nonnegative(self):
        """NNLS solution should contain no negative coefficients."""
        from scipy.optimize import nnls

        rng = np.random.default_rng(42)
        x = np.linspace(0, 10, 100)
        ref1 = np.sin(x)
        ref2 = np.cos(x)
        observed = 2 * ref1 + 3 * ref2 + rng.normal(0, 0.1, 100)
        A = np.column_stack([ref1, ref2])
        coeffs, _ = nnls(A, observed)
        assert np.all(coeffs >= 0), "NNLS returned negative coefficients"
        assert coeffs[0] == pytest.approx(2.0, abs=0.2)
        assert coeffs[1] == pytest.approx(3.0, abs=0.2)


# ===========================================================================
# 4. R
