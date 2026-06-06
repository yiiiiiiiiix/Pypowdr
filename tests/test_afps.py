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
# 1. Import tests
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

    def test_interpolate_to_grid(self):
        """Cubic spline interpolation should produce correct-length output."""
        from preprocessing import interpolate_to_grid

        tth_old = np.linspace(5, 65, 3000)
        counts = np.sin(np.radians(tth_old)) * 1000
        tth_new = np.linspace(10, 60, 2000)

        result = interpolate_to_grid(tth_new, tth_old, counts)
        assert len(result) == len(tth_new)
        assert np.all(result >= 0), "Interpolation produced negative values"

    def test_harmonise_creates_common_grid(self):
        """Harmonised output should share the same 2-theta grid."""
        from preprocessing import harmonise_data

        # harmonise_data(tth_smpl, counts_smpl, tth_lib, ref_matrix)
        tth_smpl = np.linspace(4.5, 66, 3200)
        counts_smpl = np.sin(np.radians(tth_smpl)) * 800
        tth_lib = np.linspace(5, 65, 3000)
        ref_matrix = np.column_stack([
            np.sin(np.radians(tth_lib)) * 1000,
            np.cos(np.radians(tth_lib)) * 500,
        ])

        tth_common, counts_harm, ref_harm = harmonise_data(
            tth_smpl, counts_smpl, tth_lib, ref_matrix)

        assert len(tth_common) == len(counts_harm)
        assert ref_harm.shape[0] == len(tth_common)
        assert ref_harm.shape[1] == 2

    def test_align_sample_returns_shifted_tth(self):
        """Alignment should return a shifted 2-theta axis."""
        from preprocessing import align_sample

        tth = np.linspace(5, 65, 3000)
        pattern = np.exp(-0.5 * ((tth - 35) / 2) ** 2) * 1000
        shifted = np.exp(-0.5 * ((tth - 35.1) / 2) ** 2) * 1000

        # align_sample(tth, counts, tth_std, counts_std, align, manual_align)
        tth_out, counts_out = align_sample(
            tth, shifted, tth, pattern, 0.3, False)

        assert len(tth_out) == len(tth)
        assert len(counts_out) == len(shifted)


# ===========================================================================
# 3. Fitting tests
# ===========================================================================
class TestFitting:
    """Test objective functions and NNLS fitting."""

    def test_nnls_nonnegative(self):
        """NNLS solution should contain no negative coefficients."""
        from fitting import apply_nnls

        rng = np.random.default_rng(42)
        x = np.linspace(0, 10, 100)
        ref1 = np.sin(x)
        ref2 = np.cos(x)
        observed = 2 * ref1 + 3 * ref2 + rng.normal(0, 0.1, 100)
        ref_mat = np.column_stack([ref1, ref2])

        coeffs = apply_nnls(observed, ref_mat)
        assert np.all(coeffs >= 0), "NNLS returned negative coefficients"
        assert coeffs[0] == pytest.approx(2.0, abs=0.3)
        assert coeffs[1] == pytest.approx(3.0, abs=0.3)

    def test_objective_rwp(self):
        """Rwp objective should return 0 for a perfect fit."""
        from fitting import objective

        measured = np.array([100.0, 200.0, 150.0])
        ref_mat = np.eye(3)
        coeffs = measured.copy()

        val = objective(coeffs, measured, ref_mat, "Rwp")
        assert val == pytest.approx(0.0, abs=1e-10)

    def test_compute_rwp(self):
        """Rwp should be 0 for identical patterns."""
        from fitting import compute_rwp

        pattern = np.array([100.0, 200.0, 300.0, 400.0])
        rwp = compute_rwp(pattern, pattern)
        assert rwp == pytest.approx(0.0, abs=1e-10)

    def test_compute_concentrations(self):
        """Concentrations should sum to 100 when std_conc is None."""
        from fitting import compute_concentrations

        coeffs = np.array([1.0, 2.0, 3.0])
        rirs = np.array([1.0, 1.0, 1.0])
        std_idx = 0

        concs = compute_concentrations(coeffs, rirs, std_idx, None)
        assert concs.sum() == pytest.approx(100.0, abs=0.01)


# ===========================================================================
# 4. RockJock validation
# ===========================================================================
class TestRockJockValidation:
    """Validate against the RockJock synthetic mixture dataset."""

    @pytest.fixture(autouse=True)
    def check_data_available(self):
        """Skip these tests if validation data files are missing."""
        required = [
            VALIDATION_DIR / "rockjock_patterns.csv",
            VALIDATION_DIR / "rockjock_phases.csv",
            VALIDATION_DIR / "rockjock_weights.csv",
        ]
        for f in required:
            if not f.exists():
                pytest.skip(f"Validation data not found: {f}")
        mixtures = list(VALIDATION_DIR.glob("mixture_Mix*.csv"))
        if not mixtures:
            pytest.skip("No mixture CSV files found in validation/")

    def _load_validation_data(self):
        ref_df = pd.read_csv(
            VALIDATION_DIR / "rockjock_patterns.csv", index_col=0)
        ref_df.index = ref_df.index.astype(float)
        ref_df = ref_df[ref_df.index.notna()].dropna()
        phases_df = pd.read_csv(VALIDATION_DIR / "rockjock_phases.csv")
        known_df = pd.read_csv(VALIDATION_DIR / "rockjock_weights.csv")
        return ref_df, phases_df, known_df

    def test_validation_data_loads(self):
        """Validation CSV files should load without errors."""
        ref_df, phases_df, known_df = self._load_validation_data()
        assert ref_df.shape[0] > 0
        assert len(phases_df) > 0
        assert len(known_df) > 0

    def test_reference_patterns_shape(self):
        """Reference library should have many 2-theta points and phases."""
        ref_df, _, _ = self._load_validation_data()
        n_points, n_phases = ref_df.shape
        assert n_points > 100, f"Only {n_points} 2-theta points"
        assert n_phases > 5, f"Only {n_phases} phases"

    def test_rockjock_phases_in_library(self):
        """All phase IDs in phases CSV should exist in the reference library."""
        ref_df, phases_df, _ = self._load_validation_data()
        for phase_id in phases_df["phase_id"]:
            assert phase_id in ref_df.columns, (
                f"Phase '{phase_id}' not in reference library")

    def test_rockjock_single_mixture_fit(self):
        """Fit Mix1 using the validation script's fit_sample function."""
        sys.path.insert(0, str(VALIDATION_DIR))
        from validate_rockjock import fit_sample

        ref_df, _, _ = self._load_validation_data()
        phases_df = pd.read_csv(VALIDATION_DIR / "rockjock_phases.csv")

        phase_ids = [
            "CORUNDUM", "ORDERED_MICROCLINE", "LABRADORITE",
            "KAOLINITE_DRY_BRANCH", "MONTMORILLONITE_WYO",
            "ILLITE_1M_RM30", "QUARTZ",
        ]
        phases = phases_df[phases_df["phase_id"].isin(phase_ids)].reset_index(drop=True)

        mix1 = pd.read_csv(VALIDATION_DIR / "mixture_Mix1.csv")
        if list(mix1.columns[:2]) != ["tth", "counts"]:
            mix1 = mix1.iloc[:, :2]
            mix1.columns = ["tth", "counts"]

        row = fit_sample(
            mix1["tth"].to_numpy(dtype=float),
            mix1["counts"].to_numpy(dtype=float),
            "Mix1", ref_df, phases)

        assert "Rwp" in row, "fit_sample did not return Rwp"
        assert 0.05 < row["Rwp"] < 0.25, (
            f"Mix1 Rwp = {row['Rwp']:.3f} outside expected 0.05-0.25")
