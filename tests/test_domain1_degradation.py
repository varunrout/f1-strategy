"""
Unit tests for Domain 1 core feature-engineering functions.

Run with:
    pytest tests/test_domain1_degradation.py -v
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.features.domain1_degradation import (
    MIN_STINT_LAPS,
    extract_clean_air_stints,
    fit_degradation_curves,
)


# ---------------------------------------------------------------------------
# Fixtures – synthetic DataFrames (no FastF1 required)
# ---------------------------------------------------------------------------

def _make_lap_df(
    n_drivers: int = 2,
    n_laps: int = 20,
    compound: str = "SOFT",
    include_first_lap: bool = True,
) -> pd.DataFrame:
    """Generate a minimal synthetic lap DataFrame for testing."""
    records = []
    for driver_idx in range(n_drivers):
        driver = f"D{driver_idx:02d}"
        for lap in range(1, n_laps + 1):
            # Simulate a simple linear degradation: base pace 90 s + 0.05 s/lap
            lap_time = 90.0 + 0.05 * lap + np.random.default_rng(lap + driver_idx).normal(0, 0.01)
            records.append(
                {
                    "Driver": driver,
                    "LapNumber": lap,
                    "LapTime": lap_time,
                    "Stint": 1,
                    "Compound": compound,
                    "TyreLife": lap,
                    "TrackStatus": "1",
                    "PitInTime": None,
                    "PitOutTime": None,
                    "Position": 1 + driver_idx,
                    "year": 2023,
                    "round_number": 1,
                    "EventName": "Bahrain Grand Prix",
                    "session_type": "R",
                }
            )
    return pd.DataFrame(records)


def _make_clean_air_stint_df(
    n_drivers: int = 2,
    stint_laps: int = 15,
    compound: str = "MEDIUM",
) -> pd.DataFrame:
    """Generate a clean-air stints DataFrame as produced by extract_clean_air_stints."""
    records = []
    for driver_idx in range(n_drivers):
        driver = f"D{driver_idx:02d}"
        for sl in range(1, stint_laps + 1):
            lap_time = 91.0 + 0.06 * sl + np.random.default_rng(sl * 100 + driver_idx).normal(0, 0.02)
            records.append(
                {
                    "Driver": driver,
                    "LapNumber": sl + 1,  # lap 1 already removed
                    "LapTime": lap_time,
                    "Stint": 1,
                    "stint_id": 1,
                    "stint_lap": sl,
                    "tyre_compound": compound,
                    "Compound": compound,
                    "TrackStatus": "1",
                    "PitInTime": None,
                    "PitOutTime": None,
                    "year": 2023,
                    "round_number": 1,
                    "EventName": "Bahrain Grand Prix",
                    "session_type": "R",
                }
            )
    return pd.DataFrame(records)


# ---------------------------------------------------------------------------
# Tests – extract_clean_air_stints
# ---------------------------------------------------------------------------

class TestExtractCleanAirStints:
    def test_removes_first_lap(self):
        """Lap 1 must be excluded from clean-air stints."""
        df = _make_lap_df(n_drivers=1, n_laps=10)
        result = extract_clean_air_stints(df)
        assert 1 not in result["LapNumber"].values, "Lap 1 should be removed."

    def test_assigns_stint_ids(self):
        """Every row in the output should have a non-null stint_id."""
        df = _make_lap_df(n_drivers=2, n_laps=15)
        result = extract_clean_air_stints(df)
        assert "stint_id" in result.columns
        assert result["stint_id"].notna().all(), "All rows must have a stint_id."

    def test_assigns_stint_lap(self):
        """stint_lap should be a 1-based counter within each stint."""
        df = _make_lap_df(n_drivers=1, n_laps=12)
        result = extract_clean_air_stints(df)
        assert "stint_lap" in result.columns
        assert (result["stint_lap"] >= 1).all(), "stint_lap must be >= 1."

    def test_filters_safety_car_laps(self):
        """Laps with TrackStatus != '1' should be excluded."""
        df = _make_lap_df(n_drivers=1, n_laps=15)
        # Mark laps 5-8 as safety car (status '4')
        sc_mask = df["LapNumber"].isin([5, 6, 7, 8])
        df.loc[sc_mask, "TrackStatus"] = "4"
        result = extract_clean_air_stints(df)
        assert not any(result["LapNumber"].isin([5, 6, 7, 8])), (
            "Safety car laps should be removed."
        )

    def test_filters_non_slick_compounds(self):
        """Intermediate and wet tyres should be excluded."""
        df = _make_lap_df(n_drivers=1, n_laps=15, compound="INTERMEDIATE")
        result = extract_clean_air_stints(df)
        assert result.empty or "INTERMEDIATE" not in result["tyre_compound"].values

    def test_minimum_stint_length_enforced(self):
        """Stints with fewer than MIN_STINT_LAPS clean laps should be dropped."""
        df = _make_lap_df(n_drivers=1, n_laps=MIN_STINT_LAPS - 1)
        result = extract_clean_air_stints(df)
        assert result.empty, (
            f"Stints shorter than {MIN_STINT_LAPS} laps should be removed."
        )

    def test_compound_column_propagated(self):
        """tyre_compound must be populated in the result."""
        df = _make_lap_df(n_drivers=1, n_laps=15, compound="HARD")
        result = extract_clean_air_stints(df)
        if not result.empty:
            assert "tyre_compound" in result.columns
            assert (result["tyre_compound"] == "HARD").all()

    def test_returns_dataframe_on_empty_input(self):
        """Should return an empty DataFrame gracefully on empty input."""
        result = extract_clean_air_stints(pd.DataFrame())
        assert isinstance(result, pd.DataFrame)


# ---------------------------------------------------------------------------
# Tests – fit_degradation_curves
# ---------------------------------------------------------------------------

class TestFitDegradationCurves:
    def test_returns_expected_columns(self):
        """Output must contain the core degradation metric columns."""
        df = _make_clean_air_stint_df(n_drivers=2, stint_laps=15)
        result = fit_degradation_curves(df)
        expected_cols = {
            "deg_rate_linear",
            "r2_linear",
            "r2_poly",
            "stint_length",
            "peak_lap",
        }
        missing = expected_cols - set(result.columns)
        assert not missing, f"Missing output columns: {missing}"

    def test_returns_one_row_per_stint(self):
        """There should be exactly one output row per unique driver/event/stint."""
        df = _make_clean_air_stint_df(n_drivers=3, stint_laps=12)
        result = fit_degradation_curves(df)
        n_expected = df.groupby(["Driver", "EventName", "stint_id"]).ngroups
        assert len(result) == n_expected, (
            f"Expected {n_expected} rows, got {len(result)}."
        )

    def test_handles_short_stints_gracefully(self):
        """Stints at exactly MIN_STINT_LAPS should still be processed."""
        df = _make_clean_air_stint_df(n_drivers=1, stint_laps=MIN_STINT_LAPS)
        result = fit_degradation_curves(df)
        assert len(result) == 1
        assert not np.isnan(result.iloc[0]["deg_rate_linear"])

    def test_linear_slope_positive_for_degrading_tyres(self):
        """A tyre that degrades linearly must have a positive linear slope."""
        df = _make_clean_air_stint_df(n_drivers=1, stint_laps=15, compound="SOFT")
        result = fit_degradation_curves(df)
        assert (result["deg_rate_linear"] > 0).all(), (
            "deg_rate_linear should be positive for degrading tyres."
        )

    def test_r2_linear_within_bounds(self):
        """R² values should be in a reasonable range (–∞, 1]."""
        df = _make_clean_air_stint_df(n_drivers=2, stint_laps=20)
        result = fit_degradation_curves(df)
        valid_r2 = result["r2_linear"].dropna()
        assert (valid_r2 <= 1.0).all(), "R² must be ≤ 1.0."

    def test_empty_input_returns_empty_dataframe(self):
        """Should return an empty DataFrame gracefully on empty input."""
        result = fit_degradation_curves(pd.DataFrame())
        assert isinstance(result, pd.DataFrame)
        assert result.empty

    def test_too_few_laps_skipped(self):
        """Stints with fewer than MIN_STINT_LAPS laps should produce no output row."""
        df = _make_clean_air_stint_df(n_drivers=1, stint_laps=MIN_STINT_LAPS - 1)
        result = fit_degradation_curves(df)
        assert result.empty, (
            "Sub-minimum stints should be skipped in fit_degradation_curves."
        )

    def test_polynomial_coefficients_present(self):
        """Polynomial fit coefficients should be in the output."""
        df = _make_clean_air_stint_df(n_drivers=1, stint_laps=20)
        result = fit_degradation_curves(df)
        for col in ("poly_a", "poly_b", "poly_c"):
            assert col in result.columns, f"Column '{col}' missing from output."
