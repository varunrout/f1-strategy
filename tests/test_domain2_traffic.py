"""Unit tests for Domain 2: Traffic & Spatial feature engineering."""
import numpy as np
import pandas as pd
import pytest

from src.features.domain2_traffic import (
    TRAFFIC_REGIMES,
    build_lap_positions,
    classify_traffic_regimes,
    detect_overtake_hotspots,
    detect_overtakes,
    detect_proximity_events,
    map_track_bottlenecks,
    quantify_traffic_penalties,
    run_domain2_pipeline,
    summarise_traffic_penalties,
    _derive_lap_times,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def sample_laps():
    """Minimal laps_featured fixture."""
    return pd.DataFrame(
        {
            "session_id": [1, 1, 1, 1, 1, 1],
            "driver": ["VER", "VER", "VER", "HAM", "HAM", "HAM"],
            "lap_number": [1, 2, 3, 1, 2, 3],
            "lap_time_s": [90.0, 91.0, 92.0, 91.0, 90.5, 91.5],
            "lap_time_ms": [90000, 91000, 92000, 91000, 90500, 91500],
            "tyre_age_laps": [1, 2, 3, 1, 2, 3],
            "stint": [1, 1, 1, 1, 1, 1],
            "compound": ["SOFT"] * 6,
            "position": [1, 1, 2, 2, 2, 1],
            "is_pit_lap": [0, 0, 0, 0, 0, 0],
            "track_status_cat": ["GREEN"] * 6,
            "gap_to_ahead_s": [5.0, 4.0, 1.5, 3.0, 2.0, 0.8],
        }
    )


@pytest.fixture
def sample_positions():
    """Minimal positions_raw fixture with x, y, z."""
    rows = []
    for driver, x_offset in [("VER", 0.0), ("HAM", 5.0)]:
        for lap, t_start in [(1, 0.0), (2, 90.0), (3, 181.0)]:
            for t in np.linspace(t_start, t_start + 89, 10):
                rows.append(
                    {
                        "session_id": 1,
                        "driver": driver,
                        "time_s": float(t),
                        "x": x_offset + float(t) * 0.1,
                        "y": float(t) * 0.05,
                        "z": 0.0,
                        "status": "OnTrack",
                    }
                )
    return pd.DataFrame(rows)


@pytest.fixture
def sample_lap_positions(sample_laps, sample_positions):
    return build_lap_positions(sample_positions, sample_laps)


# ---------------------------------------------------------------------------
# Tests: _derive_lap_times
# ---------------------------------------------------------------------------


def test_derive_lap_times_creates_boundaries(sample_laps):
    result = _derive_lap_times(sample_laps)
    assert "lap_start_time_s" in result.columns
    assert "lap_end_time_s" in result.columns
    ver = result[result["driver"] == "VER"].sort_values("lap_number")
    assert list(ver["lap_end_time_s"]) == sorted(ver["lap_end_time_s"])


# ---------------------------------------------------------------------------
# Tests: build_lap_positions
# ---------------------------------------------------------------------------


def test_build_lap_positions_returns_dataframe(sample_positions, sample_laps):
    result = build_lap_positions(sample_positions, sample_laps)
    assert isinstance(result, pd.DataFrame)
    assert "lap_number" in result.columns


def test_build_lap_positions_empty_positions(sample_laps):
    result = build_lap_positions(pd.DataFrame(), sample_laps)
    assert result.empty


def test_build_lap_positions_empty_laps(sample_positions):
    result = build_lap_positions(sample_positions, pd.DataFrame())
    assert result.empty


def test_build_lap_positions_lap_numbers_valid(sample_lap_positions, sample_laps):
    valid_laps = set(sample_laps["lap_number"].unique())
    assert set(sample_lap_positions["lap_number"].unique()).issubset(valid_laps)


# ---------------------------------------------------------------------------
# Tests: detect_proximity_events
# ---------------------------------------------------------------------------


def test_detect_proximity_events_returns_dataframe(sample_lap_positions):
    result = detect_proximity_events(sample_lap_positions)
    assert isinstance(result, pd.DataFrame)


def test_detect_proximity_events_columns(sample_lap_positions):
    result = detect_proximity_events(sample_lap_positions)
    expected_cols = {
        "session_id", "lap_number", "time_s",
        "driver_ahead", "driver_behind", "distance_m",
    }
    if not result.empty:
        assert expected_cols.issubset(result.columns)


def test_detect_proximity_events_empty_input():
    result = detect_proximity_events(pd.DataFrame())
    assert result.empty


def test_detect_proximity_events_missing_columns():
    df = pd.DataFrame({"session_id": [1], "driver": ["VER"], "time_s": [0.0]})
    with pytest.raises(ValueError, match="missing columns"):
        detect_proximity_events(df)


def test_detect_proximity_events_large_threshold(sample_lap_positions):
    """With very large threshold all cars should be in proximity."""
    result = detect_proximity_events(sample_lap_positions, distance_threshold_m=1e6)
    assert len(result) > 0


def test_detect_proximity_events_zero_threshold(sample_lap_positions):
    """With zero threshold no events should be found."""
    result = detect_proximity_events(sample_lap_positions, distance_threshold_m=0.0)
    assert result.empty


# ---------------------------------------------------------------------------
# Tests: classify_traffic_regimes
# ---------------------------------------------------------------------------


def test_classify_traffic_regimes_adds_regime_column(sample_laps):
    result = classify_traffic_regimes(sample_laps, pd.DataFrame())
    assert "traffic_regime" in result.columns


def test_classify_traffic_regimes_valid_values(sample_laps):
    result = classify_traffic_regimes(sample_laps, pd.DataFrame())
    assert set(result["traffic_regime"]).issubset(set(TRAFFIC_REGIMES))


def test_classify_traffic_regimes_clean_air_no_proximity(sample_laps):
    """Rows with large gap and no proximity events → CLEAN_AIR."""
    laps = sample_laps.copy()
    laps["gap_to_ahead_s"] = 10.0
    result = classify_traffic_regimes(laps, pd.DataFrame())
    assert (result["traffic_regime"] == "CLEAN_AIR").all()


def test_classify_traffic_regimes_adds_proximity_pct(sample_laps):
    result = classify_traffic_regimes(sample_laps, pd.DataFrame())
    assert "proximity_pct" in result.columns
    assert (result["proximity_pct"] >= 0).all()


# ---------------------------------------------------------------------------
# Tests: quantify_traffic_penalties
# ---------------------------------------------------------------------------


def test_quantify_traffic_penalties_adds_penalty_column(sample_laps):
    laps = classify_traffic_regimes(sample_laps, pd.DataFrame())
    result = quantify_traffic_penalties(laps)
    assert "traffic_penalty_s" in result.columns


def test_quantify_traffic_penalties_missing_columns(sample_laps):
    df = sample_laps.drop(columns=["session_id"])
    with pytest.raises(ValueError, match="missing columns"):
        quantify_traffic_penalties(df)


def test_quantify_traffic_penalties_clean_air_near_zero(sample_laps):
    """Clean air laps should have near-zero penalty on average."""
    laps = classify_traffic_regimes(sample_laps, pd.DataFrame())
    result = quantify_traffic_penalties(laps)
    clean = result[result["traffic_regime"] == "CLEAN_AIR"]["traffic_penalty_s"]
    assert abs(clean.mean()) < 2.0


def test_summarise_traffic_penalties(sample_laps):
    laps = classify_traffic_regimes(sample_laps, pd.DataFrame())
    laps = quantify_traffic_penalties(laps)
    summary = summarise_traffic_penalties(laps)
    assert set(summary.columns).issuperset(
        {"traffic_regime", "avg_penalty_s", "median_penalty_s", "n_laps"}
    )


# ---------------------------------------------------------------------------
# Tests: detect_overtakes
# ---------------------------------------------------------------------------


def test_detect_overtakes_returns_dataframe(sample_laps):
    result = detect_overtakes(sample_laps)
    assert isinstance(result, pd.DataFrame)


def test_detect_overtakes_finds_position_gain(sample_laps):
    """HAM starts in position 2 (laps 1–2) then gains to position 1 on lap 3 — a real overtake."""
    result = detect_overtakes(sample_laps)
    ham_overtakes = result[result["driver"] == "HAM"]
    assert len(ham_overtakes) >= 1


def test_detect_overtakes_missing_columns(sample_laps):
    df = sample_laps.drop(columns=["position"])
    with pytest.raises(ValueError, match="missing columns"):
        detect_overtakes(df)


def test_detect_overtakes_empty_input():
    result = detect_overtakes(pd.DataFrame())
    assert result.empty


def test_detect_overtakes_columns(sample_laps):
    result = detect_overtakes(sample_laps)
    if not result.empty:
        assert "position_change" in result.columns
        assert "position_after" in result.columns
        assert "overtake_type" in result.columns


def test_detect_overtakes_pit_lap_excluded(sample_laps):
    """Overtakes on pit laps should be excluded."""
    laps = sample_laps.copy()
    laps.loc[(laps["driver"] == "HAM") & (laps["lap_number"] == 3), "is_pit_lap"] = 1
    result = detect_overtakes(laps)
    ham_on_pit = result[
        (result["driver"] == "HAM") & (result["lap_number"] == 3)
    ]
    assert ham_on_pit.empty


# ---------------------------------------------------------------------------
# Tests: detect_overtake_hotspots
# ---------------------------------------------------------------------------


def test_detect_overtake_hotspots_empty_inputs():
    result = detect_overtake_hotspots(pd.DataFrame(), pd.DataFrame())
    assert result.empty


def test_detect_overtake_hotspots_returns_clusters(sample_laps, sample_lap_positions):
    overtakes = detect_overtakes(sample_laps)
    result = detect_overtake_hotspots(
        overtakes, sample_lap_positions, eps_m=1000.0, min_samples=1
    )
    assert isinstance(result, pd.DataFrame)


def test_detect_overtake_hotspots_columns(sample_laps, sample_lap_positions):
    overtakes = detect_overtakes(sample_laps)
    result = detect_overtake_hotspots(
        overtakes, sample_lap_positions, eps_m=1000.0, min_samples=1
    )
    if not result.empty:
        assert set(result.columns).issuperset(
            {"cluster_id", "center_x", "center_y", "n_overtakes"}
        )


# ---------------------------------------------------------------------------
# Tests: map_track_bottlenecks
# ---------------------------------------------------------------------------


def test_map_track_bottlenecks_empty_inputs():
    result = map_track_bottlenecks(pd.DataFrame(), pd.DataFrame())
    assert result.empty


def test_map_track_bottlenecks_returns_dataframe(sample_lap_positions):
    proximity = detect_proximity_events(sample_lap_positions, distance_threshold_m=1e6)
    if proximity.empty:
        pytest.skip("No proximity events generated with test data")
    result = map_track_bottlenecks(sample_lap_positions, proximity)
    assert isinstance(result, pd.DataFrame)


# ---------------------------------------------------------------------------
# Tests: run_domain2_pipeline
# ---------------------------------------------------------------------------


def test_run_domain2_pipeline_returns_dict(sample_positions, sample_laps):
    result = run_domain2_pipeline(sample_positions, sample_laps)
    assert isinstance(result, dict)
    expected_keys = {
        "lap_positions", "proximity_events", "laps_with_regimes",
        "laps_with_penalties", "overtakes", "hotspots", "bottlenecks",
    }
    assert expected_keys == set(result.keys())


def test_run_domain2_pipeline_all_dataframes(sample_positions, sample_laps):
    result = run_domain2_pipeline(sample_positions, sample_laps)
    for key, val in result.items():
        assert isinstance(val, pd.DataFrame), f"{key} should be a DataFrame"


def test_run_domain2_pipeline_laps_have_regime(sample_positions, sample_laps):
    result = run_domain2_pipeline(sample_positions, sample_laps)
    assert "traffic_regime" in result["laps_with_regimes"].columns


def test_run_domain2_pipeline_laps_have_penalty(sample_positions, sample_laps):
    result = run_domain2_pipeline(sample_positions, sample_laps)
    assert "traffic_penalty_s" in result["laps_with_penalties"].columns
