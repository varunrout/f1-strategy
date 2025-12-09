"""Tests for FastF1 utilities."""
import pytest
from src.utils.fastf1_utils import normalize_session_type


def test_normalize_session_type():
    """Test session type normalization."""
    assert normalize_session_type("FP1") == "FP1"
    assert normalize_session_type("fp1") == "FP1"
    assert normalize_session_type("Q") == "Q"
    assert normalize_session_type("qualifying") == "Q"
    assert normalize_session_type("QUALIFYING") == "Q"
    assert normalize_session_type("R") == "R"
    assert normalize_session_type("race") == "R"
    assert normalize_session_type("RACE") == "R"
    assert normalize_session_type("S") == "S"
    assert normalize_session_type("sprint") == "S"
    assert normalize_session_type("SPRINT") == "S"


# Note: We can't easily test actual FastF1 API calls without internet
# and valid F1 data, so we focus on helper functions
