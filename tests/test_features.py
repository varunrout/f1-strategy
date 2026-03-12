"""Tests for feature building - track status categorization."""
import pytest
from src.features import categorize_track_status


def test_categorize_track_status():
    """Test track status categorization."""
    assert categorize_track_status('1') == 'GREEN'
    assert categorize_track_status('') == 'GREEN'
    assert categorize_track_status(None) == 'GREEN'
    assert categorize_track_status('2') == 'YELLOW'
    assert categorize_track_status('4') == 'SC'
    assert categorize_track_status('6') == 'VSC'
    assert categorize_track_status('5') == 'RED'
    assert categorize_track_status('7') == 'UNKNOWN'
