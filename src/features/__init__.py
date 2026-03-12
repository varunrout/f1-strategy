"""Feature engineering modules."""


def categorize_track_status(track_status: str) -> str:
    """Categorize track status into simplified categories.

    Args:
        track_status: Raw track status string from FastF1

    Returns:
        Categorized status: 'GREEN', 'YELLOW', 'SC', 'VSC', 'RED', or 'UNKNOWN'
    """
    if not track_status or track_status == '1':
        return 'GREEN'
    elif '2' in track_status:
        return 'YELLOW'
    elif '4' in track_status:
        return 'SC'
    elif '6' in track_status:
        return 'VSC'
    elif '5' in track_status:
        return 'RED'
    else:
        return 'UNKNOWN'
