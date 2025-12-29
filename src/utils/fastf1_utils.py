"""FastF1 utility functions."""
import fastf1
from pathlib import Path
import logging

logger = logging.getLogger(__name__)

# Configure FastF1 cache directory
CACHE_DIR = Path(__file__).parent.parent.parent / "cache"
CACHE_DIR.mkdir(exist_ok=True)
fastf1.Cache.enable_cache(str(CACHE_DIR))


def get_session_safely(year: int, gp_name: str, session_type: str):
    """Get FastF1 session with error handling.
    
    Args:
        year: Season year
        gp_name: Grand Prix name (e.g., 'Monza', 'Silverstone')
        session_type: Session type ('FP1', 'FP2', 'FP3', 'Q', 'S', 'R')
        
    Returns:
        FastF1 Session object or None if failed
    """
    try:
        session = fastf1.get_session(year, gp_name, session_type)
        return session
    except Exception as e:
        logger.error(f"Failed to get session {year} {gp_name} {session_type}: {e}")
        return None


def load_session_safely(session, telemetry: bool = True, weather: bool = True, messages: bool = True):
    """Load FastF1 session data with error handling.
    
    Args:
        session: FastF1 Session object
        telemetry: Load telemetry/car data (default True)
        weather: Load weather data (default True)
        messages: Load race control messages (default True)
        
    Returns:
        True if loaded successfully, False otherwise
    """
    try:
        session.load(telemetry=telemetry, weather=weather, messages=messages)
        return True
    except Exception as e:
        logger.error(f"Failed to load session data: {e}")
        return False


def get_gp_list_for_year(year: int):
    """Get list of Grand Prix events for a given year.
    
    Args:
        year: Season year
        
    Returns:
        List of event names
    """
    try:
        schedule = fastf1.get_event_schedule(year)
        # Filter out testing events
        events = schedule[schedule['EventFormat'] != 'testing']
        return events['EventName'].tolist()
    except Exception as e:
        logger.error(f"Failed to get GP list for year {year}: {e}")
        return []


def normalize_session_type(session_type: str) -> str:
    """Normalize session type to FastF1 format.
    
    Args:
        session_type: Session type string
        
    Returns:
        Normalized session type
    """
    mapping = {
        'FP1': 'FP1',
        'FP2': 'FP2',
        'FP3': 'FP3',
        'Q': 'Q',
        'QUALIFYING': 'Q',
        'S': 'S',
        'SPRINT': 'S',
        'R': 'R',
        'RACE': 'R',
    }
    return mapping.get(session_type.upper(), session_type)
