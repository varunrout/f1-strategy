"""Logging utilities for the F1 strategy pipeline."""
import logging
import sys
from typing import Optional


def setup_logger(name: str, level: str = "INFO") -> logging.Logger:
    """Setup logger with consistent formatting.
    
    Args:
        name: Logger name
        level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        
    Returns:
        Configured logger instance
    """
    logger = logging.getLogger(name)
    
    # Only add handler if logger doesn't have one
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    
    logger.setLevel(getattr(logging, level.upper()))
    return logger


def log_ingestion_status(logger: logging.Logger, session_info: dict, 
                        status: str, message: Optional[str] = None) -> None:
    """Log ingestion status with consistent format.
    
    Args:
        logger: Logger instance
        session_info: Dictionary with session information
        status: Status string ('success', 'skipped', 'error')
        message: Optional message to include
    """
    log_msg = f"{session_info.get('year', 'N/A')} {session_info.get('gp_name', 'N/A')} "
    log_msg += f"{session_info.get('session_type', 'N/A')} - {status}"
    if message:
        log_msg += f": {message}"
    
    if status == 'success':
        logger.info(log_msg)
    elif status == 'skipped':
        logger.info(log_msg)
    elif status == 'error':
        logger.error(log_msg)
    else:
        logger.warning(log_msg)
