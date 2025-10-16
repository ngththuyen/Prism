"""
Logging configuration with UTF-8 support for Vietnamese characters
"""
import logging
import sys
from pathlib import Path


def setup_logging(log_file: Path = None, level=logging.INFO):
    """
    Setup logging with UTF-8 encoding support for Vietnamese characters
    
    Args:
        log_file: Path to log file (optional)
        level: Logging level (default: INFO)
    """
    # Remove any existing handlers
    root_logger = logging.getLogger()
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)
    
    # Create formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Create handlers list
    handlers = []
    
    # Console handler with UTF-8 encoding
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    
    # Try to reconfigure stream encoding to UTF-8
    try:
        if hasattr(console_handler.stream, 'reconfigure'):
            console_handler.stream.reconfigure(encoding='utf-8')
        elif hasattr(console_handler.stream, 'buffer'):
            # For Python < 3.7, wrap the buffer
            import io
            console_handler.stream = io.TextIOWrapper(
                console_handler.stream.buffer,
                encoding='utf-8',
                line_buffering=True
            )
    except Exception:
        # If reconfiguration fails, continue with default
        pass
    
    handlers.append(console_handler)
    
    # File handler with UTF-8 encoding
    if log_file:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(
            log_file,
            encoding='utf-8',
            mode='a'
        )
        file_handler.setFormatter(formatter)
        handlers.append(file_handler)
    
    # Configure root logger
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=handlers,
        force=True  # Force reconfiguration
    )
    
    return logging.getLogger()
