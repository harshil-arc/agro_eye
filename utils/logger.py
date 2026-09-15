import logging
import sys
from pathlib import Path
from config import LOG_FILE

def setup_logger(name: str = "AgroEye", log_file: str = LOG_FILE, level: int = logging.INFO) -> logging.Logger:
    """Configures and returns a centralized logger with console and file output."""
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger

    logger.setLevel(level)
    formatter = logging.Formatter(
        "[%(asctime)s] [%(levelname)s] [%(name)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )

    # Console Handler
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(level)
    ch.setFormatter(formatter)
    logger.addHandler(ch)

    # File Handler
    if log_file:
        try:
            Path(log_file).parent.mkdir(parents=True, exist_ok=True)
            fh = logging.FileHandler(log_file, encoding="utf-8")
            fh.setLevel(level)
            fh.setFormatter(formatter)
            logger.addHandler(fh)
        except Exception as e:
            print(f"Warning: Could not configure file logging to {log_file}: {e}")

    return logger

logger = setup_logger()

