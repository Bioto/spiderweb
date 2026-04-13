"""Logging configuration for Spiderweb.

Provides structured logging with file rotation and optional JSON formatting,
using structured console and rotating file handlers.
"""

import logging
import sys
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Any

from colorlog import ColoredFormatter
from pythonjsonlogger import json as jsonlogger


def setup_logging(
    log_level: str = "INFO",
    log_file_level: str = "DEBUG",
    log_dir: Path | str = "logs",
    log_file_name: str = "spiderweb.log",
    log_json_format: bool = False,
    log_max_bytes: int = 10485760,  # 10MB
    log_backup_count: int = 5,
    console_output: bool = True,
) -> None:
    """Set up logging configuration for Spiderweb.

    Args:
        log_level: Console logging level
        log_file_level: File logging level
        log_dir: Directory for log files
        log_file_name: Log file name
        log_json_format: Use JSON format for logs
        log_max_bytes: Maximum log file size in bytes
        log_backup_count: Number of backup files to keep
        console_output: Enable console logging output
    """
    # Convert log levels
    console_level = getattr(logging, log_level.upper(), logging.INFO)
    file_level = getattr(logging, log_file_level.upper(), logging.DEBUG)

    # Create log directory
    log_dir = Path(log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)

    # Create root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.DEBUG)  # Capture all levels, handlers will filter

    # Remove existing handlers
    root_logger.handlers.clear()

    # Console handler with colors
    if console_output:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(console_level)

        # Colored formatter for console
        console_format = (
            "%(log_color)s%(asctime)s%(reset)s | "
            "%(log_color)s%(levelname)-8s%(reset)s | "
            "%(cyan)s%(name)s%(reset)s | "
            "%(message)s"
        )

        console_formatter = ColoredFormatter(
            console_format,
            datefmt="%Y-%m-%d %H:%M:%S",
            log_colors={
                "DEBUG": "blue",
                "INFO": "green",
                "WARNING": "yellow",
                "ERROR": "red",
                "CRITICAL": "red,bg_white",
            },
        )

        console_handler.setFormatter(console_formatter)
        root_logger.addHandler(console_handler)

    # File handler with rotation
    log_file_path = log_dir / log_file_name
    file_handler = RotatingFileHandler(
        log_file_path,
        maxBytes=log_max_bytes,
        backupCount=log_backup_count,
        encoding="utf-8",
    )
    file_handler.setLevel(file_level)

    if log_json_format:
        # JSON formatter for structured logging
        json_format = "%(asctime)s %(name)s %(levelname)s %(message)s"
        file_formatter = jsonlogger.JsonFormatter(json_format)
    else:
        # Standard formatter
        file_format = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
        file_formatter = logging.Formatter(file_format, datefmt="%Y-%m-%d %H:%M:%S")

    file_handler.setFormatter(file_formatter)
    root_logger.addHandler(file_handler)

    # Set levels for noisy libraries
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    logging.getLogger("qdrant_client").setLevel(logging.INFO)


def get_logger(name: str, **kwargs: Any) -> logging.Logger:
    """Get a logger instance with the given name.

    Args:
        name: Logger name (typically __name__ from calling module)
        **kwargs: Additional context to include in structured logs

    Returns:
        Configured logger instance
    """
    logger = logging.getLogger(name)

    # Add extra context if provided
    if kwargs:
        logger = logging.LoggerAdapter(logger, kwargs)

    return logger
