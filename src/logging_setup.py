"""
Concept: LoggingSetup
Purpose: Provide a single point of loguru configuration for the project.
State: A configured loguru sink writing to stderr at the user-configured level.
Actions: `configure_logging(level)`.
Operational principle: Calling `configure_logging("INFO")` once at process start makes every
    subsequent `from loguru import logger` import emit consistently formatted log lines.
"""

from __future__ import annotations

import sys

from loguru import logger

_CONFIGURED: bool = False


def configure_logging(level: str | None = None) -> None:
    """Initialize the project loguru sink.

    Args:
        level: Log level string (e.g., "DEBUG", "INFO"). If None, falls back to `settings.log_level`.
    """
    global _CONFIGURED
    if _CONFIGURED:
        return

    from src.config import settings

    effective_level = (level or settings.log_level).upper()
    logger.remove()
    logger.add(
        sys.stderr,
        level=effective_level,
        format=(
            "<green>{time:YYYY-MM-DD HH:mm:ss}</green> "
            "| <level>{level: <8}</level> "
            "| <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> "
            "- <level>{message}</level>"
        ),
        backtrace=False,
        diagnose=False,
    )
    _CONFIGURED = True
