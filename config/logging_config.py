"""Central logging configuration: structured output to console + rotating file."""

from __future__ import annotations

import logging
import logging.handlers
from pathlib import Path

_LOG_FORMAT = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"

_configured = False


def setup_logging(
    level: str = "INFO",
    log_dir: str | Path = "logs",
    log_file: str = "paperpilot.log",
) -> logging.Logger:
    """Configure the root logger once (idempotent) and return it.

    Logs go to the console and to a size-rotating file under ``log_dir``.
    Calling this more than once is a no-op so library/CLI entry points can
    each call it safely.
    """
    global _configured
    root = logging.getLogger()

    if _configured:
        return root

    log_path = Path(log_dir)
    log_path.mkdir(parents=True, exist_ok=True)

    formatter = logging.Formatter(_LOG_FORMAT, datefmt=_DATE_FORMAT)

    console = logging.StreamHandler()
    console.setFormatter(formatter)

    file_handler = logging.handlers.RotatingFileHandler(
        log_path / log_file,
        maxBytes=5 * 1024 * 1024,
        backupCount=5,
        encoding="utf-8",
    )
    file_handler.setFormatter(formatter)

    root.setLevel(level)
    root.handlers.clear()
    root.addHandler(console)
    root.addHandler(file_handler)

    # Quiet down chatty third-party libraries.
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    logging.getLogger("alpaca").setLevel(logging.INFO)

    _configured = True
    root.debug("Logging configured (level=%s, dir=%s)", level, log_path)
    return root


def get_logger(name: str) -> logging.Logger:
    """Return a module-scoped logger. Use ``get_logger(__name__)``."""
    return logging.getLogger(name)
