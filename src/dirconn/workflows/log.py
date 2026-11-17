"""Logging set-up of a pipeline run: ``dirconn.log`` in the log directory plus stderr."""

from __future__ import annotations

import logging
import sys
from pathlib import Path

__all__ = ["setup_logging"]

LOGGER_NAME = "dirconn"
LOG_FORMAT = "%(asctime)s %(levelname)s %(name)s: %(message)s"
_MARK = "_dirconn_handler"


def _level(verbosity: int) -> int:
    if verbosity <= 0:
        return logging.WARNING
    if verbosity == 1:
        return logging.INFO
    return logging.DEBUG


def setup_logging(log_dir: Path, verbosity: int) -> logging.Logger:
    """Configure the ``dirconn`` logger to write to ``log_dir/dirconn.log`` and to stderr.

    Parameters
    ----------
    log_dir : Path
        Directory of the log file; created when missing.
    verbosity : int
        ``0`` logs warnings, ``1`` information, ``2`` or more debug messages.

    Returns
    -------
    logging.Logger
        The ``dirconn`` logger. Calling the function again does not add handlers: the stderr
        handler is kept, and the file handler is kept when ``log_dir`` is unchanged and replaced
        otherwise. Only handlers installed by this function are touched.
    """
    log_dir = Path(log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger(LOGGER_NAME)
    level = _level(verbosity)
    logger.setLevel(level)
    formatter = logging.Formatter(LOG_FORMAT)
    log_file = (log_dir / "dirconn.log").resolve()

    has_stream = False
    has_file = False
    for handler in list(logger.handlers):
        if not getattr(handler, _MARK, False):
            continue
        if isinstance(handler, logging.FileHandler):
            if Path(handler.baseFilename).resolve() == log_file:
                has_file = True
            else:
                logger.removeHandler(handler)
                handler.close()
        elif isinstance(handler, logging.StreamHandler):
            has_stream = True

    if not has_stream:
        stream = logging.StreamHandler(sys.stderr)
        stream.setFormatter(formatter)
        setattr(stream, _MARK, True)
        logger.addHandler(stream)
    if not has_file:
        file_handler = logging.FileHandler(log_file, encoding="utf-8")
        file_handler.setFormatter(formatter)
        setattr(file_handler, _MARK, True)
        logger.addHandler(file_handler)
    return logger
