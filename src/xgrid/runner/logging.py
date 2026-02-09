from __future__ import annotations

import logging

_LOGGER_NAME = "xgrid.runner"
_LOG_FORMAT = "%(levelname)s %(message)s"
_LOG_HANDLER_NAME = "xgrid.runner.stderr"


def configure_logging(log_level: str) -> logging.Logger:
    logger = logging.getLogger(_LOGGER_NAME)
    if not any(handler.get_name() == _LOG_HANDLER_NAME for handler in logger.handlers):
        handler = logging.StreamHandler()
        handler.set_name(_LOG_HANDLER_NAME)
        handler.setFormatter(logging.Formatter(_LOG_FORMAT))
        logger.addHandler(handler)
    logger.setLevel(_parse_log_level(log_level))
    logger.propagate = False
    return logger


def _parse_log_level(log_level: str) -> int:
    level = getattr(logging, log_level.upper(), None)
    if isinstance(level, int):
        return level
    return logging.INFO
