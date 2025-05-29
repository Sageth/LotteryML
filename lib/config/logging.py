import logging
from lib.config.custom_formatter import CustomFormatter  # Correct import

def configure_logging():
    log_level = logging.DEBUG
    handler = logging.StreamHandler()
    handler.setLevel(log_level)
    handler.setFormatter(CustomFormatter())

    logger = logging.getLogger("lottery")
    logger.setLevel(log_level)
    logger.handlers.clear()  # clear any pre-existing handlers
    logger.addHandler(handler)
    logger.propagate = False

    return logger
