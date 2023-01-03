import logging


def setup_logger(logger_name, log_file, formatter, level=logging.DEBUG):
    """
    Setup a logger with

    Args:
        logger_name: logger file name
        log_file: logging file path
        formatter: specified formatter - logging.Formatter
        level: logging level

    """
    log = logging.getLogger(logger_name)
    file_handler = logging.FileHandler(log_file, mode="w")
    file_handler.setFormatter(formatter)
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)

    log.setLevel(level)
    log.addHandler(file_handler)
    log.addHandler(stream_handler)
