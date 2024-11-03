import logging
from datetime import datetime


class LoggerSingleton:
    _loggers = {}

    @classmethod
    def get_logger(cls, name: str):
        if name not in cls._loggers:
            cls._loggers[name] = cls._create_logger(name)
        return cls._loggers[name]

    @classmethod
    def _create_logger(cls, name: str):
        logger = logging.getLogger(name)
        logger.setLevel(logging.DEBUG)
        return logger

    @classmethod
    def update_pair_name(cls, name: str, pair_name: str):
        logger = cls.get_logger(name)
        # Remove existing file handlers
        for handler in logger.handlers[:]:
            if isinstance(handler, logging.FileHandler):
                logger.removeHandler(handler)
                handler.close()

        # Create a new file handler with the updated pair_name
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_filename = f"./logs/{name}_{timestamp}-{pair_name}.log"
        file_handler = logging.FileHandler(log_filename)
        file_handler.setLevel(logging.DEBUG)

        formatter = logging.Formatter('%(message)s')
        file_handler.setFormatter(formatter)

        logger.addHandler(file_handler)
