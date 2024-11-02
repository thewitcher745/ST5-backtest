import logging
from datetime import datetime
from utils.config import Config


class LoggerSingleton:
    _instance = None

    def __new__(cls, name: str):
        if cls._instance is None:
            cls._instance = super(LoggerSingleton, cls).__new__(cls)
            cls._instance._initialize(name)
        return cls._instance

    def _initialize(self, name: str):
        self.name = name
        self.logger = logging.getLogger(name)
        self.logger.setLevel(logging.DEBUG)

    def update_pair_name(self, pair_name: str):
        # Remove existing file handlers
        for handler in self.logger.handlers[:]:
            if isinstance(handler, logging.FileHandler):
                self.logger.removeHandler(handler)
                handler.close()

        # Create a new file handler with the updated pair_name
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_filename = f"./logs/{self.name}_{timestamp}-{pair_name}.log"
        file_handler = logging.FileHandler(log_filename)
        file_handler.setLevel(logging.DEBUG)

        formatter = logging.Formatter('%(message)s')
        file_handler.setFormatter(formatter)

        self.logger.addHandler(file_handler)

    def get_logger(self):
        return self.logger
