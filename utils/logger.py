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
        pair_name = Config.get_pair_name()
        self.logger = logging.getLogger(name)
        self.logger.setLevel(logging.DEBUG)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_filename = f"../logs/{name}_{timestamp}-{pair_name}.log"
        file_handler = logging.FileHandler(log_filename)
        file_handler.setLevel(logging.DEBUG)

        formatter = logging.Formatter('%(message)s')
        file_handler.setFormatter(formatter)

        self.logger.addHandler(file_handler)

        print("Logger initiated with pair name", pair_name)

    def get_logger(self):
        return self.logger

# def get_higher_order_zigzag_logger(pair_name: str) -> logging.Logger:
#     return create_logger("ho_zigzag", pair_name)
#
#
# def get_position_formation_logger(pair_name: str) -> logging.Logger:
#     return create_logger("positions", pair_name)
