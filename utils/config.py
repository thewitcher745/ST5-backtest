# utils/config.py
# utils/config.py
class Config:
    _pair_name = None

    @classmethod
    def set_pair_name(cls, pair_name: str):
        cls._pair_name = pair_name

    @classmethod
    def get_pair_name(cls) -> str:
        return cls._pair_name
