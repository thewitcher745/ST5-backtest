from typing import NamedTuple, List, Tuple, Type, Union
import pandas as pd
from datetime import datetime


class Candle(NamedTuple):
    pair_df_index: int
    time: datetime
    open: float
    high: float
    low: float
    close: float

    @staticmethod
    def create(row: Union[pd.Series, NamedTuple]):
        if type(row) == pd.Series:
            return Candle(int(row.name), row.time, row.open, row.high, row.low, row.close)
        else:
            return Candle(row.Index, row.time, row.open, row.high, row.low, row.close)

class Pivot(NamedTuple):
    pair_df_index: int
    time: datetime
    pivot_value: float
    pivot_type: str

    @staticmethod
    def create(pivot: Union[tuple, pd.Series]):
        if type(pivot) == tuple:
            pivot_candle: Candle = pivot[0]
            pivot_type: str = pivot[1]
            pivot_value: float = pivot_candle.high if pivot_type == 'peak' else pivot_candle.low

            return Pivot(pivot_candle.pair_df_index, pivot_candle.time, pivot_value, pivot_type)

        else:
            return Pivot(pivot.pair_df_index, pivot.time, pivot.pivot_value, pivot.pivot_type)


class Leg(NamedTuple):
    start_index: int
    end_index: int
    start_time: datetime
    end_time: datetime
    start_value: float
    end_value: float
    leg_type: str

    @staticmethod
    def create(pivot_1: Pivot, pivot_2: Pivot):
        leg_type = 'bullish' if pivot_1.pivot_value < pivot_2.pivot_value else 'bearish'

        return Leg(pivot_1.pair_df_index, pivot_2.pair_df_index, pivot_1.time, pivot_2.time, pivot_1.pivot_value, pivot_2.pivot_value,
                            leg_type)


class OneDChain(NamedTuple):
    low_chain_length: int
    high_chain_length: int
    start_pair_df_index: int
    direction: str
    is_simplifying: bool

    @staticmethod
    def create(low_chain_length: int, high_chain_length: int, start_pair_df_index: int, direction: str, is_simplifying: bool):
        return OneDChain(low_chain_length, high_chain_length, start_pair_df_index, direction, is_simplifying)