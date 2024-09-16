from typing import NamedTuple, List, Tuple, Type, Union
import pandas as pd
from datetime import datetime

import general_utils as gen_utils


class Candle(NamedTuple):
    pdi: int
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
    pdi: int
    time: datetime
    pivot_value: float
    pivot_type: str

    @staticmethod
    def create(pivot: Union[tuple, pd.Series]):
        if type(pivot) == tuple:
            pivot_candle: Candle = pivot[0]
            pivot_type: str = pivot[1]
            pivot_value: float = pivot_candle.high if pivot_type == 'peak' else pivot_candle.low

            return Pivot(pivot_candle.pdi, pivot_candle.time, pivot_value, pivot_type)

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

        return Leg(pivot_1.pdi, pivot_2.pdi, pivot_1.time, pivot_2.time, pivot_1.pivot_value, pivot_2.pivot_value,
                   leg_type)


class OneDChain(NamedTuple):
    chain_length: int
    start_pair_df_index: int
    direction: str

    @staticmethod
    def create(chain_length: int, start_pair_df_index: int, direction: str):
        return OneDChain(chain_length, start_pair_df_index, direction)


class FVG(NamedTuple):
    middle_candle: int
    fvg_lower: float
    fvg_upper: float
