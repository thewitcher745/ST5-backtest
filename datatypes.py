from typing import NamedTuple, List, Tuple, Type, Union
import pandas as pd
from datetime import datetime


class CandleTupleType(NamedTuple):
    pair_df_index: int
    time: datetime
    open: float
    high: float
    low: float
    close: float


class PivotTupleType(NamedTuple):
    pair_df_index: int
    time: datetime
    pivot_value: float
    pivot_type: str


class LegTupleType(NamedTuple):
    start_index: int
    end_index: int
    start_time: datetime
    end_time: datetime
    start_value: float
    end_value: float
    leg_type: str


def create_candle_tuple(row: Union[pd.Series, CandleTupleType]) -> CandleTupleType:
    if type(row) == pd.Series:
        return CandleTupleType(int(row.name), row.time, row.open, row.high, row.low, row.close)
    else:
        return CandleTupleType(row.Index, row.time, row.open, row.high, row.low, row.close)


def create_pivot_tuple(pivot: tuple) -> PivotTupleType:
    pivot_candle: CandleTupleType = pivot[0]
    pivot_type: str = pivot[1]
    pivot_value: float = pivot_candle.high if pivot_type == "peak" else pivot_candle.low

    return PivotTupleType(pivot_candle.pair_df_index, pivot_candle.time, pivot_value, pivot_type)


def create_leg_tuple(pivot_1: PivotTupleType, pivot_2: PivotTupleType) -> LegTupleType:
    leg_type = "bullish" if pivot_1.pivot_value < pivot_2.pivot_value else "bearish"

    return LegTupleType(pivot_1.pair_df_index, pivot_2.pair_df_index, pivot_1.time, pivot_2.time, pivot_1.pivot_value, pivot_2.pivot_value, leg_type)
