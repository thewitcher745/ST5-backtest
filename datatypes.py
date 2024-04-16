from typing import NamedTuple, List, Tuple, Type, Union
import pandas as pd
from datetime import datetime

NamedTupleColumnType = List[Tuple[str, type]]

class CandleTupleType(NamedTuple):
    pair_df_index: int
    time: datetime
    open: float
    high: float
    low: float
    close: float

pivot_columns: NamedTupleColumnType = [('pair_df_index', int), ('time', datetime), ('pivot_value', float), ('pivot_type', str)]
PivotTupleType: Type[NamedTuple] = NamedTuple('Pivot', pivot_columns)


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
