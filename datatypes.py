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
    is_forming_pbos: bool

    @staticmethod
    def create(low_chain_length: int, high_chain_length: int, start_pair_df_index: int, direction: str, is_forming_pbos: bool):
        return OneDChain(low_chain_length, high_chain_length, start_pair_df_index, direction, is_forming_pbos)


class Box:
    def __init__(self, base_candle: Candle, box_type: str):
        self.start_index = base_candle.pair_df_index
        self.base_candle = base_candle
        self.type = box_type

        self.top = base_candle.high
        self.bottom = base_candle.low

        self.is_valid = True
        self.price_exit_index = None
        self.price_reentry_indices = []

    def check_box_entries(self, pair_df: pd.DataFrame) -> None:
        """
        Method to check the entries of the box and determine its validity.

        This method checks when the price candlesticks in pair_df "exit" the box and whether they re-enter the box.
        If there is a re-entry, the box is marked as invalid. All the indices are also registered.

        Parameters:
        pair_df (pd.DataFrame): The DataFrame containing the price data.

        Returns:
        None
        """

        # Get the subset of pair_df that we need to check
        check_window = pair_df.iloc[self.start_index + 1:]

        # If the box is of type "long"
        if self.type == "long":
            # Find the first index where the price exits the box
            exit_index = check_window[check_window['low'] > self.top].first_valid_index()
            if exit_index is not None:
                self.price_exit_index = exit_index
                # If an exit is found, check for a reentry into the box after the exit
                # Should use check_window.loc[exit_index:] instead of iloc because the current df is a subset of pair_df, and the indices are
                # all messed up
                reentry_check_window = check_window.loc[exit_index:]
                reentry_index = reentry_check_window.loc[reentry_check_window['low'] < self.top].first_valid_index()
                # If a reentry is found, mark the box as invalid
                if reentry_index is not None:
                    self.price_reentry_indices.append(reentry_index)
                    self.is_valid = False
        else:  # If the box is of type "short"
            # Find the first index where the price exits the box
            exit_index = check_window[check_window['high'] < self.bottom].first_valid_index()
            if exit_index is not None:
                self.price_exit_index = exit_index
                # If an exit is found, check for a reentry into the box after the exit
                # Should use check_window.loc[exit_index:] instead of iloc because the current df is a subset of pair_df, and the indices are
                # all messed up
                reentry_check_window = check_window.loc[exit_index:]
                reentry_index = reentry_check_window.loc[reentry_check_window['high'] > self.bottom].first_valid_index()
                # If a reentry is found, mark the box as invalid
                if reentry_index is not None:
                    self.price_reentry_indices.append(reentry_index)
                    self.is_valid = False
