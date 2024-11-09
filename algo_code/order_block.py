from typing import Union
import pandas as pd

from algo_code.datatypes import Candle
import utils.general_utils as gen_utils
from algo_code.position import Position


class OrderBlock:
    def __init__(self, base_candle: Union[pd.Series, Candle], icl: float, ob_type: str):
        if isinstance(base_candle, Candle):
            self.start_index = base_candle.pdi
        elif isinstance(base_candle, pd.Series):
            self.start_index = base_candle.name
        elif isinstance(base_candle, tuple):
            self.start_index = base_candle.Index

        # Identification
        self.base_candle = base_candle
        self.type = ob_type
        self.id = f"OB{self.start_index}/" + gen_utils.convert_timestamp_to_readable(base_candle.time)
        self.id += "L" if ob_type == "long" else "S"

        # Geometry
        self.top = base_candle.high
        self.bottom = base_candle.low
        self.height = self.top - self.bottom
        # ICL represents the liquidity level of the initial candle, Initial Candle Liquidity. This is the (for long positions) low value of the first
        # candle that was tested for finding the order block. This is a very important variable which is directly used for forming the positions'
        # stoploss and targets.
        self.icl = icl

        # The position formed by the OrderBLock
        self.position = Position(self)

        # Checks and flags
        self.is_valid = True
        self.price_exit_index = None
        self.price_reentry_indices = []
        self.condition_check_window = None
        self.has_reentry_condition = True
        self.has_fvg_condition = None
        self.has_stop_break_condition = None
        self.has_been_replaced = False

        # The number of times the algorithm has tried to find a replacement for this order block
        self.times_moved = 0

        # The ranking of the order block within its segment. Sequential number, meaning the first VALID OB in a segment gets assigned number 1, next
        # one gets assigned 2, and so on
        self.ranking_within_segment = 0

    def __repr__(self):
        return f"OB {self.id} ({self.type})"

    def register_exit_candle(self, pair_df: pd.DataFrame, upper_search_bound_pdi: int) -> None:
        """
        Method to check the entries of the box and determine its validity.

        This method checks when the price candlesticks in pair_df "exit" the box and whether they re-enter the box.
        If there is a re-entry, the box is marked as invalid. All the indices are also registered.

        Args:
            pair_df (pd.DataFrame): The DataFrame containing the price data.
            upper_search_bound_pdi (int): The PDI of the candle to stop the search at.
        """

        # Get the subset of pair_df that we need to check
        check_window = pair_df.iloc[self.start_index + 1:upper_search_bound_pdi + 1]

        # If the box is of type "long"
        if self.type == "long":
            # Find the first candle where a candle opens inside the OB and closes above it
            exit_index = check_window[(check_window['close'] > self.top) & (check_window['open'] <= self.top)].first_valid_index()

        else:  # If the box is of type "short"
            # Find the first candle where a candle opens inside the OB and closes below it
            exit_index = check_window[(check_window['close'] < self.bottom) & (check_window['open'] >= self.bottom)].first_valid_index()

        if exit_index is not None:
            self.price_exit_index = exit_index

    def set_condition_check_window(self, condition_check_window: pd.DataFrame) -> None:
        self.condition_check_window = condition_check_window

    def check_reentry_condition(self, reentry_check_window: pd.DataFrame):
        """
        Method to check if the price returns to the  box pre-emptively, before it is fully formed from the LPL breaking it. This check is performed
        by checking all the candles from right after the base_candle to the candle that breaks the LPL (Which is passed to this function through
        the respective segment), and in the check window, for a long order block, we check if the LOWEST LOW of all the candles in the window pierced
        the order block's top. For a short position, we check the HIGHEST HIGH and the bottom of the box. The has_reentry_condition property of the OB
        object is the flag property that keeps track of the passing of this condition.

        Args:
            reentry_check_window (pd.DataFrame): A dataframe containing the candles of the window formed starting after the base_candle and before
            the breaking of the LPL. This window will be checked for reentry in this function.

        """

        if self.type == "long":
            # Check if the lowest low in the reentry check window pierces the top of the box
            lowest_low = reentry_check_window.low.min()
            if lowest_low <= self.top:
                self.has_reentry_condition = False
        else:
            # Check if the highest high in the reentry check window pierces the bottom of the box
            highest_high = reentry_check_window.high.max()
            if highest_high >= self.bottom:
                self.has_reentry_condition = False

    def check_fvg_condition(self):
        """
        Method to check the FVG condition for the box. The method checks if the exiting candle has an FVG on it which aligns exactly with the box's
        top/bottom for long/short cases. This method should be called after the check_box_entries method. The method sets the has_fvg_cond property
        for the instance of the object if the check passes, otherwise it will remain false.

        The method aggregates the candles before and after the exit candle using min() and max() functions and identifies the fair value gaps in the
        exiting candle, if any exist. Then the values are checked for alignment with the box's top/bottom for long/short cases.
        """

        # aggregated_candle_after represents the candles after the exit candle. This would be a candle with the highest high and lowest low set as
        # its high and low.

        if self.price_exit_index is None:
            self.has_fvg_condition = False

        aggregated_candle_after_exit: list = [self.condition_check_window.loc[self.price_exit_index + 1:].low.min(),
                                              self.condition_check_window.loc[self.price_exit_index + 1:].high.max()]
        aggregated_candle_before_exit: list = [self.condition_check_window.loc[:self.price_exit_index - 1].low.min(),
                                               self.condition_check_window.loc[:self.price_exit_index - 1].high.max()]

        def find_gap(interval1, interval2):
            if interval1[1] < interval2[0] or interval2[1] < interval1[0]:
                # The intervals do not overlap, return the gap
                return [min(interval1[1], interval2[1]), max(interval1[0], interval2[0])]
            else:
                # The intervals overlap
                return None

        def find_overlap(interval1, interval2):
            if interval1[1] < interval2[0] or interval2[1] < interval1[0]:
                # The intervals do not overlap
                return None
            else:
                # The intervals overlap
                return [max(interval1[0], interval2[0]), min(interval1[1], interval2[1])]

        # If the before and after aggregated candle overlap, no FVG exists
        if find_overlap(aggregated_candle_before_exit, aggregated_candle_after_exit) is not None:
            self.has_fvg_condition = False

        else:
            fvg_exit_candle: pd.Series = self.condition_check_window.loc[self.price_exit_index]

            # The overlap between the gap between the before and after candles and the exit candle's body constitutes the FVG.
            exit_candle_body_interval: list = [min(fvg_exit_candle.open, fvg_exit_candle.close),
                                               max(fvg_exit_candle.open, fvg_exit_candle.close)]
            gap = find_gap(aggregated_candle_before_exit, aggregated_candle_after_exit)

            fvg: list = find_overlap(exit_candle_body_interval, gap)

            # If an overlap between the gap and the exit candle's body exists
            if fvg is not None:
                # Check if the FVG area aligns with the order block EXACTLY. If so, the check passes
                if self.type == "long" and min(fvg) == self.top:
                    self.has_fvg_condition = True
                elif self.type == "short" and max(fvg) == self.bottom:
                    self.has_fvg_condition = True

                # If there isn't exact alignment, the check doesn't pass.
                else:
                    self.has_fvg_condition = False

            # If the exit candle's body doesn't overlap with the gap, there is no FVG.
            else:
                self.has_fvg_condition = False

    def check_stop_break_condition(self):
        """
        This method checks if there is a price which breaks the OrderBlock's stop level, meaning the bottom in a long OrderBlock and the top in a
        short OrderBlock.

        This method should be called after the check_box_entries method. The method sets the has_stop_break_condition property for the instance of
        the object accordingly.
        """

        # Find candles which break the stop level, if any.
        stop_breaking_candles: pd.DataFrame
        if self.type == "long":
            stop_breaking_candles = self.condition_check_window[self.condition_check_window['low'] < self.bottom]
        else:
            stop_breaking_candles = self.condition_check_window[self.condition_check_window['high'] > self.top]

        # If there are any candles in the condition check window which break the order block's stop level, the check fails.
        if len(stop_breaking_candles) > 0:
            self.has_stop_break_condition = False
        else:
            self.has_stop_break_condition = True
