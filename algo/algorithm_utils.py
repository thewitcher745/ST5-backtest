from typing import Optional, Literal
import pandas as pd

import constants
from algo.datatypes import *
from utils.general_utils import log_message as log_message_general
from algo import position_prices_setup as setup
from utils.logger import LoggerSingleton

positions_logger = None


class Algo:
    def __init__(self, pair_df: pd.DataFrame,
                 symbol: str,
                 timeframe: str = "15m",
                 allowed_verbosity=constants.allowed_verbosity):
        global positions_logger

        if positions_logger is None:
            positions_logger = LoggerSingleton("positions").get_logger()

        self.allowed_verbosity = allowed_verbosity
        self.pair_df: pd.DataFrame = pair_df
        self.symbol: str = symbol
        self.zigzag_df: Optional[pd.DataFrame] = None

        # pbos_indices and choch_indices is a list which stores the PBOS and CHOCH's being moved due to shadows breaking the most recent lows/highs
        self.pbos_indices: list[int] = []
        self.choch_indices: list[int] = []

        # Indices of NON-BROKEN LPL's. This means only LPL's that get updated to the next one in the calc_broken_lpl method are added here.
        self.lpl_indices: dict[str, list] = {
            "peak": [],
            "valley": []
        }

        # h_o_indices indicates the indices of the peaks and valleys in the higher order zigzag
        self.h_o_indices: list[int] = []

        # This is a list that will contain all the segments to be processed in the final backtest. Each segment is a datatype with its start and end
        # PDI's specified, and a top and bottom price for plotting purposes. Segments end at PBOS_CLOSE events and start at the low/high before the
        # PBOS which they closed above/below (Ascending/Descending)
        self.segments: list[Segment] = []

        # starting_pdi is the starting point of the entire pattern, calculated using __init_pattern_start_pdi. This method is
        # executed in the calc_h_o_zigzag method.
        self.starting_pdi = 0

    def log_message(self, *messages, v: int = 3):
        """
        Method to log messages with a verbosity check.

        Args:
            *messages: List of messages in print() argument format
            v (int): Allowed verbosity level for the message
        """
        log_message_general(*messages, v=v, av=self.allowed_verbosity)

    def init_zigzag(self, last_pivot_type=None, last_pivot_candle_pdi=None) -> None:
        """
            Method to identify turning points in a candlestick chart.
            It compares each candle to its previous pivot to determine if it's a new pivot point.
            This implementation is less optimized than the deprecated version, as it doesn't use
            vectorized operations, but it is what it is

            Returns:
            pd.DataFrame: A DataFrame containing the identified turning points.
            """

        if last_pivot_type is None:
            # Find the first candle that has a higher high or a lower low than its previous candle
            # and set it as the first pivot. Also set the type of the pivot (peak or valley)

            last_pivot_candle_series = \
                self.pair_df[(self.pair_df['high'] > self.pair_df['high'].shift(1)) | (
                        self.pair_df['low'] < self.pair_df['low'].shift(1))].iloc[0]

            last_pivot_type: str = 'valley'
            if last_pivot_candle_series.high > self.pair_df.iloc[last_pivot_candle_series.name - 1].high:
                last_pivot_type = 'peak'

        # If a first candle is already given
        else:
            last_pivot_candle_series = self.pair_df.loc[last_pivot_candle_pdi]

        last_pivot_candle: Candle = Candle.create(last_pivot_candle_series)
        pivots: List[Pivot] = []

        # Start at the candle right after the last (first) pivot
        for row in self.pair_df.iloc[last_pivot_candle.pdi + 1:].itertuples():

            # Conditions to check if the current candle is an extension of the last pivot or a reversal
            peak_extension_condition: bool = row.high > last_pivot_candle.high and last_pivot_type == 'peak'
            valley_extension_condition: bool = row.low < last_pivot_candle.low and last_pivot_type == 'valley'

            reversal_from_peak_condition = row.low < last_pivot_candle.low and last_pivot_type == 'peak'
            reversal_from_valley_condition = row.high > last_pivot_candle.high and last_pivot_type == 'valley'

            # Does the candle register both a higher high AND a lower low?
            if (reversal_from_valley_condition and valley_extension_condition) or (
                    peak_extension_condition and reversal_from_peak_condition):

                # INITIAL NAIVE IMPLEMENTATION
                # Add the last previous pivot to the list
                # pivots.append(Pivot.create((last_pivot_candle, last_pivot_type)))

                # Update the last pivot's type and value
                # last_pivot_candle = Candle.create(row)
                # last_pivot_type = 'valley' if last_pivot_type == 'peak' else 'peak'

                # JUDGING BASED ON CANDLE COLOR
                # If the candle is green, that means the low value was probably hit before the high value
                # If the candle is red, that means the high value was probably hit before the low value
                # This means that if the candle is green, we can extend a peak, and if it's red, we can extend a valley
                # Otherwise the direction must flip
                if (row.candle_color == 'green' and last_pivot_type == 'valley') or (
                        row.candle_color == 'red' and last_pivot_type == 'peak'):
                    # Add the last previous pivot to the list of pivots
                    pivots.append(Pivot.create((last_pivot_candle, last_pivot_type)))

                    # Update the last pivot's type and value
                    last_pivot_candle = Candle.create(row)
                    last_pivot_type = 'valley' if last_pivot_type == 'peak' else 'peak'

                else:
                    last_pivot_candle = Candle.create(row)

            # Has a same direction pivot been found?
            if peak_extension_condition or valley_extension_condition:
                # Don't change the direction of the last pivot found, just update its value
                last_pivot_candle = Candle.create(row)

            # Has a pivot in the opposite direction been found?
            elif reversal_from_valley_condition or reversal_from_peak_condition:
                # Add the last previous pivot to the list of pivots
                pivots.append(Pivot.create((last_pivot_candle, last_pivot_type)))

                # Update the last pivot's type and value
                last_pivot_candle = Candle.create(row)
                last_pivot_type = 'valley' if last_pivot_type == 'peak' else 'peak'

        # Convert the pivot list to zigzag_df
        # noinspection PyTypeChecker
        zigzag_df = pd.DataFrame.from_dict(pivot._asdict() for pivot in pivots)

        self.zigzag_df = zigzag_df

    def find_relative_pivot(self, pivot_pdi: int, delta: int) -> int:
        """
        Finds the relative pivot to the pivot at the given index.

        Args:
            pivot_pdi (int): The pdi of the pivot to find the relative pivot for.
            delta (int): The distance from the pivot to the relative pivot.

        Returns:
            int: The pdi of the relative pivot.
        """

        # zigzag_idx is the zigzag_df index of the current pivot
        zigzag_idx = self.zigzag_df[self.zigzag_df.pdi == pivot_pdi].first_valid_index()

        return self.zigzag_df.iloc[zigzag_idx + delta].pdi

    def detect_first_broken_lpl(self, search_window_start_pdi: int) -> Union[None, tuple[pd.Series, int]]:
        """
        Calculates the LPL's and then broken LPL's in a series of zigzag pivots.


        An LPL (For ascending patterns) is registered when a higher high than the highest high since the last LPL is registered. If a lower low than
        the lowest low is registered, the last LPL is considered a broken LPL and registered as such.

        Args:
            search_window_start_pdi (int): The pdi of the pivot to start the search from.

        Returns:
            pd.Series: a row of zigzag_df which contains the broken LPL
            None: If no broke LPL is found
        """

        starting_pivot = self.zigzag_df[self.zigzag_df.pdi == search_window_start_pdi].iloc[0]
        trend_type = "ascending" if starting_pivot.pivot_type == "valley" else "descending"
        self.log_message("Trend type is", trend_type, v=3)

        # Breaking and extension pdi and values represent the values to surpass for registering a higher high (extension) of a lower low (breaking)
        breaking_pdi = search_window_start_pdi
        breaking_value: float = starting_pivot.pivot_value
        extension_pdi = self.find_relative_pivot(search_window_start_pdi, 1)
        extension_value: float = self.zigzag_df.loc[self.zigzag_df.pdi == extension_pdi].iloc[0].pivot_value

        check_start_pdi = self.find_relative_pivot(search_window_start_pdi, 2)

        for row in self.zigzag_df[self.zigzag_df.pdi >= check_start_pdi].iloc[:-1].itertuples():
            if trend_type == "ascending":
                extension_condition = row.pivot_type == "peak" and row.pivot_value >= extension_value
                breaking_condition = row.pivot_type == "valley" and row.pivot_value <= breaking_value
            else:
                extension_condition = row.pivot_type == "valley" and row.pivot_value <= extension_value
                breaking_condition = row.pivot_type == "peak" and row.pivot_value >= breaking_value

            # Breaking
            if breaking_condition:
                # If a breaking event has occurred, we need to find the actual CANDLE that broke the LPL, since it might have happened before the PIVOT
                # that broke the LPL, since zigzag pivots are a much more aggregated type of data compared to the candles and almost always the actual
                # candle that breaks the LPL is one of the candles before the pivot that was just found.

                # The candle search range starts at the pivot before the LPL-breaking pivot (which is typically a higher order pivot) PDI and the
                # breaking pivot PDI.
                pivot_before_breaking_pivot: int = self.find_relative_pivot(row.pdi, -1)
                breaking_candle_search_window: pd.DataFrame = self.pair_df.loc[pivot_before_breaking_pivot + 1:row.pdi + 1]

                # If the trend is ascending, it means the search window should be checked for the first candle that breaks the LPL by having a lower
                # low than the breaking_value.
                if trend_type == "ascending":
                    lpl_breaking_candles = breaking_candle_search_window[breaking_candle_search_window.low < breaking_value]

                # If the trend is descending, the breaking candle must have a higher high than the breaking value.
                elif trend_type == "descending":
                    lpl_breaking_candles = breaking_candle_search_window[breaking_candle_search_window.high > breaking_value]

                breaking_candle_pdi = lpl_breaking_candles.first_valid_index()

                # If the search window for the breaking candle is empty, return the pivot as the breaking candle
                if breaking_candle_pdi is None:
                    breaking_candle_pdi = row.pdi

                if constants.logs_format == "time":
                    self.log_message("LPL #", breaking_pdi, "broken at", self.convert_pdis_to_times(breaking_candle_pdi), v=1)
                else:
                    self.log_message("LPL #", breaking_pdi, "broken at", breaking_candle_pdi, v=1)

                return self.zigzag_df[self.zigzag_df.pdi == breaking_pdi].iloc[0], breaking_candle_pdi

            # Extension
            if extension_condition:
                # If a higher high is found, extend and update the pattern

                prev_pivot_pdi = self.find_relative_pivot(row.pdi, -1)
                prev_pivot = self.zigzag_df[self.zigzag_df.pdi == prev_pivot_pdi].iloc[0]

                if constants.logs_format == "time":
                    self.log_message("Changing breaking_pdi to", self.convert_pdis_to_times(prev_pivot.pdi))
                else:
                    self.log_message("Changing breaking_pdi to", prev_pivot.pdi)
                breaking_pdi = prev_pivot.pdi
                breaking_value = prev_pivot.pivot_value
                extension_value = row.pivot_value

            # If a break or extension has happened, the next LPL is the pivot at the breaking pivot
            if breaking_condition or extension_condition:
                if trend_type == "ascending":
                    self.lpl_indices["valley"].append(breaking_pdi)
                else:
                    self.lpl_indices["peak"].append(breaking_pdi)

        return None

    def __detect_breaking_sentiment(self, latest_pbos_value: float, latest_pbos_pdi: int, latest_choch_value: float,
                                    trend_type: str) -> dict:
        """
        Detects the breaking sentiment in the price data based on the latest PBOS and CHOCH values.

        This method identifies the candles that break the PBOS and CHOCH values either by shadow or close price. It then determines
        which breaking event occurs first and returns the sentiment associated with that event.

        Args:
            latest_pbos_value (float): The latest PBOS value.
            latest_pbos_pdi (int): The index of the latest PBOS.
            latest_choch_value (float): The latest CHOCH value.
            trend_type (str): The current trend type, either "ascending" or "descending".

        Returns:
            dict: A dictionary containing the breaking sentiment and the index of the candle that caused the break. The breaking sentiment
                  can be one of the following: "PBOS_SHADOW", "PBOS_CLOSE", "CHOCH_SHADOW", "CHOCH_CLOSE", "NONE".
        """
        search_window: pd.DataFrame = self.pair_df.iloc[latest_pbos_pdi + 1:]

        # The definition of "breaking" is different whether the PBOS is a peak or a valley
        if trend_type == "ascending":
            pbos_shadow_breaking_candles = search_window[search_window.high > latest_pbos_value]
            pbos_close_breaking_candles = search_window[search_window.close > latest_pbos_value]
            choch_shadow_breaking_candles = search_window[search_window.low < latest_choch_value]
            choch_close_breaking_candles = search_window[search_window.close < latest_choch_value]

        else:
            pbos_shadow_breaking_candles = search_window[search_window.low < latest_pbos_value]
            pbos_close_breaking_candles = search_window[search_window.close < latest_pbos_value]
            choch_shadow_breaking_candles = search_window[search_window.high > latest_choch_value]
            choch_close_breaking_candles = search_window[search_window.close > latest_choch_value]

        pbos_close_index = pbos_close_breaking_candles.first_valid_index()
        pbos_shadow_index = pbos_shadow_breaking_candles.first_valid_index()
        choch_shadow_index = choch_shadow_breaking_candles.first_valid_index()
        choch_close_index = choch_close_breaking_candles.first_valid_index()

        # The return dicts for each case
        pbos_shadow_output = {
            "sentiment": "PBOS_SHADOW",
            "pdi": pbos_shadow_index
        }
        pbos_close_output = {
            "sentiment": "PBOS_CLOSE",
            "pdi": pbos_close_index
        }
        choch_shadow_output = {
            "sentiment": "CHOCH_SHADOW",
            "pdi": choch_shadow_index
        }
        choch_close_output = {
            "sentiment": "CHOCH_CLOSE",
            "pdi": choch_close_index
        }
        none_output = {
            "sentiment": "NONE",
            "pdi": None
        }

        outputs_list: list[dict] = [pbos_shadow_output, pbos_close_output, choch_shadow_output, choch_close_output]

        # This function sorts the outputs of the breaking sentiment analysis to determine which one is reached first using the
        # sorted built-in function. It also prioritizes sentiments that have "CLOSE" in their description, because a candle closing above/below a
        # value logically takes priority over a shadow.
        def sorting_key(output_item):
            pdi = output_item["pdi"] if output_item["pdi"] is not None else 0
            has_close = 1 if "CLOSE" in output_item["sentiment"] else 2
            return pdi, has_close

        sorted_outputs: list[dict] = [output_item for output_item in sorted(outputs_list, key=sorting_key)
                                      if output_item["pdi"] is not None]

        return sorted_outputs[0] if len(sorted_outputs) > 0 else none_output

    def __calc_region_start_pdi(self, broken_lpl: pd.Series) -> int:
        """
        Initializes the starting point of the region after the broken LPL

        The region starting point is the first pivot right after the broken LPL

        Args:
            broken_lpl (pd.Series): The broken LPL
        """

        # The pivots located between the starting point and the first pivot after the broken LPL. The starting point is either
        # 1) The start of the pattern, which means we are forming the first region, or
        # 2) The start of the next section. The region_start_pdi variable determines this value.
        region_start_pdi = self.find_relative_pivot(broken_lpl.pdi, 1)

        return region_start_pdi

    def calc_h_o_zigzag(self, starting_point_pdi) -> None:
        """
        Calculates the higher order zigzag for the given starting point.

        This method sets the starting point of the higher order zigzag and adds it to the list of higher order indices. It then
        identifies the first broken LPL after the starting point and determines the trend type based on the type of the broken LPL.
        It then identifies the base of the swing (BOS) which is the pivot right after the broken LPL.

        The method then enters a loop where it checks for breaking sentiments (either PBOS_SHADOW, CHOCH_SHADOW, PBOS_CLOSE, CHOCH_CLOSE or NONE)
        and updates the latest PBOS and CHOCH thresholds accordingly if a shadow has broken a PBOS or CHOCH. If a PBOS_CLOSE or CHOCH_CLOSE sentiment
        is detected, the method identifies the extremum point and adds it to the higher order indices, and then resets the starting point for finding
        higher order pivots.

        Any PBOS_CLOSE events will trigger a segment creation. The segment is then added to the list of segments. A segment is a region within which
        the order blocks aren't invalidated. This means that the trades can be safely entered in each segment independently without worrying about
        OB updates.

        The loop continues until no more candles are found that break the PBOS or CHOCH even with a shadow.

        Args:
            starting_point_pdi (int): The starting point of the higher order zigzag.

        Returns:
            None
        """

        # Set the starting point of the HO zigzag and add it
        self.starting_pdi = starting_point_pdi
        self.h_o_indices.append(self.starting_pdi)

        # The first CHOCH is always the starting point, until it is updated when a BOS or a CHOCH is broken.
        latest_choch_pdi = self.starting_pdi
        latest_choch_threshold: float = self.zigzag_df[self.zigzag_df.pdi == self.starting_pdi].iloc[0].pivot_value
        if constants.logs_format == "time":
            self.log_message("Added starting point", self.convert_pdis_to_times(self.starting_pdi), v=1)
        else:
            self.log_message("Added starting point", self.starting_pdi, v=1)

        # The starting point of each pattern. This resets and changes whenever the pattern needs to be restarted. Unlike self.starting_pdi this DOES
        # change.
        pattern_start_pdi = self.starting_pdi

        latest_pbos_pdi = None
        latest_pbos_threshold = None

        # The loop which continues until the end of the pattern is reached.
        while True:
            # Spacing between each iteration
            if constants.logs_format == "time":
                self.log_message("", v=1)
            else:
                self.log_message("", v=1)

            # Find the first broken LPL after the starting point and the region starting point
            broken_lpl_output_set = self.detect_first_broken_lpl(pattern_start_pdi)

            # If no broken LPL can be found, just quit
            if broken_lpl_output_set is None:
                if constants.logs_format == "time":
                    self.log_message("Reached end of chart, no more broken LPL's.", v=1)
                else:
                    self.log_message("Reached end of chart, no more broken LPL's.", v=1)
                break
            else:
                broken_lpl = broken_lpl_output_set[0]
                lpl_breaking_pdi: int = broken_lpl_output_set[1]

            if constants.logs_format == "time":
                self.log_message("Starting pattern at", self.convert_pdis_to_times(pattern_start_pdi), v=3)
                self.log_message("Broken LPL is at", self.convert_pdis_to_times(broken_lpl.pdi), v=3)
            else:
                self.log_message("Starting pattern at", pattern_start_pdi, v=3)
                self.log_message("Broken LPL is at", broken_lpl.pdi, v=3)

            # If the LPL type is valley, it means the trend type is ascending
            trend_type = "ascending" if broken_lpl.pivot_type == "valley" else "descending"

            # The BOS is the pivot right after the broken LPL
            bos_pdi = int(self.__calc_region_start_pdi(broken_lpl))

            # When pattern resets, aka a new point is found OR when the pattern is initializing. Each time a restart is required in the next
            # iteration, latest_pbos_pdi is set to None.
            if latest_pbos_pdi is None:
                latest_pbos_pdi = bos_pdi
                latest_pbos_threshold = self.zigzag_df[self.zigzag_df.pdi == bos_pdi].iloc[0].pivot_value

                # Add the BOS to the HO indices
                self.h_o_indices.append(bos_pdi)
                if constants.logs_format == "time":
                    self.log_message("Added BOS", self.convert_pdis_to_times(bos_pdi), v=1)
                    self.log_message("HO indices", self.convert_pdis_to_times(self.h_o_indices), v=2)
                else:
                    self.log_message("Added BOS", bos_pdi, v=1)
                    self.log_message("HO indices", self.h_o_indices, v=2)

            if constants.logs_format == "time":
                self.log_message("CHOCH threshold is at", self.convert_pdis_to_times(latest_choch_pdi), "BOS threshold is at",
                                 self.convert_pdis_to_times(latest_pbos_pdi), v=2)
            else:
                self.log_message("CHOCH threshold is at", latest_choch_pdi, "BOS threshold is at", latest_pbos_pdi, v=2)

            # Add the first found PBOS to the list as that is needed to kickstart the h_o_zigzag
            self.pbos_indices.append(bos_pdi)

            # If the candle breaks the PBOS by its shadow, the most recent BOS threshold will be moved to that candle's high instead
            # If a candle breaks the PBOS with its close value, then the search halts
            # If the candle breaks the last CHOCH by its shadow, the CHOCH threshold will be moved to that candle's low
            # If a candle breaks the last CHOCH with its close, the direction inverts and the search halts
            # These sentiments are detected using the self.__detect_breaking_sentiment method.
            breaking_output = self.__detect_breaking_sentiment(latest_pbos_threshold, latest_pbos_pdi,
                                                               latest_choch_threshold, trend_type)
            breaking_pdi = breaking_output["pdi"]
            breaking_sentiment = breaking_output["sentiment"]

            # For brevity and simplicity, from this point on all the comments are made with the ascending pattern in mind. THe descending pattern is
            # exactly the same, just inverted.
            # If a PBOS has been broken by a shadow(And ONLY its shadow, not its close value. This is explicitly enforced in the sentiment detection
            # method, where CLOSE sentiments are given priority over SHADOW ), update the latest PBOS pdi and threshold (level). Note that since this
            # statement doesn't set latest_pbos_pdi to None, the pattern will not restart.
            if breaking_sentiment == "PBOS_SHADOW":
                if constants.logs_format == "time":
                    self.log_message("PBOS #", self.convert_pdis_to_times(latest_pbos_pdi), "broken by candle shadow at index",
                                     self.convert_pdis_to_times(breaking_pdi), v=2)
                else:
                    self.log_message("PBOS #", latest_pbos_pdi, "broken by candle shadow at index", breaking_pdi, v=2)

                latest_pbos_pdi = breaking_pdi
                latest_pbos_threshold = self.pair_df.iloc[breaking_pdi].high if trend_type == "ascending" else \
                    self.pair_df.iloc[breaking_pdi].low

            # If a candle breaks the CHOCH with its shadow (And ONLY its shadow, not its close value), update the latest CHOCH pdi and threshold
            elif breaking_sentiment == "CHOCH_SHADOW":
                if constants.logs_format == "time":
                    self.log_message("CHOCH #", self.convert_pdis_to_times(latest_choch_pdi), "broken by candle shadow at index",
                                     self.convert_pdis_to_times(breaking_pdi), v=2)
                else:
                    self.log_message("CHOCH #", latest_choch_pdi, "broken by candle shadow at index", breaking_pdi, v=2)

                latest_choch_pdi = breaking_pdi
                latest_choch_threshold = self.pair_df.iloc[breaking_pdi].low if trend_type == "ascending" else \
                    self.pair_df.iloc[breaking_pdi].high

            # If a candle CLOSES above the latest PBOS value, it means we have found an extremum, which would be the lowest low zigzag pivot between
            # the latest HO zigzag point (The initial BOS before being updated with shadows) and the candle which closed above it. After detecting
            # this extremum, we add it to HO Zigzag.
            elif breaking_sentiment == "PBOS_CLOSE":
                if constants.logs_format == "time":
                    self.log_message("Candle at index",
                                     self.convert_pdis_to_times(breaking_pdi), "broke the last PBOS #", self.convert_pdis_to_times(latest_pbos_pdi),
                                     "with its close price", v=2)
                    self.log_message("BOS #", self.convert_pdis_to_times(self.h_o_indices[-1]), "break at", self.convert_pdis_to_times(breaking_pdi),
                                     v=1)
                else:
                    self.log_message("Candle at index",
                                     breaking_pdi, "broke the last PBOS #", latest_pbos_pdi, "with its close price", v=2)
                    self.log_message("BOS #", self.h_o_indices[-1], "break at", breaking_pdi, v=1)

                # The extremum point is the point found using a "lowest low" of a "highest high" search between the last HO pivot and
                # the closing candle
                extremum_point_pivot_type = "valley" if trend_type == "ascending" else "peak"

                # extremum_point_pivots_of_type is a list of all the pivots of the right type for the extremum
                extremum_point_pivots_of_type = self.zigzag_df[
                    (self.zigzag_df.pdi >= self.h_o_indices[-1])
                    & (self.zigzag_df.pdi <= breaking_pdi)
                    & (self.zigzag_df.pivot_type == extremum_point_pivot_type)]

                # The extremum pivot is the lowest low / highest high in the region between the first PBOS and the closing candle
                if extremum_point_pivot_type == "peak":
                    extremum_pivot = extremum_point_pivots_of_type.loc[
                        extremum_point_pivots_of_type['pivot_value'].idxmax()]
                else:
                    extremum_pivot = extremum_point_pivots_of_type.loc[
                        extremum_point_pivots_of_type['pivot_value'].idxmin()]

                # Add the extremum point to the HO indices
                self.h_o_indices.append(int(extremum_pivot.pdi))
                extremum_type = "lowest low" if trend_type == "ascending" else "highest high"

                if constants.logs_format == "time":
                    self.log_message("Added extremum of type", extremum_type, "at", self.convert_pdis_to_times(extremum_pivot.pdi), v=1)
                else:
                    self.log_message("Added extremum of type", extremum_type, "at", extremum_pivot.pdi, v=1)

                # Now, we can restart finding HO pivots. Starting point is set to the last LPL of the same type BEFORE the BOS breaking candle.
                # Trend stays the same since no CHOCH has occurred.
                pivot_type = "valley" if trend_type == "ascending" else "peak"
                pivots_of_type_before_closing_candle = self.zigzag_df[(self.zigzag_df.pivot_type == pivot_type)
                                                                      & (self.zigzag_df.pdi <= breaking_pdi)]

                pattern_start_pdi = pivots_of_type_before_closing_candle.iloc[-1].pdi
                if constants.logs_format == "time":
                    self.log_message("Setting pattern start to", self.convert_pdis_to_times(pattern_start_pdi), v=1)
                else:
                    self.log_message("Setting pattern start to", pattern_start_pdi, v=1)

                # Essentially reset the algorithm
                latest_pbos_pdi = None

                # A segment is added to the list of segments here. Each segment starts at the pivot before the high that was just broken by a candle
                # closing above it. The segment ends at the PBOS_CLOSE event, at the candle that closed above the high. The -3 index is used because
                # there are two points after it: The high that was just broken, and the extremum that was added because the high was broken; therefore
                # we need the THIRD to last pivot.
                segment_to_add: Segment = Segment(start_pdi=self.h_o_indices[-3],
                                                  end_pdi=breaking_pdi - 1,
                                                  ob_leg_start_pdi=self.h_o_indices[-3],
                                                  ob_leg_end_pdi=self.h_o_indices[-2],
                                                  top_price=latest_pbos_threshold,
                                                  bottom_price=latest_choch_threshold,
                                                  ob_formation_start_pdi=lpl_breaking_pdi + 1,
                                                  broken_lpl_pdi=broken_lpl.pdi,
                                                  type=trend_type)
                self.segments.append(segment_to_add)

                # New lowest low is our CHOCH.
                latest_choch_pdi = self.h_o_indices[-1]
                latest_choch_threshold = self.zigzag_df[self.zigzag_df.pdi == latest_choch_pdi].iloc[0].pivot_value



            # If a CHOCH has happened, this means the pattern has inverted and should be restarted with the last LPL before the candle which closed
            # below the CHOCH.
            elif breaking_sentiment == "CHOCH_CLOSE":
                if constants.logs_format == "time":
                    self.log_message("Candle at index",
                                     self.convert_pdis_to_times(breaking_pdi), "broke the last CHOCH #", self.convert_pdis_to_times(latest_choch_pdi),
                                     "with its close price", v=2)
                    self.log_message("CHOCH #", self.convert_pdis_to_times(self.h_o_indices[-2]), "break at",
                                     self.convert_pdis_to_times(breaking_pdi), v=1)
                else:
                    self.log_message("Candle at index",
                                     breaking_pdi, "broke the last CHOCH #", latest_choch_pdi, "with its close price", v=2)
                    self.log_message("CHOCH #", self.h_o_indices[-2], "break at", breaking_pdi, v=1)

                trend_type = "ascending" if trend_type == "descending" else "descending"

                # Set the pattern start to the last inverse pivot BEFORE the closing candle
                pivot_type = "valley" if trend_type == "ascending" else "peak"
                pivots_of_type_before_closing_candle = self.zigzag_df[(self.zigzag_df.pivot_type == pivot_type)
                                                                      & (self.zigzag_df.pdi <= breaking_pdi)]

                pattern_start_pdi = pivots_of_type_before_closing_candle.iloc[-1].pdi
                if constants.logs_format == "time":
                    self.log_message("Setting pattern start to", self.convert_pdis_to_times(pattern_start_pdi), v=1)
                else:
                    self.log_message("Setting pattern start to", pattern_start_pdi, v=1)

                # A segment is added to the list of segments here. Each segment starts at the pivot before the low that was just broken by a candle
                # closing below it. The segment ends at the CHOCH_CLOSE event, at the candle that closed above the high.
                # we need the THIRD to last pivot. trend_type needs to be reverted because we are still working on the same positions from before
                # the CHOCH happened and in the same direction, just that the event is different...
                segment_to_add: Segment = Segment(start_pdi=self.h_o_indices[-2],
                                                  end_pdi=breaking_pdi,
                                                  ob_leg_start_pdi=self.h_o_indices[-2],
                                                  ob_leg_end_pdi=self.h_o_indices[-2],
                                                  top_price=latest_pbos_threshold,
                                                  bottom_price=latest_choch_threshold,
                                                  ob_formation_start_pdi=lpl_breaking_pdi + 1,
                                                  broken_lpl_pdi=broken_lpl.pdi,
                                                  type="ascending" if trend_type == "descending" else "descending", formation_method="choch")
                self.segments.append(segment_to_add)

                # Essentially reset the algorithm
                latest_choch_pdi = self.h_o_indices[-1]
                latest_choch_threshold = self.zigzag_df[self.zigzag_df.pdi == latest_choch_pdi].iloc[0].pivot_value

                latest_pbos_pdi = None

            # If no candles have broken the PBOS even with a shadow, break the loop
            else:
                if constants.logs_format == "time":
                    self.log_message("No more candles found. Breaking...", v=1)
                else:
                    self.log_message("No more candles found. Breaking...", v=1)
                break

        # return self.h_o_indices

    def convert_pdis_to_times(self, pdis: Union[int, list[int]]) -> Union[pd.Timestamp, list[pd.Timestamp]]:
        """
        Convert a list (or a single) of PDIs to their corresponding times using algo.pair_df.

        Args:
            pdis (list[int]): List of PDIs to convert.
            pair_df (pd.DataFrame): The pair_df DataFrame to use for the conversion.

        Returns:
            list[pd.Timestamp]: List of corresponding times.
        """

        if pdis is None:
            return None

        if not isinstance(pdis, list):
            pdis = [pdis]

        if len(pdis) == 0:
            return []

        # Map PDIs to their corresponding times
        times = [self.pair_df.iloc[pdi].time for pdi in pdis]

        # If it's a singular entry, return it as a single timestamp
        if len(times) == 1:
            return times[0]

        return list(times)


def find_last_htf_ho_pivot(htf_pair_df: pd.DataFrame,
                           ltf_start_time: pd.Timestamp,
                           backtrack_window: int = constants.starting_point_backtrack_window) -> tuple[
    pd.Timestamp, str]:
    """
    This function returns a starting point for the algorithm. It uses the algo object (kind of recursively) with a higher order pair_df and applies
    a higher order zigzag operator on it. The last point of the higher order zigzag before the original LTF data's starting point is set as the
    starting timestamp for LTF data, and the data is reshaped to account for it

    Args:
        htf_pair_df (pd.Dataframe): A dataframe containing the higher timeframe data
        ltf_start_time (pd.Timestamp): A timestamp of the beginning of the lower timeframe data
        backtrack_window (int): The size of the backtracking performed from the beginning of the original lower timeframe data.

    Returns:
        tuple: A tuple containing both the Timestamp of the last HO pivot before the LTF original starting point, and a str indicating if it's a low
               or a high
    """

    # The HTF HO zigzag calculation should start "backtrack_window" candles behind the stating timestamp of the LTF data
    htf_pair_df_candles_before_ltf_start: pd.DataFrame = htf_pair_df[htf_pair_df.time <= ltf_start_time]
    htf_starting_index: int = htf_pair_df_candles_before_ltf_start.iloc[-1].name - backtrack_window

    truncated_htf_pair_df: pd.DataFrame = htf_pair_df.iloc[htf_starting_index:].reset_index()

    # Instantiate the Algo object so the functions can be used.
    htf_algo = Algo(truncated_htf_pair_df, "HTF Data", allowed_verbosity=0)
    htf_algo.init_zigzag()
    first_zigzag_pivot_pdi: int = htf_algo.zigzag_df.iloc[0].pdi
    htf_algo.calc_h_o_zigzag(first_zigzag_pivot_pdi)
    htf_h_o_indices = htf_algo.h_o_indices

    # Convert the h_o_indices of the higher timeframe data to their respective timestamps
    timestamps = htf_algo.zigzag_df[htf_algo.zigzag_df.pdi.isin(htf_h_o_indices)]

    # Select the timestamps before the lower timeframe data's starting point
    timestamps = timestamps[timestamps.time >= ltf_start_time]

    # The last one along with its type
    last_timestamp = timestamps.iloc[0].time
    pivot_type = "low" if htf_algo.zigzag_df[htf_algo.zigzag_df.time == last_timestamp].iloc[
                              0].pivot_type == "valley" else "high"

    return last_timestamp, pivot_type


def create_filtered_pair_df_with_corrected_starting_point(htf_pair_df: pd.DataFrame,
                                                          initial_data_start_time: pd.Timestamp,
                                                          original_pair_df: pd.DataFrame,
                                                          timeframe: str, higher_timeframe: str) -> pd.DataFrame:
    """
    This function created a new pair_df using the starting timestamp found by find_last_htf_ho_pivot. It then determines whether it's a low or a high,
    and processes the data aggregated by the higher order timeframe to find the actual starting candle

    Args:
        htf_pair_df (pd.DataFrame): The higher timeframe DataFrame
        initial_data_start_time (pd.Timestamp): The start date of the uncorrected pair_df
        original_pair_df (pd.DataFrame): The complete, non-truncated version of pair_df which is used to filter the candles to create the
        final pair_df.
        timeframe (str): The original pair_df timeframe, aka the lower timeframe
        higher_timeframe (str): The higher timeframe of the pair_df, found using general_utils.find_higher_timeframe()

    Returns:
        pd.DataFrame: The filtered, corrected pair_df, ready for use in the algorithm
    """

    starting_point_output = find_last_htf_ho_pivot(htf_pair_df, initial_data_start_time)
    starting_timestamp, starting_pivot_type = starting_point_output

    # The PDI of the candle in the LTF data which corresponds to the exact time found by find_last_htf_ho_pivot. This will be used to create the
    # aggregated candles and find the lowest low/highest high candle depending on starting_pivot_type and filtering pair_df based on that.
    print(starting_timestamp)
    initial_starting_pdi = original_pair_df[original_pair_df.time == starting_timestamp].iloc[0].name

    # This parameter depends on the conversion rate between the LTF and HTF timeframes
    n_aggregated_candles = int(constants.timeframe_minutes[higher_timeframe] / constants.timeframe_minutes[timeframe])

    # The n_aggregated_candles-long window to find the lowest low/highest high.
    pair_df_window = original_pair_df.iloc[initial_starting_pdi + 1:initial_starting_pdi + n_aggregated_candles + 1]

    if starting_pivot_type == "low":
        starting_extremum_candle_pdi = pair_df_window.loc[pair_df_window.low.idxmin()].name
    else:
        starting_extremum_candle_pdi = pair_df_window.loc[pair_df_window.high.idxmax()].name

    pair_df = original_pair_df.iloc[starting_extremum_candle_pdi:].reset_index(drop=True)

    return pair_df


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

    def check_box_entries(self, pair_df: pd.DataFrame, upper_search_bound_pdi: int) -> None:
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

            if exit_index is not None:
                self.price_exit_index = exit_index
                # If an exit is found, check for a reentry into the box after the exit
                # Should use check_window.loc[exit_index:] instead of iloc because the current df is a subset of pair_df, and the indices are
                # all messed up
                reentry_check_window = check_window.loc[exit_index + 1:]
                reentry_index = reentry_check_window.loc[reentry_check_window['low'] < self.top].first_valid_index()
                # If a reentry is found, mark the box as invalid
                if reentry_index is not None:
                    self.price_reentry_indices.append(reentry_index)
                    self.is_valid = False

        else:  # If the box is of type "short"
            # Find the first candle where a candle opens inside the OB and closes below it
            exit_index = check_window[(check_window['close'] < self.bottom) & (check_window['open'] >= self.bottom)].first_valid_index()

            if exit_index is not None:
                self.price_exit_index = exit_index
                # If an exit is found, check for a reentry into the box after the exit
                # Should use check_window.loc[exit_index:] instead of iloc because the current df is a subset of pair_df, and the indices are
                # all messed up
                reentry_check_window = check_window.loc[exit_index + 1:]
                reentry_index = reentry_check_window.loc[reentry_check_window['high'] > self.bottom].first_valid_index()
                # If a reentry is found, mark the box as invalid
                if reentry_index is not None:
                    self.price_reentry_indices.append(reentry_index)
                    # self.is_valid = False

        self.form_condition_check_window(pair_df)

    def form_condition_check_window(self, pair_df: pd.DataFrame) -> None:
        """
        Method to form the condition check window for the box. This check window is used to check the order block confirmation conditions
        (FVG and price breaking, refer to the check_x_condition methods). The check_box_entries method should be called before this method.

        Args
            pair_df (pd.DataFrame): The DataFrame containing the price data.
        """

        if len(self.price_reentry_indices) > 0:
            self.condition_check_window = pair_df.iloc[self.start_index:self.price_reentry_indices[0]]
        else:
            self.condition_check_window = pair_df.iloc[self.start_index:]

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


class Segment:
    """
    A segment is a series of candles during which the order blocks specified in Segment.ob_list do not change, so it would be safe to check for
    entry to these order blocks WITHIN this segment. After the expiration candle of the segment, indicated by Segment.end_pdi, entry to the order
    blocks isn't permitted, and we move on to the next segment.
    """

    def __init__(self, start_pdi: int,
                 end_pdi: int,
                 ob_leg_start_pdi: int,
                 ob_leg_end_pdi: int,
                 top_price: float,
                 bottom_price: float,
                 ob_formation_start_pdi: int,
                 broken_lpl_pdi: int,
                 type: str,
                 formation_method: str = "bos"):
        self.end_pdi = end_pdi
        self.start_pdi = start_pdi
        self.ob_leg_start_pdi = ob_leg_start_pdi
        self.ob_leg_end_pdi = ob_leg_end_pdi
        self.top_price = top_price
        self.bottom_price = bottom_price
        self.ob_formation_start_pdi = ob_formation_start_pdi
        self.broken_lpl_pdi = broken_lpl_pdi
        self.type = type
        self.formation_method = formation_method

        if constants.logs_format != "time":
            self.id = f"SEG/{self.formation_method}/{self.start_pdi}"

        self.ob_list: list[OrderBlock] = []
        self.pair_df: pd.DataFrame = pd.DataFrame()

    def __repr__(self):
        return f"{self.type.capitalize()} segment starting at {self.start_pdi} ending at {self.end_pdi} OB formation at {self.ob_formation_start_pdi}"

    def filter_candlestick_range(self, algo: Algo):
        """
        This method defines the range of pair_df which is used to find box entries. This is useful for checking order block entries. This section is
        defined as the candles between the OB formation start and the end of the segment, inclusive. The inclusivity is important because in the code
        a segment's bounds are defined as such.
        """
        self.pair_df = algo.pair_df.iloc[self.ob_formation_start_pdi:self.end_pdi + 1]

        if constants.logs_format == "time":
            self.id = f"SEG/{self.formation_method}/{algo.pair_df.loc[self.start_pdi].time}"

    def find_order_blocks(self, algo: Algo):
        """
        This method identifies the order blocks specific to the segment by taking the entire Algo object as an input, as many of its properties and
        methods are useful here, and it would be redundant to pass around multiple inputs and methods. This method populates the Segment.ob_list
        object with a list of order blocks that are only valid within this segment.

        Args:
            algo: The Algo object
        """
        positions_logger.debug(f"Finding order blocks for segment {self.id}")

        # For testing and safety purposes, the ob_list property is reset.
        self.ob_list = []

        # base_candle_type is the type of the pivot that is used to filter the zigzag_df dataframe for the correct pivot type. In ascending segments
        # (patterns) the type is valley, and in descending segments it's peak.
        base_pivot_type = "valley" if self.type == "ascending" else "peak"

        # This variable is used to keep track of how many valid order blocks have been found. It is then assigned to each order block within the
        # segment, so it's ranking in the segment is recorded.
        valid_ob_counter = 0

        # Filter pivots of the correct type (valley for ascending, peak for descending) and pivots that are within the first leg. Also omit the pivots
        # that have a higher PDI than the broken LPL PDI, meaning the boxes that form above the broken LPL in ascending and below the LPL in
        # descending

        for pivot in algo.zigzag_df[(algo.zigzag_df.pivot_type == base_pivot_type) &
                                    (self.ob_leg_start_pdi <= algo.zigzag_df.pdi) &
                                    (algo.zigzag_df.pdi < self.broken_lpl_pdi)].itertuples():

            if constants.logs_format == "time":
                positions_logger.debug(f"\tFinding OBs for lower order leg starting at {algo.convert_pdis_to_times(pivot.pdi)}")
            else:
                positions_logger.debug(f"\tFinding OBs for lower order leg starting at {pivot.pdi}")

            # This try-except block is used to determine the window that is used for finding replacement order blocks in the chart. Currently, the
            # window spans from the very first base candle (the pivot found using the outer loop) to the lower-order pivot immediately after it.
            # The except clause catches the error in case we reach the end of the chart and no more next pivots exist, in which case the end of the
            # search window is set to the last candle of the whole dataset.
            try:
                next_pivot_pdi = algo.find_relative_pivot(pivot.pdi, 1)
                replacement_ob_threshold_pdi = next_pivot_pdi
            except IndexError:
                replacement_ob_threshold_pdi = algo.pair_df.last_valid_index()

            if constants.logs_format == "time":
                positions_logger.debug(f"\tReplacement OB search threshold set up to {algo.convert_pdis_to_times(replacement_ob_threshold_pdi)}")
            else:
                positions_logger.debug(f"\tReplacement OB search threshold set up to {replacement_ob_threshold_pdi}")

            # times_moved indicates the times the algorithm had to move the base candle to find a replacement order block.
            times_moved = 0

            # The stoploss is set at the pivot value of the INITIAL box that was found, since that's the box which has the liquidity. This value is
            # passed to the OB instantiation line as the stoploss value, which in turn goes to the Position attribute within it.
            initial_pivot_candle_liquidity = pivot.pivot_value

            for base_candle_pdi in range(pivot.pdi, replacement_ob_threshold_pdi):
                base_candle = algo.pair_df.iloc[base_candle_pdi]
                ob = OrderBlock(base_candle=base_candle,
                                icl=initial_pivot_candle_liquidity,
                                ob_type="long" if base_pivot_type == "valley" else "short")

                if constants.logs_format == "time":
                    positions_logger.debug(f"\t\tInvestigating base candle at {algo.convert_pdis_to_times(base_candle_pdi)}")
                else:
                    positions_logger.debug(f"\t\tInvestigating base candle at {base_candle_pdi}")

                # the check_box_entries method finds any entry point to each box. These entry points can later be used and we can check if the entries
                # are within the respective segments. This method also sets the condition check window at the end, which is used to check if the boxes
                # satisfy the confirmation conditions. The search upper bound is the last candle of the segment.
                ob.check_box_entries(algo.pair_df, self.end_pdi)

                # The reentry window dataframe is used to check whether the price returned to the box in the span between the exit candle and the LPL
                # breaking candle. This is checked using the check_reentry_condition() method of the OrderBlock object. The reentry dataframe is
                # passed as an argument to the method.
                if ob.price_exit_index is not None:
                    reentry_check_window: pd.DataFrame = algo.pair_df.iloc[ob.price_exit_index + 1:self.ob_formation_start_pdi]

                    # Log the exit candle location
                    if constants.logs_format == "time":
                        positions_logger.debug(
                            f"\t\t\tExit candle found at {algo.convert_pdis_to_times(ob.price_exit_index)}")
                    else:
                        positions_logger.debug(f"\t\t\tExit candle found at {ob.price_exit_index}")

                    if constants.logs_format == "time":
                        positions_logger.debug(
                            f"\t\t\tReentry check window set up from {algo.convert_pdis_to_times(ob.price_exit_index + 1)} to {algo.convert_pdis_to_times(self.ob_formation_start_pdi - 1)}")
                    else:
                        positions_logger.debug(
                            f"\t\t\tReentry check window set up from {ob.price_exit_index + 1} to {self.ob_formation_start_pdi - 1}")

                # This else statement is implemented to account for boxes which don't have an exit candle which opens inside and closes outside of
                # them, automatically making them invalid and prompting considering another replacement.
                else:
                    positions_logger.debug("\t\t\tNo exit candle found. OB is invalid, looking for a replacement further in time.")
                    continue

                # This check ensures that the order block being processed is totally valid to be used AFTER the formation of the pattern, that means
                # that the order block has either A) had no reentry at all or B) has had its reentry after the formation of the pattern.
                # ob_is_valid_in_formation_region = len(ob.price_reentry_indices) == 0 or ob.price_reentry_indices[0] > self.ob_formation_start_pdi
                ob.check_reentry_condition(reentry_check_window)
                ob.check_fvg_condition()
                ob.check_stop_break_condition()

                positions_logger.debug(f"\t\t\tReentry check status: {ob.has_reentry_condition}")
                positions_logger.debug(f"\t\t\tFVG check status: {ob.has_fvg_condition}")
                positions_logger.debug(f"\t\t\tStop break check status: {ob.has_stop_break_condition}")
                if ob.has_reentry_condition and ob.has_fvg_condition and ob.has_stop_break_condition:
                    positions_logger.debug(f"\t\t\tAll checks passed, adding OB with ID {ob.id}")
                    valid_ob_counter += 1

                    ob.ranking_within_segment = valid_ob_counter

                    ob.times_moved = times_moved
                    ob.has_been_replaced = False
                    self.ob_list.append(ob)
                    break

                else:
                    positions_logger.debug("\t\t\tOne or more checks didn't pass, moving to next candle...")
                    ob.has_been_replaced = True

                times_moved += 1

        positions_logger.debug(f"End of finding order blocks for segment {self.id}")
        positions_logger.debug("")


class Position:
    def __init__(self, parent_ob: OrderBlock):

        self.parent_ob = parent_ob
        self.entry_price = parent_ob.top if parent_ob.type == "long" else parent_ob.bottom

        # Calculation of stoploss is done using the distance from the entry of the box to the initial candle that was checked for OB, before being
        # potentially replaced. This distance is denoted as EDICL, entry distance from initial candle liquidity.
        self.edicl = abs(parent_ob.icl - self.entry_price)

        self.type = parent_ob.type

        self.status: str = "ACTIVE"
        self.entry_pdi = None
        self.qty: float = 0
        self.highest_target: int = 0
        self.target_hit_pdis: list[int] = []
        self.exit_pdi = None
        self.portioned_qty = []
        self.net_profit = None

        self.target_list = []
        self.stoploss = None
        # Set up the targt list nd stoploss using a function which operates on the "self" object and directly manipulates the instance.
        setup.default_357(self)

    def find_entry_within_segment(self, segment: Segment) -> Union[int, None]:
        """
        This method analyzes the candles within the segment's entry region to see if any candle enters the position.

        Args:
            segment (Segment): The segment with the filtered candles ready to be checked for entry

        Returns:
            Union[int, None]: The index of the candle entering the position, or None if no candle enters the position
        """
        if self.type == "long":
            entering_candles: pd.DataFrame = segment.pair_df[segment.pair_df.low <= self.entry_price]
        else:
            entering_candles: pd.DataFrame = segment.pair_df[segment.pair_df.high >= self.entry_price]

        if len(entering_candles) > 0:
            return entering_candles.first_valid_index()
        else:
            return None

    def enter(self, entry_pdi: int):
        """
        Method to enter the OB. This method sets the current OB status to "ENTERED", and registers the entry PDI, entry price, and quantity of the
        entry.

        Args:
            entry_pdi (int): The PDI at which the entry is made
        """

        self.entry_pdi = entry_pdi
        self.qty = constants.used_capital / self.entry_price
        self.status = "ENTERED"

    def register_target(self, target_id: int, target_registry_pdi: int):
        """
        This method registers a new highest target on the position to later use in calculating the PNL. If the target being registered is the highest
        target, it also triggers an exit command.

        Args:
            target_id (int): The ID of the target to register. Must be higher than 0 since the default value is zero.
            target_registry_pdi (int): The PDI of the candle registering the target(s)
        """

        # First, for safety, check if the target being registered is actually higher than the highest registered target
        if target_id > self.highest_target:
            # Register all the non-hit targets with the PDI of the candle hitting them
            self.target_hit_pdis.extend([target_registry_pdi] * (target_id - self.highest_target))

            self.highest_target = target_id

    def register_stoploss(self):
        """
        This method triggers a stoploss registration on the position. The exit_code STOPLOSS is then used to call the exit function and calculate the
        PNL
        """
        self.exit(exit_code="STOPLOSS")

    def exit(self, exit_code: Literal["STOPLOSS", "FULL_TARGET"], exit_pdi: int):
        """
        This method exits an entered order block with an exit code. If the exit code is "STOPLOSS" that means the position is exiting due to hitting
        the stoploss level. Otherwise, if the exit code is "FULL_TARGET" that means the last target has been hit and therefore the maximum possible
        profit should be registered. If a "STOPLOSS" event happens, the profit is calculated using the highest registered target, accounting for
        losses from the stoploss and gains from the targets separately. The net profit is then registered into the Position.net_profit property.

        Args:
            exit_code (str): How the position has been exit.
            exit_pdi (int): At which candle the exit happens
        """

        self.exit_pdi = exit_pdi

        # Even distribution of quantities
        self.portioned_qty = [self.qty / len(self.target_list) for target in self.target_list]

        n_targets = len(self.target_list)
        # If the position is exiting due to hitting a stoploss
        if exit_code == "STOPLOSS":

            # If we do have any registered targets, set the highest registered target as the final status
            if self.highest_target > 0:
                self.status = f"TARGET_{self.highest_target}"

            # Otherwise, just report a STOPLOSS
            else:
                self.status = "STOPLOSS"

            # If the position is long, this means that we have one loss: a loss from purchasing the asset at entry, and we have two gains: a loss
            # from selling the remainder of the asset at stoploss and another for selling each portioned quantity at each target hit.
            if self.type == "long":
                loss_from_entry = self.entry_price * self.qty
                gain_from_stop = sum(self.portioned_qty[self.highest_target:]) * self.stoploss
                gain_from_targets = sum([self.portioned_qty[i] * self.target_list[i] for i in range(self.highest_target)])

                total_position_gain = gain_from_stop + gain_from_targets
                total_position_loss = loss_from_entry

            # If the position is short, this means that we have one gain: a gain from selling the asset at entry, and we have two losses: a loss from
            # buying the remainder of the asset at stoploss and another for buying each portioned quantity at each target hit.
            else:
                gain_from_entry = self.entry_price * self.qty
                loss_from_stop = sum(self.portioned_qty[self.highest_target:]) * self.stoploss
                loss_from_targets = sum([self.portioned_qty[i] * self.target_list[i] for i in range(self.highest_target)])

                total_position_gain = gain_from_entry
                total_position_loss = loss_from_stop + loss_from_targets

        # If a full target has been hit, report it as such
        elif exit_code == "FULL_TARGET":
            self.status = f"FULL_TARGET_{self.highest_target}"

            # If the position has achieved full targets, we have the same codes for calculating net profit, only with the omission of stoploss
            # loss/gains. All the target calculations will also use the entire target_list property instead of the spliced version
            if self.type == "long":
                total_position_loss = self.entry_price * self.qty
                total_position_gain = sum([qty_target[0] * qty_target[1] for qty_target in zip(self.portioned_qty, self.target_list)])

            else:
                total_position_gain = self.entry_price * self.qty
                total_position_loss = sum([qty_target[0] * qty_target[1] for qty_target in zip(self.portioned_qty, self.target_list)])

        self.net_profit = total_position_gain - total_position_loss

    def does_candle_stop(self, candle):
        """
        This method checks if the candle stops the position. This is done by checking if the candle's low is lower than the stoploss in the case of
        long positions, and if the candle's high is higher than the stoploss in the case of short positions.

        Args:
            candle (pd.Series): The candle to check for stopping

        Returns:
            bool: True if the candle stops the position, False otherwise
        """

        if self.type == "long":
            return candle.low <= self.stoploss
        else:
            return candle.high >= self.stoploss

    def detect_candle_sentiment(self, candle: pd.Series) -> tuple[str, Union[int, None]]:
        """
        This method checks which target (or stoploss) the candle argument breaks. The method is used to determine if the position should be exited.
        This method uses the candle's color to determine which of the stoploss or targets were hit first.

        Args:
            candle (pd.Series): The candle to check for target/stoploss

        Returns:
            tuple: A tuple containing a sentiment ("TARGET" , "FULL_TARGET", "STOPLOSS" or "NONE") and an int, for the case where the candle registers a
            target. If a candle registers a stoploss, the int is 0.
        """

        def last_element_bigger_than(targets: list[float], price: float):
            for i in reversed(range(len(targets))):
                if targets[i] >= price:
                    return i + 1
            return 0

        def last_element_smaller_than(targets: list[float], price: float):
            for i in reversed(range(len(targets))):
                if targets[i] <= price:
                    return i + 1
            return 0

        # Long order blocks
        if self.type == "long":
            highest_target = last_element_smaller_than(self.target_list, candle.high)
        # Short order blocks
        else:
            highest_target = last_element_bigger_than(self.target_list, candle.low)

        # If the candle is green, it means the price is going up, and the bottom of the box should be checked first
        if candle.close > candle.open:
            if self.does_candle_stop(candle):
                return "STOPLOSS", None

            if highest_target > self.highest_target:
                if highest_target < len(self.target_list):
                    return "TARGET", highest_target

                elif highest_target == len(self.target_list):
                    return "FULL_TARGET", None

        # If the candle is red, it means the price is going down, and the top of the box should be checked first
        else:
            if highest_target > self.highest_target:
                if highest_target < len(self.target_list):
                    return "TARGET", highest_target

                elif highest_target == len(self.target_list):
                    return "FULL_TARGET", None

            if self.does_candle_stop(candle):
                return "STOPLOSS", None

        return "NONE", None
