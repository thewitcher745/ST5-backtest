from logging import Logger
from typing import Optional
import pandas as pd

import constants
from algo_code.datatypes import *
from algo_code.segment import Segment
from utils.logger import LoggerSingleton

# noinspection PyTypeChecker
ho_zigzag_logger: Logger = None


class Algo:
    def __init__(self, pair_df: pd.DataFrame,
                 symbol: str,
                 timeframe: str = "15m"):
        global ho_zigzag_logger

        if ho_zigzag_logger is None:
            ho_zigzag_logger = LoggerSingleton.get_logger("ho_zigzag")

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
        ho_zigzag_logger.debug(f"Trend type is {trend_type}")

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
                # If a breaking event has occurred, we need to find the actual CANDLE that broke the LPL, since it might have happened before the
                # PIVOT that broke the LPL, since zigzag pivots are a much more aggregated type of data compared to the candles and almost always
                # the actual candle that breaks the LPL is one of the candles before the pivot that was just found.

                # The candle search range starts at the pivot before the LPL-breaking pivot (which is typically a higher order pivot) PDI and the
                # breaking pivot PDI.
                pivot_before_breaking_pivot: int = self.find_relative_pivot(row.pdi, -1)
                breaking_candle_search_window: pd.DataFrame = self.pair_df.loc[pivot_before_breaking_pivot + 1:row.pdi + 1]

                # If the trend is ascending, it means the search window should be checked for the first candle that breaks the LPL by having a lower
                # low than the breaking_value.
                if trend_type == "ascending":
                    lpl_breaking_candles = breaking_candle_search_window[breaking_candle_search_window.low < breaking_value]

                # If the trend is descending, the breaking candle must have a higher high than the breaking value.
                else:
                    lpl_breaking_candles = breaking_candle_search_window[breaking_candle_search_window.high > breaking_value]

                breaking_candle_pdi = lpl_breaking_candles.first_valid_index()

                # If the search window for the breaking candle is empty, return the pivot as the breaking candle
                if breaking_candle_pdi is None:
                    breaking_candle_pdi = row.pdi

                if constants.logs_format == "time":
                    ho_zigzag_logger.debug(
                        f"LPL #{self.convert_pdis_to_times(breaking_pdi)} broken at {self.convert_pdis_to_times(breaking_candle_pdi)}")
                else:
                    ho_zigzag_logger.debug(f"LPL #{breaking_pdi} broken at {breaking_candle_pdi}")

                return self.zigzag_df[self.zigzag_df.pdi == breaking_pdi].iloc[0], breaking_candle_pdi

            # Extension
            if extension_condition:
                # If a higher high is found, extend and update the pattern

                prev_pivot_pdi = self.find_relative_pivot(row.pdi, -1)
                prev_pivot = self.zigzag_df[self.zigzag_df.pdi == prev_pivot_pdi].iloc[0]

                if constants.logs_format == "time":
                    ho_zigzag_logger.debug(f"Changing breaking_pdi to {self.convert_pdis_to_times(prev_pivot.pdi)}")
                else:
                    ho_zigzag_logger.debug(f"Changing breaking_pdi to {prev_pivot.pdi}")
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
            ho_zigzag_logger.debug(f"Added starting point {self.convert_pdis_to_times(self.starting_pdi)}")
        else:
            ho_zigzag_logger.debug(f"Added starting point {self.starting_pdi}")

        # The starting point of each pattern. This resets and changes whenever the pattern needs to be restarted. Unlike self.starting_pdi this DOES
        # change.
        pattern_start_pdi = self.starting_pdi

        latest_pbos_pdi = None
        latest_pbos_threshold = None

        # The loop which continues until the end of the pattern is reached.
        while True:
            # Spacing between each iteration
            ho_zigzag_logger.debug("")

            # Find the first broken LPL after the starting point and the region starting point
            broken_lpl_output_set = self.detect_first_broken_lpl(pattern_start_pdi)

            # If no broken LPL can be found, just quit
            if broken_lpl_output_set is None:
                if constants.logs_format == "time":
                    ho_zigzag_logger.debug("Reached end of chart, no more broken LPL's.")
                else:
                    ho_zigzag_logger.debug("Reached end of chart, no more broken LPL's.")
                break

            else:
                broken_lpl = broken_lpl_output_set[0]
                lpl_breaking_pdi: int = broken_lpl_output_set[1]

            if constants.logs_format == "time":
                ho_zigzag_logger.debug(f"Starting pattern at {self.convert_pdis_to_times(pattern_start_pdi)}")
                ho_zigzag_logger.debug(f"Broken LPL is at {self.convert_pdis_to_times(broken_lpl.pdi)}")
            else:
                ho_zigzag_logger.debug(f"Starting pattern at {pattern_start_pdi}")
                ho_zigzag_logger.debug(f"Broken LPL is at {broken_lpl.pdi}")

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
                    ho_zigzag_logger.debug(f"Added BOS {self.convert_pdis_to_times(bos_pdi)}")
                    ho_zigzag_logger.debug(f"HO indices {self.convert_pdis_to_times(self.h_o_indices)}")
                else:
                    ho_zigzag_logger.debug(f"Added BOS {bos_pdi}")
                    ho_zigzag_logger.debug(f"HO indices {self.h_o_indices}")

            if constants.logs_format == "time":
                ho_zigzag_logger.debug(
                    f"CHOCH threshold is at {self.convert_pdis_to_times(latest_choch_pdi)} BOS threshold is at "
                    f"{self.convert_pdis_to_times(latest_pbos_pdi)}")
            else:
                ho_zigzag_logger.debug(f"CHOCH threshold is at {latest_choch_pdi} BOS threshold is at {latest_pbos_pdi}")

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
                    ho_zigzag_logger.debug(
                        f"PBOS # {self.convert_pdis_to_times(latest_pbos_pdi)} broken by candle shadow at index"
                        f" {self.convert_pdis_to_times(breaking_pdi)}")
                else:
                    ho_zigzag_logger.debug(f"PBOS # {latest_pbos_pdi} broken by candle shadow at index {breaking_pdi}")

                latest_pbos_pdi = breaking_pdi
                latest_pbos_threshold = self.pair_df.iloc[breaking_pdi].high if trend_type == "ascending" else \
                    self.pair_df.iloc[breaking_pdi].low

            # If a candle breaks the CHOCH with its shadow (And ONLY its shadow, not its close value), update the latest CHOCH pdi and threshold
            elif breaking_sentiment == "CHOCH_SHADOW":
                if constants.logs_format == "time":
                    ho_zigzag_logger.debug(
                        f"CHOCH # {self.convert_pdis_to_times(latest_choch_pdi)} broken by candle shadow at index"
                        f" {self.convert_pdis_to_times(breaking_pdi)}")
                else:
                    ho_zigzag_logger.debug(f"CHOCH # {latest_choch_pdi} broken by candle shadow at index {breaking_pdi}")

                latest_choch_pdi = breaking_pdi
                latest_choch_threshold = self.pair_df.iloc[breaking_pdi].low if trend_type == "ascending" else \
                    self.pair_df.iloc[breaking_pdi].high

            # If a candle CLOSES above the latest PBOS value, it means we have found an extremum, which would be the lowest low zigzag pivot between
            # the latest HO zigzag point (The initial BOS before being updated with shadows) and the candle which closed above it. After detecting
            # this extremum, we add it to HO Zigzag.
            elif breaking_sentiment == "PBOS_CLOSE":
                if constants.logs_format == "time":
                    ho_zigzag_logger.debug(
                        f"Candle at index {self.convert_pdis_to_times(breaking_pdi)} broke the last PBOS # "
                        f"{self.convert_pdis_to_times(latest_pbos_pdi)} with its close price")
                    ho_zigzag_logger.debug(
                        f"BOS # {self.convert_pdis_to_times(self.h_o_indices[-1])} break at {self.convert_pdis_to_times(breaking_pdi)}")
                else:
                    ho_zigzag_logger.debug(f"Candle at index {breaking_pdi} broke the last PBOS # {latest_pbos_pdi} with its close price")
                    ho_zigzag_logger.debug(f"BOS # {self.h_o_indices[-1]} break at {breaking_pdi}")

                # The extremum point is the point found using a "lowest low" of a "highest high" search between the last HO pivot and
                # the closing candle
                extremum_point_pivot_type = "valley" if trend_type == "ascending" else "peak"

                # extremum_point_pivots_of_type is a list of all the pivots of the right type for the extremum
                extremum_point_pivots_of_type = self.zigzag_df[
                    (self.zigzag_df.pdi >= self.h_o_indices[-1])
                    & (self.zigzag_df.pdi <= breaking_pdi)
                    & (self.zigzag_df.pivot_type == extremum_point_pivot_type)]

                # The extremum pivot is the lowest low / the highest high in the region between the first PBOS and the closing candle
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
                    ho_zigzag_logger.debug(f"Added extremum of type {extremum_type} at {self.convert_pdis_to_times(extremum_pivot.pdi)}")
                else:
                    ho_zigzag_logger.debug(f"Added extremum of type {extremum_type} at {extremum_pivot.pdi}")

                # Now, we can restart finding HO pivots. Starting point is set to the last LPL of the same type BEFORE the BOS breaking candle.
                # Trend stays the same since no CHOCH has occurred.
                pivot_type = "valley" if trend_type == "ascending" else "peak"
                pivots_of_type_before_closing_candle = self.zigzag_df[(self.zigzag_df.pivot_type == pivot_type)
                                                                      & (self.zigzag_df.pdi <= breaking_pdi)]

                pattern_start_pdi = pivots_of_type_before_closing_candle.iloc[-1].pdi
                if constants.logs_format == "time":
                    ho_zigzag_logger.debug(f"Setting pattern start to {self.convert_pdis_to_times(pattern_start_pdi)}")
                else:
                    ho_zigzag_logger.debug(f"Setting pattern start to {pattern_start_pdi}")

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
                    ho_zigzag_logger.debug(
                        f"Candle at index {self.convert_pdis_to_times(breaking_pdi)} broke the last CHOCH # "
                        f"{self.convert_pdis_to_times(latest_choch_pdi)} with its close price")
                    ho_zigzag_logger.debug(
                        f"CHOCH # {self.convert_pdis_to_times(self.h_o_indices[-2])} break at {self.convert_pdis_to_times(breaking_pdi)}")
                else:
                    ho_zigzag_logger.debug(f"Candle at index {breaking_pdi} broke the last CHOCH # {latest_choch_pdi} with its close price")
                    ho_zigzag_logger.debug(f"CHOCH # {self.h_o_indices[-2]} break at {breaking_pdi}")

                trend_type = "ascending" if trend_type == "descending" else "descending"

                # Set the pattern start to the last inverse pivot BEFORE the closing candle
                pivot_type = "valley" if trend_type == "ascending" else "peak"
                pivots_of_type_before_closing_candle = self.zigzag_df[(self.zigzag_df.pivot_type == pivot_type)
                                                                      & (self.zigzag_df.pdi <= breaking_pdi)]

                pattern_start_pdi = pivots_of_type_before_closing_candle.iloc[-1].pdi
                if constants.logs_format == "time":
                    ho_zigzag_logger.debug(f"Setting pattern start to {self.convert_pdis_to_times(pattern_start_pdi)}")
                else:
                    ho_zigzag_logger.debug(f"Setting pattern start to {pattern_start_pdi}")

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
                ho_zigzag_logger.debug("No more candles found. Breaking...")
                break

        # return self.h_o_indices

    def convert_pdis_to_times(self, pdis: Union[int, list[int]]) -> Union[pd.Timestamp, list[pd.Timestamp], None]:
        """
        Convert a list (or a single) of PDIs to their corresponding times using algo_code.pair_df.

        Args:
            pdis (list[int]): List of PDIs to convert.

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
                           backtrack_window: int = constants.starting_point_backtrack_window) -> tuple[pd.Timestamp, str]:
    """
    This function returns a starting point for the algorithm. It uses the algo_code object (kind of recursively) with a higher order pair_df and applies
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
    htf_algo = Algo(truncated_htf_pair_df, "HTF Data")
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
    # aggregated candles and find the lowest low/the highest high candle depending on starting_pivot_type and filtering pair_df based on that.

    initial_starting_pdi = original_pair_df[original_pair_df.time == starting_timestamp].iloc[0].name

    # This parameter depends on the conversion rate between the LTF and HTF timeframes
    n_aggregated_candles = int(constants.timeframe_minutes[higher_timeframe] / constants.timeframe_minutes[timeframe])

    # The n_aggregated_candles-long window to find the lowest low/the highest high.
    pair_df_window = original_pair_df.iloc[initial_starting_pdi + 1:initial_starting_pdi + n_aggregated_candles + 1]

    if starting_pivot_type == "low":
        starting_extremum_candle_pdi = pair_df_window.loc[pair_df_window.low.idxmin()].name
    else:
        starting_extremum_candle_pdi = pair_df_window.loc[pair_df_window.high.idxmax()].name

    pair_df = original_pair_df.iloc[starting_extremum_candle_pdi:].reset_index(drop=True)

    return pair_df
