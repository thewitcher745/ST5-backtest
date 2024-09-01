from typing import Optional
import pandas as pd
from intervals import Interval, IllegalArgument, AbstractInterval

from datatypes import *
from general_utils import log_message


class Algo:
    def __init__(self, pair_df, symbol, pattern_limit=None):
        self.pair_df: pd.DataFrame = pair_df
        self.symbol: str = symbol
        self.zigzag_df: Optional[pd.DataFrame] = None

        # This variable indicates whether only the first pattern is generated, for testing purposes
        self.pattern_limit = pattern_limit

        # pbos_indices and choch_indices is a list which stores the PBOS and CHOCH's being moved due to shadows breaking the most recent lows/highs
        self.pbos_indices = []
        self.choch_indices = []

        # h_o_indices indicates the indices of the peaks and valleys in the higher order zigzag
        self.h_o_indices = []

        # starting_pdi is the starting point of the entire pattern, calculated using __init_pattern_start_pdi. This method is
        # executed in the calc_h_o_zigzag method.
        self.starting_pdi = 0

        # A list of FVG's in the entire pair_df dataframe, which will get populated by the identify_fvgs method with the
        # FVG object from datatypes.py
        self.fvg_list = []

    def init_zigzag(self) -> None:
        """
            Method to identify turning points in a candlestick chart.
            It compares each candle to its previous pivot to determine if it's a new pivot point.
            This implementation is less optimized than the deprecated version, as it doesn't use
            vectorized operations, but it is what it is

            Returns:
            pd.DataFrame: A DataFrame containing the identified turning points.
            """

        # Find the first candle that has a higher high or a lower low than its previous candle
        # and set it as the first pivot. Also set the type of the pivot (peak or valley)
        last_pivot_candle_series = \
            self.pair_df[(self.pair_df['high'] > self.pair_df['high'].shift(1)) | (self.pair_df['low'] < self.pair_df['low'].shift(1))].iloc[0]
        last_pivot_type: str = 'valley'
        if last_pivot_candle_series.high > self.pair_df.iloc[last_pivot_candle_series.name - 1].high:
            last_pivot_type = 'peak'

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
            if (reversal_from_valley_condition and valley_extension_condition) or (peak_extension_condition and reversal_from_peak_condition):

                # INITIAL NAIVE IMPLEMENTATION
                # Add the last previous pivot to the list
                # pivots.append(Pivot.create((last_pivot_candle, last_pivot_type)))

                # Update the last pivot's type and value
                # last_pivot_candle = Candle.create(row)
                # last_pivot_type = 'valley' if last_pivot_type == 'peak' else 'peak'

                # JUDGING BASED ON CANDLE COLOR
                # If the candle is green, that means the low value was probably hit before the high value
                # If the candle is red, that means the high value was probably hit before the low value
                # This means that if the candle is green, we can extend a valley, and if it's red, we can extend a peak
                # Otherwise the direction must flip
                if (row.candle_color == 'green' and last_pivot_type == 'valley') or (row.candle_color == 'red' and last_pivot_type == 'peak'):
                    last_pivot_candle = Candle.create(row)

                else:
                    # Add the last previous pivot to the list of pivots
                    pivots.append(Pivot.create((last_pivot_candle, last_pivot_type)))

                    # Update the last pivot's type and value
                    last_pivot_candle = Candle.create(row)
                    last_pivot_type = 'valley' if last_pivot_type == 'peak' else 'peak'

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

    def __find_relative_pivot(self, pivot_pdi: int, delta: int) -> int:
        """
        Finds the relative pivot to the pivot at the given index.

        Parameters:
        pivot_pdi (int): The pdi of the pivot to find the relative pivot for.
        delta (int): The distance from the pivot to the relative pivot.

        Returns:
        int: The pdi of the relative pivot.
        """

        # zigzag_idx is the zigzag_df index of the current pivot
        zigzag_idx = self.zigzag_df[self.zigzag_df.pdi == pivot_pdi].first_valid_index()

        return self.zigzag_df.iloc[zigzag_idx + delta].pdi

    def detect_first_broken_lpl(self, search_window_start_pdi: int) -> Union[None, pd.Series]:
        """
        Calculates the LPL's and then broken LPL's in a series of zigzag pivots.

        An LPL (For ascending patterns) is registered when a higher high than the highest high since the last LPL is registered. If a lower low than
        the lowest low is registered, the last LPL is considered a broken LPL and registered as such.

        Parameters:
        None

        Returns:
        pd.Series: a row of zigzag_df which contains the broken LPL
        None: If no broke LPL is found
        """

        starting_pivot = self.zigzag_df[self.zigzag_df.pdi == search_window_start_pdi].iloc[0]
        trend_type = "ascending" if starting_pivot.pivot_type == "valley" else "descending"
        log_message(3, "Trend type is", trend_type)

        # Breaking and extension pdi and values represent the values to surpass for registering a higher high (extension) of a lower low (breaking)
        breaking_pdi = search_window_start_pdi
        breaking_value: float = starting_pivot.pivot_value
        extension_pdi = self.__find_relative_pivot(search_window_start_pdi, 1)
        extension_value: float = self.zigzag_df.loc[self.zigzag_df.pdi == extension_pdi].iloc[0].pivot_value

        check_start_pdi = self.__find_relative_pivot(search_window_start_pdi, 2)

        for row in self.zigzag_df[self.zigzag_df.pdi > check_start_pdi].iloc[:-1].itertuples():
            if trend_type == "ascending":
                extension_condition = row.pivot_type == "peak" and row.pivot_value >= extension_value
                breaking_condition = row.pivot_type == "valley" and row.pivot_value <= breaking_value
            else:
                extension_condition = row.pivot_type == "valley" and row.pivot_value <= extension_value
                breaking_condition = row.pivot_type == "peak" and row.pivot_value >= breaking_value

            # Breaking
            if breaking_condition:
                # Return the LPL which was broken to the list of valley LPL's
                return self.zigzag_df[self.zigzag_df.pdi == breaking_pdi].iloc[0]

            # Extension
            if extension_condition:
                # If a higher high is found, extend and update the pattern
                prev_pivot_pdi = self.__find_relative_pivot(row.pdi, -1)
                prev_pivot = self.zigzag_df[self.zigzag_df.pdi == prev_pivot_pdi].iloc[0]

                breaking_pdi = prev_pivot.pdi
                breaking_value = prev_pivot.pivot_value
                extension_value = row.pivot_value

        return None

    def __detect_breaking_sentiment(self, latest_pbos_value: float, latest_pbos_pdi: int, latest_choch_value: float, trend_type: str) -> dict:
        """
        Detects the sentiment of the market by checking if the latest Potential BOS (PBOS) or CHOCH (Change of Character) value is broken by any
        subsequent candles.

        The method checks both the shadows (highs for peaks and lows for valleys) and the closing values of the candles.
        If a candle breaks the PBOS with its shadow, the sentiment is "PBOS_SHADOW".
        If a candle breaks the PBOS with its close value, the sentiment is "PBOS_CLOSE".
        If a candle breaks the CHOCH with its close value, the sentiment is "CHOCH_CLOSE".
        If no candles break the PBOS or CHOCH, the sentiment is "NONE".

        Parameters:
        latest_pbos_value (float): The value of the latest PBOS.
        latest_pbos_pdi (int): The index of the latest PBOS in the pair DataFrame.
        pbos_type (str): The type of the PBOS, either "peak" or "valley".

        Returns:
        dict: A dictionary containing the sentiment ("NONE", "SHADOW", or "CLOSE") and the index of the breaking candle, if any.
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
        # sorted built-in function. It also prioritizes sentiments that have "CLOSE" in their description.
        def sorting_key(output_item):
            pdi = output_item["pdi"] if output_item["pdi"] is not None else 0
            has_close = 1 if "CLOSE" in output_item["sentiment"] else 2
            return pdi, has_close

        sorted_outputs: list[dict] = [output_item for output_item in sorted(outputs_list, key=sorting_key)
                                      if output_item["pdi"] is not None]

        return sorted_outputs[0] if len(sorted_outputs) > 0 else none_output

    def __calc_region_start_pdi(self, broken_lpl: pd.Series) -> int:
        """
        Initializes the starting point of the region after the first potential BOS.

        The region starting point is the first pivot right after the first broken LPL

        Note: This method does not return anything.
        """

        # The pivots located between the starting point and the first pivot after the broken LPL. The starting point is either
        # 1) The start of the pattern, which means we are forming the first region, or
        # 2) The start of the next section. The region_start_pdi variable determines this value.
        region_start_pdi = self.__find_relative_pivot(broken_lpl.pdi, 1)

        return region_start_pdi

    def calc_h_o_zigzag(self, starting_point_pdi) -> pd.DataFrame:
        # Set the starting point of the HO zigzag and add it
        self.starting_pdi = starting_point_pdi
        self.h_o_indices.append(self.starting_pdi)
        latest_choch_pdi = self.starting_pdi
        latest_choch_threshold: float = self.zigzag_df[self.zigzag_df.pdi == self.starting_pdi].iloc[0].pivot_value
        log_message("Added starting point", self.starting_pdi, v=1)

        pattern_start_pdi = self.starting_pdi

        latest_pbos_pdi = None
        latest_pbos_threshold = None

        while True:
            log_message("", v=1)
            # Find the first broken LPL after the starting point and the region starting point
            broken_lpl = self.detect_first_broken_lpl(pattern_start_pdi)

            # If no broken LPL can be found, just quit
            if broken_lpl is None:
                log_message("Reached end of chart, no more broken LPL's.", v=1)
                break

            log_message("Starting pattern at", pattern_start_pdi, v=3)
            log_message("Broken LPL is at", broken_lpl.pdi, v=3)

            # If the LPL type is valley, it means the trend type is ascending
            trend_type = "ascending" if broken_lpl.pivot_type == "valley" else "descending"

            # The BOS is the pivot right after the broken LPL
            bos_pdi = int(self.__calc_region_start_pdi(broken_lpl))

            # When pattern resets, aka a new point is found OR when the pattern is initializing
            if latest_pbos_pdi is None:
                latest_pbos_pdi = bos_pdi
                latest_pbos_threshold = self.zigzag_df[self.zigzag_df.pdi == bos_pdi].iloc[0].pivot_value

                # Add the BOS to the HO indices
                self.h_o_indices.append(bos_pdi)
                log_message("Added BOS", bos_pdi, v=1)

                log_message("HO indices", self.h_o_indices, v=2)

            log_message("CHOCH threshold is at", latest_choch_pdi, "BOS threshold is at", latest_pbos_pdi, v=2)

            # Add the first found PBOS to the list as that is needed to kickstart the h_o_zigzag
            self.pbos_indices.append(bos_pdi)

            # If the candle breaks the PBOS by its shadow, the most recent BOS threshold will be moved to that candle's high instead
            # If a candle breaks the PBOS with its close value, then the search halts
            # If the candle breaks the last CHOCH by its shadow, the CHOCH threshold will be moved to that candle's low
            # If a candle breaks the last CHOCH with its close, the direction inverts and the search halts

            # have its effect first. This whole logic is implemented in the __detect_breaking_sentiment method.
            breaking_output = self.__detect_breaking_sentiment(latest_pbos_threshold, latest_pbos_pdi, latest_choch_threshold, trend_type)

            breaking_pdi = breaking_output["pdi"]
            breaking_sentiment = breaking_output["sentiment"]

            if breaking_sentiment == "PBOS_SHADOW":
                log_message("PBOS #", latest_pbos_pdi, "broken by candle shadow at index", breaking_pdi, v=2)

                latest_pbos_pdi = breaking_pdi
                latest_pbos_threshold = self.pair_df.iloc[breaking_pdi].high if trend_type == "ascending" else self.pair_df.iloc[breaking_pdi].low

            elif breaking_sentiment == "CHOCH_SHADOW":
                log_message("CHOCH #", latest_choch_pdi, "broken by candle shadow at index", breaking_pdi, v=2)

                latest_choch_pdi = breaking_pdi
                latest_choch_threshold = self.pair_df.iloc[breaking_pdi].low if trend_type == "ascending" else self.pair_df.iloc[breaking_pdi].high

            elif breaking_sentiment == "PBOS_CLOSE":
                log_message("Candle at index",
                            breaking_pdi, "broke the last PBOS #", latest_pbos_pdi, "with its close price", v=2)
                log_message("BOS #", self.h_o_indices[-1], "break at", breaking_pdi, v=1)

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
                    extremum_pivot = extremum_point_pivots_of_type.loc[extremum_point_pivots_of_type['pivot_value'].idxmax()]
                else:
                    extremum_pivot = extremum_point_pivots_of_type.loc[extremum_point_pivots_of_type['pivot_value'].idxmin()]

                # Add the extremum point to the HO indices
                self.h_o_indices.append(int(extremum_pivot.pdi))
                extremum_type = "lowest low" if trend_type == "ascending" else "highest high"
                log_message("Added extremum of type", extremum_type, "at", extremum_pivot.pdi, v=1)

                # Now, we can restart finding HO pivots. Starting point is set to the last LPL of the same type BEFORE the BOS breaking candle.
                # Trend stays the same since no CHOCH has occurred.
                pivot_type = "valley" if trend_type == "ascending" else "peak"
                pivots_of_type_before_closing_candle = self.zigzag_df[(self.zigzag_df.pivot_type == pivot_type)
                                                                      & (self.zigzag_df.pdi <= breaking_pdi)]

                pattern_start_pdi = pivots_of_type_before_closing_candle.iloc[-1].pdi
                log_message("Setting pattern start to", pattern_start_pdi, v=1)

                # Essentially reset the algorithm
                latest_pbos_pdi = None
                latest_choch_pdi = self.h_o_indices[-1]
                latest_choch_threshold = self.zigzag_df[self.zigzag_df.pdi == latest_choch_pdi].iloc[0].pivot_value

            elif breaking_sentiment == "CHOCH_CLOSE":
                log_message("Candle at index",
                            breaking_pdi, "broke the last CHOCH #", latest_choch_pdi, "with its close price", v=2)
                log_message("CHOCH #", self.h_o_indices[-2], "break at", breaking_pdi, v=1)

                trend_type = "ascending" if trend_type == "descending" else "descending"

                # Set the pattern start to the last inverse pivot BEFORE the closing candle
                pivot_type = "valley" if trend_type == "ascending" else "peak"
                pivots_of_type_before_closing_candle = self.zigzag_df[(self.zigzag_df.pivot_type == pivot_type)
                                                                      & (self.zigzag_df.pdi <= breaking_pdi)]

                pattern_start_pdi = pivots_of_type_before_closing_candle.iloc[-1].pdi
                log_message("Setting pattern start to", pattern_start_pdi, v=1)

                # Essentially reset the algorithm
                latest_choch_pdi = pattern_start_pdi
                latest_choch_threshold = self.zigzag_df[self.zigzag_df.pdi == latest_choch_pdi].iloc[0].pivot_value

                latest_pbos_pdi = None

            # If no candles have broken the PBOS even with a shadow, break the loop
            else:
                log_message("No more candles found. Breaking...", v=1)

                break

        return

    def identify_fvgs(self):

        """
        This method identifies and stores the Fair Value Gaps (FVGs) in the pair DataFrame.

        An FVG is a gap between two candles that is not filled by the body of a third candle. This method calculates FVGs by creating a rolling window of 3 candles at a time and checking for the existence of an FVG in each window.

        If an FVG is found, it is added to the `fvg_list` attribute of the class instance.

        Note: This method does not return anything.
        """

        def calc_fvg(candles_df) -> Union[None, AbstractInterval]:
            """
            This helper function calculates the FVG for a given window of 3 candles.

            It first creates intervals for each candle's range (high to low) and body (open to close). It then checks if there is an overlap between
            the first and third candle. If there is, it returns None as there is no FVG.

            If there is no overlap, it calculates the gap between the first and third candle and calculates the intersection of the gap with the
            second candle's body. THe intersection is the FVG.

            Parameters:
            candles_df (pd.DataFrame): A DataFrame containing 3 consecutive candles.

            Returns:
            AbstractInterval: An interval representing the FVG, or None if there is no FVG.
            """

            # Get each candle from the DataFrame
            candle1 = candles_df.iloc[0]
            candle2 = candles_df.iloc[1]
            candle3 = candles_df.iloc[2]

            # Create intervals for each candle's range and body
            candle1_interval: AbstractInterval = Interval([candle1.low, candle1.high])
            candle2_body_interval: AbstractInterval = Interval([min(candle2.open, candle2.close), max(candle2.open, candle2.close)])
            candle3_interval: AbstractInterval = Interval([candle3.low, candle3.high])

            try:
                # Check for overlap between the first and third candle
                overlap = candle1_interval & candle3_interval
                return None

            except IllegalArgument:
                # If there is no overlap, calculate the gap
                if candle1.high < candle3.low:
                    gap = Interval([candle1.high, candle3.low])
                elif candle1.low >= candle3.high:
                    gap = Interval([candle3.high, candle1.low])
                else:
                    return None

                try:
                    # Check if the second candle's body fills the gap
                    fvg: AbstractInterval = candle2_body_interval & gap
                    return fvg

                except IllegalArgument:
                    return None

        # Create a rolling window of 3 candles
        windows = self.pair_df.rolling(3)

        for window in windows:
            # Skip windows with less than 3 candles
            if len(window) < 3:
                continue

            # Calculate the FVG for the current window
            fvg = calc_fvg(window)

            # If there is an FVG, add it to the list
            if fvg is not None:
                self.fvg_list.append(FVG(middle_candle=window.iloc[1].name, fvg_lower=float(fvg.lower), fvg_upper=float(fvg.upper)))
