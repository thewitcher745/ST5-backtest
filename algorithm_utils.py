from typing import Optional

import pandas as pd

from datatypes import *


def zigzag(pair_df: pd.DataFrame) -> pd.DataFrame:
    """
        Function to identify turning points in a candlestick chart.
        It compares each candle to its previous pivot to determine if it's a new pivot point.
        This implementation is less optimized than the deprecated version, as it doesn't use
        vectorized operations, but it is what it is

        Parameters:
        pair_df (pd.DataFrame): The input DataFrame containing the candlestick data.

        Returns:
        pd.DataFrame: A DataFrame containing the identified turning points.
        """

    # Find the first candle that has a higher high or a lower low than its previous candle
    # and set it as the first pivot. Also set the type of the pivot (peak or valley)
    last_pivot_candle_series = pair_df[(pair_df['high'] > pair_df['high'].shift(1)) | (pair_df['low'] < pair_df['low'].shift(1))].iloc[0]
    last_pivot_type: str = 'valley'
    if last_pivot_candle_series.high > pair_df.iloc[last_pivot_candle_series.name - 1].high:
        last_pivot_type = 'peak'

    last_pivot_candle: Candle = Candle.create(last_pivot_candle_series)
    pivots: List[Pivot] = []

    # Start at the candle right after the last (first) pivot
    for row in pair_df.iloc[last_pivot_candle.pdi + 1:].itertuples():

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

    return zigzag_df


class Algo:
    def __init__(self, pair_df, symbol, pattern_limit=None):
        self.pair_df: pd.DataFrame = pair_df
        self.symbol: str = symbol
        self.zigzag_df: Optional[pd.DataFrame] = None

        # This variable indicates whether only the first pattern is generated, for testing purposes
        self.pattern_limit = pattern_limit

        # pbos_indices is a list which stores the PBOS's being moved due to shadows breaking the most recent PBOS
        self.pbos_indices = []

        # h_o_indices indicates the indices of the peaks and valleys in the higher order zigzag
        self.h_o_indices = []

        # Peak and valley LPL's are calculated using the calc_lpls method
        self.peak_lpls: Optional[pd.DataFrame] = None
        self.valley_lpls: Optional[pd.DataFrame] = None

        # Brokeb peak and valley LPL's are calculated using the calc_broken_lpls method, then combined into self.broken_lpls
        self.valley_broken_lpls: Optional[pd.DataFrame] = None
        self.peak_broken_lpls: Optional[pd.DataFrame] = None
        self.broken_lpls: Optional[pd.DataFrame] = None

        # starting_pdi is the starting point of the entire pattern, calculated using __init_pattern_start_pdi. This method is
        # executed in the calc_h_o_zigzag method.
        self.starting_pdi = 0
        self.candles_starting_idx = 0

    def init_zigzag(self) -> None:
        """
            Method to identify turning points in a candlestick chart.
            It compares each candle to its previous pivot to determine if it's a new pivot point.
            This implementation is less optimized than the deprecated version, as it doesn't use
            vectorized operations, but it is what it is

            Parameters:
            pair_df (pd.DataFrame): The input DataFrame containing the candlestick data.

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

    def calc_lpls(self):
        """
        Calculates the Last Pivot Liquidities (LPLs) for both bullish and bearish trends.

        For a bullish trend, an LPL is any instance of a valley which has a higher value than the one before it.
        For a bearish trend, an LPL is any instance of a peak which has a lower value than the one before it.

        The method updates the instance variables `valley_lpls` and `peak_lpls` with the calculated LPLs.

        Note: This method does not return anything.
        """

        # Bullish LPL's
        valleys = self.zigzag_df[self.zigzag_df.pivot_type == 'valley']
        self.valley_lpls = valleys[valleys["pivot_value"] > valleys["pivot_value"].shift(1)]

        # Bearish LPL's
        peaks = self.zigzag_df[self.zigzag_df.pivot_type == 'peak']
        self.peak_lpls = peaks[peaks["pivot_value"] < peaks["pivot_value"].shift(1)]

    def calc_broken_lpls(self):
        """
        Calculates the Broken LPL's for both bullish and bearish trends. Broken means the LPL is the last LPL which obeys the
        LPL formula, which is higher lows for bullish trends and lower highs for bearish trends.

        The method updates the instance variables `valley_broken_lpls`, `peak_broken_lpls` and `broken_lpls` with the calculated broken LPLs.

        Note: This method does not return anything.
        """

        # Bullish LPL's
        valleys = self.zigzag_df[self.zigzag_df.pivot_type == 'valley']
        self.valley_broken_lpls = valleys[(valleys["pivot_value"] > valleys["pivot_value"].shift(1)) &
                                          (valleys["pivot_value"] > valleys["pivot_value"].shift(-1))]

        # Bearish LPL's
        peaks = self.zigzag_df[self.zigzag_df.pivot_type == 'peak']
        self.peak_broken_lpls = peaks[(peaks["pivot_value"] < peaks["pivot_value"].shift(1)) &
                                      (peaks["pivot_value"] < peaks["pivot_value"].shift(-1))]

        self.broken_lpls = pd.concat([self.valley_broken_lpls, self.peak_broken_lpls]).sort_values("pdi", inplace=False)

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

    def __detect_breaking_sentiment(self, latest_pbos_value: float, latest_pbos_pdi: int, pbos_type: str) -> dict:
        """
        Detects the sentiment of the market by checking if the latest Potential Box Start (PBOS) value is broken by any subsequent candles.

        The method checks both the shadows (highs for peaks and lows for valleys) and the closing values of the candles.
        If a candle breaks the PBOS with its shadow, the sentiment is "SHADOW".
        If a candle breaks the PBOS with its close value, the sentiment is "CLOSE".
        If no candles break the PBOS, the sentiment is "NONE".

        Parameters:
        latest_pbos_value (float): The value of the latest PBOS.
        latest_pbos_pdi (int): The index of the latest PBOS in the pair DataFrame.
        pbos_type (str): The type of the PBOS, either "peak" or "valley".

        Returns:
        dict: A dictionary containing the sentiment ("NONE", "SHADOW", or "CLOSE") and the index of the breaking candle, if any.
        """

        search_window: pd.DataFrame = self.pair_df.iloc[latest_pbos_pdi + 1:]

        # The definition of "breaking" is different whether the PBOS is a peak or a valley
        if pbos_type == "peak":
            shadow_breaking_candles = search_window[search_window.high > latest_pbos_value]
            close_breaking_candles = search_window[search_window.close > latest_pbos_value]
        else:
            shadow_breaking_candles = search_window[search_window.low < latest_pbos_value]
            close_breaking_candles = search_window[search_window.close < latest_pbos_value]

        if shadow_breaking_candles.first_valid_index() is None:
            return {
                "sentiment": "NONE",
                "pdi": None
            }

        elif close_breaking_candles.first_valid_index() is None and shadow_breaking_candles.first_valid_index() is not None:
            return {
                "sentiment": "SHADOW",
                "pdi": shadow_breaking_candles.first_valid_index()
            }

        elif close_breaking_candles.first_valid_index() is not None:
            if close_breaking_candles.first_valid_index() <= shadow_breaking_candles.first_valid_index():
                return {
                    "sentiment": "CLOSE",
                    "pdi": close_breaking_candles.first_valid_index()
                }
            else:
                return {
                    "sentiment": "SHADOW",
                    "pdi": shadow_breaking_candles.first_valid_index()
                }

    def __find_last_mid_region_lpl(self, mid_region_pivot: pd.Series, lpls_df: pd.DataFrame, breaking_pdi: int) -> int:
        last_mid_region_lpl = lpls_df[(lpls_df.pdi >= mid_region_pivot.pdi) & (lpls_df.pdi <= breaking_pdi)].iloc[-1].pdi

        return last_mid_region_lpl

    def __find_last_mid_region_lpl_chain(self, mid_region_pivot: pd.Series, lpls_df: pd.DataFrame, breaking_pdi: int) -> int:
        last_mid_region_lpl = self.__find_last_mid_region_lpl(mid_region_pivot, lpls_df, breaking_pdi)

        # Finding the longest consecutive chain of LPL's including and after the last_mid_region_lpl.
        last_found_lpl_index = lpls_df[lpls_df.pdi == last_mid_region_lpl].first_valid_index() \
            # Continue adding +2 to the LPL pdi's, until the +2 value doesn't exist in lpls_df. Report the most recent value  as output
        while True:
            # The pivots are positioned as VALLEY, PEAK, VALLEY, PEAK. So if the index of the last_mid_region_lpl + 2
            # is in lpls_df, that means there is another LPL located right after. If so, we extend the found LPL to
            if last_found_lpl_index + 2 not in lpls_df.index:
                break
            else:
                last_found_lpl_index += 2

        return self.zigzag_df.iloc[last_found_lpl_index].pdi

    def __init_pattern_start_pdi(self, lpls_df: pd.DataFrame):
        """
        Initializes the starting point of the pattern based on the DataFrame of LPL's.

        The starting point is determined as the same-type pivot right before the first LPL, therefore the pivot 2 behind
        the first LPL

        The method updates the instance variables `starting_pdi` and `h_o_indices` with the calculated starting point.

        Parameters:
        lpls_df (pd.DataFrame): DataFrame containing the LPLs.
        broken_lpl (pd.Series): Series representing the broken LPL.

        Note: This method does not return anything.
        """

        first_lpl_pdi = lpls_df.iloc[0].pdi
        print("Starting point is 2", lpls_df.iloc[0].pivot_type, "'s behind #", first_lpl_pdi)
        self.starting_pdi = self.__find_relative_pivot(first_lpl_pdi, -2)
        print("Starting point is at", self.starting_pdi)

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

    def calc_h_o_zigzag(self) -> pd.DataFrame:
        # end_of_h_o is used to terminate the search for a new h_o_zigzag.
        end_of_h_o = False

        pattern_count = 0
        while not end_of_h_o:
            broken_lpl: pd.Series = self.broken_lpls[self.broken_lpls.pdi > self.candles_starting_idx].iloc[0]
            lpls_df = self.valley_lpls.copy() if broken_lpl.pivot_type == "valley" else self.peak_lpls.copy()
            lpls_df = lpls_df[lpls_df.pdi > self.candles_starting_idx]

            print("Broken LPL is at", broken_lpl.pdi)
            # The starting point of the zigzag is denoted as the same-type pivot right before the first LPL

            self.__init_pattern_start_pdi(lpls_df)

            region_start_pdi = self.__calc_region_start_pdi(broken_lpl)
            # Add the first found PBOS to the list as that is needed to kickstart the h_o_zigzag
            self.pbos_indices.append(region_start_pdi)

            # If the LPL type is valley, it means the PBOS's are peak, and vice versa
            pbos_type = "valley" if broken_lpl.pivot_type == "peak" else "peak"

            while True:
                latest_pbos_pdi = self.pbos_indices[-1]
                latest_pbos_candle = self.pair_df.iloc[latest_pbos_pdi]

                # The latest_pbos_value has to be calculated using the candle value because it isn't necessarily located on a pivot
                # so using .pivotvalue isn't an option
                latest_pbos_value = latest_pbos_candle.high if pbos_type == "peak" else latest_pbos_candle.low

                # If the candle breaks the PBOS by its shadow, the most recent PBOS will be moved to that candle instead
                # If a candle breaks the PBOS with its close value, then the search halts

                # The close-value-breaking should have priority over shadow-breaking, that means whichever occurs earlier will
                # have its effect first. This whole logic is implemented in the __detect_breaking_sentiment method.
                breaking_output = self.__detect_breaking_sentiment(latest_pbos_value, latest_pbos_pdi, pbos_type)
                breaking_pdi = breaking_output["pdi"]
                breaking_sentiment = breaking_output["sentiment"]

                if breaking_sentiment == "SHADOW":
                    self.pbos_indices.append(breaking_pdi)
                    print("PBOS #", latest_pbos_pdi, "broken by candle shadow at index", breaking_pdi)

                elif breaking_sentiment == "CLOSE":
                    print("Candle at index",
                          breaking_pdi, "broke the last PBOS #", latest_pbos_pdi, "with its close price")

                    # Only when the first PBOS has a candle close above it, is the starting point set. Then the first found PBOS is also
                    # added to the higher order zigzag
                    self.pbos_indices.append(region_start_pdi)
                    self.h_o_indices.append(self.starting_pdi)
                    self.h_o_indices.append(self.pbos_indices[0])

                    print("added points", self.pbos_indices[0], self.starting_pdi)

                    # The mid-region pivot is the pivot which transports the first now-confirmed PBOS (confirmed by the candle close-breaking
                    # it) to the last pivot of the region, completing one round of the H-O zigzag formation.

                    # The mid-region pivot should have a pivot_Type opposite to that of the original PBOS being studied.
                    mid_region_pivot_type = "valley" if pbos_type == "peak" else "peak"

                    # mid_region_pivots_of_type is a list of all the pivots of the right type for the middle region
                    mid_region_pivots_of_type = self.zigzag_df[
                        (self.zigzag_df.pdi >= self.pbos_indices[0])
                        & (self.zigzag_df.pdi <= breaking_pdi)
                        & (self.zigzag_df.pivot_type == mid_region_pivot_type)]

                    # The mid-region pivot is the lowest low/highest high in the region between the first PBOS and the closing candle
                    if mid_region_pivot_type == "peak":
                        mid_region_pivot = mid_region_pivots_of_type.loc[mid_region_pivots_of_type['pivot_value'].idxmax()]
                    else:
                        mid_region_pivot = mid_region_pivots_of_type.loc[mid_region_pivots_of_type['pivot_value'].idxmin()]

                    last_region_lpl_pdi = self.__find_last_mid_region_lpl(mid_region_pivot, lpls_df, breaking_pdi)
                    last_mid_region_chain_lpl_pdi = self.__find_last_mid_region_lpl_chain(mid_region_pivot, lpls_df, breaking_pdi)
                    last_region_lpl_next_pivot_pdi = self.__find_relative_pivot(last_mid_region_chain_lpl_pdi, 1)

                    # Finally, after the LPL has been found, the highest point between the first LPL and the last BOS is set
                    # as the last point in the region
                    print("Looking for last HO pivot in range", last_region_lpl_pdi, last_region_lpl_next_pivot_pdi)
                    mid_region_pivot_pdi = mid_region_pivot.pdi

                    if pbos_type == "peak":
                        last_region_point_idx = self.zigzag_df[
                            (mid_region_pivot_pdi <= self.zigzag_df.pdi) & (
                                    self.zigzag_df.pdi <= last_region_lpl_next_pivot_pdi)].pivot_value.idxmax()
                    else:
                        last_region_point_idx = self.zigzag_df[
                            (mid_region_pivot_pdi <= self.zigzag_df.pdi) & (
                                    self.zigzag_df.pdi <= last_region_lpl_next_pivot_pdi)].pivot_value.idxmin()

                    last_region_point_pdi = self.zigzag_df.iloc[last_region_point_idx].pdi

                    self.h_o_indices.append(mid_region_pivot.pdi)
                    self.h_o_indices.append(last_region_point_pdi)

                    # Clear the list of recent PBOS's
                    print("Finished one round of HO Zigzag, resetting PBOS and changing broken_lpl")
                    print()

                    broken_lpl = \
                        self.broken_lpls[(self.broken_lpls.pdi > last_region_point_pdi)].iloc[0]

                    self.pbos_indices = []
                    self.candles_starting_idx = last_region_point_pdi + 1

                    pattern_count += 1

                    break

                # If no candles have broken the PBOS even with a shadow, break the loop
                else:
                    print("No more candles found. Breaking...")

                    end_of_h_o = True
                    break

            if self.pattern_limit:
                if pattern_count >= self.pattern_limit:
                    break
        return
