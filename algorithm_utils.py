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
    for row in pair_df.iloc[last_pivot_candle.pair_df_index + 1:].itertuples():

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
    zigzag_df = pd.DataFrame.from_dict(pivot._asdict() for pivot in pivots)

    return zigzag_df


def find_longest_chain(values: pd.Series, direction: str = 'ascending') -> int:
    """
        Finds the longest chain of ascending or descending values in a list, starting at the first index

        Parameters:
        values (pd.Series): The list of values to search.
        direction (str): The direction to search for. Can be 'ascending' or 'descending'.

        Returns:
        int: The end index of the longest chain.
    """

    values_list = values.to_list()

    # Initialize the end_index as 0
    end_index = 0

    # Store the first value in the list
    last_value = values_list[0]

    # Iterate over the rest of the values in the list
    for i, value in enumerate(values_list[1:]):
        # If the current value is greater than the last value, and we're looking for an ascending chain
        # Or if the current value is less than the last value, and we're looking for a descending chain
        if (value > last_value and direction == 'ascending') or (value < last_value and direction == 'descending'):
            # Update the end index of the longest chain
            end_index = i + 1
            # Update the last value
            last_value = value
        else:
            # If the current value breaks the chain, return the longest chain found so far
            return end_index

    return end_index


def find_ascending_valleys(zigzag_df: pd.DataFrame, start_pair_df_index: int) -> OneDChain:
    """
        Function to find the longest chain of higher lows in a zigzag DataFrame.
        It starts from a given index and goes forward, looking for valleys with ascending pivot values.
        It stops once it finds a lower low than the previous.

        Parameters:
        zigzag_df (pd.DataFrame): The DataFrame containing the zigzag data.
        start_pair_df_index (int): The index to start the search from.

        Returns:
        OneDChain: A one directional chain object representing the chain found
    """

    # If the selected start_index isn't a valley, throw an error
    if zigzag_df[zigzag_df.pair_df_index == start_pair_df_index].iloc[0].pivot_type != 'valley':
        raise ValueError('The start index must be a valley.')

    # The slice of the zigzag_df dataframe that needs to be searched
    search_window: pd.DataFrame = zigzag_df[zigzag_df.pair_df_index >= start_pair_df_index]

    search_window_valleys: pd.Series = search_window[search_window.pivot_type == 'valley'].pivot_value

    chain_length: int = find_longest_chain(search_window_valleys, 'ascending')

    return OneDChain.create(chain_length, start_pair_df_index, 'ascending')


def find_descending_peaks(zigzag_df: pd.DataFrame, start_pair_df_index: int) -> OneDChain:
    """
        Function to find the longest chain of lower highs in a zigzag DataFrame.
        It starts from a given index and goes forward, looking for peaks with descending pivot values.
        It stops once it finds a higher high than the previous.

        Parameters:
        zigzag_df (pd.DataFrame): The DataFrame containing the zigzag data.
        start_pair_df_index (int): The index to start the search from.

        Returns:
        OneDChain: A one directional chain object representing the chain found
    """

    # If the selected start_index isn't a valley, throw an error
    if zigzag_df[zigzag_df.pair_df_index == start_pair_df_index].iloc[0].pivot_type != 'peak':
        raise ValueError('The start index must be a peak.')

    # The slice of the zigzag_df dataframe that needs to be searched
    search_window: pd.DataFrame = zigzag_df[zigzag_df.pair_df_index >= start_pair_df_index]

    search_window_peaks: pd.Series = search_window[search_window.pivot_type == 'peak'].pivot_value

    chain_length: int = find_longest_chain(search_window_peaks, 'descending')

    return OneDChain.create(chain_length, start_pair_df_index, 'descending')


class Algo:
    def __init__(self, pair_df, symbol):
        self.pair_df: pd.DataFrame = pair_df
        self.symbol: str = symbol
        self.zigzag_df = None
        self.pbos_indices = []
        self.pbos_list = []
        self.is_start_established = False
        self.h_o_indices = []

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
        for row in self.pair_df.iloc[last_pivot_candle.pair_df_index + 1:].itertuples():

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
        zigzag_df = pd.DataFrame.from_dict(pivot._asdict() for pivot in pivots)

        self.zigzag_df = zigzag_df

    def calc_h_o_zigzag(self) -> pd.DataFrame:
        # The current_pivot_index represents the entire zigzag. We only want to search for new things until we reach the end of the
        # currently available zigzag. Note that this is the original zigzag_df DataFrame index, not the pair_df_index
        current_pivot_index = 0
        # lpl_l_indices = []
        # lpl_s_indices = []
        while current_pivot_index < len(self.zigzag_df):
            # current_pivot is basically the row from the zigzag_df that we're currently looking at
            current_pivot = self.zigzag_df.iloc[current_pivot_index]

            # Use previously defined functions to find the longest chains of ascending valleys and descending peaks
            if current_pivot.pivot_type == "valley":
                chain = find_ascending_valleys(self.zigzag_df, current_pivot.pair_df_index)

            else:
                chain = find_descending_peaks(self.zigzag_df, current_pivot.pair_df_index)

            # If the chain is longer than 0, add its last pivot + 1 to the list of PBOS's
            # The coefficient 2 is required because there is as many peaks as there are valleys
            current_pivot_index += 2 * chain.chain_length + 1

            # If there is a non-single chain (which would have a length of 0 if single), add the last pivot to the list of PBOS's
            if chain.chain_length > 0:
                # This snippet basically skips the first PBOS which is actually the starting index. Add its pair_df_index to h_o_zigzag,
                # effectively starting it
                if not self.is_start_established:
                    self.h_o_indices.append(current_pivot.pair_df_index)
                    self.is_start_established = True
                    continue

                self.pbos_indices.append(current_pivot.pair_df_index)

                # This section will include an inner loop, which checks to see how the PBOS (or the LPL?) get broken by the candles
                while True:
                    latest_pbos = self.pair_df.iloc[self.pbos_indices[-1]]
                    # print(self.pbos_indices)
                    latest_pbos_value = latest_pbos.high if current_pivot.pivot_type == "peak" else latest_pbos.low
                    # Now we need to check if the most recently added PBOS has a candle after it which breaks it by closing,
                    # or by having a shadow which breaks it

                    # If the candle breaks the PBOS by its shadow, the most recent PBOS will be moved to that candle instead
                    search_window: pd.DataFrame = self.pair_df.iloc[current_pivot.pair_df_index + 1:]

                    # The definition of "breaking" is different whether the PBOS is a peak or a valley
                    if current_pivot.pivot_type == "peak":
                        shadow_breaking_candles = search_window[search_window.high > latest_pbos_value]
                        close_breaking_candles = search_window[search_window.close > latest_pbos_value]
                    else:
                        shadow_breaking_candles = search_window[search_window.low < latest_pbos_value]
                        close_breaking_candles = search_window[search_window.close < latest_pbos_value]

                    if shadow_breaking_candles.first_valid_index() is not None and close_breaking_candles.first_valid_index() is not None:
                        # If the shadow breaking the PBOS happens before the close breaking the PBOS
                        if shadow_breaking_candles.first_valid_index() < close_breaking_candles.first_valid_index():
                            print("PBOS #", latest_pbos.name, "broken by candle shadow at index", shadow_breaking_candles.first_valid_index())
                            self.pbos_indices.append(shadow_breaking_candles.first_valid_index())

                        # Otherwise, if a candle breaks the PBOS with its close price, register a new PBOSRegion object, basically
                        # ending the search for a PBOS
                        else:
                            print("Candle at index",
                                  close_breaking_candles.first_valid_index(), "broke the last PBOS #", latest_pbos.name, "with its close price")

                            pbos_type = current_pivot.pivot_type

                            pbos_region = PBOSRegion.create(self.pbos_indices[0], self.pbos_indices[-1], close_breaking_candles.first_valid_index(),
                                                            pbos_type)

                            # To find the inverse of the PBOS, which makes the middle point of the h_o zigzag between the PBOS region start and end,
                            # we need to find the lowest valley between the two peaks (for peak PBOS's) or
                            # the highest peak between the two valleys (for valley PBOS's)
                            inverse_pivot_type = "valley" if pbos_type == "peak" else "peak"

                            # h_o_inverse_pivots is a list of all the peaks/valleys, which we later find the max/min from in
                            # minmax_inverse_pivot_value_row
                            h_o_inverse_pivots = self.zigzag_df[
                                (self.zigzag_df.pair_df_index >= self.pbos_indices[0])
                                & (self.zigzag_df.pair_df_index <= close_breaking_candles.first_valid_index())
                                & (self.zigzag_df.pivot_type == inverse_pivot_type)]

                            if inverse_pivot_type == "peak":
                                region_inverse_pivot = h_o_inverse_pivots.loc[h_o_inverse_pivots['pivot_value'].idxmax()]
                            else:
                                region_inverse_pivot = h_o_inverse_pivots.loc[h_o_inverse_pivots['pivot_value'].idxmin()]

                            self.h_o_indices.append(pbos_region.start_pbos)
                            self.h_o_indices.append(region_inverse_pivot.pair_df_index)
                            # self.pbos_indices = []
                            break

                    elif shadow_breaking_candles.first_valid_index() is None:
                        print("No more candles found. Breaking...")
                        break

                break
        # lpl_s_df: pd.DataFrame = zigzag_df[zigzag_df.pair_df_index.isin(lpl_s_indices)].copy()
        # lpl_l_df: pd.DataFrame = zigzag_df[zigzag_df.pair_df_index.isin(lpl_l_indices)].copy()
        #
        # lpls_df: pd.DataFrame = pd.concat([lpl_s_df, lpl_l_df], axis=0).iloc[:-1]
        # lpls_df.sort_values(by='pair_df_index', inplace=True)
        # return lpls_df
