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


def find_ascending_chains(zigzag_df: pd.DataFrame, start_pair_df_index: int) -> OneDChain:
    """
        Function to find the longest chain of higher lows in a zigzag DataFrame.
        It starts from a given index and goes forward, looking for valleys with ascending pivot values.
        It stops once it finds a lower low than the previous.

        Parameters:
        zigzag_df (pd.DataFrame): The DataFrame containing the zigzag data.
        start_pair_df_index (int): The index to start the search from.

        Returns:
        OneDChain: The start and end indices of the chain, along with information on if the chain simplifies an LO zigzag
    """

    # If the selected start_index isn't a valley, throw an error
    if zigzag_df[zigzag_df.pair_df_index == start_pair_df_index].iloc[0].pivot_type != 'valley':
        raise ValueError('The start index must be a valley.')

    # The slice of the zigzag_df dataframe that needs to be searched
    search_window: pd.DataFrame = zigzag_df[zigzag_df.pair_df_index >= start_pair_df_index]

    search_window_peaks: pd.Series = search_window[search_window.pivot_type == 'peak'].pivot_value
    search_window_valleys: pd.Series = search_window[search_window.pivot_type == 'valley'].pivot_value

    higher_low_chain_length: int = find_longest_chain(search_window_valleys, 'ascending')
    higher_high_chain_length: int = find_longest_chain(search_window_peaks, 'ascending')

    # A variable which indicates if the OneDChain is actually simplifying a set of legs, which is used to find BOS points
    is_simplifying: bool = higher_low_chain_length > 0 and higher_high_chain_length > 0

    return OneDChain.create(higher_low_chain_length, higher_high_chain_length, start_pair_df_index, 'ascending', is_simplifying)


def find_descending_chains(zigzag_df: pd.DataFrame, start_pair_df_index: int) -> OneDChain:
    if zigzag_df[zigzag_df.pair_df_index == start_pair_df_index].iloc[0].pivot_type != 'peak':
        raise ValueError('The start index must be a peak.')

    search_window: pd.DataFrame = zigzag_df[zigzag_df.pair_df_index >= start_pair_df_index]

    search_window_peaks: pd.Series = search_window[search_window.pivot_type == 'peak'].pivot_value
    search_window_valleys: pd.Series = search_window[search_window.pivot_type == 'valley'].pivot_value

    lower_low_chain_length: int = find_longest_chain(search_window_valleys, 'descending')
    lower_high_chain_length: int = find_longest_chain(search_window_peaks, 'descending')

    is_simplifying: bool = lower_low_chain_length > 0 and lower_high_chain_length > 0

    return OneDChain.create(lower_low_chain_length, lower_high_chain_length, start_pair_df_index, 'descending', is_simplifying)


def simplify_chain(input_zigzag_df: pd.DataFrame,
                   high_low_chains: OneDChain) -> dict:
    """
        Function that takes a chain of higher low/higher highs (ascending), or lower low/lower highs (descending) and
        returns a dict object with instructions on forming a higher order zigzag with it.

        Parameters:
        zigzag_df (pd.DataFrame): The DataFrame containing the zigzag data.
        high_low_chains (OneDChain): A OneDChain object containing the chain lengths and the direction.

        Returns:
        dict: A dict object containing the pair_df_indices of the start and end of the chain, as well as a bool representing if the chain has
        simplified the lower order zigzag
    """
    zigzag_df = input_zigzag_df.copy()

    # Filter out the rows that are before the start index.
    zigzag_df = zigzag_df[zigzag_df.pair_df_index >= high_low_chains.start_pair_df_index]

    # The chain will start at the first pair_df_index in zigzag_df, regardless of direction
    chain_start_pair_df_index = zigzag_df.reset_index(drop=True).iloc[0].pair_df_index
    if high_low_chains.direction == 'ascending':
        # If the chain is ascending, it starts at a valley and ends on a peak and vice versa. The ending peak/valley would be the
        # minimum of the two chain end lengths, as that is where the pattern is either ended by the higher highs coming to an end or
        # the lower lows
        peaks_df: pd.DataFrame = zigzag_df[zigzag_df.pivot_type == 'peak']
        total_chain_end_index = min(high_low_chains.low_chain_length, high_low_chains.high_chain_length)

        # The reset_index function is used because the chain length is calculated from the flattened form of a pandas Series in
        # find_longest_chain without inherited indices from the parent dataframe.

        chain_end_peak_pair_df_index = peaks_df.reset_index(drop=True).iloc[total_chain_end_index].pair_df_index

        return {
            "start_index": chain_start_pair_df_index,
            "end_index": chain_end_peak_pair_df_index,
            "is_simplifying": high_low_chains.is_simplifying
        }

    elif high_low_chains.direction == 'descending':
        valleys_df: pd.DataFrame = zigzag_df[zigzag_df.pivot_type == 'valley']
        total_chain_end_index = min(high_low_chains.low_chain_length, high_low_chains.high_chain_length)

        chain_end_valley_pair_df_index = valleys_df.reset_index(drop=True).iloc[total_chain_end_index].pair_df_index

        return {
            "start_index": chain_start_pair_df_index,
            "end_index": chain_end_valley_pair_df_index,
            "is_simplifying": high_low_chains.is_simplifying
        }


def generate_h_o_zigzag(zigzag_df: pd.DataFrame) -> pd.DataFrame:
    """
        Function to generate a higher order zigzag from a given zigzag DataFrame.
        It iterates through the DataFrame, starting from the first pivot, and finds chains of ascending or descending pivots.
        Then it turns those chains into straight zigzags and appends them to a list.
        It stops when it reaches the last pivot in the DataFrame.

        Parameters:
        zigzag_df (pd.DataFrame): The DataFrame containing the zigzag data, most importantly pivot_value, pivot_type and pair_df_index columns

        Returns:
        pd.DataFrame: A list of tuples, where each tuple contains the start and end indices of a chain.
    """

    # Initialize an empty list to store the chains
    h_o_zigzag_indices: list[int] = []
    pbos_indices: list[int] = []

    # Get the maximum and minimum pair_df_index in the DataFrame, used as the start and end to the while loop
    max_pair_df_index: int = zigzag_df.pair_df_index.max()
    current_pair_df_index: int = zigzag_df.pair_df_index.min()

    # Loop until the current pair_df_index reaches the maximum
    while current_pair_df_index != max_pair_df_index:
        # Get the pivot at the current pair_df_index
        pivot = Pivot.create(zigzag_df[zigzag_df.pair_df_index == current_pair_df_index].iloc[0])

        # If the pivot is a peak, find descending chains; if it's a valley, find ascending chains
        # Since an ascending chain can only start at a valley, and a descending chain can only start at a peak
        if pivot.pivot_type == 'peak':
            chains = find_descending_chains(zigzag_df, pivot.pair_df_index)
        else:
            chains = find_ascending_chains(zigzag_df, pivot.pair_df_index)

        simplified_leg = simplify_chain(zigzag_df, chains)
        # Simplify the unidirectional chain and add it to the h_o_zigzag_indices. If the list is empty, also append the starting index of the leg
        if len(h_o_zigzag_indices) == 0:
            h_o_zigzag_indices.append(simplified_leg["start_index"])

        h_o_zigzag_indices.append(simplified_leg["end_index"])
        if simplified_leg["is_simplifying"]:
            pbos_indices.append(simplified_leg["end_index"])

        # Update the current pair_df_index to the end index ([1]) of the last chain ([-1])
        current_pair_df_index = simplified_leg["end_index"]

    # Convert the list of pair_df_indices to a pandas dataframe containing the values, types and times of the pivots
    h_o_zigzag_df: pd.DataFrame = zigzag_df[zigzag_df.pair_df_index.isin(h_o_zigzag_indices)].copy()
    h_o_zigzag_df.loc[:, "is_pbos"] = h_o_zigzag_df.pair_df_index.isin(pbos_indices)
    return h_o_zigzag_df


def is_pbos_confirmed(bos_value, bos_type, confirmation_check_window) -> bool:
    """
    Function to check if a potential breakout or breakdown (PBOS) is confirmed.

    A PBOS is confirmed if there are any candles in the confirmation check window that break through the PBOS value.
    If the PBOS is a peak, a confirmation is a close price greater than the PBOS value.
    If the PBOS is a valley, a confirmation is a close price less than the PBOS value.

    Parameters:
    bos_value (float): The value of the potential breakout or breakdown (PBOS).
    bos_type (str): The type of the PBOS. Can be 'peak' or 'valley'.
    confirmation_check_window (pd.DataFrame): The DataFrame containing the candlestick data to check for confirmation.

    Returns:
    bool: True if the PBOS is confirmed, False otherwise.
    """
    if bos_type == "peak":
        breaking_candles = confirmation_check_window.loc[confirmation_check_window.close > bos_value]
    else:
        breaking_candles = confirmation_check_window.loc[confirmation_check_window.close < bos_value]

    if len(breaking_candles) > 0:
        return True
    return False


def find_confirmed_boss(pbos_df: pd.DataFrame, pair_df: pd.DataFrame) -> pd.DataFrame:
    """
    Function to find confirmed breakouts or breakdowns (BOS) from a DataFrame of potential BOS (PBOS).

    A BOS is confirmed if there are any candles in the confirmation check window that break through the PBOS value.
    The function iterates over the PBOS DataFrame and checks each PBOS for confirmation.
    If a PBOS is confirmed, its index is added to a list of confirmed BOS indices.
    Finally, a DataFrame of confirmed BOS is returned.

    Parameters:
    pbos_df (pd.DataFrame): The DataFrame containing the potential breakouts or breakdowns (PBOS).
    pair_df (pd.DataFrame): The DataFrame containing the candlestick data to check for confirmation.

    Returns:
    pd.DataFrame: A DataFrame containing the confirmed breakouts or breakdowns (BOS).
    """

    confirmed_bos_indices: list[int] = []
    for pbos_row in pbos_df.itertuples():
        confirmation_check_window = pair_df.iloc[pbos_row.pair_df_index + 1:]
        if is_pbos_confirmed(pbos_row.pivot_value, pbos_row.pivot_type, confirmation_check_window):
            confirmed_bos_indices.append(pbos_row.pair_df_index)

    bos_df = pbos_df.loc[pbos_df.pair_df_index.isin(confirmed_bos_indices)]
    return bos_df


def find_lpls(bos_df: pd.DataFrame, zigzag_df: pd.DataFrame) -> pd.DataFrame:
    """
    Function to find the last pivot liquidities (LPLs) in a zigzag DataFrame.

    The function identifies the indices of the confirmed breakouts or breakdowns (BOS) in the zigzag DataFrame.
    It then finds the indices of the pivots that are immediately before these BOS indices, which are the LPLs.
    Finally, it returns a DataFrame containing the LPLs.

    Parameters:
    bos_df (pd.DataFrame): The DataFrame containing the confirmed breakouts or breakdowns (BOS).
    zigzag_df (pd.DataFrame): The DataFrame containing the zigzag data.

    Returns:
    pd.DataFrame: A DataFrame containing the last pivot liquidities (LPLs).
    """

    bos_indices = bos_df.pair_df_index
    lpl_indices = [index - 1 for index in zigzag_df.loc[zigzag_df.pair_df_index.isin(bos_indices)].index]
    return zigzag_df.iloc[lpl_indices]


def find_lplbs(bos_df: pd.DataFrame, zigzag_df: pd.DataFrame) -> pd.DataFrame:
    """
    Function to find the last pivot liquidity breakouts (LPLBs) in a zigzag DataFrame.

    The function identifies the indices of the confirmed breakouts or breakdowns (BOS) in the zigzag DataFrame.
    It then finds the indices of the pivots that are immediately after these BOS indices, which are the LPLBs.
    Finally, it returns a DataFrame containing the LPLBs.

    Parameters:
    bos_df (pd.DataFrame): The DataFrame containing the confirmed breakouts or breakdowns (BOS).
    zigzag_df (pd.DataFrame): The DataFrame containing the zigzag data.

    Returns:
    pd.DataFrame: A DataFrame containing the last pivot liquidity breakouts (LPLBs).
    """

    bos_indices = bos_df.pair_df_index
    lplb_indices = [index + 1 for index in zigzag_df.loc[zigzag_df.pair_df_index.isin(bos_indices)].index]
    return zigzag_df.iloc[lplb_indices]
