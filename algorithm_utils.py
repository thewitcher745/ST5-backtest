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
        Tuple[int, int]: The start and end indices of the chain.
    """

    # If the selected start_index isn't a valley, throw an error
    if zigzag_df[zigzag_df.pair_df_index == start_pair_df_index].iloc[0].pivot_type != 'valley':
        raise ValueError("The start index must be a valley.")

    # The slice of the zigzag_df dataframe that needs to be searched
    search_window: pd.DataFrame = zigzag_df[zigzag_df.pair_df_index >= start_pair_df_index]

    search_window_peaks: pd.Series = search_window[search_window.pivot_type == 'peak'].pivot_value
    search_window_valleys: pd.Series = search_window[search_window.pivot_type == 'valley'].pivot_value

    higher_low_chain_length: int = find_longest_chain(search_window_valleys, 'ascending')
    higher_high_chain_length: int = find_longest_chain(search_window_peaks, 'ascending')

    return OneDChain.create(higher_low_chain_length, higher_high_chain_length, start_pair_df_index, 'ascending')


def find_descending_chains(zigzag_df: pd.DataFrame, start_pair_df_index: int) -> OneDChain:
    """
        Function to find the longest chain of lower lows in a zigzag DataFrame.
        It starts from a given index and goes forward, looking for peaks with descending pivot values.
        It stops once it finds a higher high than the previous.

        Parameters:
        zigzag_df (pd.DataFrame): The DataFrame containing the zigzag data.
        start_pair_df_index (int): The index to start the search from.

        Returns:
        Tuple[int, int]: The start and end indices of the chain.
    """

    # If the selected start_index isn't a peak, throw an error
    if zigzag_df[zigzag_df.pair_df_index == start_pair_df_index].iloc[0].pivot_type != 'peak':
        raise ValueError("The start index must be a peak.")

    # The slice of the zigzag_df dataframe that needs to be searched
    search_window: pd.DataFrame = zigzag_df[zigzag_df.pair_df_index >= start_pair_df_index]

    search_window_peaks: pd.Series = search_window[search_window.pivot_type == 'peak'].pivot_value
    search_window_valleys: pd.Series = search_window[search_window.pivot_type == 'valley'].pivot_value

    lower_low_chain_length: int = find_longest_chain(search_window_valleys, 'descending')
    lower_high_chain_length: int = find_longest_chain(search_window_peaks, 'descending')

    return OneDChain.create(lower_low_chain_length, lower_high_chain_length, start_pair_df_index, 'descending')


def simplify_chain(input_zigzag_df: pd.DataFrame,
                   high_low_chains: OneDChain) -> tuple:
    """
        Function that takes a chain of higher low/higher highs (ascending), or lower low/lower highs (descending) and
        returns a dict object with instructions on forming a higher order zigzag with it.

        Parameters:
        zigzag_df (pd.DataFrame): The DataFrame containing the zigzag data.
        high_low_chains (OneDChain): A OneDChain object containing the chain lengths and the direction.

        Returns:
        tuple: A tuple object containing the pair_df_indices of the start and end of the chain
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

        return chain_start_pair_df_index, chain_end_peak_pair_df_index

    elif high_low_chains.direction == 'descending':
        valleys_df: pd.DataFrame = zigzag_df[zigzag_df.pivot_type == 'valley']
        total_chain_end_index = min(high_low_chains.low_chain_length, high_low_chains.high_chain_length)

        chain_end_valley_pair_df_index = valleys_df.reset_index(drop=True).iloc[total_chain_end_index].pair_df_index

        return chain_start_pair_df_index, chain_end_valley_pair_df_index


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
    h_o_zigzag_indices = []

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
            h_o_zigzag_indices.append(simplified_leg[0])

        h_o_zigzag_indices.append(simplified_leg[1])

        # Update the current pair_df_index to the end index ([1]) of the last chain ([-1])
        current_pair_df_index = simplified_leg[1]

    # Convert the list of pair_df_indices to a pandas dataframe containing the values, types and times of the pivots
    h_o_zigzag_df: pd.DataFrame = zigzag_df[zigzag_df.pair_df_index.isin(h_o_zigzag_indices)]

    return h_o_zigzag_df

