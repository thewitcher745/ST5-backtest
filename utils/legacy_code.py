from algo_code.datatypes import *


# def zigzag_deprecated(input_df: pd.DataFrame) -> pd.DataFrame:
#     """
#     Deprecated function to identify turning points in a candlestick chart.
#     It compares each candle to its previous candle to determine if the candle is a pivot.
#     This function has been replaced with the zigzag() function.
#
#     Parameters:
#     input_df (pd.DataFrame): The input DataFrame containing the candlestick data.
#
#     Returns:
#     pd.DataFrame: A DataFrame containing the identified turning points.
#     """
#
#     pair_df = input_df.copy()
#     # Identify peaks in the high prices and valleys in low prices
#     pair_df['peak'] = pair_df.high[(pair_df.high.shift(1) < pair_df.high)]
#     pair_df['valley'] = pair_df.low[(pair_df.low.shift(1) > pair_df.low)]
#
#     peaks_df = pair_df[pair_df['peak'].notnull()][['time', 'peak']]
#     valleys_df = pair_df[pair_df['valley'].notnull()][['time', 'valley']]
#
#     # Concatenate the peaks and valleys DataFrames and sort by time
#     zigzag_df = pd.concat([peaks_df, valleys_df]).sort_values(by='time')
#
#     # Combine the peak and valley values into a single column, 'pivot_value'
#     # If a peak value is not null, it is used; otherwise, the valley value is used
#     zigzag_df['pivot_value'] = zigzag_df['peak'].combine_first(zigzag_df['valley'])
#
#     # Create a 'pivot_type' column that indicates whether the pivot_value is a peak or a valley or both
#     # zigzag_df['pivot_type'] = np.where(zigzag_df['valley'].notnull() & zigzag_df['peak'].notnull(), 'both', '')
#     zigzag_df['pivot_type'] = np.where(zigzag_df['valley'].isnull(), 'peak', 'valley')
#
#     # Continue to remove rows with duplicate 'time' values and rows where the pivot_type is the same as the pivot_type in the next row, until no more
#     # such rows exist
#     while True:
#         # Remove rows where the pivot_type is the same as the pivot_type in the next row
#         zigzag_df = zigzag_df[zigzag_df['pivot_type'].shift(-1) != zigzag_df['pivot_type']]
#         # Remove consecutive rows with the same 'time' value
#         zigzag_df_before = zigzag_df.copy()
#         zigzag_df = zigzag_df.loc[~zigzag_df['time'].duplicated(keep='first')]
#
#         # If the DataFrame did not change in the last iteration, break the loop
#         if zigzag_df.equals(zigzag_df_before):
#             break
#
#     zigzag_df.drop(['peak', 'valley'], axis=1, inplace=True)
#     first_row_series: pd.Series = pair_df.iloc[0].copy()
#     first_row_series['pivot_value'] = first_row_series['high'] if zigzag_df.iloc[0].pivot_type == 'valley' else first_row_series['low']
#     first_row_series['pivot_type'] = 'peak' if zigzag_df.iloc[0].pivot_type == 'valley' else 'valley'
#
#     first_row_df: pd.DataFrame = first_row_series.to_frame().T
#     first_row_df.drop(['open', 'high', 'low', 'close', 'peak', 'valley'], axis=1, inplace=True)
#
#     zigzag_df: pd.DataFrame = pd.concat([first_row_df, zigzag_df], axis=0)
#
#     # Reset the index column, and rename it to pair_df_index
#     zigzag_df.reset_index(inplace=True)
#     zigzag_df.rename(columns={'index': 'pair_df_index'}, inplace=True)
#
#     return zigzag_df


# def find_patterns(zigzag_df: pd.DataFrame) -> dict[str, list[PatternTupleType]]:
#     """
#     Function to find patterns from the zigzag DataFrame.
#     Bullish pattern: A registered low and the high after it, then look for a higher high, without a lower low in between
#     Bearish pattern: A registered high and the low after it, then look for a lower low, without a higher high in between
#
#     Parameters:
#     zigzag_df (pd.DataFrame): The DataFrame containing the zigzag data.
#
#     Returns:
#     pd.DataFrame: A DataFrame containing the identified patterns.
#     """
#
#     patterns: dict[str, list[PatternTupleType]] = {"bullish": [],
#                                                    "bearish": []}
#
#     # Form the bullish patterns
#     valleys_df: pd.DataFrame = zigzag_df[zigzag_df.pivot_type == 'valley']
#     peaks_df: pd.DataFrame = zigzag_df[(zigzag_df.pivot_type == 'peak') & (zigzag_df.pair_df_index > valleys_df.iloc[0].pair_df_index)]
#
#     for peak in peaks_df.itertuples():
#         try:
#             # Find the first higher peak after the current one
#             first_higher_peak: pd.Series = peaks_df[(peaks_df.pair_df_index > peak.pair_df_index) & (peaks_df.pivot_value > peak.pivot_value)].iloc[0]
#
#             # Find the valley right before the peak
#             starting_leg_valley: pd.Series = zigzag_df.iloc[peak.Index - 1]
#
#             # Check if between the two peaks exists a lower low than the valley right before the peak
#             if zigzag_df[
#                 (zigzag_df.time > peak.time) & (zigzag_df.time < first_higher_peak.time)].pivot_value.min() > starting_leg_valley.pivot_value:
#                 patterns["bullish"].append(create_pattern_tuple(starting_leg_valley.pair_df_index, first_higher_peak.pair_df_index, 'bullish'))
#
#         except IndexError:
#             continue
#
#         # Form the bearish patterns
#     peaks_df: pd.DataFrame = zigzag_df[zigzag_df.pivot_type == 'peak']
#     valleys_df: pd.DataFrame = zigzag_df[(zigzag_df.pivot_type == 'valley') & (zigzag_df.pair_df_index > peaks_df.iloc[0].pair_df_index)]
#
#     for valley in valleys_df.itertuples():
#         try:
#             # Find the first lower valley after the current one
#             first_lower_valley: pd.Series = \
#                 valleys_df[(valleys_df.pair_df_index > valley.pair_df_index) & (valleys_df.pivot_value < valley.pivot_value)].iloc[0]
#
#             # Find the peak right before the valley
#             starting_leg_peak: pd.Series = zigzag_df.iloc[valley.Index - 1]
#
#             # Check if between the two valleys exists a higher high than the peak right before the valley
#             if zigzag_df[
#                 (zigzag_df.time > valley.time) & (zigzag_df.time < first_lower_valley.time)].pivot_value.max() < starting_leg_peak.pivot_value:
#                 patterns["bearish"].append(create_pattern_tuple(starting_leg_peak.pair_df_index, first_lower_valley.pair_df_index, 'bearish'))
#
#         except IndexError:
#             continue
#
#     return patterns


# def merge_overlapping_patterns(patterns: dict[str, list[PatternTupleType]]) -> dict[str, list[PatternTupleType]]:
#     merged_patterns = {
#         'bullish': [],
#         'bearish': []
#     }
#
#     for pattern_type in patterns.keys():
#         # Set the start of the first merged pattern to the starting index of the first found pattern of
#         # type pattern_type
#         start_index = patterns[pattern_type][0].start_index
#         end_index = patterns[pattern_type][0].end_index
#
#         for i, pattern in enumerate(patterns[pattern_type][1:]):
#             # If the current pattern is completely WITHIN the most recent start and end, go to the next pattern, as this one doesn't matter
#             if pattern.start_index <= end_index and pattern.end_index <= end_index:
#                 continue
#
#             # If the current pattern has a start_index higher than the one from the most recent one, extend
#             # the end_index to include the current pattern
#             if pattern.start_index <= end_index:
#                 end_index = pattern.end_index
#
#             # If the current pattern is disconnected from the previously found range, add the previous range to the merged patterns list and reset the
#             # most recent start and end index.
#             else:
#                 merged_patterns[pattern_type].append(create_pattern_tuple(start_index, end_index, pattern_type))
#                 start_index = pattern.start_index
#                 end_index = pattern.end_index
#
#         merged_patterns[pattern_type].append(create_pattern_tuple(start_index, end_index, pattern_type))
#
#     return merged_patterns


# def calculate_higher_order_zigzag(zigzag_df: pd.DataFrame, patterns: dict[str, list[PatternTupleType]]):
#     """
#     Function to calculate a higher order zigzag based on the identified patterns.
#     It takes the zigzag DataFrame and the patterns dictionary as input and returns a new DataFrame
#     with the higher order zigzag data.
#
#     Parameters:
#     zigzag_df (pd.DataFrame): The DataFrame containing the zigzag data.
#     patterns (dict[str, list]): The dictionary containing the identified patterns.
#
#     Returns:
#     pd.DataFrame: A DataFrame containing the higher order zigzag data.
#     """
#
#     higher_order_zigzag_df = zigzag_df.copy()
#
#     # Identify continuous bullish or bearish zones in patterns
#     merged_patterns = merge_overlapping_patterns(patterns)
#
#     return merged_patterns


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
            "is_forming_pbos": high_low_chains.is_forming_pbos
        }

    elif high_low_chains.direction == 'descending':
        valleys_df: pd.DataFrame = zigzag_df[zigzag_df.pivot_type == 'valley']
        total_chain_end_index = min(high_low_chains.low_chain_length, high_low_chains.high_chain_length)

        chain_end_valley_pair_df_index = valleys_df.reset_index(drop=True).iloc[total_chain_end_index].pair_df_index

        return {
            "start_index": chain_start_pair_df_index,
            "end_index": chain_end_valley_pair_df_index,
            "is_forming_pbos": high_low_chains.is_forming_pbos
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
        if simplified_leg["is_forming_pbos"]:
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

# def calculate_FVG_from_candles(candles: ) -> Tuple:

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
