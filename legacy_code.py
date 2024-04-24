import pandas as pd

from datatypes import *


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