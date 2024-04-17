import pandas as pd
import numpy as np
from typing import *

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



