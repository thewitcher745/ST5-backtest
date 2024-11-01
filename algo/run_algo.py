from dotenv import set_key
import os
import pandas as pd
from openpyxl import load_workbook

import utils.general_utils as gu
from utils.config import Config
from algo.algorithm_utils import Algo, create_filtered_pair_df_with_corrected_starting_point
from utils.logger import LoggerSingleton


def run_algo(pair_name: str, timeframe: str):
    higher_timeframe = gu.find_higher_timeframe(timeframe)

    # --------------------------------------------------------
    # Set up the logger module

    # Initialize the logger
    positions_logger = LoggerSingleton("positions").get_logger()

    testing_length = 0
    start_index = -8 * testing_length - 1

    original_pair_df: pd.DataFrame = gu.load_local_data(pair_name=pair_name, timeframe=timeframe)

    # --------------------------------------------------------
    # Set up the pair data dataframe for the algo
    if testing_length > 0:
        pair_df: pd.DataFrame = original_pair_df.iloc[start_index:start_index + testing_length].reset_index(drop=True)
    else:
        pair_df: pd.DataFrame = original_pair_df

    # --------------------------------------------------------
    # Higher timeframe data for determining starting point
    htf_pair_df: pd.DataFrame = gu.load_higher_tf_data(pair_name=pair_name, timeframe=higher_timeframe)

    print("LTF length is", len(pair_df))
    print("HTF length is", len(htf_pair_df))

    gu.reset_logs()

    print("Original pair_df starting point is at time", pair_df.iloc[0].time)

    # --------------------------------------------------------
    # Calculate the starting point and higher order zigzag
    algo = Algo(pair_df, pair_name, allowed_verbosity=0)
    initial_data_start_time = algo.pair_df.iloc[0].time

    corrected_pair_df = create_filtered_pair_df_with_corrected_starting_point(htf_pair_df, initial_data_start_time,
                                                                              original_pair_df, timeframe, higher_timeframe)

    print("Updated pair_df starting point to", corrected_pair_df.iloc[0].time)

    algo = Algo(corrected_pair_df, "BTCUSDT", allowed_verbosity=0)

    algo.init_zigzag(last_pivot_type="valley", last_pivot_candle_pdi=0)
    h_o_starting_point: int = algo.zigzag_df.iloc[0].pdi

    algo.calc_h_o_zigzag(h_o_starting_point)

    # --------------------------------------------------------
    # Use the segments from the algo object to calculate position entries and exits
    for segment in algo.segments:
        print(segment)
        segment.filter_candlestick_range(algo)
        segment.find_order_blocks(algo)
        for ob in segment.ob_list:
            position_entry_pdi = ob.position.find_entry_within_segment(segment)
            if position_entry_pdi is not None:
                ob.position.enter(position_entry_pdi)

        for ob in segment.ob_list:
            if ob.position.status != "ENTERED":
                continue

            exit_check_candles: pd.DataFrame = algo.pair_df.iloc[ob.position.entry_pdi:]
            candle_sentiments: pd.Series = exit_check_candles.apply(lambda candle: ob.position.detect_candle_sentiment(candle), axis=1)

            # Find the first instance where a STOPLOSS or FULL_TARGET happens in the candle_sentiments Series.
            mask = candle_sentiments.apply(lambda row: row[0] == "FULL_TARGET" or row[0] == "STOPLOSS")
            exit_index = mask.idxmax() if mask.any() else None

            if exit_index is not None:
                exit_code = candle_sentiments.loc[exit_index][0]
                # If the first exit event to happen is a full target, that means we have the maximum profit, and we exit the position.
                if exit_code == "FULL_TARGET":
                    # This snippet is used to register each target individually, before the full target event happens. This is used for validating target
                    # hits and troubleshoot bugs later using the ob.position.target_hit_pdis property which stores the PDI's of the candles that hit
                    # the targets.
                    targets_before_full_target_cum_max: pd.Series = candle_sentiments.loc[:exit_index].apply(
                        lambda row: row[1] if row[0] == "TARGET" else 0).cummax()
                    max_target_changes: pd.Series = targets_before_full_target_cum_max[targets_before_full_target_cum_max.diff() > 0]

                    # Register the targets before the final full target
                    for target_id, target_hit_pdi in zip(max_target_changes.values, max_target_changes.index):
                        ob.position.register_target(target_id, target_hit_pdi)

                    # Register the final, full target
                    ob.position.register_target(len(ob.position.target_list), exit_index)

                    # Exit the position
                    ob.position.exit(exit_code="FULL_TARGET", exit_pdi=exit_index)


                # If the first exit event to happen is a stoploss, we must check what the highest target reached was.
                elif exit_code == "STOPLOSS":
                    # This snippet is used to register each target individually, before the stoploss event happens.
                    targets_before_stoploss_cum_max: pd.Series = candle_sentiments.loc[:exit_index].apply(
                        lambda row: row[1] if row[0] == "TARGET" else 0).cummax()
                    max_target_changes: pd.Series = targets_before_stoploss_cum_max[targets_before_stoploss_cum_max.diff() > 0]

                    # Register the targets before the final full target
                    for target_id, target_hit_pdi in zip(max_target_changes.values, max_target_changes.index):
                        ob.position.register_target(target_id, target_hit_pdi)

                    ob.position.exit(exit_code="STOPLOSS", exit_pdi=exit_index)

            # print(ob)
            # print("TARGETS", ob.position.target_list)
            # print("STOPLOSS", ob.position.stoploss)
            # print("FIRST EXIT INDEX", exit_index)
            # print(candle_sentiments.iloc[exit_index])

    # --------------------------------------------------------
    # Generate the Excel file report
    def generate_positions_excel(output_file):
        # List to store position data
        positions_data = []

        # Iterate through each segment in the Algo object
        for segment in algo.segments:
            # Iterate through each order block in the segment
            for ob in segment.ob_list:
                # Extract the position information
                position = ob.position
                if len(position.target_hit_pdis) == 1:
                    target_hit_times_list = [algo.convert_pdis_to_times(position.target_hit_pdis)]

                else:
                    target_hit_times_list = algo.convert_pdis_to_times(position.target_hit_pdis)

                position_data = {
                    'Pair name': pair_name,
                    'Position ID': position.parent_ob.id,
                    'Segment ID': segment.id,
                    'Status': position.status,
                    'Net profit': position.net_profit,
                    # 'Has been replaced?': position.parent_ob.has_been_replaced,
                    'Ranking within segment': position.parent_ob.ranking_within_segment,
                    'Quantity': position.qty,
                    'Entry time': algo.convert_pdis_to_times(position.entry_pdi),
                    'Exit time': algo.convert_pdis_to_times(position.exit_pdi),
                    'Target hit times': [time.strftime("%Y-%m-%d %X") for time in target_hit_times_list],
                    'Type': position.type,
                    'Entry price': position.entry_price,
                    'Stoploss': position.stoploss,
                    'Target list': [round(float(target), 6) for target in position.target_list]
                }
                # Append the position data to the list
                positions_data.append(position_data)

        # Convert the list of dictionaries to a pandas DataFrame
        df = pd.DataFrame(positions_data)

        # Write the DataFrame to an Excel file
        df.to_excel(output_file, index=False)

        # Adjust column widths
        adjust_column_widths(output_file)

    def adjust_column_widths(output_file):
        # Load the workbook and select the active worksheet
        wb = load_workbook(output_file)
        ws = wb.active

        # Adjust column widths
        for col in ws.columns:
            max_length = 0
            column = col[0].column_letter  # Get the column name
            for cell in col:
                try:
                    if len(str(cell.value)) > max_length:
                        max_length = len(cell.value)
                except:
                    pass
            adjusted_width = (max_length + 2)
            ws.column_dimensions[column].width = adjusted_width

        # Save the workbook
        wb.save(output_file)

    generate_positions_excel(f'./reports/positions-{pair_name}.xlsx')

    # --------------------------------------------------------
    # Generate a summary of the data in the Excel file
    positions_data_df = pd.read_excel(f'./reports/positions-{pair_name}.xlsx')

    total_positions_found = len(positions_data_df)
    entered_positions = len(positions_data_df[positions_data_df["Status"] != "ACTIVE"])
    not_exit_positions = len(positions_data_df[positions_data_df["Status"] == "ENTERED"])

    positive_positions = len(positions_data_df[positions_data_df["Net profit"] > 0])
    negative_positions = len(positions_data_df[positions_data_df["Net profit"] < 0])
    winrate = positive_positions / entered_positions

    full_target_positions = len(positions_data_df[positions_data_df["Status"].str.startswith("FULL_TARGET")])
    no_target_positions = len(positions_data_df[positions_data_df["Status"].str.startswith("STOPLOSS")])
    target_1_positions = len(positions_data_df[positions_data_df["Status"].str.startswith("TARGET_1")])
    target_2_positions = len(positions_data_df[positions_data_df["Status"].str.startswith("TARGET_2")])

    long_positions = len(positions_data_df[positions_data_df["Type"] == "short"])
    short_positions = len(positions_data_df[positions_data_df["Type"] == "short"])

    total_profit = positions_data_df["Net profit"].sum()
    total_negative = positions_data_df[positions_data_df["Net profit"] < 0]["Net profit"].sum()
    total_positive = positions_data_df[positions_data_df["Net profit"] > 0]["Net profit"].sum()

    # Print the report
    print(pair_name, "\n")
    print("Over 2 years (Oct 12th 2022 - Oct 12th 2024)\n")
    print(f"{total_positions_found} total positions found")
    print(f"{entered_positions} total positions entered - {not_exit_positions if not_exit_positions > 0 else 'no'} positions without exit\n")
    print(f"{positive_positions} positive PnL")
    print(f"{negative_positions} negative PnL")
    print(f"({winrate:.0%} winrate)\n")
    print(f"{full_target_positions} full targets achieved")
    print(f"{no_target_positions} no target hit")
    print(f"{target_1_positions} target_1 achieved")
    print(f"{target_2_positions} target_2 achieved\n")
    print(f"{long_positions} long positions")
    print(f"{short_positions} short positions\n")
    print(f"{total_profit:.2f} total profit")
    print(f"{total_negative:.2f} total negative")
    print(f"{total_positive:.2f} total positive")
