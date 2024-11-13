from logging import Logger
import pandas as pd

import constants
from algo_code.order_block import OrderBlock
from utils.logger import LoggerSingleton

# noinspection PyTypeChecker
positions_logger: Logger = None


class Segment:
    """
    A segment is a series of candles during which the order blocks specified in Segment.ob_list do not change, so it would be safe to check for
    entry to these order blocks WITHIN this segment. After the expiration candle of the segment, indicated by Segment.end_pdi, entry to the order
    blocks isn't permitted, and we move on to the next segment.
    """

    def __init__(self, start_pdi: int,
                 end_pdi: int,
                 ob_leg_start_pdi: int,
                 ob_leg_end_pdi: int,
                 top_price: float,
                 bottom_price: float,
                 ob_formation_start_pdi: int,
                 broken_lpl_pdi: int,
                 type: str,
                 formation_method: str = "bos"):

        global positions_logger

        if positions_logger is None:
            positions_logger = LoggerSingleton.get_logger("positions")

        self.end_pdi = end_pdi
        self.start_pdi = start_pdi
        self.ob_leg_start_pdi = ob_leg_start_pdi
        self.ob_leg_end_pdi = ob_leg_end_pdi
        self.top_price = top_price
        self.bottom_price = bottom_price
        self.ob_formation_start_pdi = ob_formation_start_pdi
        self.broken_lpl_pdi = broken_lpl_pdi
        self.type = type
        self.formation_method = formation_method

        if constants.logs_format != "time":
            self.id = f"SEG/{self.formation_method}/{self.start_pdi}"

        self.ob_list: list[OrderBlock] = []
        self.pair_df: pd.DataFrame = pd.DataFrame()

    def __repr__(self):
        return f"{self.type.capitalize()} segment starting at {self.start_pdi} ending at {self.end_pdi} OB formation at {self.ob_formation_start_pdi}"

    def filter_candlestick_range(self, algo):
        """
        This method defines the range of pair_df which is used to find box entries. This is useful for checking order block entries. This section is
        defined as the candles between the OB formation start and the end of the segment, inclusive. The inclusivity is important because in the code
        a segment's bounds are defined as such.
        """
        self.pair_df = algo.pair_df.iloc[self.ob_formation_start_pdi:self.end_pdi + 1]

        if constants.logs_format == "time":
            self.id = f"SEG/{self.formation_method}/{algo.pair_df.loc[self.start_pdi].time}"

    # noinspection LongLine
    def find_order_blocks(self, algo):
        """
        This method identifies the order blocks specific to the segment by taking the entire Algo object as an input, as many of its properties and
        methods are useful here, and it would be redundant to pass around multiple inputs and methods. This method populates the Segment.ob_list
        object with a list of order blocks that are only valid within this segment.

        Args:
            algo: The Algo object
        """
        positions_logger.debug(f"Finding order blocks for segment {self.id}")

        # For testing and safety purposes, the ob_list property is reset.
        self.ob_list = []

        # base_candle_type is the type of the pivot that is used to filter the zigzag_df dataframe for the correct pivot type. In ascending segments
        # (patterns) the type is valley, and in descending segments it's peak.
        base_pivot_type = "valley" if self.type == "ascending" else "peak"

        # This variable is used to keep track of how many valid order blocks have been found. It is then assigned to each order block within the
        # segment, so it's ranking in the segment is recorded.
        valid_ob_counter = 0

        # Filter pivots of the correct type (valley for ascending, peak for descending) and pivots that are within the first leg. Also omit the pivots
        # that have a higher PDI than the broken LPL PDI, meaning the boxes that form above the broken LPL in ascending and below the LPL in
        # descending

        for pivot in algo.zigzag_df[(algo.zigzag_df.pivot_type == base_pivot_type) &
                                    (self.ob_leg_start_pdi <= algo.zigzag_df.pdi) &
                                    (algo.zigzag_df.pdi < self.broken_lpl_pdi)].itertuples():

            if constants.logs_format == "time":
                positions_logger.debug(f"\tFinding OBs for lower order leg starting at {algo.convert_pdis_to_times(pivot.pdi)}")
            else:
                positions_logger.debug(f"\tFinding OBs for lower order leg starting at {pivot.pdi}")

            # This try-except block is used to determine the window that is used for finding replacement order blocks in the chart. Currently, the
            # window spans from the very first base candle (the pivot found using the outer loop) to the lower-order pivot immediately after it.
            # The except clause catches the error in case we reach the end of the chart and no more next pivots exist, in which case the end of the
            # search window is set to the last candle of the whole dataset.
            try:
                next_pivot_pdi = algo.find_relative_pivot(pivot.pdi, 1)
                replacement_ob_threshold_pdi = next_pivot_pdi
            except IndexError:
                replacement_ob_threshold_pdi = algo.pair_df.last_valid_index()

            if constants.logs_format == "time":
                positions_logger.debug(f"\tReplacement OB search threshold set up to {algo.convert_pdis_to_times(replacement_ob_threshold_pdi)}")
            else:
                positions_logger.debug(f"\tReplacement OB search threshold set up to {replacement_ob_threshold_pdi}")

            # times_moved indicates the times the algorithm had to move the base candle to find a replacement order block.
            times_moved = 0

            # The stoploss is set at the pivot value of the INITIAL box that was found, since that's the box which has the liquidity. This value is
            # passed to the OB instantiation line as the stoploss value, which in turn goes to the Position attribute within it.
            initial_pivot_candle_liquidity = pivot.pivot_value

            for base_candle_pdi in range(pivot.pdi, replacement_ob_threshold_pdi):
                base_candle = algo.pair_df.iloc[base_candle_pdi]
                ob = OrderBlock(base_candle=base_candle,
                                icl=initial_pivot_candle_liquidity,
                                ob_type="long" if base_pivot_type == "valley" else "short")

                ob.register_exit_candle(algo.pair_df, self.ob_formation_start_pdi)

                if constants.logs_format == "time":
                    positions_logger.debug(f"\t\tInvestigating base candle at {algo.convert_pdis_to_times(base_candle_pdi)}")
                else:
                    positions_logger.debug(f"\t\tInvestigating base candle at {base_candle_pdi}")

                # The reentry window dataframe is used to check whether the price returned to the box in the span between the exit candle and the LPL
                # breaking candle. This is checked using the check_reentry_condition() method of the OrderBlock object. The reentry dataframe is
                # passed as an argument to the method.
                if ob.price_exit_index is not None:
                    reentry_check_window: pd.DataFrame = algo.pair_df.iloc[ob.price_exit_index + 1:self.ob_formation_start_pdi]

                    # Log the exit candle location
                    if constants.logs_format == "time":
                        positions_logger.debug(
                            f"\t\t\tExit candle found at {algo.convert_pdis_to_times(ob.price_exit_index)}")
                    else:
                        positions_logger.debug(f"\t\t\tExit candle found at {ob.price_exit_index}")

                    if constants.logs_format == "time":
                        positions_logger.debug(
                            f"\t\t\tReentry check window set up from {algo.convert_pdis_to_times(ob.price_exit_index + 1)} to {algo.convert_pdis_to_times(self.ob_formation_start_pdi - 1)}")
                    else:
                        positions_logger.debug(
                            f"\t\t\tReentry check window set up from {ob.price_exit_index + 1} to {self.ob_formation_start_pdi - 1}")

                # This else statement is implemented to account for boxes which don't have an exit candle which opens inside and closes outside of
                # them, automatically making them invalid and prompting considering another replacement.
                else:
                    positions_logger.debug("\t\t\tNo exit candle found. OB is invalid, looking for a replacement further in time.")
                    continue

                # This check ensures that the order block being processed is totally valid to be used AFTER the formation of the pattern, that means
                # that the order block has either A: had no reentry at all or B: has had its reentry after the formation of the pattern.
                ob.check_reentry_condition(reentry_check_window)

                conditions_check_window: pd.DataFrame = algo.pair_df[ob.start_index:self.ob_formation_start_pdi]
                ob.set_condition_check_window(conditions_check_window)

                ob.check_fvg_condition()
                ob.check_stop_break_condition()

                positions_logger.debug(f"\t\t\tReentry check status: {ob.has_reentry_condition}")
                positions_logger.debug(f"\t\t\tFVG check status: {ob.has_fvg_condition} ({ob.fvg_fail_message})")
                positions_logger.debug(f"\t\t\tStop break check status: {ob.has_stop_break_condition}")
                if ob.has_reentry_condition and ob.has_fvg_condition and ob.has_stop_break_condition:
                    positions_logger.debug(f"\t\t\tAll checks passed, adding OB with ID {ob.id}")
                    valid_ob_counter += 1

                    ob.ranking_within_segment = valid_ob_counter

                    ob.times_moved = times_moved
                    ob.has_been_replaced = False
                    self.ob_list.append(ob)
                    break

                else:
                    positions_logger.debug("\t\t\tOne or more checks didn't pass, moving to next candle...")
                    ob.has_been_replaced = True

                times_moved += 1

        positions_logger.debug(f"End of finding order blocks for segment {self.id}")
        positions_logger.debug("")
