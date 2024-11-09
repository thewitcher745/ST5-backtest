from typing import Union, Literal
import pandas as pd

import algo_code.position_prices_setup as setup
import constants


class Position:
    def __init__(self, parent_ob):
        self.parent_ob = parent_ob
        self.entry_price = parent_ob.top if parent_ob.type == "long" else parent_ob.bottom

        # Calculation of stoploss is done using the distance from the entry of the box to the initial candle that was checked for OB, before being
        # potentially replaced. This distance is denoted as EDICL, entry distance from initial candle liquidity.
        self.edicl = abs(parent_ob.icl - self.entry_price)

        self.type = parent_ob.type

        self.status: str = "ACTIVE"
        self.entry_pdi = None
        self.qty: float = 0
        self.highest_target: int = 0
        self.target_hit_pdis: list[int] = []
        self.exit_pdi = None
        self.portioned_qty = []
        self.net_profit = None

        self.target_list = []
        self.stoploss = None
        # Set up the target list nd stoploss using a function which operates on the "self" object and directly manipulates the instance.
        setup.default_357(self)

    def find_entry_within_segment(self, segment) -> Union[int, None]:
        """
        This method analyzes the candles within the segment's entry region to see if any candle enters the position.

        Args:
            segment (Segment): The segment with the filtered candles ready to be checked for entry

        Returns:
            Union[int, None]: The index of the candle entering the position, or None if no candle enters the position
        """
        if self.type == "long":
            entering_candles: pd.DataFrame = segment.pair_df[segment.pair_df.low <= self.entry_price]
        else:
            entering_candles: pd.DataFrame = segment.pair_df[segment.pair_df.high >= self.entry_price]

        if len(entering_candles) > 0:
            return entering_candles.first_valid_index()
        else:
            return None

    def enter(self, entry_pdi: int):
        """
        Method to enter the OB. This method sets the current OB status to "ENTERED", and registers the entry PDI, entry price, and quantity of the
        entry.

        Args:
            entry_pdi (int): The PDI at which the entry is made
        """

        self.entry_pdi = entry_pdi
        self.qty = constants.used_capital / self.entry_price
        self.status = "ENTERED"

    def register_target(self, target_id: int, target_registry_pdi: int):
        """
        This method registers a new highest target on the position to later use in calculating the PNL. If the target being registered is the highest
        target, it also triggers an exit command.

        Args:
            target_id (int): The ID of the target to register. Must be higher than 0 since the default value is zero.
            target_registry_pdi (int): The PDI of the candle registering the target(s)
        """

        # First, for safety, check if the target being registered is actually higher than the highest registered target
        if target_id > self.highest_target:
            # Register all the non-hit targets with the PDI of the candle hitting them
            self.target_hit_pdis.extend([target_registry_pdi] * (target_id - self.highest_target))

            self.highest_target = target_id

    def exit(self, exit_code: Literal["STOPLOSS", "FULL_TARGET"], exit_pdi: int):
        """
        This method exits an entered order block with an exit code. If the exit code is "STOPLOSS" that means the position is exiting due to hitting
        the stoploss level. Otherwise, if the exit code is "FULL_TARGET" that means the last target has been hit and therefore the maximum possible
        profit should be registered. If a "STOPLOSS" event happens, the profit is calculated using the highest registered target, accounting for
        losses from the stoploss and gains from the targets separately. The net profit is then registered into the Position.net_profit property.

        Args:
            exit_code (str): How the position has been exit.
            exit_pdi (int): At which candle the exit happens
        """

        self.exit_pdi = exit_pdi

        # Even distribution of quantities
        self.portioned_qty = [self.qty / len(self.target_list) for target in self.target_list]

        # If the position is exiting due to hitting a stoploss
        if exit_code == "STOPLOSS":
            # If we do have any registered targets, set the highest registered target as the final status
            if self.highest_target > 0:
                self.status = f"TARGET_{self.highest_target}"

            # Otherwise, just report a STOPLOSS
            else:
                self.status = "STOPLOSS"

            # If the position is long, this means that we have one loss: a loss from purchasing the asset at entry, and we have two gains: a loss
            # from selling the remainder of the asset at stoploss and another for selling each portioned quantity at each target hit.
            if self.type == "long":
                loss_from_entry = self.entry_price * self.qty
                gain_from_stop = sum(self.portioned_qty[self.highest_target:]) * self.stoploss
                gain_from_targets = sum([self.portioned_qty[i] * self.target_list[i] for i in range(self.highest_target)])

                total_position_gain = gain_from_stop + gain_from_targets
                total_position_loss = loss_from_entry

            # If the position is short, this means that we have one gain: a gain from selling the asset at entry, and we have two losses: a loss from
            # buying the remainder of the asset at stoploss and another for buying each portioned quantity at each target hit.
            else:
                gain_from_entry = self.entry_price * self.qty
                loss_from_stop = sum(self.portioned_qty[self.highest_target:]) * self.stoploss
                loss_from_targets = sum([self.portioned_qty[i] * self.target_list[i] for i in range(self.highest_target)])

                total_position_gain = gain_from_entry
                total_position_loss = loss_from_stop + loss_from_targets

        # If a full target has been hit, report it as such
        elif exit_code == "FULL_TARGET":
            self.status = f"FULL_TARGET_{self.highest_target}"

            # If the position has achieved full targets, we have the same codes for calculating net profit, only with the omission of stoploss
            # loss/gains. All the target calculations will also use the entire target_list property instead of the spliced version
            if self.type == "long":
                total_position_loss = self.entry_price * self.qty
                total_position_gain = sum([qty_target[0] * qty_target[1] for qty_target in zip(self.portioned_qty, self.target_list)])

            else:
                total_position_gain = self.entry_price * self.qty
                total_position_loss = sum([qty_target[0] * qty_target[1] for qty_target in zip(self.portioned_qty, self.target_list)])

        self.net_profit = total_position_gain - total_position_loss

    def does_candle_stop(self, candle):
        """
        This method checks if the candle stops the position. This is done by checking if the candle's low is lower than the stoploss in the case of
        long positions, and if the candle's high is higher than the stoploss in the case of short positions.

        Args:
            candle (pd.Series): The candle to check for stopping

        Returns:
            bool: True if the candle stops the position, False otherwise
        """

        if self.type == "long":
            return candle.low <= self.stoploss
        else:
            return candle.high >= self.stoploss

    def detect_candle_sentiment(self, candle: pd.Series) -> tuple[str, Union[int, None]]:
        """
        This method checks which target (or stoploss) the candle argument breaks. The method is used to determine if the position should be exited.
        This method uses the candle's color to determine which of the stoploss or targets were hit first.

        Args:
            candle (pd.Series): The candle to check for target/stoploss

        Returns:
            tuple: A tuple containing a sentiment ("TARGET" , "FULL_TARGET", "STOPLOSS" or "NONE") and an int, for the case where the candle registers
            a target. If a candle registers a stoploss, the int is 0.
        """

        def last_element_bigger_than(targets: list[float], price: float):
            for i in reversed(range(len(targets))):
                if targets[i] >= price:
                    return i + 1
            return 0

        def last_element_smaller_than(targets: list[float], price: float):
            for i in reversed(range(len(targets))):
                if targets[i] <= price:
                    return i + 1
            return 0

        # Long order blocks
        if self.type == "long":
            highest_target = last_element_smaller_than(self.target_list, candle.high)
        # Short order blocks
        else:
            highest_target = last_element_bigger_than(self.target_list, candle.low)

        # If the candle is green, it means the price is going up, and the bottom of the box should be checked first
        if candle.close > candle.open:
            if self.does_candle_stop(candle):
                return "STOPLOSS", None

            if highest_target > self.highest_target:
                if highest_target < len(self.target_list):
                    return "TARGET", highest_target

                elif highest_target == len(self.target_list):
                    return "FULL_TARGET", None

        # If the candle is red, it means the price is going down, and the top of the box should be checked first
        else:
            if highest_target > self.highest_target:
                if highest_target < len(self.target_list):
                    return "TARGET", highest_target

                elif highest_target == len(self.target_list):
                    return "FULL_TARGET", None

            if self.does_candle_stop(candle):
                return "STOPLOSS", None

        return "NONE", None
