import constants


def default_357(position):
    if position.type == "long":
        position.stoploss = position.entry_price - constants.stoploss_coeff * position.edicl
        position.target_list = [
            position.entry_price + 3 * position.edicl,
            position.entry_price + 5 * position.edicl,
            position.entry_price + 7 * position.edicl,
        ]
    else:
        position.stoploss = position.entry_price + constants.stoploss_coeff * position.edicl
        position.target_list = [
            position.entry_price - 3 * position.edicl,
            position.entry_price - 5 * position.edicl,
            position.entry_price - 7 * position.edicl,
        ]


def all_on_7(position):
    if position.type == "long":
        position.stoploss = position.entry_price - constants.stoploss_coeff * position.edicl
        position.target_list = [
            position.entry_price + 7 * position.edicl,
        ]
    else:
        position.stoploss = position.entry_price + constants.stoploss_coeff * position.edicl
        position.target_list = [
            position.entry_price - 7 * position.edicl,
        ]


def all_on_5(position):
    if position.type == "long":
        position.stoploss = position.entry_price - constants.stoploss_coeff * position.edicl
        position.target_list = [
            position.entry_price + 7 * position.edicl,
        ]
    else:
        position.stoploss = position.entry_price + constants.stoploss_coeff * position.edicl
        position.target_list = [
            position.entry_price - 7 * position.edicl,
        ]
