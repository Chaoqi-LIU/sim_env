import numpy as np
from typing import List

PICK_AND_PLACE = [
    'PnPCounterToCab',          # H = 500
    'PnPCabToCounter',          # H = 500
    'PnPCounterToSink',         # H = 700
    'PnPSinkToCounter',         # H = 500
    'PnPCounterToMicrowave',    # H = 600
    'PnPMicrowaveToCounter',    # H = 500
    'PnPCounterToStove',        # H = 500
    'PnPStoveToCounter',        # H = 500
]

OPEN_CLOSE_DOOR = [
    'OpenSingleDoor',           # H = 500
    'CloseSingleDoor',          # H = 500
    'OpenDoubleDoor',           # H = 1000
    'CloseDoubleDoor',          # H = 700
]

OPEN_CLOSE_DRAWER = [
    'OpenDrawer',               # H = 500
    'CloseDrawer',              # H = 500
]

TURN_LEVER = [
    'TurnOnSinkFaucet',         # H = 500
    'TurnOffSinkFaucet',        # H = 500
    'TurnSinkSpout',            # H = 500
]

TWIST_KNOB = [
    'TurnOnStove',              # H = 500
    'TurnOffStove',             # H = 500
]

INSERTION = [
    'CoffeeSetupMug',           # H = 600
    'CoffeeServeMug',           # H = 600
]

PRESS_BUTTON = [
    'CoffeePressButton',        # H = 300
    'TurnOnMicrowave',          # H = 500
    'TurnOffMicrowave',         # H = 500
]


MT_TASKS = {
    'pnp': PICK_AND_PLACE,
    'door': OPEN_CLOSE_DOOR,
    'drawer': OPEN_CLOSE_DRAWER,
    'lever': TURN_LEVER,
    'knob': TWIST_KNOB,
    'insert': INSERTION,
    'button': PRESS_BUTTON,
    'mt4': [
        'CloseDrawer',
        'TurnOffSinkFaucet',
        'CoffeePressButton',
        'TurnOffMicrowave',
    ],
    'mt5': OPEN_CLOSE_DRAWER + PRESS_BUTTON,
    'mt8': OPEN_CLOSE_DRAWER + TURN_LEVER + PRESS_BUTTON
}


def is_multitask(task_name: str) -> bool:
    return task_name in MT_TASKS


def get_subtasks(task_name: str) -> List[str]:
    if is_multitask(task_name):
        return MT_TASKS[task_name]
    else:
        return [task_name]
