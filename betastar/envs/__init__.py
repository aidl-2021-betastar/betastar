from gym.envs.registration import register
from pysc2.lib import actions
from betastar.envs.env import PySC2Env
from betastar.envs.move_env import MoveEnv

ACTIONS_MINIGAMES =  [0, 1, 2, 3, 4, 6, 7, 12, 13, 42, 44, 50, 91, 183, 234, 309, 331, 332, 333, 334, 451, 452, 490]
ACTIONS_MINIGAMES_ALL = ACTIONS_MINIGAMES + [11, 71, 72, 73, 74, 79, 140, 168, 239, 261, 264, 269, 274, 318, 335, 336, 453, 477]
ACTIONS_ALL = [f.id for f in actions.FUNCTIONS] # type: ignore

register(
    id='SC2MoveToBeacon-v0',
    entry_point='betastar.envs:PySC2Env',
    kwargs={
        'map_name': "MoveToBeacon",
        "action_ids": ACTIONS_MINIGAMES
    }
)

register(
    id='SC2MoveToBeaconSimple-v0',
    entry_point='betastar.envs:MoveEnv',
    kwargs={
        'map_name': "MoveToBeacon",
        "action_ids": [
            0, # no op
            7, # select army
            331 # move_screen
        ]
    }
)

register(
    id='SC2CollectMineralShards-v0',
    entry_point='betastar.envs:PySC2Env',
    kwargs={
        'map_name': "CollectMineralShards",
        "action_ids": ACTIONS_MINIGAMES
    }
)

register(
    id='SC2FindAndDefeatZerglings-v0',
    entry_point='betastar.envs:PySC2Env',
    kwargs={
        'map_name': "FindAndDefeatZerglings",
        "action_ids": ACTIONS_MINIGAMES
    }
)

register(
    id='SC2DefeatRoaches-v0',
    entry_point='betastar.envs:PySC2Env',
    kwargs={
        'map_name': "DefeatRoaches",
        "action_ids": ACTIONS_MINIGAMES
    }
)

register(
    id='SC2DefeatZerglingsAndBanelings-v0',
    entry_point='betastar.envs:PySC2Env',
    kwargs={
        'map_name': "DefeatZerglingsAndBanelings",
        "action_ids": ACTIONS_MINIGAMES
    }
)

register(
    id='SC2CollectMineralsAndGas-v0',
    entry_point='betastar.envs:PySC2Env',
    kwargs={
        'map_name': "CollectMineralsAndGas",
        "action_ids": ACTIONS_MINIGAMES
    }
)

register(
    id='SC2BuildMarines-v0',
    entry_point='betastar.envs:PySC2Env',
    kwargs={
        'map_name': "BuildMarines",
        "action_ids": ACTIONS_MINIGAMES
    }
)