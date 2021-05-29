from gym.envs.registration import register
from betastar.envs.sc2_game import SC2GameEnv
from betastar.envs.move_to_beacon import MoveToBeacon1dEnv
from betastar.envs.move_to_beacon import MoveToBeacon2dEnv
from betastar.envs.collect_mineral_shards import CollectMineralShards1dEnv
from betastar.envs.collect_mineral_shards import CollectMineralShards2dEnv
from betastar.envs.collect_mineral_shards import CollectMineralShardsGroupsEnv

register(
    id='SC2Game-v0',
    entry_point='betastar.envs:SC2GameEnv',
    kwargs={}
)

register(
    id='SC2MoveToBeacon-v0',
    entry_point='betastar.envs:MoveToBeacon1dEnv',
    kwargs={}
)

register(
    id='SC2MoveToBeacon-v1',
    entry_point='betastar.envs:MoveToBeacon2dEnv',
    kwargs={}
)

register(
    id='SC2CollectMineralShards-v0',
    entry_point='betastar.envs:CollectMineralShards1dEnv',
    kwargs={}
)

register(
    id='SC2CollectMineralShards-v1',
    entry_point='betastar.envs:CollectMineralShards2dEnv',
    kwargs={}
)

register(
    id='SC2CollectMineralShards-v2',
    entry_point='betastar.envs:CollectMineralShardsGroupsEnv',
    kwargs={}
)

register(
    id='SC2FindAndDefeatZerglings-v0',
    entry_point='betastar.envs:SC2GameEnv',
    kwargs={
        'map_name': 'FindAndDefeatZerglings'
    }
)

register(
    id='SC2DefeatRoaches-v0',
    entry_point='betastar.envs:SC2GameEnv',
    kwargs={
        'map_name': 'DefeatRoaches'
    }
)

register(
    id='SC2DefeatZerglingsAndBanelings-v0',
    entry_point='betastar.envs:SC2GameEnv',
    kwargs={
        'map_name': 'DefeatZerglingsAndBanelings'
    }
)

register(
    id='SC2CollectMineralsAndGas-v0',
    entry_point='betastar.envs:SC2GameEnv',
    kwargs={
        'map_name': 'CollectMineralsAndGas'
    }
)

register(
    id='SC2BuildMarines-v0',
    entry_point='betastar.envs:SC2GameEnv',
    kwargs={
        'map_name': 'BuildMarines'
    }
)