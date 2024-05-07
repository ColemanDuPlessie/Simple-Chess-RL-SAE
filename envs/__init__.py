from gym.envs.registration import register
from envs.chess_env import RookCheckmateEnv

register(
    id='RookCheckmate-v0',
    entry_point='envs:RookCheckmateEnv', # TODO
    max_episode_steps=100 # Yes, this is flaunting the 50-move rule, but I wanted to give the agent a bit more time
    )
