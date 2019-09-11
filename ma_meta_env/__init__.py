import logging
import gym
from gym.envs.registration import register

logger = logging.getLogger(__name__)

"""
We bypass gym.TimeLimit by not explicitly specifying `max_episode_steps`

This allows having more flexible env.step(action, **kwargs)
with additional arguments
"""

register(
    id="MultiagentHeavyObject-2-v0",
    entry_point="ma_meta_env.envs.heavy_object:HeavyObjectEnv",
    kwargs={},
)
