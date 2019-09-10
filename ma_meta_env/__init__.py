import logging
import gym
from gym.envs.registration import register

logger = logging.getLogger(__name__)

# for i, observability in enumerate([False, True]):
register(
    id="MultiagentHeavyObject-2-v0",
    entry_point="ma_meta_env.envs.heavy_object:HeavyObjectEnv",
    max_episode_steps=20,
    kwargs={},
)
