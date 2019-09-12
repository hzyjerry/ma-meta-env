import numpy as np
from ma_meta_env.envs.heavy_object import HeavyObjectEnv


def run_env(fix_goal, num_agents):
    env = HeavyObjectEnv(fix_goal=fix_goal, num_agents=num_agents)
    env.reset()
    step = 0
    obs_space = env.observation_space
    acs_space = env.action_space
    total_reward = 0
    while True:
        #ac = np.array([[8, 0.5]] * env.num_agents)
        ac = np.array([ac_space.sample() for ac_space in acs_space])
        ob, rew, done, _ = env.step(ac)
        total_reward += rew[0]
        #env.render()
        step += 1
        if done[0]:
            print("Total rew", total_reward)
            total_reward = 0
            ob = env.reset()

if __name__ == "__main__":
    run_env(True, 3)
