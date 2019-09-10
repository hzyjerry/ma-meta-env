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
        ac = np.array([[8, 0.5]] * env.num_agents)
        ob, rew, done, _ = env.step(ac)
        env.render()
        step += 1
        total_reward += rew
        if done:
            print("step {} total {}".format(step, total_reward))
            break


if __name__ == "__main__":
    run_env(True, 3)
