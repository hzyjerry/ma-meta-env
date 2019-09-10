import os
import gym
import datetime
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from gym.spaces import Box
from ma_meta_env.envs.heavy_object.utils import *


"""
Multiple Agents carry one heavy object
The object is too heavy for anyone to singly carry

Author: Zhiyang He, Erdem Biyik
"""

PAPER_FIX_DIST = 3.5


class HeavyObjectEnv(gym.Env):
    """
    Distinct agent
    """

    def __init__(
        self,
        goal=None,
        observable_target=False,
        fix_goal=False,
        fix_dist=None,
        num_agents=2,
        centralized=False,
    ):
        # Properties
        self.observable_target = observable_target
        self.centralized = centralized
        assert num_agents in [1, 2, 3]
        self.num_agents = num_agents
        self.goal = goal
        self.fix_goal = fix_goal
        self.fix_dist = fix_dist

        # Environment & Episode settings
        self.map_W = 10.0
        self.map_H = 10.0
        self.stick_length = 1.0
        self.stick_mass = 0.2 * num_agents
        self.inertia = 0.1 * num_agents
        self.max_robot_force = 1
        self.dt = 0.2
        self.angle_offsets = self.get_angle_offsets()
        self.r_length = self.get_r_length()

        # Book keepings
        self._last_value = 0
        self._step_count = 0
        self._total_reward = 0
        self._fig = None  # only for visualization
        self._fig_folder = None
        self._hist = {}  # only for making figure
        self._last_hist = {}  # only for making figure
        self._recordings = {}  # only for making figure
        self._last_actions = None
        self._state = None
        self._last_state = None

        # Action space
        self.action_low = np.array([-self.max_robot_force, -np.pi])
        self.action_high = np.array([self.max_robot_force, np.pi])
        if self.centralized:
            # Centralized control: x, y, r
            self.action_low = np.array(
                [-self.max_robot_force, -self.max_robot_force, -self.max_robot_force]
            )
            self.action_high = np.array(
                [self.max_robot_force, self.max_robot_force, self.max_robot_force]
            )

        # Observation space
        self.obs_low = np.array([self.map_W / 2, -self.map_H / 2, -np.pi])
        self.obs_high = np.array([self.map_W / 2, self.map_H / 2, np.pi])
        if self.observable_target:
            self.obs_low = np.concatenate([self.obs_low, self.obs_low])
            self.obs_high = np.concatenate([self.obs_high, self.obs_high])

    def get_angle_offsets(self):
        """ Map one angle to three angles """
        if self.num_agents == 1:
            return np.array([0.0])
        elif self.num_agents == 2:
            return np.array([0.0, np.pi])
        elif self.num_agents == 3:
            return np.array([-np.pi / 6, np.pi / 2, 7 * np.pi / 6])
        else:
            raise NotImplementedError(f"{self.num_agents} agents not supported")

    def get_r_length(self):
        """ Object radius """
        if self.num_agents == 1:
            return 0.0
        elif self.num_agents == 2:
            return self.stick_length / 2
        elif self.num_agents == 3:
            return self.stick_length / np.sqrt(3)
        else:
            raise NotImplementedError(f"{self.num_agents} agents not supported")

    @property
    def observation_space(self):
        if self.centralized:
            return [
                Box(
                    [self.obs_low] * self.num_agents,
                    [self.obs_high] * self.num_agents,
                    dtype=np.float32,
                )
            ]
        else:
            return [
                Box(self.obs_low, self.obs_high, dtype=np.float32)
                for _ in range(self.num_agents)
            ]

    @property
    def action_space(self):
        if self.centralized:
            return [Box(self.action_low, self.action_high, dtype=np.float32)]
        else:
            return [
                Box(self.action_low, self.action_high, dtype=np.float32)
                for _ in range(self.num_agents)
            ]

    @property
    def num_decentralized(self):
        return self.num_agents

    @property
    def identity_space(self):
        low = [0]
        high = [self.num_agents]
        return Box(low=np.array(low), high=np.array(high), dtype=np.float32)

    def generate_random_goal(self, fix_dist=None):
        if fix_dist is not None:
            assert type(fix_dist) == float
            angle = np.random.uniform(low=-np.pi, high=np.pi)
            theta = np.random.uniform(low=-np.pi, high=np.pi)
            pair = np.array([fix_dist * np.cos(angle), fix_dist * np.sin(angle), theta])
        else:
            pair = np.random.uniform(
                low=[-self.map_W / 2, -self.map_H / 2, -np.pi],
                high=[self.map_W / 2, self.map_H / 2, np.pi],
            )
        return pair

    def generate_default_goal(self):
        return [0, 0, 0]

    def sample_paper_goals(self):
        goals = None
        if self.num_agents == 2:
            num_goals = 5
            angle = np.array(
                [0, np.pi * 2 / 5, np.pi * 4 / 5, np.pi * 6 / 5, np.pi * 8 / 5]
            )
            theta = np.random.uniform(low=[-np.pi], high=[np.pi], size=(num_goals,))
            goals = np.array(
                [PAPER_FIX_DIST * np.cos(angle), PAPER_FIX_DIST * np.sin(angle), theta]
            ).T
        elif self.num_agents == 3:
            num_goals = 5
            angle = np.array(
                [np.pi * 2 / 5, np.pi * 4 / 5, np.pi * 6 / 5, np.pi * 8 / 5, 0]
            )
            theta = np.random.uniform(low=[-np.pi], high=[np.pi], size=(num_goals,))
            goals = np.array(
                [PAPER_FIX_DIST * np.cos(angle), PAPER_FIX_DIST * np.sin(angle), theta]
            ).T
        return goals

    def sample_table_goals(self):
        goals = None
        num_goals = 1
        angle = np.random.uniform(low=[-np.pi], high=[np.pi], size=(num_goals,))
        theta = np.random.uniform(low=[-np.pi], high=[np.pi], size=(num_goals,))
        goals = np.array(
            [PAPER_FIX_DIST * np.cos(angle), PAPER_FIX_DIST * np.sin(angle), theta]
        ).T
        return goals

    def sample_goals(self, num_goals, make_figure=False, make_table=False):
        goals = np.zeros((num_goals, 3), dtype=np.float32)
        if make_figure:
            goals = self.sample_paper_goals()
        elif make_table:
            goals = self.sample_table_goals()
        else:
            for i in range(num_goals):
                if self.fix_goal:
                    goals[i, :3] = [4, 4, np.pi / 2]
                elif self.fix_dist is not None:
                    goals[i, :3] = self.generate_random_goal(fix_dist=self.fix_dist)
                else:
                    goals[i, :3] = self.generate_random_goal()
        return goals

    def sample_tasks(self, num_tasks, **kargs):
        return self.sample_goals(num_tasks, **kargs)

    def reset_task(self, task):
        self.reset_goal(task)

    def reset_goal(self, goal):
        self.goal = goal

    def reset(self, goal=None):
        self.goal = goal if goal is not None else self.sample_goals(1)[0]
        self._state = np.zeros((3,), dtype=np.float32)
        self._state[:3] = self.generate_default_goal()
        self._last_value = 0
        self._step_count = 0
        self._total_reward = 0
        self._last_actions = None
        self._last_state = self._state
        obs = self._get_obs()
        return obs

    def _get_obs(self):
        state = self._state.copy()
        xs, ys, gxs, gys, angles, goal_angles = self.get_pos_gpos()
        obs = np.stack([xs, ys, angles]).T
        if self.observable_target:
            goal_obs = np.stack([gxs, gys, goal_angles]).T
            obs = np.concatenate([obs, goal_obs], axis=1)
        return obs.astype(np.float32)

    def action_reward(self, actions):
        assert actions is not None
        new_state = self._get_new_state(actions)
        rew = self._get_reward(new_state)
        return rew

    def _get_reward(self, state=None):
        if state is None:
            state = self._state
        xs, ys, gxs, gys, _, _ = self.get_pos_gpos(state=state)
        total_dist = np.sum(np.sqrt(np.square(xs - gxs) + np.square(ys - gys)))
        current_value = -np.mean(total_dist)
        self._last_value = current_value
        reward = current_value
        return reward

    def _clip_actions(self, actions):
        return np.clip(actions, self.action_low, self.action_high)

    def step(self, actions, acts=None, probs=None):
        # Book keeping
        actions = self._clip_actions(np.array(actions))
        self._last_state = self._state

        # Get reward
        new_cx, new_cy, new_angle = self._get_new_state(actions)
        rew = self.action_reward(actions)
        rews = [rew] * self.num_agents
        advs = [rew] * self.num_agents
        if acts is not None:
            advs = self.current_adv_counterfactual(actions, acts=acts)

        # Update state
        self._state = np.array([new_cx, new_cy, new_angle])
        self._step_count += 1
        done = False
        dones = [done] * self.num_agents
        obs = self._get_obs()
        self._total_reward += rew
        self._last_actions = actions

        info = {"goal": self.goal, "rew_shaped": [0] * self.num_agents}
        for i in range(self.num_agents):
            info["rew_shaped"][i] = np.float32(advs[i])
        return obs, rews, dones, info

    def current_adv_counterfactual(self, actions, acts=None):
        """ Use counterfactual reward as proxy for counterfactual advantage """
        debug = False
        rew = self.action_reward(actions)
        rews = [rew] * self.num_agents
        if acts is not None:
            for agent_i, actsi in enumerate(acts):
                counter_rews = []
                for ai in actsi:
                    new_a = np.array(actions, copy=True)
                    new_a[agent_i] = ai
                    a_value = self.action_reward(new_a)
                    if debug:
                        print(
                            " Agent({}) A:{} aval:{}/{}".format(
                                agent_i, ai, a_value, len(actsi)
                            )
                        )
                    counter_rews.append(a_value)
                    # rews[agent_i] -= a_value / len(actsi)
                # print(" Agent({}) rew:{:.4f} mean:{:.4f} std:{:.4f}".format(agent_i, rews[agent_i], np.mean(counter_rews), np.std(counter_rews)))
                rews[agent_i] -= np.mean(counter_rews)
        # print("advs", rews)
        return rews

    def _convert_f_xyr(self, state, actions):
        """
        Return
        : state : environment state (can use past states)
        : F_x   : joint force along x
        : F_y   : joint force along y
        : F_r   : joint force to rotate the stick
        """
        angles = state[2] + self.angle_offsets
        if self.centralized:
            F_xs = actions[:, 0] * np.cos(angles[0]) - actions[:, 1] * np.sin(angles[1])
            F_ys = actions[:, 0] * np.sin(angles[0]) + actions[:, 1] * np.cos(angles[1])
            F_rs = actions[:, 1]
        else:
            F_xs = actions[:, 0] * np.cos(angles + actions[:, 1])
            F_ys = actions[:, 0] * np.sin(angles + actions[:, 1])
            F_rs = actions[:, 0] * np.sin(actions[:, 1])

        F_xs, F_ys, F_rs = F_xs.T, F_ys.T, F_rs.T
        return F_xs, F_ys, F_rs

    def _get_new_state(self, actions):
        cx, cy, angle = self._state
        F_xs, F_ys, F_rs = self._convert_f_xyr(self._state, actions)
        F_x, F_y, F_r = np.mean(F_xs), np.mean(F_ys), np.mean(F_rs)

        new_cx = cx + (F_x / self.stick_mass) * self.dt
        new_cy = cy + (F_y / self.stick_mass) * self.dt
        new_angle = regulate_radians(angle + (F_r / self.inertia) * self.dt)

        optimal_action = False  # report optimal rew in paper
        if optimal_action:
            ds = (
                (self.max_robot_force * self.num_agents / self.stick_mass) * self.dt / 2
            )
            gx, gy, ga = self.goal
            dx = gx - cx
            dy = gy - cy
            new_cx = cx + ds * dx / (np.sqrt(dx * dx + dy * dy) + 1e-5)
            new_cy = cy + ds * dy / (np.sqrt(dx * dx + dy * dy) + 1e-5)
            new_angle = ga
        return new_cx, new_cy, new_angle

    def get_pos_gpos(self, state=None):
        """
        Get current position and goal positions
        """
        state = state if state is not None else self._state
        cx, cy, angle = state
        gx, gy, goal_angle = self.goal
        angles = regulate_radians(angle + self.angle_offsets)
        goal_angles = regulate_radians(goal_angle + self.angle_offsets)

        xs = cx + self.r_length * np.cos(angles)
        ys = cy + self.r_length * np.sin(angles)
        gxs = gx + self.r_length * np.cos(goal_angles)
        gys = gy + self.r_length * np.sin(goal_angles)
        return xs, ys, gxs, gys, angles, goal_angles

    def render(
        self,
        title=None,
        n_frame=10,
        show_before=False,
        task_num=0,
        save_viz=False,
        iteration=0,
        **kargs,
    ):
        t_delta = 0.001
        if self.num_agents == 1:
            colors = ["#000075"]
        elif self.num_agents == 2:
            colors = ["#000075", "#e6194B"]
        else:
            colors = ["#000075", "#e6194B", "#3cb44b"]

        curr_state = self._state
        show_state = self._state
        if n_frame != 1:
            show_state = self._last_state

        for i_frame in range(n_frame):
            # Refresh
            if self._fig is None:
                self._fig = plt.figure(figsize=(8, 8), dpi=80)
            else:
                self._fig.clear()
            ax = plt.gca()

            # Object visualization
            show_state = show_state + (curr_state - show_state) / n_frame
            xs, ys, gxs, gys, angles, _ = self.get_pos_gpos(show_state)

            if self.num_agents == 2:
                object_plt = plt.Line2D(
                    xs,
                    ys,
                    lw=10.0,
                    ls="-",
                    marker=".",
                    color="grey",
                    markersize=0,
                    markerfacecolor="r",
                    markeredgecolor="r",
                    markerfacecoloralt="k",
                    alpha=0.7,
                )
                ax.add_line(object_plt)
            elif self.num_agents == 3:
                object_plt = plt.Polygon(
                    list(map(list, zip(xs, ys))), alpha=0.7, color="grey"
                )
                ax.add_line(object_plt)

            for i, c in zip(range(self.num_agents), colors):
                plt.scatter(xs[i], ys[i], c=c, marker="o", s=140, zorder=10, alpha=0.7)

            # Before adaptation visualization
            if show_before:
                if self.num_agents == 2:
                    object_plt = plt.Line2D(
                        before_xs,
                        before_ys,
                        lw=10.0,
                        ls="-",
                        marker=".",
                        color="grey",
                        markersize=0,
                        markerfacecolor="r",
                        markeredgecolor="r",
                        markerfacecoloralt="k",
                        alpha=0.35,
                    )
                    ax.add_line(object_plt)
                elif self.num_agents == 3:
                    object_plt = plt.Polygon(
                        list(map(list, zip(before_xs, before_ys))),
                        alpha=0.35,
                        color="grey",
                    )
                    ax.add_line(object_plt)
                for i, c in zip(range(self.num_agents), colors):
                    plt.scatter(
                        before_xs[i],
                        before_ys[i],
                        c=c,
                        marker="o",
                        s=140,
                        zorder=10,
                        alpha=0.35,
                    )

            # Goal visualization
            markers = ["$" + s + "$" for s in "ABC"[: self.num_agents]]
            sm_marker = 50
            lg_marker = 80
            if gxs is not None and gys is not None:
                gxs += [gxs[0]]
                gys += [gys[0]]
            ax.plot(gxs, gys, ":", lw=2, alpha=1, color="grey")
            if self.num_agents == 2:
                goal_plt = plt.Line2D(
                    gxs,
                    gys,
                    lw=10.0,
                    ls="-",
                    marker=".",
                    color="grey",
                    markersize=0,
                    markerfacecolor="r",
                    markeredgecolor="r",
                    markerfacecoloralt="k",
                    alpha=0.3,
                )
            else:
                goal_plt = plt.Polygon(
                    list(map(list, zip(gxs, gys))), alpha=0.3, color="grey"
                )
            ax.add_line(goal_plt)
            for i, c in zip(range(self.num_agents), colors):
                plt.scatter(
                    gxs[i], gys[i], c=c, marker="o", s=140, zorder=10, alpha=0.3
                )

            # Action visualization
            fs = self._last_actions
            F_xs, F_ys, F_rs = [], [], []
            if fs is not None:
                F_xs, F_ys, F_rs = self._convert_f_xyr(self._last_state, fs)
                fs_const = 0.3
                F_xs *= fs_const
                F_ys *= fs_const
                lengths = np.sqrt(np.square(F_xs) + np.square(F_ys))
                for i in range(self.num_agents):
                    plt.arrow(
                        xs[i],
                        ys[i],
                        F_xs[i],
                        F_ys[i],
                        fc=colors[i],
                        ec="none",
                        alpha=0.7,
                        width=0.06,
                        head_width=0.1,
                        head_length=0.2,
                        zorder=8,
                    )
            sns.despine(offset=10, trim=True)
            ax.set_xlim([-(PAPER_FIX_DIST + 0.75), (PAPER_FIX_DIST + 0.75)])
            ax.set_ylim([-(PAPER_FIX_DIST + 0.75), (PAPER_FIX_DIST + 0.75)])
            plt.xticks(fontsize=15)
            plt.yticks(fontsize=15)
            plt.axis("off")
            for item in (
                [ax.title, ax.xaxis.label, ax.yaxis.label]
                + ax.get_xticklabels()
                + ax.get_yticklabels()
            ):
                item.set_fontname("Times New Roman")
            if save_viz:
                if not self._fig_folder:
                    self._fig_folder = "figures/stick-{s}-{date:%Y-%m-%d %H:%M:%S}".format(
                        s=self.num_agents, date=datetime.datetime.now()
                    ).replace(
                        " ", "_"
                    )
                    os.makedirs(self._fig_folder, exist_ok=True)
                self._fig.savefig(
                    os.path.join(
                        self._fig_folder,
                        "stick{}_task{}_itr{}_frame{:04d}.png".format(
                            self.num_agents,
                            task_num,
                            iteration,
                            self._step_count * n_frame + i_frame,
                        ),
                    )
                )
            self._fig.show()
            plt.pause(t_delta / n_frame)

    def record_hist(self, done=False, task_num=0, reward=-1):
        if task_num not in self._hist.keys():
            self._hist[task_num] = dict(score=-np.inf, hist=[])
        xs, ys, gxs, gys, _, _ = self.get_pos_gpos()
        self._hist[task_num]["hist"].append((xs, ys, gxs, gys))
        if done:
            if reward > self._hist[task_num]["score"]:
                self._hist[task_num]["score"] = reward
                self._recordings[task_num] = self._hist[task_num]["hist"]
            self._hist[task_num]["hist"] = []

    def make_figure(self, interpolate=1, after=True):
        assert self._recordings is not None
        palatte = sns.color_palette("deep", len(self._recordings.keys()))
        # colors = ["windows blue", "#34495e", "greyish", "faded green", "dusty purple"]
        # palatte = sns.xkcd_palette(colors)
        if self._fig is None:
            self._fig = plt.figure(figsize=(8, 8), dpi=80)
        else:
            self._fig.clear()
        initial_pos = self._recordings[0][0]
        for task_i, record in self._recordings.items():
            for ts in record[-1::-interpolate]:
                xs, ys, gxs, gys = ts
                self._render_pos(xs, ys, gxs, gys, obj_alpha=0.3, color=palatte[task_i])
            print("task {}".format(task_i))
        sns.despine(offset=10, trim=True)
        self._render_pos(
            initial_pos[0],
            initial_pos[1],
            None,
            None,
            obj_alpha=1.0,
            color="k",
            fixed_pos=True,
        )

        ax = self._fig.axes[0]
        plt.xticks(fontsize=15)
        plt.yticks(fontsize=15)
        for item in (
            [ax.title, ax.xaxis.label, ax.yaxis.label]
            + ax.get_xticklabels()
            + ax.get_yticklabels()
        ):
            item.set_fontname("Times New Roman")
        self._fig.show()
        plt.savefig(
            "HeavyObject-{}-{}.png".format(
                self.num_agents, "after" if after else "before"
            ),
            bbox_inches="tight",
        )
        plt.pause(100)

    def _render_pos(self, xs, ys, gxs, gys, obj_alpha=1.0, color=None, fixed_pos=False):
        ax = self._fig.gca()
        w = self.stick_mass
        ax.set_xlim([-(PAPER_FIX_DIST + 0.5), (PAPER_FIX_DIST + 0.5)])
        ax.set_ylim([-(PAPER_FIX_DIST + 0.5), (PAPER_FIX_DIST + 0.5)])
        markers = ["$" + s + "$" for s in "ABC"[: self.num_agents]]
        sm_marker = 50
        lg_marker = 80
        xs += [xs[0]]
        ys += [ys[0]]
        ax.plot(xs, ys, "-", lw=2 + w, alpha=obj_alpha, color=color if color else "k")
        for xi, yi, m in zip(xs, ys, markers):
            plt.scatter(
                xi,
                yi,
                marker=m,
                c=color if color else "k",
                s=lg_marker if fixed_pos else sm_marker,
            )
        if gxs and gys:
            if gxs and gys:
                gxs += [gxs[0]]
                gys += [gys[0]]
            ax.plot(gxs, gys, ":", lw=2 + w, alpha=1, color=color if color else "b")
            for gxi, gyi, m in zip(gxs, gys, markers):
                plt.scatter(gxi, gyi, marker=m, c=color if color else "b", s=lg_marker)
