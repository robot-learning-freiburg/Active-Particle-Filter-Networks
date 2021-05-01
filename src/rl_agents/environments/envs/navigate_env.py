#!/usr/bin/env python3

from collections import OrderedDict
from gibson2.envs.igibson_env import iGibsonEnv
import gym
import numpy as np

class NavigateGibsonEnv(iGibsonEnv):
    """
    Custom implementation of navigation task based on iGibsonEnv
    """

    def __init__(
        self,
        config_file,
        scene_id=None,
        mode='headless',
        action_timestep=1 / 10.0,
        physics_timestep=1 / 240.0,
        device_idx=0,
        render_to_tensor=False,
        automatic_reset=False,
    ):
        """
        :param config_file: config_file path
        :param scene_id: override scene_id in config file
        :param mode: headless, gui, iggui
        :param action_timestep: environment executes action per action_timestep second
        :param physics_timestep: physics timestep for pybullet
        :param device_idx: which GPU to run the simulation and rendering on
        :param render_to_tensor: whether to render directly to pytorch tensors
        :param automatic_reset: whether to automatic reset after an episode finishes
        """

        super(NavigateGibsonEnv, self).__init__(
                        config_file=config_file,
                        scene_id=scene_id,
                        mode=mode,
                        action_timestep=action_timestep,
                        physics_timestep=physics_timestep,
                        device_idx=device_idx,
                        render_to_tensor=render_to_tensor,
                        automatic_reset=automatic_reset)

        # override observation_space
        # task_obs_dim = robot_prorpio_state (18) + goal_coords (2)
        TASK_OBS_DIM = 20

        # custom tf_agents we are using supports dict() type observations
        observation_space = OrderedDict()
        self.custom_output = ['task_obs', ]

        if 'task_obs' in self.custom_output:
            observation_space['task_obs'] = gym.spaces.Box(
                    low=-np.inf, high=+np.inf,
                    shape=(TASK_OBS_DIM,),
                    dtype=np.float32)
        # image_height and image_width are obtained from env config file
        if 'rgb_obs' in self.custom_output:
            observation_space['rgb'] = gym.spaces.Box(
                    low=0.0, high=1.0,
                    shape=(self.image_height, self.image_width, 3),
                    dtype=np.float32)

        self.observation_space = gym.spaces.Dict(observation_space)

        self.pfnet_model = None

    def step(self, action):
        """
        Apply robot's action.
        Returns the next state, reward, done and info,
        following OpenAI Gym's convention

        :param action: robot actions

        :return: state: next observation
        :return: reward: reward of this time step
        :return: done: whether the episode is terminated
        :return: info: info dictionary with any useful information
        """

        state, reward, done, info = super(NavigateGibsonEnv, self).step(action)

        custom_state = self.process_state(state)

        return custom_state, reward, done, info

    def reset(self):
        """
        Reset episode

        :return: state: new observation
        """

        """
        manually select random simple fixed target goal
        """
        if np.random.uniform() < 0.5:
            self.task.target_pos = np.array([0.3, 0.9, 0.0])
        else:
            self.task.target_pos = np.array([0.2, -0.2, 0.0])

        state = super(NavigateGibsonEnv, self).reset()

        custom_state = self.process_state(state)

        return custom_state

    def process_state(self, state):
        """
        Perform additional processing.

        :param state: env observations

        :return: processed_state: processed env observations
        """

        # process and return only output we are expecting to
        processed_state = OrderedDict()
        if 'task_obs' in self.custom_output:
            processed_state['task_obs'] = np.concatenate([
                    self.robots[0].calc_state(),    # robot proprioceptive state
                    self.task.get_task_obs(self)[:-2], # goal x, y relative distance
            ])
        if 'rgb_obs' in self.custom_output:
            processed_state['rgb'] = state['rgb']  # [0, 1] range rgb image

        return processed_state
