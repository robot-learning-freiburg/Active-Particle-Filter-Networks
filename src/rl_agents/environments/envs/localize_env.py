#!/usr/bin/env python3

from collections import OrderedDict
from gibson2.envs.igibson_env import iGibsonEnv
import gym
import numpy as np

class LocalizeGibsonEnv(iGibsonEnv):
    """
    Custom implementation of localization task based on iGibsonEnv
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

        super(LocalizeGibsonEnv, self).__init__(
                        config_file=config_file,
                        scene_id=scene_id,
                        mode=mode,
                        action_timestep=action_timestep,
                        physics_timestep=physics_timestep,
                        device_idx=device_idx,
                        render_to_tensor=render_to_tensor,
                        automatic_reset=automatic_reset)

        # manually remove point navigation task termination and reward conditions
        del self.task.termination_conditions[-1]
        del self.task.reward_functions[-1]

        # override observation_space
        # task_obs_dim = robot_prorpio_state (18) + goal_coords (2)
        TASK_OBS_DIM = 18

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

    def load_miscellaneous_variables(self):
        """
        Load miscellaneous variables for book keeping
        """

        super(LocalizeGibsonEnv, self).load_miscellaneous_variables()

        self.obstacle_map = None
        self.floor_map = None

        self.curr_pfnet_state = None
        self.curr_rgb_obs = None
        self.curr_gt_pose = None

    def reset_variables(self):
        """
        Reset bookkeeping variables for the next new episode
        """

        super(LocalizeGibsonEnv, self).reset_variables()

        self.obstacle_map = None
        self.floor_map = None

        self.curr_pfnet_state = None
        self.curr_rgb_obs = None
        self.curr_gt_pose = None

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

        state, reward, done, info = super(LocalizeGibsonEnv, self).step(action)

        custom_state = self.process_state(state)

        return custom_state, reward, done, info

    def reset(self):
        """
        Reset episode

        :return: state: new observation
        """

        state = super(LocalizeGibsonEnv, self).reset()

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
            ])
        if 'rgb_obs' in self.custom_output:
            custom_state['rgb'] = state['rgb']  # [0, 1] range rgb image

        return processed_state

    def step_pfnet(self, new_rgb_obs):
        """
        """

        old_rgb_obs = self.curr_rgb_obs
        old_gt_pose = self.curr_gt_pose[0].numpy()
        old_pfnet_state = self.curr_pfnet_state

        # get new robot state
        new_robot_state = self.robots[0].calc_state()

        # process new rgb observation: convert to [-1, +1] range
        new_rgb_obs = datautils.process_raw_image(new_rgb_obs)

        # process new robot state: convert coords to pixel space
        new_pose = self.get_robot_pose(robot_state, floor_map.shape)

        # calculate actual odometry b/w old pose and new pose
        assert list(old_pose.shape) == [3] and list(new_pose.shape) == [3]
        new_odom = datautils.calc_odometry(old_pose, new_pose)

        # convert to tensor
        new_rgb_obs = tf.expand_dims(
                    tf.convert_to_tensor(new_rgb_obs, dtype=tf.float32)
                    , axis=0)
        new_odom = tf.expand_dims(
                    tf.convert_to_tensor(new_odom, dtype=tf.float32)
                    , axis=0)
        new_pose = tf.expand_dims(
                        tf.convert_to_tensor(new_pose, dtype=tf.float32)
                        , axis=0)

        # construct pfnet input
        odometry = tf.expand_dims(new_odom, axis=1)
        observation = tf.expand_dims(old_rgb_obs, axis=1)
        input = [observation, odometry]
        model_input = (input, old_pfnet_state)


        """
        ## HACK:
            if stateful: reset RNN s.t. initial_state is set to initial particles and weights
                start of each trajectory
            if non-stateful: pass the state explicity every step
        """
        if self.stateful:
            self.pfnet_model.layers[-1].reset_states(old_pfnet_state)    # RNN layer

        # forward pass pfnet
        # output: contains particles and weights before transition update
        # pfnet_state: contains particles and weights after transition update
        output, new_pfnet_state = self.pfnet_model(model_input, training=False)

        # compute pfnet loss
        particles, particle_weights = output
        true_old_pose = tf.expand_dims(self.curr_gt_pose, axis=1)
        loss_dict = pfnet_loss.compute_loss(particles, particle_weights, true_old_pose, self.map_pixel_in_meters)

        ## TODO: may need better reward
        # compute reward
        reward = reward - tf.squeeze(loss_dict['coords']).numpy()

        self.curr_pfnet_state = new_pfnet_state
        self.curr_gt_pose = new_pose
        self.curr_rgb_obs = new_rgb_obs

    def reset_pfnet(self, new_rgb_obs):
        """
        """

        # get new robot state
        new_robot_state = self.robots[0].calc_state()

        # process new rgb observation: convert to [-1, +1] range
        new_rgb_obs = datautils.process_raw_image(new_rgb_obs)

        # process new robot state: convert coords to pixel space
        new_pose = self.get_robot_pose(robot_state, floor_map.shape)

        # process new env map
        new_floor_map = self.get_floor_map()
        new_obstacle_map = self.get_obstacle_map()

        # get random particles based on init distribution conditions
        rnd_particles = self.get_random_particles(
                                num_particles,
                                particles_distr,
                                true_pose.numpy(),
                                floor_map[0],
                                particles_cov)

        # convert to tensor
        new_rgb_obs = tf.expand_dims(
                    tf.convert_to_tensor(new_rgb_obs, dtype=tf.float32)
                    , axis=0)
        new_pose = tf.expand_dims(
                        tf.convert_to_tensor(new_pose, dtype=tf.float32)
                        , axis=0)
        new_floor_map = tf.expand_dims(
                    tf.convert_to_tensor(new_floor_map, dtype=tf.float32)
                    , axis=0)
        new_obstacle_map = tf.expand_dims(
                    tf.convert_to_tensor(new_obstacle_map, dtype=tf.float32)
                    , axis=0)
        new_init_particles = tf.convert_to_tensor(rnd_particles, dtype=tf.float32)
        new_init_particle_weights = tf.constant(
                                    np.log(1.0/float(num_particles)),
                                    shape=(batch_size, num_particles),
                                    dtype=tf.float32)

        self.obstacle_map = obstacle_map
        self.floor_map = floor_map
        self.curr_pfnet_state = [init_particles, init_particle_weights, obstacle_map]
        self.curr_gt_pose = new_pose
        self.curr_rgb_obs = new_rgb_obs
