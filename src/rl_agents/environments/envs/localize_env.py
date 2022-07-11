#!/usr/bin/env python3

import os
from collections import OrderedDict
from datetime import datetime

import cv2
import gym
import matplotlib.pyplot as plt
import numpy as np
from igibson.envs.igibson_env import iGibsonEnv
from igibson.utils.utils import l2_distance
from matplotlib.backends.backend_agg import FigureCanvasAgg

from igibson.utils.utils import l2_distance, quatToXYZW, rotate_vector_2d, rotate_vector_3d
from igibson.reward_functions.collision_reward import CollisionReward
from igibson.termination_conditions.out_of_bound import OutOfBound
from igibson.termination_conditions.timeout import Timeout

from ..env_utils import datautils, render

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
from pathlib import Path
from pfnetwork import pfnet
import tensorflow as tf


class LocalizeGibsonEnv(iGibsonEnv):
    """
    Custom implementation of localization task extending iGibsonEnv's functionality
    """

    def __init__(
            self,
            config_file,
            scene_id=None,
            mode='headless',
            pfnet_model=None,
            action_timestep=1 / 10.0,
            physics_timestep=1 / 240.0,
            device_idx=0,
            render_to_tensor=False,
            automatic_reset=False,
            pf_params=None,
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
        :param pf_params: argparse.Namespace parsed command-line arguments to initialize pfnet
        """
        self.pf_params = pf_params

        if hasattr(pf_params, "high_res") and pf_params.high_res == True:
            config_file = config_file.replace('.yaml', '_high_res.yaml')

        if isinstance(scene_id, (list, np.ndarray)):
            self.scene_ids = scene_id
            scene_id = np.random.choice(scene_id)
        else:
            self.scene_ids = []

        super(LocalizeGibsonEnv, self).__init__(
            config_file=config_file,
            scene_id=scene_id,
            mode=mode,
            action_timestep=action_timestep,
            physics_timestep=physics_timestep,
            device_idx=device_idx,
            render_to_tensor=render_to_tensor,
            automatic_reset=automatic_reset)

        # For the igibson maps, originally each pixel represents 0.01m, and the center of the image correspond to (0,0)
        # self.map_pixel_in_meters = 0.01
        # in igibson we work with rescaled 0.01m to 0.1m maps to sample robot poses
        self.trav_map_resolution = self.config['trav_map_resolution']
        assert self.trav_map_resolution == pf_params.map_pixel_in_meters
        self.depth_th = 3.
        self.robot_size_px = 0.3 / self.trav_map_resolution  # 0.03m
        assert self.config['max_step'] // pf_params.loop == pf_params.trajlen, (
        self.config['max_step'], pf_params.trajlen)

        self.pfnet_model = pfnet_model
        assert pf_params is not None
        self.use_pfnet = self.pfnet_model is not None
        if self.pf_params.use_plot:
            self.init_pfnet_plots()

        observation_space = OrderedDict()

        task_obs_dim = 7 + self.pf_params.observe_steps  # robot_prorpio_state (18)
        if 'task_obs' in self.pf_params.custom_output:
            observation_space['task_obs'] = gym.spaces.Box(
                low=-1000.0, high=+1000.0,
                shape=(task_obs_dim,),
                dtype=np.float32)
        # image_height and image_width are obtained from env config file
        if 'rgb_obs' in self.pf_params.custom_output:
            observation_space['rgb_obs'] = gym.spaces.Box(
                low=0.0, high=1.0,
                shape=(self.image_height, self.image_width, 3),
                dtype=np.float32)
        if 'depth_obs' in self.pf_params.custom_output:
            observation_space['depth_obs'] = gym.spaces.Box(
                low=0.0, high=1.0,
                shape=(self.image_height, self.image_width, 1),
                dtype=np.float32)
        if 'floor_map' in self.pf_params.custom_output:
            observation_space['floor_map'] = gym.spaces.Box(
                low=0.0, high=1.0,
                shape=self.pf_params.global_map_size,
                dtype=np.float32)
        if 'likelihood_map' in self.pf_params.custom_output:
            if np.max(self.pf_params.global_map_size) > 100:
                shape = [50, 50]
            else:
                shape = self.pf_params.global_map_size[:2]
            observation_space['likelihood_map'] = gym.spaces.Box(
                low=-10.0, high=+10.0,
                shape=(*shape, 4),
                dtype=np.float32)
        if "occupancy_grid" in self.pf_params.custom_output:
            self.grid_resolution = self.config.get("grid_resolution", 128)
            observation_space['occupancy_grid'] = gym.spaces.Box(
                low=0.0, high=1.0,
                shape=(self.grid_resolution, self.grid_resolution, 1),
                dtype=np.float32)
        if "scan_obs" in self.pf_params.custom_output:
            observation_space["scan_obs"] = gym.spaces.Box(
                low=0.0, high=1.0,
                shape=(self.n_horizontal_rays * self.n_vertical_beams, 1),
                dtype=np.float32)

        self.observation_space = gym.spaces.Dict(observation_space)

        print("=====> LocalizeGibsonEnv initialized")

        self.last_reward = 0.0

    def load(self):
        self.config["max_step"] = self.pf_params.trajlen * self.pf_params.loop
        return super(LocalizeGibsonEnv, self).load()

    def load_task_setup(self):
        super(LocalizeGibsonEnv, self).load_task_setup()
        # override task rewards as we don't want to use all of them
        self.task.termination_conditions = [
            # MaxCollision(self.config),
            Timeout(self.config),
            OutOfBound(self.config),
            # PointGoal(self.config),
        ]
        self.task.reward_functions = [
            # PotentialReward(self.config),
            CollisionReward(self.config),
            # PointGoalReward(self.config),
        ]

    @staticmethod
    def _get_empty_env_plots():
        return {
            'map_plt': None,
            'robot_gt_plt': {
                'position_plt': None,
                'heading_plt': None,
            },
            'robot_est_plt': {
                'position_plt': None,
                'heading_plt': None,
                'particles_plt': None,
            },
            'step_txt_plt': None,
        }

    def init_pfnet_plots(self):
        """
        Initialize Particle Filter

        :param pf_params: argparse.Namespace parsed command-line arguments to initialize pfnet
        """
        # code related to displaying/storing results in matplotlib
        self.fig = plt.figure(figsize=(len(self.observation_space) * 6, 7))
        self.plt_ax = None
        self.env_plts = self._get_empty_env_plots()

        # FigureCanvasAgg and ion is not working together
        current_time = datetime.now().strftime("%Y%m%d-%H%M%S")
        self.out_folder = os.path.join(self.pf_params.root_dir, f'episode_run_{current_time}')
        Path(self.out_folder).mkdir(parents=True, exist_ok=True)
        if self.pf_params.store_plot:
            self.canvas = FigureCanvasAgg(self.fig)
        else:
            plt.ion()
            plt.show()

    def _reset_vars(self):
        self.floor_map = None

        self.eps_obs = {
            'rgb': [],
            'depth': [],
            'occupancy_grid': []
        }
        self.curr_plt_images = []
        self.curr_pfnet_state = None
        self.curr_obs = None
        self.curr_gt_pose = None
        self.curr_est_pose = None

    def load_miscellaneous_variables(self):
        """
        Load miscellaneous variables for book keeping
        """

        super(LocalizeGibsonEnv, self).load_miscellaneous_variables()
        self._reset_vars()

    def reset_variables(self):
        """
        Reset bookkeeping variables for the next new episode
        """

        super(LocalizeGibsonEnv, self).reset_variables()
        self._reset_vars()

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

        # we use low update frequency
        collisions = 0
        for _ in range(self.pf_params.loop):
            state, reward_unused, done, info = super(LocalizeGibsonEnv, self).step(action)
            collisions += len(self.collision_links) > 0
        # NOTE: completely overriding the env reward here !
        has_collision = (collisions > 0)
        reward = - self.pf_params.collision_reward_weight * has_collision
        info['collision_penalty'] = reward  # contains only collision reward per step

        # perform particle filter update
        if self.use_pfnet:
            loss_dict = self.step_pfnet(state)
            info['pred'] = loss_dict['pred'].cpu().numpy()
            info['coords'] = loss_dict['coords'].cpu().numpy()
            info['orient'] = loss_dict['orient'].cpu().numpy()

            # include pfnet's estimate in environment's reward
            if self.pf_params.reward == 'pred':
                rescale = 1
                reward -= tf.squeeze(loss_dict['pred']).cpu().numpy() / rescale
            elif self.pf_params.reward == 'belief':
                particles = self.curr_pfnet_state[0]

                dist_thresh = 0.4
                angle_thresh = 0.2
                correct_dist = np.linalg.norm(particles[..., :2] - self.curr_gt_pose[..., :2], axis=-1) < dist_thresh
                correct_angle = np.abs(particles[..., 2] - self.curr_gt_pose[..., 2]) < angle_thresh
                correct_share = np.sum(np.logical_and(correct_dist, correct_angle)) / self.pf_params.num_particles

                reward = 10 * correct_share
            else:
                raise ValueError(self.pf_params.reward)

        # just for the render function
        self.last_reward = reward
        info['reward'] = reward

        # return custom environment observation
        custom_state = self.process_state(state)
        return custom_state, reward, done, info

    def reset(self):
        """
        Reset episode

        :return: state: new observation
        """
        if self.pf_params.use_plot:
            self.last_video_path = self.store_results()
            # self.store_obs()

            # clear subplots
            self.fig.clear()

            num_subplots = 6

            self.plt_ax = [self.fig.add_subplot(1, num_subplots, i + 1) for i in range(num_subplots)]
            [ax.set_axis_off() for ax in self.plt_ax]
            self.env_plts = self._get_empty_env_plots()

        if self.scene_ids:
            scene_id = np.random.choice(self.scene_ids)
            if scene_id != self.config["scene_id"]:
                curr_ep, curr_step = self.current_episode, self.current_step
                self.reload_model(scene_id)
                self.current_episode, self.current_step = curr_ep, curr_step
        state = super(LocalizeGibsonEnv, self).reset()

        # process new env map
        self.floor_map, self.org_map_shape, self.trav_map = self.get_floor_map(
            pad_map_size=self.pf_params.global_map_size)

        # perform particle filter update
        if self.use_pfnet:
            # get latest robot state
            new_robot_state = self.robots[0].calc_state()
            # process new robot state: convert coords to pixel space
            self.curr_gt_pose = self.get_robot_pose(new_robot_state)
            init_particles, init_particle_weights = pfnet.PFCell.reset(robot_pose_pixel=self.curr_gt_pose, env=self,
                                                                       params=self.pf_params)
            self.curr_est_pose = pfnet.PFCell.get_est_pose(particles=init_particles,
                                                           particle_weights=init_particle_weights)

            self.curr_pfnet_state = [init_particles, init_particle_weights, self.floor_map[None]]

        # return custom environment observation
        custom_state = self.process_state(state)
        return custom_state

    def process_state(self, state):
        """
        Perform additional processing of environment's observation.

        :param state: env observations

        :return: processed_state: processed env observations
        """
        assert np.min(state['rgb']) >= 0. and np.max(state['rgb']) <= 1., (np.min(state['rgb']), np.max(state['rgb']))
        assert np.min(state['depth']) >= 0. and np.max(state['depth']) <= 1., (
        np.min(state['depth']), np.max(state['depth']))

        self.eps_obs['rgb'].append((state['rgb'] * 255).astype(np.uint8))
        self.eps_obs['depth'].append(cv2.applyColorMap((state['depth'] * 255).astype(np.uint8), cv2.COLORMAP_JET))
        self.eps_obs['occupancy_grid'].append(
            cv2.cvtColor((state['occupancy_grid'] * 255).astype(np.uint8), cv2.COLOR_GRAY2BGR))

        # process and return only output we are expecting to
        processed_state = OrderedDict()
        if 'task_obs' in self.pf_params.custom_output:
            rpy = self.robots[0].get_rpy()
            # rotate linear and angular velocities to local frame
            lin_vel = rotate_vector_3d(self.robots[0].get_linear_velocity(), *rpy)
            ang_vel = rotate_vector_3d(self.robots[0].get_angular_velocity(), *rpy)
            steps_remaining = max(self.pf_params.trajlen - self.current_step, 0)
            # processed_state['task_obs'] = self.robots[0].calc_state()  # robot proprioceptive state
            if self.pf_params.observe_steps:
                processed_state['task_obs'] = np.concatenate(
                    [lin_vel, ang_vel, [len(self.collision_links) > 0], [steps_remaining]])
            else:
                processed_state['task_obs'] = np.concatenate([lin_vel, ang_vel, [len(self.collision_links) > 0]])
        if 'rgb_obs' in self.pf_params.custom_output:
            processed_state['rgb_obs'] = state['rgb']  # [0, 1] range rgb image
        if 'depth_obs' in self.pf_params.custom_output:
            processed_state['depth_obs'] = state['depth']  # [0, 1] range depth image
        if 'scan_obs' in self.pf_params.custom_output:
            processed_state['scan_obs'] = state['scan']
        if 'occupancy_grid' in self.pf_params.custom_output:
            # robot is at center facing right in grid
            processed_state['occupancy_grid'] = state['occupancy_grid']  # [0: occupied, 0.5: unknown, 1: free]
        if 'floor_map' in self.pf_params.custom_output:
            if self.floor_map is None:
                floor_map, self.org_map_shape = self.get_floor_map(pad_map_size=self.pf_params.global_map_size)
                processed_state['floor_map'] = floor_map
            else:
                processed_state['floor_map'] = self.floor_map  # [0, 2] range floor map
        if 'likelihood_map' in self.pf_params.custom_output:
            lmap = pfnet.PFCell.get_likelihood_map(particles=self.curr_pfnet_state[0],
                                                   particle_weights=self.curr_pfnet_state[1],
                                                   floor_map=self.floor_map)
            if np.max(self.pf_params.global_map_size) > 100:
                # take a cut-out around the current best estimate
                lmap = pfnet.PFCell.transform_maps(global_map=tf.convert_to_tensor(lmap[None], tf.float32),
                                                   particle_states=tf.convert_to_tensor(self.curr_est_pose, tf.float32)[
                                                       None, None],
                                                   local_map_size=(50, 50),
                                                   window_scaler=self.pf_params.window_scaler,
                                                   agent_at_bottom=False,
                                                   flip_map=True)
                lmap = np.squeeze(tf.stop_gradient(lmap))
            processed_state['likelihood_map'] = lmap
        if 'obstacle_obs' in self.pf_params.custom_output:
            # check for close obstacles to robot
            min_depth = np.min(state['depth'] * 100, axis=0)
            s = min_depth.shape[0] // 4
            left = np.min(min_depth[:s]) < self.depth_th
            left_front = np.min(min_depth[s:2 * s]) < self.depth_th
            right_front = np.min(min_depth[2 * s:3 * s]) < self.depth_th
            right = np.min(min_depth[3 * s:]) < self.depth_th
            processed_state['obstacle_obs'] = np.array([left, left_front, right_front, right])

        return processed_state

    def step_pfnet(self, new_state):
        """
        Perform one particle filter update step
        :param new_obs: latest observation from env.step()
        :return loss_dict: dictionary of total loss and coordinate loss (in meters)
        """

        batch_size = self.pf_params.batch_size
        num_particles = self.pf_params.num_particles
        obs_ch = self.pf_params.obs_ch
        obs_mode = self.pf_params.obs_mode

        # get latest robot state
        new_robot_state = self.robots[0].calc_state()

        def process_image(img, resize=None):
            if resize is not None:
                img = cv2.resize(img, resize)
            return np.atleast_3d(img.astype(np.float32))

        # process new robot state: convert coords to pixel space
        new_gt_pose = self.get_robot_pose(new_robot_state)

        # calculate actual odometry b/w old pose and new pose
        old_gt_pose = self.curr_gt_pose
        assert list(old_gt_pose.shape) == [3] and list(new_gt_pose.shape) == [
            3], f'{old_gt_pose.shape}, {new_gt_pose.shape}'
        odometry = datautils.calc_odometry(old_gt_pose, new_gt_pose)

        # add traj_dim
        new_depth_obs = process_image(new_state['depth'], resize=(56, 56))
        new_rgb_obs = process_image(new_state['rgb'], resize=(56, 56))
        if obs_mode == 'rgb-depth':
            observation = tf.concat([new_rgb_obs, new_depth_obs], axis=-1)
        elif obs_mode == 'depth':
            observation = new_depth_obs
        elif obs_mode == 'rgb':
            observation = new_rgb_obs
        elif obs_mode == 'occupancy_grid':
            if self.pf_params.likelihood_model == 'learned':
                observation = process_image(new_state['occupancy_grid'], resize=(56, 56))
            else:
                observation = process_image(new_state['occupancy_grid'], resize=None)
        else:
            raise ValueError(obs_mode)

        # sanity check
        def _add_batch_dim(x, add_traj_dim: bool = False):
            x = tf.expand_dims(tf.convert_to_tensor(x, dtype=tf.float32), axis=0)
            if add_traj_dim:
                x = tf.expand_dims(x, 1)
            return x

        odometry = _add_batch_dim(odometry, add_traj_dim=True)
        observation = _add_batch_dim(observation, add_traj_dim=True)

        old_pfnet_state = self.curr_pfnet_state
        trajlen = 1
        assert list(odometry.shape) == [batch_size, trajlen, 3], f'{odometry.shape}'
        assert list(observation.shape) in [[batch_size, trajlen, 56, 56, obs_ch],
                                           [batch_size, trajlen, 128, 128, obs_ch]], f'{observation.shape}'
        assert list(old_pfnet_state[0].shape) == [batch_size, num_particles, 3], f'{old_pfnet_state[0].shape}'
        assert list(old_pfnet_state[1].shape) == [batch_size, num_particles], f'{old_pfnet_state[1].shape}'
        assert list(old_pfnet_state[2].shape) == [batch_size] + list(
            self.floor_map.shape), f'{old_pfnet_state[2].shape}'

        curr_input = [observation, odometry]
        output, new_pfnet_state = self.pfnet_model((curr_input, old_pfnet_state), training=False)
        particles, particle_weights = output
        self.curr_est_pose = tf.stop_gradient(
            pfnet.PFCell.get_est_pose(particles=particles, particle_weights=particle_weights))

        loss_dict = pfnet.PFCell.compute_mse_loss(particles=particles,
                                                  particle_weights=particle_weights,
                                                  true_states=_add_batch_dim(new_gt_pose),
                                                  trav_map_resolution=self.trav_map_resolution)

        # latest robot's pose, observation and particle filter state
        self.curr_pfnet_state = new_pfnet_state
        self.curr_gt_pose = new_gt_pose

        return loss_dict

    def set_scene(self, scene_id, floor_num):
        """
        Override the task floor number

        :param str: scene id
        :param int: task floor number
        """
        self.config['scene_id'] = scene_id
        self.task.floor_num = floor_num

    def get_floor_map(self, scene_id=None, floor_num=None, pad_map_size=None):
        """
        Get the scene floor map (traversability map + obstacle map)

        :param str: scene id
        :param int: task floor number
        :return ndarray: floor map of current scene (H, W, 1)
        """
        if scene_id is not None and floor_num is not None:
            self.set_scene(scene_id, floor_num)

        occupancy_map_small, occupancy_map_small_shape, trav_map = datautils.get_floor_map(
            scene_id=self.config.get('scene_id'),
            floor_num=self.task.floor_num,
            trav_map_resolution=self.trav_map_resolution,
            trav_map_erosion=self.config.get('trav_map_erosion', 2),
            pad_map_size=pad_map_size)

        self.trav_map_size = np.array(trav_map.shape[:2])
        return occupancy_map_small, occupancy_map_small_shape, trav_map

    def map_to_world(self, xy):
        """
        Transforms a 2D point in map reference frame into world (simulator) reference frame

        :param xy: 2D location in map reference frame (image)
        :return: 2D location in world reference frame (metric)
        """
        # axis = 0 if len(xy.shape) == 1 else 1
        # return np.flip((xy - self.trav_map_size / 2.0) * self.trav_map_resolution, axis=axis)
        return self.scene.map_to_world(xy)

    def world_to_map(self, xy):
        """
        Transforms a 2D point in world (simulator) reference frame into map reference frame

        :param xy: 2D location in world reference frame (metric)
        :return: 2D location in map reference frame (image)
        """
        # return np.flip((np.array(xy) / self.trav_map_resolution + self.trav_map_size / 2.0)).astype(int)
        return self.scene.world_to_map(xy)

    def get_random_points_map(self, npoints, true_mask=None):
        """
        Sample a random point on the given floor number. If not given, sample a random floor number.

        :param floor: floor number
        :return floor: floor number
        :return point: randomly sampled point in [x, y, z]
        """
        return datautils.get_random_points_map(npoints=npoints, trav_map=self.trav_map, true_mask=true_mask)

    def get_robot_pose(self, robot_state):
        """
        Transform robot's pose from coordinate space to pixel space.
        """
        robot_pos = robot_state[0:3]  # [x, y, z]
        robot_orn = robot_state[3:6]  # [r, p, y]

        # transform from co-ordinate space [x, y] to pixel space [col, row]
        robot_pose_px = np.array([*self.world_to_map(robot_pos[:2]), robot_orn[2]])  # [x, y, theta]

        return robot_pose_px

    def plot_robot_pose(self, pose=None, floor_map=None):
        if pose is None:
            pose = self.curr_gt_pose
        if floor_map is None:
            floor_map = self.floor_map
        f, ax = plt.subplots(1, 1)
        render.draw_floor_map(floor_map, self.org_map_shape, ax, None)
        render.draw_robot_pose(pose, '#7B241C', floor_map.shape, ax, None, None, plt_path=True)
        return f, ax

    def render(self, mode='human', particles=None, particle_weights=None, floor_map=None, observation=None,
               gt_pose=None, current_step=None, est_pose=None, info=None):
        """
        Render plots
        """
        # TODO: remove the whole render function because now have render_paper()?
        if self.pf_params.use_plot:
            if particles is None:
                if self.curr_pfnet_state is not None:
                    particles, particle_weights, floor_map = self.curr_pfnet_state
                    floor_map = np.squeeze(floor_map, 0)
                else:
                    floor_map = self.floor_map
                if gt_pose is None:
                    gt_pose = self.curr_gt_pose
                if est_pose is None:
                    est_pose = np.squeeze(self.curr_est_pose) if (self.curr_est_pose is not None) else None
                current_step = self.current_step
            else:
                est_pose = np.squeeze(pfnet.PFCell.get_est_pose(particles=particles, particle_weights=particle_weights))

            # if self.use_pfnet and ("likelihood_map" in self.pf_params.custom_output):
            likelihood_map = pfnet.PFCell.get_likelihood_map(particles=particles,
                                                             particle_weights=particle_weights,
                                                             floor_map=floor_map)

            map_plt = self.env_plts['map_plt']
            self.env_plts['map_plt'] = render.draw_floor_map(0.025 * likelihood_map[..., 0] + likelihood_map[..., 1],
                                                             self.org_map_shape, self.plt_ax[0], map_plt, cmap=None)

            # ground truth robot pose and heading
            color = '#7B241C'
            position_plt = self.env_plts['robot_gt_plt']['position_plt']
            heading_plt = self.env_plts['robot_gt_plt']['heading_plt']
            position_plt, heading_plt = render.draw_robot_pose(
                gt_pose,
                color,
                floor_map.shape,
                self.plt_ax[0],
                position_plt,
                heading_plt,
                plt_path=True)
            self.env_plts['robot_gt_plt']['position_plt'] = position_plt
            self.env_plts['robot_gt_plt']['heading_plt'] = heading_plt

            # estimated robot pose and heading
            if est_pose is not None:
                color = '#515A5A'
                # est_pose = np.squeeze(self.curr_est_pose[0].cpu().numpy())
                position_plt = self.env_plts['robot_est_plt']['position_plt']
                heading_plt = self.env_plts['robot_est_plt']['heading_plt']
                position_plt, heading_plt = render.draw_robot_pose(
                    est_pose,
                    color,
                    floor_map.shape,
                    self.plt_ax[0],
                    position_plt,
                    heading_plt,
                    plt_path=False)
                self.env_plts['robot_est_plt']['position_plt'] = position_plt
                self.env_plts['robot_est_plt']['heading_plt'] = heading_plt

            # pose mse in mts
            if (est_pose is not None) and (gt_pose is not None):
                gt_pose_mts = np.array([*self.map_to_world(gt_pose[:2]), gt_pose[2]])
                est_pose_mts = np.array([*self.map_to_world(est_pose[:2]), est_pose[2]])
                pose_diff = gt_pose_mts - est_pose_mts
                pose_diff[-1] = datautils.normalize(pose_diff[-1])  # normalize
                pose_error = np.linalg.norm(pose_diff[..., :2])
            else:
                pose_error = 0.
            has_collision = ' True' if len(self.collision_links) > 0 else 'False'

            step_txt_plt = self.env_plts['step_txt_plt']
            itext = f"i: {float(info['coords']):.3f}, {float(info['orient']):.3f}, {float(info['pred']):.3f}\n" if info is not None else ""
            step_txt_plt = render.draw_text(
                f"pose mse: {pose_error:.3f},{pose_diff[-1]:.3f}\n{itext}current step: {current_step // self.pf_params.loop:02.0f}\nlast reward: {np.squeeze(self.last_reward):.3f}\ncollision: {has_collision}",
                '#FFFFFF', self.plt_ax[0], step_txt_plt,
                alpha=1.0, x=1.0, y=-0.1)
            self.env_plts['step_txt_plt'] = step_txt_plt

            self.plt_ax[0].legend([self.env_plts['robot_gt_plt']['position_plt'],
                                   self.env_plts['robot_est_plt']['position_plt']],
                                  ["GT Pose", "Est Pose"], loc='upper left', fontsize=12)

            next_subplot = 1
            if (observation is not None) and isinstance(observation, dict):
                if 'likelihood_map' in observation:
                    self.plt_ax[next_subplot].imshow(
                        0.025 * observation['likelihood_map'][..., 0] + observation['likelihood_map'][..., 1])
                    next_subplot += 1
            if (observation is not None) and self.pf_params.obs_mode == "rgb-depth":
                self.plt_ax[next_subplot + 1].imshow(np.squeeze(observation[..., 3]))
                observation = observation[..., :3]
                self.plt_ax[next_subplot].imshow(np.squeeze(observation))
            else:
                if self.eps_obs.get('occupancy_grid', None):
                    self.plt_ax[next_subplot].imshow(np.rot90(self.eps_obs['occupancy_grid'][-1], 1))
                    next_subplot += 1
                if self.eps_obs.get('rgb', None):
                    self.plt_ax[next_subplot].imshow(self.eps_obs['rgb'][-1])
                    next_subplot += 1
                if self.eps_obs.get('depth', None):
                    self.plt_ax[next_subplot].imshow(self.eps_obs['depth'][-1])
                    next_subplot += 1

            # plot the local map extracted for the ground-truth pose
            local_map = pfnet.PFCell.transform_maps(global_map=tf.convert_to_tensor(floor_map[None], tf.float32),
                                                    particle_states=tf.convert_to_tensor(gt_pose, tf.float32)[
                                                        None, None],
                                                    local_map_size=(28, 28),
                                                    window_scaler=self.pf_params.window_scaler,
                                                    agent_at_bottom=True,
                                                    flip_map=True)
            self.plt_ax[next_subplot].imshow(np.squeeze(local_map))

            # remove any potential padding / empty space around it
            l = .025 * likelihood_map[..., 0] + likelihood_map[..., 1]
            bb = pfnet.PFCell.bounding_box(l != 0)
            ylim = [max(bb[0] - 5, 0), min(bb[1] + 5, l.shape[0])]
            xlim = [max(bb[2] - 5, 0), min(bb[3] + 5, l.shape[0])]
            # make quadratic to fit plot nicely
            sz = max([ylim[1] - ylim[0], xlim[1] - xlim[0]])
            for lim in (ylim, xlim):
                diff = (lim[1] - lim[0]) // 2
                mid = lim[0] + diff
                lim[0], lim[1] = mid - sz // 2, mid + sz // 2
            self.plt_ax[0].set_ylim(ylim)
            self.plt_ax[0].set_xlim(xlim)

            if self.pf_params.store_plot:
                self.canvas.draw()
                plt_img = np.array(self.canvas.renderer._renderer)
                plt_img = cv2.cvtColor(plt_img, cv2.COLOR_RGB2BGR)
                self.curr_plt_images.append(plt_img)
            else:
                plt.draw()
                plt.pause(0.00000000001)

    def close(self):
        """
        environment close()
        """
        super(LocalizeGibsonEnv, self).close()

        # store the plots as video
        if self.pf_params.use_plot:
            if self.pf_params.store_plot:
                self.store_results()
                # self.store_obs()
            else:
                # to prevent plot from closing after environment is closed
                plt.ioff()
                plt.show()

        print("=====> iGibsonEnv closed")

    @staticmethod
    def convert_imgs_to_video(images, file_path):
        """
        Convert images to video
        """
        fps = 5
        frame_size = (images[0].shape[1], images[0].shape[0])
        out = cv2.VideoWriter(file_path,
                              cv2.VideoWriter_fourcc(*'MP4V'),
                              fps,
                              frame_size)
        for img in images:
            out.write(img)
        out.release()

    def store_obs(self):
        """
        Store the episode environment's observations as video
        """
        for m in ['rgb', 'depth', 'occupancy_grid']:
            if len(self.eps_obs[m]) > 1:
                file_path = os.path.join(self.out_folder, f'{m}_episode_run_{self.current_episode}.mp4')
                self.convert_imgs_to_video(self.eps_obs[m], file_path)
                print(f'stored {m} imgs {len(self.eps_obs[m])} to {file_path}')
                self.eps_obs[m] = []

    def store_results(self):
        """
        Store the episode environment's belief map/likelihood map as video
        """
        if len(self.curr_plt_images) > 0:
            file_path = os.path.join(self.out_folder, f'episode_run_{self.current_episode}.mp4')
            self.convert_imgs_to_video(self.curr_plt_images, file_path)
            print(f'stored img results {len(self.curr_plt_images)} eps steps to {file_path}')
            self.curr_plt_images = []
            return file_path
        else:
            print('no plots available to store, check if env.render() is being called')

    def __del__(self):
        # if len(self.curr_plt_images) > 0:
        self.close()
