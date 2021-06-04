#!/usr/bin/env python3

from absl import flags
import argparse
from collections import OrderedDict
import copy
import cv2
from ..env_utils import datautils
from ..env_utils import pfnet_loss
from ..env_utils import render
from gibson2.envs.igibson_env import iGibsonEnv
from gibson2.utils.assets_utils import get_scene_path
import gym
from matplotlib.backends.backend_agg import FigureCanvasAgg
import matplotlib.pyplot as plt
import numpy as np
import os
from pathlib import Path
from PIL import Image
from pfnetwork import pfnet
import tensorflow as tf


class LocalizeGibsonEnv(iGibsonEnv):
    """
    Custom implementation of localization task based on iGibsonEnv
    """

    def __init__(
            self,
            config_file,
            scene_id=None,
            mode='headless',
            use_tf_function=True,
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

        argparser = argparse.ArgumentParser()
        self.pf_params = argparser.parse_args([])
        self.use_pfnet = flags.FLAGS.use_pfnet
        self.use_tf_function = use_tf_function
        if self.use_pfnet:
            self.init_pfnet(flags.FLAGS)
            task_obs_dim = 18 + 3 # robot_prorpio_state (18) + est_pose (3)
        else:
            task_obs_dim = 18 # robot_prorpio_state (18)
            self.pf_params.use_plot = False
            self.pf_params.store_plot = False

        # custom tf_agents we are using supports dict() type observations
        observation_space = OrderedDict()
        self.custom_output = ['task_obs', ]

        if 'task_obs' in self.custom_output:
            # HACK: use [-1k, +1k] range for TanhNormalProjectionNetwork to work
            observation_space['task_obs'] = gym.spaces.Box(
                low=-1000.0, high=+1000.0,
                shape=(task_obs_dim,),
                dtype=np.float32)
        # image_height and image_width are obtained from env config file
        if 'rgb_obs' in self.custom_output:
            observation_space['rgb_obs'] = gym.spaces.Box(
                low=0.0, high=1.0,
                shape=(self.image_height, self.image_width, 3),
                dtype=np.float32)

        self.observation_space = gym.spaces.Dict(observation_space)

        print("=====> LocalizeGibsonEnv initialized")

    def init_pfnet(self, FLAGS):
        """
        Initialize Particle Filter
        """

        assert 0.0 <= FLAGS.alpha_resample_ratio <= 1.0
        assert FLAGS.init_particles_distr in ['gaussian', 'uniform']
        assert len(FLAGS.transition_std) == len(FLAGS.init_particles_std) == 2

        self.pf_params.init_particles_distr = FLAGS.init_particles_distr
        self.pf_params.init_particles_std = np.array(FLAGS.init_particles_std, dtype=np.float32)
        self.pf_params.num_particles = FLAGS.num_particles
        self.pf_params.resample = FLAGS.resample
        self.pf_params.alpha_resample_ratio = FLAGS.alpha_resample_ratio
        self.pf_params.transition_std = np.array(FLAGS.transition_std, dtype=np.float32)
        self.pf_params.pfnet_load = FLAGS.pfnet_load
        self.pf_params.use_plot = FLAGS.use_plot
        self.pf_params.store_plot = FLAGS.store_plot

        self.pf_params.map_pixel_in_meters = 0.1
        self.pf_params.batch_size = 1
        self.pf_params.trajlen = 1
        self.pf_params.return_state = True
        self.pf_params.stateful = False
        self.pf_params.global_map_size = [1000, 1000, 1]
        self.pf_params.window_scaler = 8.0

        # build initial covariance matrix of particles, in pixels and radians
        particle_std2 = np.square(self.pf_params.init_particles_std.copy())  # variance
        self.pf_params.init_particles_cov = np.diag(particle_std2[(0, 0, 1),])

        # Create a new pfnet model instance
        self.pfnet_model = pfnet.pfnet_model(self.pf_params)
        print("=====> pfnet initialized")

        # load model from checkpoint file
        if self.pf_params.pfnet_load:
            self.pfnet_model.load_weights(self.pf_params.pfnet_load)
            print("=====> loaded pf model checkpoint " + self.pf_params.pfnet_load)

        if self.use_tf_function:
            print("=====> wrapped pfnet in tf.graph")
            self.pfnet_model = tf.function(self.pfnet_model)

        if self.pf_params.use_plot:
            # code related to displaying results in matplotlib
            self.fig = plt.figure(figsize=(7, 7))
            self.plt_ax = None
            self.env_plts = {
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

            # HACK FigureCanvasAgg and ion is not working together
            if self.pf_params.store_plot:
                self.canvas = FigureCanvasAgg(self.fig)
                self.out_folder = os.path.join('./', 'episode_runs')
                Path(self.out_folder).mkdir(parents=True, exist_ok=True)
            else:
                plt.ion()
                plt.show()

    def load_miscellaneous_variables(self):
        """
        Load miscellaneous variables for book keeping
        """

        super(LocalizeGibsonEnv, self).load_miscellaneous_variables()

        self.obstacle_map = None
        self.floor_map = None

        self.curr_plt_images = []
        self.curr_pfnet_state = None
        self.curr_rgb_obs = None
        self.curr_gt_pose = None
        self.curr_est_pose = None

    def reset_variables(self):
        """
        Reset bookkeeping variables for the next new episode
        """

        super(LocalizeGibsonEnv, self).reset_variables()

        self.obstacle_map = None
        self.floor_map = None

        self.curr_plt_images = []
        self.curr_pfnet_state = None
        self.curr_rgb_obs = None
        self.curr_gt_pose = None
        self.curr_est_pose = None

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
        if self.use_pfnet:
            new_rgb_obs = copy.deepcopy(state['rgb'])
            reward = self.step_pfnet(new_rgb_obs, reward)

        custom_state = self.process_state(state)
        return custom_state, reward, done, info

    def reset(self):
        """
        Reset episode

        :return: state: new observation
        """

        state = super(LocalizeGibsonEnv, self).reset()
        if self.use_pfnet:
            new_rgb_obs = copy.deepcopy(state['rgb'])
            self.reset_pfnet(new_rgb_obs)

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
            if self.use_pfnet:
                processed_state['task_obs'] = np.concatenate([
                    self.robots[0].calc_state(),  # robot proprioceptive state
                    self.curr_est_pose[0].numpy()  # gaussian mean of particles (x,y, theta)
                ])
            else:
                processed_state['task_obs'] = np.concatenate([
                    self.robots[0].calc_state(),  # robot proprioceptive state
                ])
            # print(np.min(processed_state['task_obs']), np.max(processed_state['task_obs']))
        if 'rgb_obs' in self.custom_output:
            processed_state['rgb_obs'] = state['rgb']  # [0, 1] range rgb image
            # cv2.imwrite('./test.png', processed_state['rgb_obs'] * 255)

        return processed_state

    def step_pfnet(self, new_rgb_obs, reward):
        """
        """

        trajlen = self.pf_params.trajlen
        batch_size = self.pf_params.batch_size
        num_particles = self.pf_params.num_particles
        pfnet_stateful = self.pf_params.stateful
        map_pixel_in_meters = self.pf_params.map_pixel_in_meters

        floor_map = self.floor_map[0]
        old_rgb_obs = self.curr_rgb_obs
        old_pose = self.curr_gt_pose[0].numpy()
        old_pfnet_state = self.curr_pfnet_state

        # get new robot state
        new_robot_state = self.robots[0].calc_state()

        # process new rgb observation: convert to [-1, +1] range
        new_rgb_obs = datautils.process_raw_image(new_rgb_obs)

        # process new robot state: convert coords to pixel space
        new_pose = self.get_robot_pose(new_robot_state, floor_map.shape)

        # calculate actual odometry b/w old pose and new pose
        assert list(old_pose.shape) == [3] and list(new_pose.shape) == [3]
        new_odom = datautils.calc_odometry(old_pose, new_pose)

        # convert to tensor
        new_rgb_obs = tf.expand_dims(
            tf.convert_to_tensor(new_rgb_obs, dtype=tf.float32), axis=0)
        new_odom = tf.expand_dims(
            tf.convert_to_tensor(new_odom, dtype=tf.float32), axis=0)
        new_pose = tf.expand_dims(
            tf.convert_to_tensor(new_pose, dtype=tf.float32), axis=0)
        odometry = tf.expand_dims(new_odom, axis=1)
        observation = tf.expand_dims(old_rgb_obs, axis=1)

        # sanity check
        assert list(odometry.shape) == [batch_size, trajlen, 3]
        assert list(observation.shape) == [batch_size, trajlen, 56, 56, 3]
        assert list(old_pfnet_state[0].shape) == [batch_size, num_particles, 3]
        assert list(old_pfnet_state[1].shape) == [batch_size, num_particles]

        # construct pfnet input
        curr_input = [observation, odometry]
        model_input = (curr_input, old_pfnet_state)

        """
        ## HACK:
            if stateful: reset RNN s.t. initial_state is set to initial particles and weights
                start of each trajectory
            if non-stateful: pass the state explicity every step
        """
        if pfnet_stateful:
            self.pfnet_model.layers[-1].reset_states(old_pfnet_state)  # RNN layer

        # forward pass pfnet
        # output: contains particles and weights before transition update
        # pfnet_state: contains particles and weights after transition update
        output, new_pfnet_state = self.pfnet_model(model_input, training=False)

        # compute pfnet loss
        particles, particle_weights = output
        true_old_pose = tf.expand_dims(self.curr_gt_pose, axis=1)

        assert list(true_old_pose.shape) == [batch_size, trajlen, 3]
        assert list(particles.shape) == [batch_size, trajlen, num_particles, 3]
        assert list(particle_weights.shape) == [batch_size, trajlen, num_particles]
        loss_dict = pfnet_loss.compute_loss(particles, particle_weights, true_old_pose, map_pixel_in_meters)

        # TODO: may need better reward
        # compute reward and normalize to range [-10, 0]
        reward = reward - tf.squeeze(loss_dict['coords']).numpy()
        reward = np.clip(reward / 100, -10, 0)

        self.curr_pfnet_state = new_pfnet_state
        self.curr_gt_pose = new_pose
        self.curr_est_pose = self.get_est_pose()
        self.curr_rgb_obs = new_rgb_obs

        return reward

    def reset_pfnet(self, new_rgb_obs):
        """
        obstacle_map: used as particle filter state
        floor_map: used for sampling init random particles
        """

        batch_size = self.pf_params.batch_size
        map_size = self.pf_params.global_map_size
        num_particles = self.pf_params.num_particles
        init_particles_cov = self.pf_params.init_particles_cov
        init_particles_distr = self.pf_params.init_particles_distr

        if self.pf_params.use_plot:
            # clear subplots
            plt.clf()
            self.plt_ax = self.fig.add_subplot(111)
            self.env_plts = {
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

            self.store_results()

        # get new robot state
        new_robot_state = self.robots[0].calc_state()

        # process new env map
        floor_map = self.get_floor_map()
        obstacle_map = self.get_obstacle_map()

        # process new rgb observation: convert to [-1, +1] range
        new_rgb_obs = datautils.process_raw_image(new_rgb_obs)

        # process new robot state: convert coords to pixel space
        new_pose = self.get_robot_pose(new_robot_state, floor_map.shape)

        # convert to tensor
        new_rgb_obs = tf.expand_dims(
            tf.convert_to_tensor(new_rgb_obs, dtype=tf.float32), axis=0)
        new_pose = tf.expand_dims(
            tf.convert_to_tensor(new_pose, dtype=tf.float32), axis=0)
        floor_map = tf.expand_dims(
            tf.convert_to_tensor(floor_map, dtype=tf.float32), axis=0)
        obstacle_map = tf.expand_dims(
            tf.convert_to_tensor(obstacle_map, dtype=tf.float32), axis=0)

        # get random particles and weights based on init distribution conditions
        init_particles = tf.cast(tf.convert_to_tensor(
            self.get_random_particles(
                num_particles,
                init_particles_distr,
                new_pose.numpy(),
                floor_map[0],
                init_particles_cov)), dtype=tf.float32)
        init_particle_weights = tf.constant(
            np.log(1.0 / float(num_particles)),
            shape=(batch_size, num_particles),
            dtype=tf.float32)

        # sanity check
        assert list(new_pose.shape) == [batch_size, 3]
        assert list(new_rgb_obs.shape) == [batch_size, 56, 56, 3]
        assert list(init_particles.shape) == [batch_size, num_particles, 3]
        assert list(init_particle_weights.shape) == [batch_size, num_particles]
        assert list(floor_map.shape) == [batch_size, *map_size]
        assert list(obstacle_map.shape) == [batch_size, *map_size]

        self.floor_map = floor_map
        self.obstacle_map = obstacle_map
        self.curr_pfnet_state = [init_particles, init_particle_weights, obstacle_map]
        self.curr_gt_pose = new_pose
        self.curr_est_pose = self.get_est_pose()
        self.curr_rgb_obs = new_rgb_obs

    def get_obstacle_map(self):
        """
        Get the scene obstacle map

        :return ndarray: obstacle map of current scene (H, W, 1)
        """
        obstacle_map = np.array(Image.open(
            os.path.join(get_scene_path(self.config.get('scene_id')),
                         f'floor_{self.task.floor_num}.png')
        ))

        # process image for training
        obstacle_map = datautils.process_floor_map(obstacle_map)

        return obstacle_map

    def get_floor_map(self):
        """
        Get the scene floor map (traversability map + obstacle map)

        :return ndarray: floor map of current scene (H, W, 1)
        """

        obstacle_map = np.array(Image.open(
            os.path.join(get_scene_path(self.config.get('scene_id')),
                         f'floor_{self.task.floor_num}.png')
        ))

        trav_map = np.array(Image.open(
            os.path.join(get_scene_path(self.config.get('scene_id')),
                         f'floor_trav_{self.task.floor_num}.png')
        ))

        trav_map[obstacle_map == 0] = 0

        trav_map_erosion = self.config.get('trav_map_erosion', 2)
        trav_map = cv2.erode(trav_map, np.ones((trav_map_erosion, trav_map_erosion)))
        trav_map[trav_map < 255] = 0

        # process image for training
        floor_map = datautils.process_floor_map(trav_map)

        return floor_map

    def get_random_particles(self, num_particles, particles_distr, robot_pose, scene_map, particles_cov):
        """
        Sample random particles based on the scene

        :param particles_distr: string type of distribution, possible value: [gaussian, uniform]
        :param robot_pose: ndarray indicating the robot pose ([batch_size], 3) in pixel space
            if None, random particle poses are sampled using unifrom distribution
            otherwise, sampled using gaussian distribution around the robot_pose
        :param particles_cov: for tracking Gaussian covariance matrix (3, 3)
        :param num_particles: integer indicating the number of random particles per batch
        :param scene_map: floor map to sample valid random particles

        :return ndarray: random particle poses  (batch_size, num_particles, 3) in pixel space
        """

        assert list(robot_pose.shape) == [1, 3]
        assert list(particles_cov.shape) == [3, 3]

        particles = []
        batches = robot_pose.shape[0]
        if particles_distr == 'uniform':
            # iterate per batch_size
            for b_idx in range(batches):
                sample_i = 0
                b_particles = []

                # get bounding box for more efficient sampling
                # rmin, rmax, cmin, cmax = self.bounding_box(scene_map)
                rmin, rmax, cmin, cmax = self.bounding_box(scene_map, robot_pose[b_idx], lmt=100)

                while sample_i < num_particles:
                    particle = np.random.uniform(low=(cmin, rmin, 0.0), high=(cmax, rmax, 2.0 * np.pi), size=(3,))
                    # reject if mask is zero
                    if not scene_map[int(np.rint(particle[1])), int(np.rint(particle[0]))]:
                        continue
                    b_particles.append(particle)

                    sample_i = sample_i + 1
                particles.append(b_particles)
        elif particles_distr == 'gaussian':
            # iterate per batch_size
            for b_idx in range(batches):
                # sample offset from the Gaussian
                center = np.random.multivariate_normal(mean=robot_pose[b_idx], cov=particles_cov)

                # sample particles from the Gaussian, centered around the offset
                particles.append(np.random.multivariate_normal(mean=center, cov=particles_cov, size=num_particles))
        else:
            raise ValueError

        particles = np.stack(particles)  # [batch_size, num_particles, 3]
        return particles

    def bounding_box(self, img, robot_pose=None, lmt=100):
        """
        Bounding box of non-zeros in an array.

        :param img: numpy array
        :param robot_pose: numpy array of robot pose
        :param lmt: integer representing width/length of bounding box

        :return (int, int, int, int): bounding box indices top_row, bottom_row, left_column, right_column
        """
        rows = np.any(img, axis=1)
        cols = np.any(img, axis=0)
        rmin, rmax = np.where(rows)[0][[0, -1]]
        cmin, cmax = np.where(cols)[0][[0, -1]]

        if robot_pose is not None:
            # futher constraint the bounding box
            x, y, _ = robot_pose

            rmin = np.rint(y - lmt) if (y - lmt) > rmin else rmin
            rmax = np.rint(y + lmt) if (y + lmt) < rmax else rmax
            cmin = np.rint(x - lmt) if (x - lmt) > cmin else cmin
            cmax = np.rint(x + lmt) if (x + lmt) < cmax else cmax

        return rmin, rmax, cmin, cmax

    def get_robot_pose(self, robot_state, floor_map_shape):
        robot_pos = robot_state[0:3]  # [x, y, z]
        robot_orn = robot_state[3:6]  # [r, p, y]

        # transform from co-ordinate space to pixel space
        robot_pos_xy = datautils.transform_pose(robot_pos[:2], floor_map_shape,
                                                self.scene.trav_map_resolution ** 2)  # [x, y]
        robot_pose = np.array([robot_pos_xy[0], robot_pos_xy[1], robot_orn[2]])  # [x, y, theta]

        return robot_pose

    def get_est_pose(self):

        batch_size = self.pf_params.batch_size
        num_particles = self.pf_params.num_particles
        particles, particle_weights, _ = self.curr_pfnet_state  # after transition update
        lin_weights = tf.nn.softmax(particle_weights, axis=-1)

        assert list(particles.shape) == [batch_size, num_particles, 3]
        assert list(lin_weights.shape) == [batch_size, num_particles]

        est_pose = tf.math.reduce_sum(tf.math.multiply(
            particles[:, :, :], lin_weights[:, :, None]
        ), axis=1)
        assert list(est_pose.shape) == [batch_size, 3]

        # normalize between [-pi, +pi]
        part_x, part_y, part_th = tf.unstack(est_pose, axis=-1, num=3)  # (k, 3)
        part_th = tf.math.floormod(part_th + np.pi, 2 * np.pi) - np.pi
        est_pose = tf.stack([part_x, part_y, part_th], axis=-1)

        return est_pose

    def render(self, mode='human'):
        """
        Render plots
        """
        # super(LocalizeGibsonEnv, self).render(mode)

        if self.use_pfnet and self.pf_params.use_plot:
            # environment map
            floor_map = self.floor_map[0].numpy()
            map_plt = self.env_plts['map_plt']
            map_plt = render.draw_floor_map(floor_map, self.plt_ax, map_plt)
            self.env_plts['map_plt'] = map_plt

            # ground truth robot pose and heading
            color = '#7B241C'
            robot_pose = self.curr_gt_pose[0].numpy()
            position_plt = self.env_plts['robot_gt_plt']['position_plt']
            heading_plt = self.env_plts['robot_gt_plt']['heading_plt']
            position_plt, heading_plt = render.draw_robot_pose(
                robot_pose,
                color,
                floor_map.shape,
                self.plt_ax,
                position_plt,
                heading_plt)
            self.env_plts['robot_gt_plt']['position_plt'] = position_plt
            self.env_plts['robot_gt_plt']['heading_plt'] = heading_plt

            # estimated robot pose and heading
            color = '#515A5A'
            est_pose = self.curr_est_pose[0].numpy()
            position_plt = self.env_plts['robot_est_plt']['position_plt']
            heading_plt = self.env_plts['robot_est_plt']['heading_plt']
            position_plt, heading_plt = render.draw_robot_pose(
                est_pose,
                color,
                floor_map.shape,
                self.plt_ax,
                position_plt,
                heading_plt)
            self.env_plts['robot_est_plt']['position_plt'] = position_plt
            self.env_plts['robot_est_plt']['heading_plt'] = heading_plt

            # particles color coded using weights
            particles, particle_weights, _ = self.curr_pfnet_state  # after transition update
            lin_weights = tf.nn.softmax(particle_weights, axis=-1)
            particles_plt = self.env_plts['robot_est_plt']['particles_plt']
            particles_plt = render.draw_particles_pose(
                particles[0].numpy(),
                lin_weights[0].numpy(),
                floor_map.shape,
                particles_plt)
            self.env_plts['robot_est_plt']['particles_plt'] = particles_plt

            # episode info
            step_txt_plt = self.env_plts['step_txt_plt']
            step_txt_plt = render.draw_text(
                f'episode: {self.current_episode}, step: {self.current_step}',
                '#7B241C', self.plt_ax, step_txt_plt)
            self.env_plts['step_txt_plt'] = step_txt_plt

            self.plt_ax.legend([self.env_plts['robot_gt_plt']['position_plt'],
                                self.env_plts['robot_est_plt']['position_plt']],
                               ["gt_pose", "est_pose"], loc='upper left')

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

        if self.use_pfnet and self.pf_params.use_plot:
            if self.pf_params.store_plot:
                self.store_results()
            else:
                # to prevent plot from closing after environment is closed
                plt.ioff()
                plt.show()

        print("=====> iGibsonEnv closed")

    def store_results(self):
        if len(self.curr_plt_images) > 0:
            fps = 30
            frame_size = (self.curr_plt_images[0].shape[0], self.curr_plt_images[0].shape[1])
            file_path = os.path.join(self.out_folder, f'episode_run_{self.current_episode}.avi')
            out = cv2.VideoWriter(file_path,
                                  cv2.VideoWriter_fourcc(*'XVID'),
                                  fps, frame_size)

            for img in self.curr_plt_images:
                out.write(img)
            out.release()
            print(f'stored img results {len(self.curr_plt_images)} to {file_path}')
            self.curr_plt_images = []

    def __del__(self):
        if len(self.curr_plt_images) > 0:
            self.close()
