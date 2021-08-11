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
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
from pathlib import Path
from PIL import Image
from pfnetwork import pfnet
from sklearn.cluster import KMeans
import tensorflow as tf
tf.get_logger().setLevel('ERROR')


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
            init_pfnet=False,
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

        # For the igibson maps, each pixel represents 0.01m, and the center of the image correspond to (0,0)
        self.map_pixel_in_meters = 0.01
        self.depth_th = 3.
        self.robot_size_px = 0.4/self.map_pixel_in_meters # 0.4m

        argparser = argparse.ArgumentParser()
        self.pf_params = argparser.parse_args([])
        self.use_pfnet = init_pfnet
        self.use_tf_function = use_tf_function
        if self.use_pfnet:
            print("=====> LocalizeGibsonEnv's pfnet initializing....")
            self.init_pfnet(pf_params)
        else:
            self.pf_params.use_plot = False
            self.pf_params.store_plot = False
            self.pf_params.num_clusters = pf_params.num_clusters
            self.pf_params.global_map_size = pf_params.global_map_size
            self.pf_params.custom_output = pf_params.custom_output

        # custom tf_agents we are using supports dict() type observations
        observation_space = OrderedDict()

        task_obs_dim = 18 # robot_prorpio_state (18)
        if 'task_obs' in self.pf_params.custom_output:
            # HACK: use [-1k, +1k] range for TanhNormalProjectionNetwork to work
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
        if 'kmeans_cluster' in self.pf_params.custom_output:
            observation_space['kmeans_cluster'] = gym.spaces.Box(
                low=-1000.0, high=+1000.0,
                shape=(self.pf_params.num_clusters,4),
                dtype=np.float32)
        if 'raw_particles' in self.pf_params.custom_output:
            observation_space['raw_particles'] = gym.spaces.Box(
                low=-1000.0, high=+1000.0,
                shape=(self.pf_params.num_particles,4),
                dtype=np.float32)
        if 'obstacle_map' in self.pf_params.custom_output:
            observation_space['obstacle_map'] = gym.spaces.Box(
                low=0.0, high=1.0,
                shape=self.pf_params.global_map_size,
                dtype=np.float32)
        if 'likelihood_map' in self.pf_params.custom_output:
            observation_space['likelihood_map'] = gym.spaces.Box(
                low=-10.0, high=+10.0,
                shape=(*self.pf_params.global_map_size[:2], 3),
                dtype=np.float32)

        self.observation_space = gym.spaces.Dict(observation_space)

        print("=====> LocalizeGibsonEnv initialized")


    def init_pf_params(self, FLAGS):
        """
        Initialize Particle Filter parameters
        """

        assert 0.0 <= FLAGS.alpha_resample_ratio <= 1.0
        assert FLAGS.init_particles_distr in ['gaussian', 'uniform']
        assert len(FLAGS.transition_std) == len(FLAGS.init_particles_std) == 2
        assert len(FLAGS.global_map_size) == 3

        self.pf_params.init_particles_distr = FLAGS.init_particles_distr
        self.pf_params.init_particles_std = np.array(FLAGS.init_particles_std, dtype=np.float32)
        self.pf_params.particles_range = FLAGS.particles_range
        self.pf_params.num_particles = FLAGS.num_particles
        self.pf_params.resample = FLAGS.resample
        self.pf_params.alpha_resample_ratio = FLAGS.alpha_resample_ratio
        self.pf_params.transition_std = np.array(FLAGS.transition_std, dtype=np.float32)
        self.pf_params.pfnet_loadpath = FLAGS.pfnet_load
        self.pf_params.use_plot = FLAGS.use_plot
        self.pf_params.store_plot = FLAGS.store_plot

        self.pf_params.return_state = True
        self.pf_params.stateful = False
        self.pf_params.global_map_size = np.array(FLAGS.global_map_size, dtype=np.int32)
        self.pf_params.window_scaler = FLAGS.window_scaler
        self.pf_params.max_step = self.config.get('max_step', 500)

        self.pf_params.transition_std[0] = self.pf_params.transition_std[0] / self.map_pixel_in_meters  # convert meters to pixels
        self.pf_params.init_particles_std[0] = self.pf_params.init_particles_std[0] / self.map_pixel_in_meters  # convert meters to pixels

        self.pf_params.obs_ch = FLAGS.obs_ch
        self.pf_params.obs_mode = FLAGS.obs_mode
        self.pf_params.num_clusters = FLAGS.num_clusters
        self.pf_params.custom_output = FLAGS.custom_output

        # build initial covariance matrix of particles, in pixels and radians
        particle_std2 = np.square(self.pf_params.init_particles_std.copy())  # variance
        self.pf_params.init_particles_cov = np.diag(particle_std2[(0, 0, 1),])


    def init_pfnet(self, pf_params):
        """
        Initialize Particle Filter
        """

        if pf_params is not None:
            self.pf_params = pf_params
        else:
            self.init_pf_params(flags.FLAGS)

        # HACK:
        self.pf_params.batch_size = 1
        self.pf_params.trajlen = 1

        # Create a new pfnet model instance
        self.pfnet_model = pfnet.pfnet_model(self.pf_params)
        print(self.pf_params)
        print("=====> LocalizeGibsonEnv's pfnet initialized")

        # load model from checkpoint file
        if self.pf_params.pfnet_loadpath:
            self.pfnet_model.load_weights(self.pf_params.pfnet_loadpath)
            print("=====> loaded pf model checkpoint " + self.pf_params.pfnet_loadpath)

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
        self.curr_obs = None
        self.curr_gt_pose = None
        self.curr_est_pose = None
        self.curr_cluster = None


    def reset_variables(self):
        """
        Reset bookkeeping variables for the next new episode
        """

        super(LocalizeGibsonEnv, self).reset_variables()

        self.obstacle_map = None
        self.floor_map = None

        self.curr_plt_images = []
        self.curr_pfnet_state = None
        self.curr_obs = None
        self.curr_gt_pose = None
        self.curr_est_pose = None
        self.curr_cluster = None


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
            new_rgb_obs = copy.deepcopy(state['rgb']*255) # [0, 255]
            new_depth_obs = copy.deepcopy(state['depth']*100) # [0, 100]
            pose_mse = self.step_pfnet([
                new_rgb_obs,
                new_depth_obs
            ])['pred'].cpu().numpy()
            # TODO: may need better reward
            # compute reward and normalize to range [-10, 0]
            reward = np.clip(reward-pose_mse, -10, 0)

        custom_state = self.process_state(state)
        return custom_state, reward, done, info


    def reset(self):
        """
        Reset episode

        :return: state: new observation
        """

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

        state = super(LocalizeGibsonEnv, self).reset()
        if self.use_pfnet:
            new_rgb_obs = copy.deepcopy(state['rgb']*255) # [0, 255]
            new_depth_obs = copy.deepcopy(state['depth']*100) # [0, 100]
            pose_mse = self.reset_pfnet([
                new_rgb_obs,
                new_depth_obs
            ])['pred'].cpu().numpy()

        custom_state = self.process_state(state)
        return custom_state


    def process_state(self, state):
        """
        Perform additional processing.

        :param state: env observations

        :return: processed_state: processed env observations
        """
        assert np.min(state['rgb'])>=0. and np.max(state['rgb'])<=1.
        assert np.min(state['depth'])>=0. and np.max(state['depth'])<=1.

        # # HACK: to collect data
        # new_rgb_obs = copy.deepcopy(state['rgb']*255) # [0, 1] ->[0, 255]
        # new_depth_obs = copy.deepcopy(state['depth']*100) # [0, 1] ->[0, 100]
        #
        # # check for close obstacles to robot
        # min_depth = np.min(new_depth_obs, axis=0)
        # left = np.min(min_depth[:64]) < self.depth_th
        # left_front = np.min(min_depth[64:128]) < self.depth_th
        # right_front = np.min(min_depth[128:192]) < self.depth_th
        # right = np.min(min_depth[192:]) < self.depth_th
        #
        # # process new rgb, depth observation: convert [0, 255] to [-1, +1] range
        # return [
        #         datautils.process_raw_image(new_rgb_obs),
        #         datautils.process_raw_image(new_depth_obs),
        #         np.array([left, left_front, right_front, right])
        #     ]

        # process and return only output we are expecting to
        processed_state = OrderedDict()
        if 'task_obs' in self.pf_params.custom_output:
            processed_state['task_obs'] = self.robots[0].calc_state()  # robot proprioceptive state
        if 'rgb_obs' in self.pf_params.custom_output:
            processed_state['rgb_obs'] = state['rgb']  # [0, 1] range rgb image
        if 'depth_obs' in self.pf_params.custom_output:
            processed_state['depth_obs'] = state['depth']  # [0, 1] range depth image
        if 'kmeans_cluster' in self.pf_params.custom_output:
            if self.curr_cluster is not None:
                cluster_centers, cluster_weights = self.curr_cluster
                particle_cluster = []
                floor_map = self.get_floor_map()
                for i, cluster_center in enumerate(cluster_centers):
                    pose_mts = datautils.inv_transform_pose(cluster_center, floor_map.shape, self.map_pixel_in_meters)
                    particle_cluster.append(np.append(pose_mts, cluster_weights[i]))
                processed_state['kmeans_cluster'] = np.stack(particle_cluster) # particle_cluster [x, y, theta, weight]
            else:
                processed_state['kmeans_cluster'] = None
        if 'raw_particles' in self.pf_params.custom_output:
            particles, particle_weights, _ = self.curr_pfnet_state  # after transition update
            lin_weights = tf.nn.softmax(particle_weights, axis=-1)  # normalize weights
            processed_state['raw_particles'] = np.append(particles[0].cpu().numpy(), lin_weights[0].cpu().numpy())
        if 'obstacle_map' in self.pf_params.custom_output:
            processed_state['obstacle_map'] = self.get_obstacle_map() # [0, 2] range floor map
        if 'likelihood_map' in self.pf_params.custom_output:

            obstacle_map = np.squeeze(self.get_obstacle_map(), axis=-1) # [0, 2] range floor map
            particles, particle_weights, _ = self.curr_pfnet_state  # after transition update
            particles = particles[0].cpu().numpy()
            lin_weights = tf.nn.softmax(particle_weights, axis=-1)[0].cpu().numpy()  # normalize weights
            likelihood_map = np.zeros(list(obstacle_map.shape)[:2] + [3])

            # update obstacle map channel
            likelihood_map[:, :, 0] = np.where( obstacle_map/2. > 0.5, 1, 0) # clip to 0 or 1
            for idx in range(self.pf_params.num_particles):
                x, y, orn = particles[idx]
                wt = lin_weights[idx]

                # update weights channel
                likelihood_map[
                    int(np.rint(x-self.robot_size_px/2.)):int(np.rint(x+self.robot_size_px/2.))+1,
                    int(np.rint(y-self.robot_size_px/2.)):int(np.rint(y+self.robot_size_px/2.))+1, 1] += wt

                # update orientation channel
                likelihood_map[
                    int(np.rint(x-self.robot_size_px/2.)):int(np.rint(x+self.robot_size_px/2.))+1,
                    int(np.rint(y-self.robot_size_px/2.)):int(np.rint(y+self.robot_size_px/2.))+1, 2] += wt*datautils.normalize(orn)
            # weighed mean of orientation channel w.r.t weights channel
            indices = likelihood_map[:, :, 1] > 0.
            likelihood_map[indices, 2] /= likelihood_map[indices, 1]

            processed_state['likelihood_map'] = likelihood_map


        return processed_state


    def step_pfnet(self, new_obs):
        """
        Perform one particle filter step
        """

        trajlen = self.pf_params.trajlen
        batch_size = self.pf_params.batch_size
        num_particles = self.pf_params.num_particles
        pfnet_stateful = self.pf_params.stateful
        obs_ch = self.pf_params.obs_ch
        obs_mode = self.pf_params.obs_mode

        floor_map = self.floor_map[0]
        old_rgb_obs, old_depth_obs = self.curr_obs
        old_pose = self.curr_gt_pose[0].cpu().numpy()
        old_pfnet_state = self.curr_pfnet_state

        # get new robot state
        new_robot_state = self.robots[0].calc_state()

        new_rgb_obs, new_depth_obs = new_obs
        # process new rgb observation: convert [0, 255] to [-1, +1] range
        new_rgb_obs = datautils.process_raw_image(new_rgb_obs)

        # process new depth observation: convert [0, 100] to [-1, +1] range
        new_depth_obs = datautils.process_raw_image(new_depth_obs)

        # process new robot state: convert coords to pixel space
        new_pose = self.get_robot_pose(new_robot_state, floor_map.shape)

        # calculate actual odometry b/w old pose and new pose
        assert list(old_pose.shape) == [3] and list(new_pose.shape) == [3], f'{old_pose.shape}, {new_pose.shape}'
        new_odom = datautils.calc_odometry(old_pose, new_pose)

        # convert to tensor, add batch_dim
        new_rgb_obs = tf.expand_dims(
            tf.convert_to_tensor(new_rgb_obs, dtype=tf.float32), axis=0)
        new_depth_obs = tf.expand_dims(
            tf.convert_to_tensor(new_depth_obs, dtype=tf.float32), axis=0)
        new_odom = tf.expand_dims(
            tf.convert_to_tensor(new_odom, dtype=tf.float32), axis=0)
        new_pose = tf.expand_dims(
            tf.convert_to_tensor(new_pose, dtype=tf.float32), axis=0)
        odometry = tf.expand_dims(new_odom, axis=1)

        # add traj_dim
        if obs_mode == 'rgb-depth':
            observation = tf.concat([
                tf.expand_dims(old_rgb_obs, axis=1),
                tf.expand_dims(old_depth_obs, axis=1),
            ], axis=-1)
        elif obs_mode == 'depth':
            observation = tf.expand_dims(old_depth_obs, axis=1)
        else:
            observation = tf.expand_dims(old_rgb_obs, axis=1)

        # sanity check
        assert list(odometry.shape) == [batch_size, trajlen, 3], f'{odometry.shape}'
        assert list(observation.shape) == [batch_size, trajlen, 56, 56, obs_ch], f'{observation.shape}'
        assert list(old_pfnet_state[0].shape) == [batch_size, num_particles, 3], f'{old_pfnet_state[0].shape}'
        assert list(old_pfnet_state[1].shape) == [batch_size, num_particles], f'{old_pfnet_state[1].shape}'

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

        # compute pfnet loss, add traj_dim
        particles, particle_weights = output
        true_old_pose = tf.expand_dims(self.curr_gt_pose, axis=1)

        assert list(true_old_pose.shape) == [batch_size, trajlen, 3], f'{true_old_pose.shape}'
        assert list(particles.shape) == [batch_size, trajlen, num_particles, 3], f'{particles.shape}'
        assert list(particle_weights.shape) == [batch_size, trajlen, num_particles], f'{particle_weights.shape}'
        loss_dict = pfnet_loss.compute_loss(particles, particle_weights, true_old_pose, self.map_pixel_in_meters)

        self.curr_pfnet_state = new_pfnet_state
        self.curr_gt_pose = new_pose
        self.curr_est_pose = self.get_est_pose()
        self.curr_obs = [
            new_rgb_obs,
            new_depth_obs
        ]
        self.curr_cluster = self.compute_kmeans()

        return loss_dict


    def reset_pfnet(self, new_obs):
        """
        obstacle_map: used as particle filter state
        floor_map: used for sampling init random particles
        """

        trajlen = self.pf_params.trajlen
        batch_size = self.pf_params.batch_size
        map_size = self.pf_params.global_map_size
        num_particles = self.pf_params.num_particles
        init_particles_cov = self.pf_params.init_particles_cov
        init_particles_distr = self.pf_params.init_particles_distr
        particles_range = self.pf_params.particles_range

        # get new robot state
        new_robot_state = self.robots[0].calc_state()

        # process new env map
        floor_map = self.get_floor_map()
        obstacle_map = self.get_obstacle_map()

        new_rgb_obs, new_depth_obs = new_obs
        # process new rgb observation: convert [0, 255] to [-1, +1] range
        new_rgb_obs = datautils.process_raw_image(new_rgb_obs)

        # process new depth observation: convert [0, 100] to [-1, +1] range
        new_depth_obs = datautils.process_raw_image(new_depth_obs)

        # process new robot state: convert coords to pixel space
        new_pose = self.get_robot_pose(new_robot_state, floor_map.shape)

        # convert to tensor, add batch_dim
        new_rgb_obs = tf.expand_dims(
            tf.convert_to_tensor(new_rgb_obs, dtype=tf.float32), axis=0)
        new_depth_obs = tf.expand_dims(
            tf.convert_to_tensor(new_depth_obs, dtype=tf.float32), axis=0)
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
                new_pose.cpu().numpy(),
                floor_map[0],
                init_particles_cov,
                particles_range)), dtype=tf.float32)
        init_particle_weights = tf.constant(
            np.log(1.0 / float(num_particles)),
            shape=(batch_size, num_particles),
            dtype=tf.float32)

        # sanity check
        assert list(new_pose.shape) == [batch_size, 3], f'{new_pose.shape}'
        assert list(new_rgb_obs.shape) == [batch_size, 56, 56, 3], f'{new_rgb_obs.shape}'
        assert list(new_depth_obs.shape) == [batch_size, 56, 56, 1], f'{new_depth_obs.shape}'
        assert list(init_particles.shape) == [batch_size, num_particles, 3], f'{init_particles.shape}'
        assert list(init_particle_weights.shape) == [batch_size, num_particles], f'{init_particle_weights.shape}'
        assert list(floor_map.shape) == [batch_size, *map_size], f'{floor_map.shape}'
        assert list(obstacle_map.shape) == [batch_size, *map_size], f'{obstacle_map.shape}'

        # compute pfnet loss, add traj_dim
        particles = tf.expand_dims(init_particles, axis=1)
        particle_weights = tf.expand_dims(init_particle_weights, axis=1)
        true_old_pose = tf.expand_dims(new_pose, axis=1)

        assert list(true_old_pose.shape) == [batch_size, trajlen, 3], f'{true_old_pose.shape}'
        assert list(particles.shape) == [batch_size, trajlen, num_particles, 3], f'{particles.shape}'
        assert list(particle_weights.shape) == [batch_size, trajlen, num_particles], f'{particle_weights.shape}'
        loss_dict = pfnet_loss.compute_loss(particles, particle_weights, true_old_pose, self.map_pixel_in_meters)

        self.floor_map = floor_map
        self.obstacle_map = obstacle_map
        self.curr_pfnet_state = [init_particles, init_particle_weights, obstacle_map]
        self.curr_gt_pose = new_pose
        self.curr_est_pose = self.get_est_pose()
        self.curr_obs = [
            new_rgb_obs,
            new_depth_obs
        ]
        self.curr_cluster = self.compute_kmeans()

        return loss_dict

    def compute_kmeans(self, num_iterations=1):
        """
        """

        num_clusters = self.pf_params.num_clusters
        particles, particle_weights, _ = self.curr_pfnet_state  # after transition update
        lin_weights = tf.nn.softmax(particle_weights, axis=-1)[0].cpu().numpy()
        particles = particles[0].cpu().numpy()

        if self.curr_cluster is None:
            # random initialization
            kmeans = KMeans(n_clusters=num_clusters, n_init=10)
        else:
            # previous cluster center initialization
            prev_cluster_centers, _ = self.curr_cluster
            assert list(prev_cluster_centers.shape) == [num_clusters, 3]
            kmeans = KMeans(n_clusters=num_clusters, init=prev_cluster_centers, n_init=1)
        kmeans.fit_predict(particles)
        cluster_indices = kmeans.labels_
        cluster_centers = kmeans.cluster_centers_
        cluster_weights = np.zeros(num_clusters)

        # kmeans = tf.compat.v1.estimator.experimental.KMeans(
        #     num_clusters=num_clusters,
        #     use_mini_batch=False
        # )
        # def particles_ds():
        #     return tf.compat.v1.data.Dataset.from_tensors(
        #         tf.convert_to_tensor(particles, dtype=tf.float32)
        #     )
        #
        # previous_centers = np.zeros((num_clusters, 3))
        # for _ in range(num_iterations):
        #     kmeans.train(particles_ds)
        #     cluster_centers = kmeans.cluster_centers()
        #     if np.linalg.norm(cluster_centers - previous_centers) < 1e-2:
        #         break
        #     else:
        #         previous_centers = cluster_centers
        #
        # cluster_centers = kmeans.cluster_centers()
        # cluster_indices = list(kmeans.predict_cluster_index(particles_ds))
        # cluster_weights = np.zeros(num_clusters)

        for i, particle in enumerate(particles):
            cluster_index = cluster_indices[i]
            cluster_weights[cluster_index] += lin_weights[i]

        return cluster_centers, cluster_weights

    def set_scene(self, scene_id, floor_num):
        """
        Override the task floor number

        :param str: scene id
        :param int: task floor number
        """
        self.config['scene_id'] = scene_id
        self.task.floor_num = floor_num


    def get_obstacle_map(self, scene_id=None, floor_num=None):
        """
        Get the scene obstacle map

        :param str: scene id
        :param int: task floor number
        :return ndarray: obstacle map of current scene (H, W, 1)
        """
        if scene_id is not None and floor_num is not None:
            self.set_scene(scene_id, floor_num)

        obstacle_map = np.array(Image.open(
            os.path.join(get_scene_path(self.config.get('scene_id')),
                         f'floor_{self.task.floor_num}.png')
        ))

        # process new obstacle map: convert [0, 255] to [0, 2] range
        obstacle_map = datautils.process_raw_map(obstacle_map)

        return obstacle_map


    def get_floor_map(self, scene_id=None, floor_num=None):
        """
        Get the scene floor map (traversability map + obstacle map)

        :param str: scene id
        :param int: task floor number
        :return ndarray: floor map of current scene (H, W, 1)
        """
        if scene_id is not None and floor_num is not None:
            self.set_scene(scene_id, floor_num)

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

        # process new obstacle map: convert [0, 255] to [0, 2] range
        floor_map = datautils.process_raw_map(trav_map)

        return floor_map


    def get_random_particles(self, num_particles, particles_distr, robot_pose, scene_map, particles_cov, particles_range=100):
        """
        Sample random particles based on the scene

        :param particles_distr: string type of distribution, possible value: [gaussian, uniform]
        :param robot_pose: ndarray indicating the robot pose ([batch_size], 3) in pixel space
            if None, random particle poses are sampled using unifrom distribution
            otherwise, sampled using gaussian distribution around the robot_pose
        :param particles_cov: for tracking Gaussian covariance matrix (3, 3)
        :param num_particles: integer indicating the number of random particles per batch
        :param scene_map: floor map to sample valid random particles
        :param particles_range: particles range in pixels for uniform distribution

        :return ndarray: random particle poses  (batch_size, num_particles, 3) in pixel space
        """

        assert list(robot_pose.shape[1:]) == [3], f'{robot_pose.shape}'
        assert list(particles_cov.shape) == [3, 3], f'{particles_cov.shape}'
        assert list(scene_map.shape[2:]) == [1], f'{scene_map.shape}'

        particles = []
        batches = robot_pose.shape[0]
        if particles_distr == 'uniform':
            # iterate per batch_size
            for b_idx in range(batches):
                sample_i = 0
                b_particles = []

                # sample offset from the Gaussian ground truth
                center = np.random.multivariate_normal(mean=robot_pose[b_idx], cov=particles_cov)

                # get bounding box, centered around the offset, for more efficient sampling
                # rmin, rmax, cmin, cmax = self.bounding_box(scene_map)
                rmin, rmax, cmin, cmax = self.bounding_box(scene_map, center, particles_range)

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
                # sample offset from the Gaussian ground truth
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
        :param lmt: integer representing width/length of bounding box in pixels

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
        robot_pos_xy = datautils.transform_position(robot_pos[:2], floor_map_shape, self.map_pixel_in_meters)  # [x, y]
        robot_pose = np.array([robot_pos_xy[0], robot_pos_xy[1], robot_orn[2]])  # [x, y, theta]

        return robot_pose


    def get_est_pose(self):

        batch_size = self.pf_params.batch_size
        num_particles = self.pf_params.num_particles
        particles, particle_weights, _ = self.curr_pfnet_state  # after transition update
        lin_weights = tf.nn.softmax(particle_weights, axis=-1)

        assert list(particles.shape) == [batch_size, num_particles, 3], f'{particles.shape}'
        assert list(lin_weights.shape) == [batch_size, num_particles], f'{lin_weights.shape}'

        est_pose = tf.math.reduce_sum(tf.math.multiply(
            particles[:, :, :], lin_weights[:, :, None]
        ), axis=1)
        assert list(est_pose.shape) == [batch_size, 3], f'{est_pose.shape}'

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
            floor_map = self.floor_map[0].cpu().numpy()
            map_plt = self.env_plts['map_plt']
            map_plt = render.draw_floor_map(floor_map, floor_map.shape, self.plt_ax, map_plt)
            self.env_plts['map_plt'] = map_plt

            # ground truth robot pose and heading
            color = '#7B241C'
            gt_pose = self.curr_gt_pose[0].cpu().numpy()
            position_plt = self.env_plts['robot_gt_plt']['position_plt']
            heading_plt = self.env_plts['robot_gt_plt']['heading_plt']
            position_plt, heading_plt = render.draw_robot_pose(
                gt_pose,
                color,
                floor_map.shape,
                self.plt_ax,
                position_plt,
                heading_plt)
            self.env_plts['robot_gt_plt']['position_plt'] = position_plt
            self.env_plts['robot_gt_plt']['heading_plt'] = heading_plt

            # estimated robot pose and heading
            color = '#515A5A'
            est_pose = self.curr_est_pose[0].cpu().numpy()
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

            # # particles color coded using weights
            # particles, particle_weights, _ = self.curr_pfnet_state  # after transition update
            # lin_weights = tf.nn.softmax(particle_weights, axis=-1)
            # particles_plt = self.env_plts['robot_est_plt']['particles_plt']
            # particles_plt = render.draw_particles_pose(
            #     particles[0].cpu().numpy(),
            #     lin_weights[0].cpu().numpy(),
            #     floor_map.shape,
            #     particles_plt)
            # self.env_plts['robot_est_plt']['particles_plt'] = particles_plt

            # kmeans-cluster particles color coded using weights
            cc_particles, cc_weights = self.curr_cluster
            particles_plt = self.env_plts['robot_est_plt']['particles_plt']
            particles_plt = render.draw_particles_pose(
                cc_particles,
                cc_weights,
                floor_map.shape,
                particles_plt)
            self.env_plts['robot_est_plt']['particles_plt'] = particles_plt

            # # episode info
            # step_txt_plt = self.env_plts['step_txt_plt']
            # step_txt_plt = render.draw_text(
            #     f'episode: {self.current_episode}, step: {self.current_step}',
            #     '#7B241C', self.plt_ax, step_txt_plt)
            # self.env_plts['step_txt_plt'] = step_txt_plt

            # pose mse in mts
            gt_pose_mts = datautils.inv_transform_pose(gt_pose, floor_map.shape, self.map_pixel_in_meters)
            est_pose_mts = datautils.inv_transform_pose(est_pose, floor_map.shape, self.map_pixel_in_meters)
            pose_diff = gt_pose_mts-est_pose_mts
            pose_diff[-1] = datautils.normalize(pose_diff[-1]) # normalize

            step_txt_plt = self.env_plts['step_txt_plt']
            step_txt_plt = render.draw_text(
                f'pose mse: {np.linalg.norm(pose_diff):02.3f} ',
                '#7B241C', self.plt_ax, step_txt_plt)
            self.env_plts['step_txt_plt'] = step_txt_plt
            # print(f'gt_pose: {gt_pose_mts}, est_pose: {est_pose_mts} in mts')

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
            print(f'stored img results {len(self.curr_plt_images)} eps steps to {file_path}')
            self.curr_plt_images = []
        else:
            print('no plots available to store, check if env.render() is being called')


    def __del__(self):
        if len(self.curr_plt_images) > 0:
            self.close()
