#!/usr/bin/env python3

import argparse
import cv2
import matplotlib.pyplot as plt
import numpy as np
import os
import random
import rospy
import tensorflow as tf

import sys
sys.path.append("/media/neo/robotics/deep-activate-localization/src/rl_agents/")
from environments.env_utils import datautils
from environments.envs.localize_env import LocalizeGibsonEnv

np.set_printoptions(suppress=True, precision=3)
def parse_args():
    """
    Parse command line arguments

    :return: argparse.Namespace
        parsed command-line arguments passed to *.py
    """

    # initialize parser
    arg_parser = argparse.ArgumentParser()

    # define testing parameters
    arg_parser.add_argument(
        '--obs_mode',
        type=str,
        default='rgb-depth',
        help='Observation input type. Possible values: rgb / depth / rgb-depth / occupancy_grid.'
    )
    arg_parser.add_argument(
        '--custom_output',
        nargs='*',
        default=['rgb_obs', 'depth_obs', 'occupancy_grid', 'floor_map', 'kmeans_cluster', 'likelihood_map'],
        help='A comma-separated list of env observation types.'
    )
    arg_parser.add_argument(
        '--root_dir',
        type=str,
        default='./test_output',
        help='Root directory for logs/summaries/checkpoints.'
    )
    arg_parser.add_argument(
        '--num_eval_samples',
        type=int,
        default=1,
        help='Total number of samples to use for evaluation. Total evaluation samples will be num_eval_samples=num_eval_batches*batch_size'
    )
    arg_parser.add_argument(
        '--testfiles',
        nargs='*',
        default=['./test.tfrecord'],
        help='Data file(s) for validation or evaluation (tfrecord).'
    )
    arg_parser.add_argument(
        '--batch_size',
        type=int,
        default=1,
        help='Minibatch size for training'
    )
    arg_parser.add_argument(
        '--pfnet_loadpath',
        type=str,
        default='',
        help='Load a previously trained pfnet model from a checkpoint file.'
    )
    arg_parser.add_argument(
        '--seed',
        type=int,
        default=1,
        help='Fix the random seed'
    )
    arg_parser.add_argument(
        '--device_idx',
        type=int,
        default='0',
        help='use gpu no. to test'
    )

    # define particle parameters
    arg_parser.add_argument(
        '--init_particles_distr',
        type=str,
        default='gaussian',
        help='Distribution of initial particles. Possible values: gaussian / uniform.'
    )
    arg_parser.add_argument(
        '--init_particles_std',
        nargs='*',
        default=["0.15", "0.523599"],
        help='Standard deviations for generated initial particles for tracking distribution.'
             'Values: translation std (meters), rotation std (radians)'
    )
    arg_parser.add_argument(
        '--particles_range',
        type=int,
        default=100,
        help='Pixel range to limit uniform distribution sampling +/- box particles_range center at robot pose'
    )
    arg_parser.add_argument(
        '--num_particles',
        type=int,
        default=30,
        help='Number of particles in Particle Filter'
    )
    arg_parser.add_argument(
        '--transition_std',
        nargs='*',
        default=["0.01", "0.0872665"],
        help='Standard deviations for transition model. Values: translation std (meters), rotation std (radians)'
    )
    arg_parser.add_argument(
        '--resample',
        type=str,
        default='false',
        help='Resample particles in Particle Filter'
    )
    arg_parser.add_argument(
        '--alpha_resample_ratio',
        type=float,
        default=1.0,
        help='Trade-off parameter for soft-resampling in PF-net. Only effective if resample == true.'
             'Assumes values 0.0 < alpha <= 1.0. Alpha equal to 1.0 corresponds to hard-resampling'
    )
    arg_parser.add_argument(
        '--global_map_size',
        nargs='*',
        default=["100", "100", "1"],
        help='Global map size in pixels (H, W, C)'
    )
    arg_parser.add_argument(
        '--window_scaler',
        type=float,
        default=1.0,
        help='Rescale factor for extracing local map'
    )

    # define igibson env parameters
    arg_parser.add_argument(
        '--config_file',
        type=str,
        default=os.path.join(
            '/media/neo/robotics/deep-activate-localization/src/rl_agents/',
            'configs',
            'locobot_pfnet_nav.yaml'
        ),
        help='Config file for the experiment'
    )
    arg_parser.add_argument(
        '--scene_id',
        type=str,
        default='Rs',
        help='Environment scene'
    )
    arg_parser.add_argument(
        '--action_timestep',
        type=float,
        default=1.0 / 10.0,
        help='Action time step for the simulator'
    )
    arg_parser.add_argument(
        '--physics_timestep',
        type=float,
        default=1.0 / 40.0,
        help='Physics time step for the simulator'
    )

    # parse parameters
    params = arg_parser.parse_args()

    # For the igibson maps, originally each pixel represents 0.01m, and the center of the image correspond to (0,0)
    params.map_pixel_in_meters = 0.01
    # in igibson we work with rescaled 0.01m to 0.1m maps to sample robot poses
    params.trav_map_resolution = 0.1

    # post-processing
    params.num_eval_batches = params.num_eval_samples//params.batch_size

    # compute observation channel dim
    if params.obs_mode == 'rgb-depth':
        params.obs_ch = 4
    elif params.obs_mode == 'rgb':
        params.obs_ch = 3
    elif params.obs_mode == 'depth' or params.obs_mode == 'occupancy_grid':
        params.obs_ch = 1
    else:
        raise ValueError

    # convert multi-input fields to numpy arrays
    params.transition_std = np.array(params.transition_std, np.float32)
    params.init_particles_std = np.array(params.init_particles_std, np.float32)
    params.global_map_size = np.array(params.global_map_size, np.int32)

    params.transition_std[0] = (params.transition_std[0] / params.map_pixel_in_meters) * params.trav_map_resolution # convert meters to pixels and rescale to trav map resolution
    params.init_particles_std[0] = (params.init_particles_std[0] / params.map_pixel_in_meters) * params.trav_map_resolution  # convert meters to pixels and rescale to trav map resolution

    # build initial covariance matrix of particles, in pixels and radians
    particle_std = params.init_particles_std.copy()
    particle_std2 = np.square(particle_std)  # variance
    params.init_particles_cov = np.diag(particle_std2[(0, 0, 1), ])

    if params.resample not in ['false', 'true']:
        raise ValueError
    else:
        params.resample = (params.resample == 'true')

    # use RNN as stateful/non-stateful
    params.stateful = False
    params.return_state = True

    # HACK:
    params.loop = 6
    params.use_tf_function = True
    params.init_env_pfnet = False
    params.store_results = True
    params.num_clusters = 10

    params.env_mode = 'headless'
    os.environ['CUDA_VISIBLE_DEVICES'] = str(params.device_idx)
    os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'

    # set random seeds
    random.seed(params.seed)
    np.random.seed(params.seed)
    tf.random.set_seed(params.seed)

    return params

def rt_pfnet_test(arg_params):
    """
    A simple test for particle filter network real-time for igibson

    :param arg_params:
        parsed command-line arguments
    :return:
    """

    # HACK:
    agent = 'manual'
    arg_params.use_plot = True
    arg_params.store_plot = False
    arg_params.init_env_pfnet = False

    # create igibson env which is used "only" to sample particles
    env = LocalizeGibsonEnv(
        config_file=arg_params.config_file,
        scene_id=arg_params.scene_id,
        mode=arg_params.env_mode,
        use_tf_function=arg_params.use_tf_function,
        init_pfnet=arg_params.init_env_pfnet,
        action_timestep=arg_params.action_timestep,
        physics_timestep=arg_params.physics_timestep,
        device_idx=arg_params.device_idx,
        pf_params=arg_params
    )
    obs = env.reset()
    env.render('human')
    assert arg_params.trav_map_resolution == env.trav_map_resolution

    trajlen = env.config.get('max_step', 500)//arg_params.loop
    max_lin_vel = env.config.get("linear_velocity", 0.5)
    max_ang_vel = env.config.get("angular_velocity", np.pi/2)
    for _ in range(trajlen-1):
        if agent == 'manual':
            action = datautils.get_discrete_action(max_lin_vel, max_ang_vel)
        else:
            # default random action forward: 0.7, turn: 0.3, backward:0., do_nothing:0.0
            action = np.random.choice(5, p=[0.7, 0.0, 0.15, 0.15, 0.0])
            # action = env.action_space.sample()

        # take action and get new observation
        obs, reward, done, info = env.step(action)
        env.render('human')

    env.close()

if __name__ == '__main__':
    parsed_params = parse_args()

    # rt_pfnet_test(parsed_params)
    img = cv2.imread("/opt/igibson/igibson/data/g_dataset/Rs/floor_0.png")
    cv2.imwrite("floor_0_fliped.png", cv2.flip(img, 0))
