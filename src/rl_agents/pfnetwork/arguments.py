#!/usr/bin/env python3

import os
import argparse
import numpy as np
import random
import tensorflow as tf
from pathlib import Path

np.set_printoptions(precision=3, suppress=True)

DEFAULTS = {
    'house3d': {
        'transition_std': ["0.0", "0.0"],
        'init_particles_distr': "tracking",
        'init_particles_std': ["0.3", "0.523599"],
        'batch_size': 24,
        'learning_rate': 0.0025,
        'global_map_size': [4000, 4000, 1],
        'window_scaler': 8,
        'map_pixel_in_meters': 0.02,
        'trajlen': 24,
    },
    'igibson': {
        'transition_std': ["0.01", "0.0872665"],
        'init_particles_distr': "gaussian",
        'init_particles_std': ["0.15", "0.523599"],
        'batch_size': 1,
        'learning_rate': 2.5e-3,
        'global_map_size': [100, 100, 1],
        'window_scaler': 1,
        'map_pixel_in_meters': 0.1,
        'trajlen': 25,
    }
}


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def particle_std_to_covariance(init_particles_std, map_pixel_in_meters: float):
    init_particles_std = np.array(init_particles_std, float)
    init_particles_std[0] = init_particles_std[0] / map_pixel_in_meters
    # build initial covariance matrix of particles, in pixels and radians
    particle_var = np.square(init_particles_std)  # variance
    init_particles_cov = np.diag(particle_var[(0, 0, 1),])
    return init_particles_cov


def parse_common_args(env_name: str, collect_data: bool = False, add_rl_args: bool = False):
    assert env_name in ['igibson', 'house3d'], env_name
    argparser = argparse.ArgumentParser()

    argparser.add_argument('--num_train_samples', type=int, default=7500,
                           help='Total number of samples to use for training. Total training samples will be num_train_samples=num_train_batches*batch_size')
    argparser.add_argument('--num_eval_samples', type=int, default=816,
                           help='Total number of samples to use for evaluation. Total evaluation samples will be num_eval_samples=num_eval_batches*batch_size')
    argparser.add_argument('--num_test_samples', type=int, default=816,
                           help='Total number of samples to use for testing. Total testing samples will be num_test_samples=num_test_batches*batch_size')

    argparser.add_argument('--num_particles', type=int, default=30, help='Number of particles in Particle Filter.')
    argparser.add_argument('--transition_std', nargs='*', default=DEFAULTS[env_name]['transition_std'],
                           help='Standard deviations for transition model. Values: translation std (meters), rotation std (radians)')
    argparser.add_argument('--resample', type=str2bool, nargs='?', const=True, default=False,
                           help='Resample particles in Particle Filter. Possible values: true / false.')
    argparser.add_argument('--alpha_resample_ratio', type=float, default=1.0,
                           help='Trade-off parameter for soft-resampling in PF-net. Only effective if resample == true. Assumes values 0.0 < alpha <= 1.0. Alpha equal to 1.0 corresponds to hard-resampling.')
    argparser.add_argument('--init_particles_distr', type=str, default=DEFAULTS[env_name]['init_particles_distr'],
                           help='Distribution of initial particles. Possible values: tracking / one-room.')
    argparser.add_argument('--init_particles_std', nargs='*', default=DEFAULTS[env_name]['init_particles_std'],
                           help='Standard deviations for generated initial particles for tracking distribution. Values: translation std (meters), rotation std (radians)')

    argparser.add_argument('--likelihood_model', type=str, default='learned', choices=['learned', 'scan_correlation'])
    argparser.add_argument('--obs_mode', type=str, choices=["rgb", "depth", "rgb-depth", "occupancy_grid"],
                           default='rgb-depth', help='Observation input type.')

    argparser.add_argument('--batch_size', type=int, default=DEFAULTS[env_name]['batch_size'],
                           help='Minibatch size for training.')
    argparser.add_argument('--learning_rate', type=float, default=DEFAULTS[env_name]['batch_size'],
                           help='Initial learning rate for training.')
    argparser.add_argument('--epochs', type=int, default=100, help='Number of epochs for training.')
    argparser.add_argument('--trajlen', type=int, default=DEFAULTS[env_name]['trajlen'], help='Length of trajectories.')

    argparser.add_argument('--s_buffer_size', type=int, default=10_000, help='Buffer size for shuffling data')
    argparser.add_argument('--seed', type=int, default='42', help='Fix the random seed of numpy and tensorflow.')
    argparser.add_argument('--device_idx', type=int, default='0', help='Use gpu no. to train/eval')
    argparser.add_argument('--pfnet_loadpath', type=str, default='',
                           help='Load a previously trained pfnet model from a checkpoint file.')

    argparser.add_argument('--global_map_size', nargs='*', default=DEFAULTS[env_name]['global_map_size'],
                           help='Global map size in pixels (H, W, C)')
    argparser.add_argument('--map_pixel_in_meters', type=float, default=DEFAULTS[env_name]['map_pixel_in_meters'],
                           help='The width (and height) of a pixel of the map in meters. Defaults to 0.02 for House3D data.')
    argparser.add_argument('--window_scaler', type=float, default=DEFAULTS[env_name]['window_scaler'],
                           help='Rescale factor for extracing local map')

    argparser.add_argument('--multiple_gpus', type=str2bool, nargs='?', const=True, default=False,
                           help="Use multiple GPUs.")
    argparser.add_argument('--resume_id', type=str, default=None, help='wandb id to resume')
    argparser.add_argument('--resume_model_name', type=str, default="model.zip",
                           help='wandb model to restore if resume_id is set')
    argparser.add_argument('--eval_only', type=str2bool, nargs='?', const=True, default=False,
                           help='Whether to run evaluation only on trained checkpoints')

    argparser.add_argument('--reward', type=str, default="pred", choices=['pred', 'belief'],
                           help='wandb model to restore if resume_id is set')
    argparser.add_argument('--collision_reward_weight', type=float, default=0.1, help='Penalty per collision.')

    if env_name == 'house3d':
        argparser.add_argument('--trainfiles', nargs='*', default=['/data2/honerkam/pfnet_data/train.tfrecords'],
                               help='Data file(s) for training (tfrecord).')
        argparser.add_argument('--evalfiles', nargs='*', default=['/data2/honerkam/pfnet_data/valid.tfrecords'],
                               help='Data file(s) for validation or evaluation (tfrecord).')
        argparser.add_argument('--testfiles', nargs='*', default=['/data2/honerkam/pfnet_data/test.tfrecords'],
                               help='Data file(s) for testing (tfrecord).')
        argparser.add_argument('--logpath', type=str, default='./log/', help='Specify path for logs.')
    elif env_name == 'igibson':
        argparser.add_argument('--tfrecordpath', type=str, default='/data/honerkam/pfnet_data/',
                               help='Folder path to training/evaluation/testing (tfrecord).')
        argparser.add_argument('--root_dir', type=str, default='./train_output',
                               help='Root directory for logs/summaries/checkpoints.')
        argparser.add_argument('--custom_output', nargs='*',
                               default=['rgb_obs', 'depth_obs', 'occupancy_grid', 'floor_map', 'likelihood_map',
                                        'obstacle_obs'],
                               choices=['rgb_obs', 'depth_obs', 'occupancy_grid', 'floor_map', 'likelihood_map',
                                        'obstacle_obs', 'task_obs'],
                               help='A comma-separated list of env observation types.')
        argparser.add_argument('--config_file', type=str,
                               default=str(Path(__file__).parent.parent / 'configs' / 'locobot_pfnet_nav.yaml'),
                               help='Config file for the experiment')
        argparser.add_argument('--scene_id', type=str, default='Rs', help='Environment scene')
        argparser.add_argument('--action_timestep', type=float, default=1.0 / 10.0,
                               help='Action time step for the simulator')
        argparser.add_argument('--physics_timestep', type=float, default=1.0 / 40.0,
                               help='Physics time step for the simulator')
        argparser.add_argument('--env_mode', type=str, default='headless', choices=['headless', 'gui'],
                               help='igibson mode')
        argparser.add_argument('--loop', type=int, default=6, help='action repeat for igibson')
        argparser.add_argument('--particles_range', type=int, default=100,
                               help='Pixel range to limit uniform distribution sampling +/- box particles_range center at robot pose')
        argparser.add_argument('--store_plot', type=str2bool, nargs='?', const=True, default=False,
                               help="Whether to store igibson plots.")
        argparser.add_argument('--use_plot', type=str2bool, nargs='?', const=True, default=False,
                               help="Whether to plot igibson stuff.")
        argparser.add_argument('--debug', type=str2bool, nargs='?', const=True, default=False,
                               help="Helper for debugging.")
        argparser.add_argument('--agent', type=str, default='avoid_agent',
                               choices=[None, 'manual_agent', 'rnd_agent', 'avoid_agent', 'goalnav_agent', 'turn_agent',
                                        'rl'], help='Agent Behavior')
        argparser.add_argument('--observe_steps', type=str2bool, nargs='?', const=True, default=True,
                               help="Whether to observe the remaining number of steps.")

    if collect_data:
        argparser.add_argument('--filename', type=str, default='./test.tfrecord', help='The tf record')
        argparser.add_argument('--num_records', type=int, default=10, help='The number of episode data')

    if add_rl_args:
        argparser.add_argument('--num_iterations', type=int, default=1_000_000,
                               help='Total number train/eval iterations to perform.')
        argparser.add_argument('--initial_collect_steps', type=int, default=1000,
                               help='Number of steps to collect at the beginning of training using random policy')
        argparser.add_argument('--collect_steps_per_iteration', type=int, default=1,
                               help='Number of steps to collect and be added to the replay buffer after every training iteration')
        argparser.add_argument('--train_steps_per_iteration', type=int, default=1,
                               help='Number of training steps in every training iteration')

        argparser.add_argument('--replay_buffer_capacity', type=int, default=1_000_000,
                               help='Replay buffer capacity per env.')
        argparser.add_argument('--rl_batch_size', type=int, default=256,
                               help='Batch size for each training step. For each training iteration, we first collect collect_steps_per_iteration steps to the replay buffer. Then we sample batch_size steps from the replay buffer and train the model for train_steps_per_iteration times.')
        argparser.add_argument('--gamma', type=float, default=0.99, help='Discount_factor for the environment')
        argparser.add_argument('--actor_learning_rate', type=float, default=3e-4, help='Actor learning rate')
        # argparser.add_argument('--critic_learning_rate', type=float, default=3e-4, help='Critic learning rate')
        # argparser.add_argument('--alpha_learning_rate', type=float, default=3e-4, help='Alpha learning rate')
        argparser.add_argument('--rl_architecture', type=int, default=1, help='Architecture version.')
        argparser.add_argument('--ent_coef', type=str, default='auto',
                               help='SAC entropy coefficient: auto to learn, auto_0.1 to learn with initial value 0.1')

        # argparser.add_argument('--use_parallel_envs', type=str2bool, nargs='?', const=True, default=False, help='Whether to use parallel env or not')
        # argparser.add_argument('--eval_deterministic', type=str2bool, nargs='?', const=True, default=False, help='Whether to run evaluation using a deterministic policy')
        argparser.add_argument('--num_eval_episodes', type=int, default=10,
                               help='The number of episodes to run eval on.')
        argparser.add_argument('--eval_interval', type=int, default=5000,
                               help='Run eval every eval_interval train steps')
        argparser.add_argument('--num_parallel_environments', type=int, default=1,
                               help='Number of environments to run in parallel')
        # argparser.add_argument('--num_parallel_environments_eval', type=int, default=1, help='Number of environments to run in parallel for eval')

    params = argparser.parse_args()

    if params.debug:
        tf.config.run_functions_eagerly(True)
        params.use_tf_function = False
        tf.data.experimental.enable_debug_mode()
    else:
        params.use_tf_function = True

    # compute observation channel dim
    if params.obs_mode == 'rgb-depth':
        params.obs_ch = 4
    elif params.obs_mode == 'rgb':
        params.obs_ch = 3
    elif params.obs_mode == 'depth' or params.obs_mode == 'occupancy_grid':
        params.obs_ch = 1
    else:
        raise ValueError(params.obs_mode)

    params.stateful = False
    params.return_state = True

    # convert multi-input fields to numpy arrays
    params.transition_std = np.array(params.transition_std, np.float32)
    params.global_map_size = np.array(params.global_map_size, np.int32)

    # convert meters to pixels
    params.transition_std[0] = params.transition_std[0] / params.map_pixel_in_meters

    # build initial covariance matrix of particles, in pixels and radians
    params.init_particles_cov = particle_std_to_covariance(params.init_particles_std,
                                                           map_pixel_in_meters=params.map_pixel_in_meters)

    params.num_train_batches = params.num_train_samples // params.batch_size
    params.num_eval_batches = params.num_eval_samples // params.batch_size
    params.num_test_batches = params.num_test_samples // params.batch_size

    if params.resume_id:
        assert params.eval_only, "Continuing to train not supported atm (replay buffer doesn't get saved)"

    if params.likelihood_model == 'scan_correlation':
        assert params.obs_mode == 'occupancy_grid', params.obs_mode

    random.seed(params.seed)
    np.random.seed(params.seed)
    tf.random.set_seed(params.seed)

    # filter out info and warning messages
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

    if params.multiple_gpus:
        os.environ['CUDA_VISIBLE_DEVICES'] = '0,1'
    else:
        os.environ['CUDA_VISIBLE_DEVICES'] = str(params.device_idx)
    os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'

    return params


if __name__ == '__main__':
    # parse_args()
    parse_common_args('house3d')
