#!/usr/bin/env python3

import argparse
import glob
import matplotlib.pyplot as plt
import numpy as np
import os
import random
import tensorflow as tf
from tensorflow import keras
from tqdm import tqdm

from pfnetwork import pfnet
from environments.env_utils import datautils, pfnet_loss, render
from environments.envs.localize_env import LocalizeGibsonEnv

np.set_printoptions(suppress=True)
def parse_args():
    """
    Parse command line arguments

    :return: argparse.Namespace
        parsed command-line arguments passed to *.py
    """

    # initialize parser
    arg_parser = argparse.ArgumentParser()

    # define training parameters
    arg_parser.add_argument(
        '--root_dir',
        type=str,
        default='./train_output',
        help='Root directory for logs/summaries/checkpoints.'
    )
    arg_parser.add_argument(
        '--tfrecordpath',
        type=str,
        default='./data',
        help='Folder path to training/evaluation/testing (tfrecord).'
    )
    arg_parser.add_argument(
        '--num_train_samples',
        type=int,
        default=1,
        help='Total number of samples to use for training. Total training samples will be num_train_samples=num_train_batches*batch_size'
    )
    arg_parser.add_argument(
        '--batch_size',
        type=int,
        default=1,
        help='Minibatch size for training'
    )
    arg_parser.add_argument(
        '--s_buffer_size',
        type=int,
        default=500,
        help='Buffer size for shuffling data'
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
        help='use gpu no. to train/eval'
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
        default=["1000", "1000", "1"],
        help='Global map size in pixels (H, W, C)'
    )
    arg_parser.add_argument(
        '--window_scaler',
        type=float,
        default=8.0,
        help='Rescale factor for extracing local map'
    )

    # define igibson env parameters
    arg_parser.add_argument(
        '--config_file',
        type=str,
        default=os.path.join(
            os.path.dirname(os.path.realpath(__file__)),
            'configs',
            'turtlebot_pfnet_nav.yaml'
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

    # For the igibson maps, each pixel represents 0.01m, and the center of the image correspond to (0,0)
    params.map_pixel_in_meters = 0.01

    # post-processing
    params.num_train_batches = params.num_train_samples//params.batch_size

    # convert multi-input fields to numpy arrays
    params.transition_std = np.array(params.transition_std, np.float32)
    params.init_particles_std = np.array(params.init_particles_std, np.float32)
    params.global_map_size = np.array(params.global_map_size, np.int32)

    params.transition_std[0] = params.transition_std[0] / params.map_pixel_in_meters  # convert meters to pixels
    params.init_particles_std[0] = params.init_particles_std[0] / params.map_pixel_in_meters  # convert meters to pixels

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
    params.use_tf_function = False
    params.init_env_pfnet = False
    params.store_results = True

    params.env_mode = 'headless'
    os.environ['CUDA_VISIBLE_DEVICES'] = str(params.device_idx)
    os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'

    # set random seeds
    random.seed(params.seed)
    np.random.seed(params.seed)
    tf.random.set_seed(params.seed)

    return params

def draw_floor_map(floor_map, map_shape, plt_ax, map_plt):
    """
    Render the scene floor map
    :param ndarray floor_map: environment scene floor map
    :param ndarray map_shape: (unpadded) map shape [H, W, C]
    :param matplotlib.axes.Axes plt_ax: figure sub plot instance
    :return matplotlib.image.AxesImage: updated plot of scene floor map
    """

    origin_x, origin_y = map_shape[1]/2, map_shape[0]/2
    if map_plt is None:
        # draw floor map
        map_plt = plt_ax.imshow(floor_map, cmap='gray', origin='lower')
        plt.scatter(origin_x, origin_y, s=10, c='black', marker='x', alpha=1)
    else:
        # do nothing
        pass
    return map_plt

def draw_robot_pose(true_states, plt_ax):

    heading_len  = robot_radius = 10.0
    heading_plt = None
    for idx in range(true_states.shape[0]):
        x, y, heading = true_states[idx]

        xdata = [x, x + (robot_radius + heading_len) * np.cos(heading)]
        ydata = [y, y + (robot_radius + heading_len) * np.sin(heading)]
        if heading_plt == None:
            heading_plt, = plt_ax.plot(xdata, ydata, color='blue', alpha=0.5)
            heading_plt = None
        else:
            heading_plt.update({'xdata': xdata, 'ydata': ydata})

def display_data(arg_params):
    """
    """
    root_dir = os.path.expanduser(arg_params.root_dir)
    train_dir = os.path.join(root_dir, 'train')
    eval_dir = os.path.join(root_dir, 'eval')

    # training data
    filenames = list(glob.glob(os.path.join(arg_params.tfrecordpath, 'train', '*.tfrecord')))
    train_ds = datautils.get_dataflow(filenames, arg_params.batch_size, arg_params.s_buffer_size, is_training=True)
    print(f'train data: {filenames}')

    # create igibson env which is used "only" to sample particles
    env = LocalizeGibsonEnv(
        config_file=arg_params.config_file,
        scene_id=arg_params.scene_id,
        mode=arg_params.env_mode,
        use_tf_function=arg_params.use_tf_function,
        init_pfnet=arg_params.init_env_pfnet,
        action_timestep=arg_params.action_timestep,
        physics_timestep=arg_params.physics_timestep,
        device_idx=arg_params.device_idx
    )
    env.reset()
    arg_params.trajlen = env.config.get('max_step', 500)
    arg_params.floors = 2

    b_idx = 0
    fig = plt.figure(figsize=(14, 14))
    plts = {}
    for idx in range(arg_params.floors):
        plts[idx] = fig.add_subplot(1,arg_params.floors,idx+1)

    # run training over all training samples in an epoch
    train_itr = train_ds.as_numpy_iterator()
    for idx in tqdm(range(arg_params.num_train_batches)):

        parsed_record = next(train_itr)
        batch_sample = datautils.transform_raw_record(env, parsed_record, arg_params)

        observation = tf.convert_to_tensor(batch_sample['observation'], dtype=tf.float32)
        odometry = tf.convert_to_tensor(batch_sample['odometry'], dtype=tf.float32)
        true_states = tf.convert_to_tensor(batch_sample['true_states'], dtype=tf.float32)
        obstacle_map = tf.convert_to_tensor(batch_sample['obstacle_map'], dtype=tf.float32)
        org_map_shape = batch_sample['org_map_shape']

        scene_id = parsed_record['scene_id'][b_idx][0].decode('utf-8')
        floor_num = parsed_record['floor_num'][b_idx][0]
        key = scene_id + '_' + str(floor_num)
        plt_ax = plts[floor_num]

        map_plt = draw_floor_map(obstacle_map[b_idx].numpy(), org_map_shape[b_idx], plt_ax, None)

        draw_robot_pose(true_states[b_idx].numpy(), plt_ax)

    plt.tight_layout()
    for key, plt_ax in plts.items():
        extent = plt_ax.get_tightbbox(fig.canvas.get_renderer()).transformed(fig.dpi_scale_trans.inverted())
        fig.savefig(f'{key}.png', bbox_inches=extent)
    fig.savefig('full_figure.png')

if __name__ == '__main__':
    parsed_params = parse_args()
    display_data(parsed_params)
