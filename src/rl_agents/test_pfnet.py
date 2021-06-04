#!/usr/bin/env python3

import argparse
import cv2
import glob
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg
import numpy as np
import os
from pathlib import Path
import random
import tensorflow as tf

from pfnetwork import pfnet
from environments.env_utils import datautils, pfnet_loss, render
from environments.envs.localize_env import LocalizeGibsonEnv


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
        '--root_dir',
        type=str,
        default='./test_output',
        help='Root directory for logs/summaries/checkpoints.'
    )
    arg_parser.add_argument(
        '--num_eval_episodes',
        type=int,
        default=1,
        help='The number of episodes to run eval on.'
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
        default=os.path.join(
            os.path.dirname(os.path.realpath(__file__)),
            'pfnetwork/checkpoints/pfnet_igibson_data/checkpoint_87_5.830',
            'pfnet_checkpoint'),
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
        help='use gpu no. to train'
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
        default=["15", "0.523599"],
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
        default=["0.0", "0.0"],
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

    # define igibson env parameters
    arg_parser.add_argument(
        '--config_file',
        type=str,
        default=os.path.join(
            os.path.dirname(os.path.realpath(__file__)),
            'configs',
            'turtlebot_random_nav.yaml'
        ),
        help='Config file for the experiment'
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

    # post-processing

    # convert multi-input fields to numpy arrays
    params.transition_std = np.array(params.transition_std, np.float32)
    params.init_particles_std = np.array(params.init_particles_std, np.float32)

    # build initial covariance matrix of particles, in pixels and radians
    particle_std = params.init_particles_std.copy()
    # particle_std[0] = particle_std[0] / params.map_pixel_in_meters  # convert meters to pixels
    particle_std2 = np.square(particle_std)  # variance
    params.init_particles_cov = np.diag(particle_std2[(0, 0, 1), ])

    if params.resample not in ['false', 'true']:
        raise ValueError
    else:
        params.resample = (params.resample == 'true')

    # use RNN as stateful/non-stateful
    params.stateful = False
    params.return_state = True

    # HACK: hardcoded values for floor map/obstacle map
    params.map_pixel_in_meters = 0.1
    params.global_map_size = [1000, 1000, 1]
    params.window_scaler = 8.0

    params.use_pfnet = False
    params.store_results = False

    gpu_num = 0
    os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_num)
    os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'

    # set random seeds
    random.seed(params.seed)
    np.random.seed(params.seed)
    tf.random.set_seed(params.seed)

    return params


def store_results(idx, obstacle_map, particle_states, particle_weights, true_states, params):
    trajlen = params.trajlen

    fig = plt.figure(figsize=(7, 7))
    plt_ax = fig.add_subplot(111)
    canvas = FigureCanvasAgg(fig)

    lin_weights = tf.nn.softmax(particle_weights, axis=-1)
    est_states = tf.math.reduce_sum(tf.math.multiply(
        particle_states[:, :, :, :], lin_weights[:, :, :, None]
    ), axis=2)

    # normalize between [-pi, +pi]
    part_x, part_y, part_th = tf.unstack(est_states, axis=-1, num=3)  # (k, 3)
    part_th = tf.math.floormod(part_th + np.pi, 2 * np.pi) - np.pi
    est_states = tf.stack([part_x, part_y, part_th], axis=-1)

    # plot map
    floor_map = obstacle_map[0].numpy()  # [H, W, 1]
    map_plt = render.draw_floor_map(floor_map, plt_ax, None)

    images = []
    gt_plt = {
        'robot_position': None,
        'robot_heading': None,
    }
    est_plt = {
        'robot_position': None,
        'robot_heading': None,
        'particles': None,
    }
    for traj in range(trajlen):
        true_state = true_states[:, traj, :]
        est_state = est_states[:, traj, :]
        particle_state = particle_states[:, traj, :, :]
        lin_weight = lin_weights[:, traj, :]

        # plot true robot pose
        position_plt, heading_plt = gt_plt['robot_position'], gt_plt['robot_heading']
        gt_plt['robot_position'], gt_plt['robot_heading'] = render.draw_robot_pose(
            true_state[0], '#7B241C', floor_map.shape, plt_ax,
            position_plt, heading_plt)

        # plot est robot pose
        position_plt, heading_plt = est_plt['robot_position'], est_plt['robot_heading']
        est_plt['robot_position'], est_plt['robot_heading'] = render.draw_robot_pose(
            est_state[0], '#515A5A', floor_map.shape, plt_ax,
            position_plt, heading_plt)

        # plot est pose particles
        particles_plt = est_plt['particles']
        est_plt['particles'] = render.draw_particles_pose(
            particle_state[0], lin_weight[0],
            floor_map.shape, particles_plt)

        plt_ax.legend([gt_plt['robot_position'], est_plt['robot_position']], ["gt_pose", "est_pose"])

        canvas.draw()
        img = np.array(canvas.renderer._renderer)
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        images.append(img)

    print(f'{idx} True Pose: {true_state[0]}, Estimated Pose: {est_state[0]}')

    size = (images[0].shape[0], images[0].shape[1])
    out = cv2.VideoWriter(params.out_folder + f'result_{idx}.avi', cv2.VideoWriter_fourcc(*'XVID'), 30, size)

    for i in range(len(images)):
        out.write(images[i])
        # cv2.imwrite(params.out_folder + f'result_img_{i}.png', images[i])
    out.release()


def pfnet_test(arg_params):
    """
    A simple test for particle filter network

    :param arg_params:
        parsed command-line arguments
    :return:
    """

    root_dir = os.path.expanduser(arg_params.root_dir)
    log_dir = os.path.join(arg_params.root_dir, 'log_dir')

    # evaluation data
    filenames = list(glob.glob(arg_params.testfiles[0]))
    test_ds = datautils.get_dataflow(filenames, arg_params.batch_size, is_training=False)
    print(f'test data: {filenames}')

    # create igibson env which is used "only" to sample particles
    env = LocalizeGibsonEnv(
        config_file=arg_params.config_file,
        scene_id=None,
        mode='headless',
        use_tf_function=True,
        use_pfnet=arg_params.use_pfnet,
        action_timestep=arg_params.action_timestep,
        physics_timestep=arg_params.physics_timestep,
        device_idx=arg_params.device_idx
    )
    env.reset()
    arg_params.trajlen = env.config.get('max_step', 500)

    # create particle filter net model
    pfnet_model = pfnet.pfnet_model(arg_params)
    print("=====> Created pf model ")

    # load model from checkpoint file
    if arg_params.pfnet_loadpath:
        pfnet_model.load_weights(arg_params.pfnet_loadpath)
        print("=====> Loaded pf model from: " + arg_params.pfnet_loadpath)

    trajlen = arg_params.trajlen
    batch_size = arg_params.batch_size
    num_particles = arg_params.num_particles

    test_summary_writer = tf.summary.create_file_writer(log_dir)
    with test_summary_writer.as_default():
        # run over all evaluation samples in an epoch
        mse_list = []
        success_list = []
        itr = test_ds.as_numpy_iterator()
        for idx in range(arg_params.num_eval_episodes):
            parsed_record = next(itr)
            batch_sample = datautils.transform_raw_record(env, parsed_record, arg_params)

            observation = tf.convert_to_tensor(batch_sample['observation'], dtype=tf.float32)
            odometry = tf.convert_to_tensor(batch_sample['odometry'], dtype=tf.float32)
            true_states = tf.convert_to_tensor(batch_sample['true_states'], dtype=tf.float32)
            floor_map = tf.convert_to_tensor(batch_sample['floor_map'], dtype=tf.float32)
            obstacle_map = tf.convert_to_tensor(batch_sample['obstacle_map'], dtype=tf.float32)
            init_particles = tf.convert_to_tensor(batch_sample['init_particles'], dtype=tf.float32)
            init_particle_weights = tf.constant(np.log(1.0 / float(num_particles)),
                                                shape=(batch_size, num_particles), dtype=tf.float32)

            # start trajectory with initial particles and weights
            state = [init_particles, init_particle_weights, obstacle_map]

            # if stateful: reset RNN s.t. initial_state is set to initial particles and weights
            # if non-stateful: pass the state explicity every step
            if arg_params.stateful:
                pfnet_model.layers[-1].reset_states(state)  # RNN layer

            pf_input = [observation, odometry]
            model_input = (pf_input, state)

            # forward pass
            output, state = pfnet_model(model_input, training=False)

            # compute loss
            particle_states, particle_weights = output
            loss_dict = pfnet_loss.compute_loss(particle_states, particle_weights, true_states,
                                                arg_params.map_pixel_in_meters)

            # we have squared differences along the trajectory
            mse = np.mean(loss_dict['coords'])
            mse_list.append(mse)

            # log
            tf.summary.scalar('eps_mean_rmse', np.sqrt(mse), step=idx)
            tf.summary.scalar('eps_final_rmse', np.sqrt(loss_dict['coords'][0][-1]), step=idx)

            # localization is successfull if the rmse error is below 1m for the last 25% of the trajectory
            successful = np.all(loss_dict['coords'][-trajlen // 4:] < 1.0 ** 2)  # below 1 meter
            success_list.append(successful)

            if arg_params.store_results:
                # store results as video
                params.out_folder = os.path.join(arg_params.root_dir, f'output_{idx}')
                Path(params.out_folder).mkdir(parents=True, exist_ok=True)
                store_results(idx, obstacle_map, particle_states, particle_weights, true_states, arg_params)

        # report results
        mean_rmse = np.mean(np.sqrt(mse_list)) * 100
        total_rmse = np.sqrt(np.mean(mse_list)) * 100
        mean_success = np.mean(np.array(success_list, 'i')) * 100
        print(f'Mean RMSE (average RMSE per trajectory) = {mean_rmse:03.3f} cm')
        print(f'Overall RMSE (reported value) = {total_rmse:03.3f} cm')
        print(f'Success rate = {mean_success:03.3f} %')

    # close gym env
    env.close()


if __name__ == '__main__':
    parsed_params = parse_args()
    pfnet_test(parsed_params)
