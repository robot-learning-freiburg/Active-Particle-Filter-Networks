#!/usr/bin/env python3

import argparse
import cv2
import glob
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg
import numpy as np
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
from pathlib import Path
import random
import tensorflow as tf

from pfnetwork import pfnet
from environments.env_utils import datautils, pfnet_loss, render
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
            os.path.dirname(os.path.realpath(__file__)),
            'configs',
            'turtlebot_random_nav.yaml'
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


def store_results(eps_idx, floor_map, org_map_shape, particle_states, particle_weights, true_states, env, params):
    trajlen = params.trajlen
    b_idx = 0

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
    floor_map = floor_map[b_idx].numpy()  # [H, W, 1]
    pad_map_shape = floor_map.shape
    o_map_shape = org_map_shape[b_idx]

    # HACK:
    plt_ax.set_yticks(np.arange(0, pad_map_shape[0], pad_map_shape[0]//10))
    plt_ax.set_xticks(np.arange(0, pad_map_shape[1], pad_map_shape[1]//10))

    map_plt = render.draw_floor_map(floor_map, o_map_shape, plt_ax, None)

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
        particle_weight = particle_weights[:, traj, :]

        # plot true robot pose
        position_plt, heading_plt = gt_plt['robot_position'], gt_plt['robot_heading']
        gt_plt['robot_position'], gt_plt['robot_heading'] = render.draw_robot_pose(
            true_state[b_idx], '#7B241C', pad_map_shape, plt_ax,
            position_plt, heading_plt)

        # plot est robot pose
        position_plt, heading_plt = est_plt['robot_position'], est_plt['robot_heading']
        est_plt['robot_position'], est_plt['robot_heading'] = render.draw_robot_pose(
            est_state[b_idx], '#515A5A', pad_map_shape, plt_ax,
            position_plt, heading_plt)

        # plot est pose particles
        particles_plt = est_plt['particles']
        est_plt['particles'] = render.draw_particles_pose(
            particle_state[b_idx], particle_weight[b_idx],
            pad_map_shape, particles_plt)

        plt_ax.legend([gt_plt['robot_position'], est_plt['robot_position']], ["gt_pose", "est_pose"])

        canvas.draw()
        img = np.array(canvas.renderer._renderer)
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        images.append(img)

    gt_pose_mts = np.array([*env.scene.map_to_world(true_state[b_idx][:2]), true_state[b_idx][2]])
    est_pose_mts = np.array([*env.scene.map_to_world(est_state[b_idx][:2]), est_state[b_idx][2]])
    print(f'{eps_idx} End True Pose: {gt_pose_mts}, End Estimated Pose: {est_pose_mts} in mts')
    print(f'{eps_idx} End True Pose: {true_state[b_idx]}, End Estimated Pose: {est_state[b_idx]} in px')

    size = (images[0].shape[0], images[0].shape[1])
    out = cv2.VideoWriter(
            os.path.join(params.out_folder, f'result_{eps_idx}.avi'),
            cv2.VideoWriter_fourcc(*'XVID'), 30, size)

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
    test_dir = os.path.join(root_dir, 'test')

    # evaluation data
    filenames = list(glob.glob(arg_params.testfiles[0]))
    test_ds = datautils.get_dataflow(filenames, arg_params.batch_size, is_training=False)
    print(f'test data: {filenames}')

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
    env.reset()
    arg_params.trajlen = env.config.get('max_step', 500)//arg_params.loop
    assert arg_params.trav_map_resolution == env.trav_map_resolution

    # create particle filter net model
    pfnet_model = pfnet.pfnet_model(arg_params)
    print("=====> Created pf model ")

    # load model from checkpoint file
    if arg_params.pfnet_loadpath:
        pfnet_model.load_weights(arg_params.pfnet_loadpath)
        print("=====> Loaded pf model from: " + arg_params.pfnet_loadpath)

    if arg_params.use_tf_function:
        pfnet_model = tf.function(pfnet_model)
        print("=====> wrapped pfnet in tf.graph")

    trajlen = arg_params.trajlen
    batch_size = arg_params.batch_size
    num_particles = arg_params.num_particles

    print(arg_params)
    test_summary_writer = tf.summary.create_file_writer(test_dir)
    with test_summary_writer.as_default():
        # run over all evaluation samples in an epoch
        mse_list = []
        init_mse_list = []
        success_list = []
        itr = test_ds.as_numpy_iterator()
        for eps_idx in range(arg_params.num_eval_batches):
            parsed_record = next(itr)
            batch_sample = datautils.transform_raw_record(env, parsed_record, arg_params)

            observation = tf.convert_to_tensor(batch_sample['observation'], dtype=tf.float32)
            odometry = tf.convert_to_tensor(batch_sample['odometry'], dtype=tf.float32)
            true_states = tf.convert_to_tensor(batch_sample['true_states'], dtype=tf.float32)
            floor_map = tf.convert_to_tensor(batch_sample['floor_map'], dtype=tf.float32)
            org_map_shape = batch_sample['org_map_shape']
            init_particles = tf.convert_to_tensor(batch_sample['init_particles'], dtype=tf.float32)
            init_particle_weights = tf.constant(np.log(1.0 / float(num_particles)),
                                                shape=(batch_size, num_particles), dtype=tf.float32)
            # compute initial loss
            init_loss_dict = pfnet_loss.compute_loss(tf.expand_dims(init_particles, axis=1),
                            tf.expand_dims(init_particle_weights, axis=1),
                            tf.expand_dims(true_states[:, 0], axis=1),
                            arg_params.trav_map_resolution)
            # we have squared differences along the trajectory
            init_mse = np.mean(init_loss_dict['coords'])
            init_mse_list.append(init_mse)

            # start trajectory with initial particles and weights
            state = [init_particles, init_particle_weights, floor_map]

            # if stateful: reset RNN s.t. initial_state is set to initial particles and weights
            # if non-stateful: pass the state explicity every step
            if arg_params.stateful:
                pfnet_model.layers[-1].reset_states(state)  # RNN layer

            pf_input = [observation, odometry]
            model_input = (pf_input, state)

            # forward pass
            output, state = pfnet_model(model_input, training=False)
            particle_states, particle_weights = output

            # sanity check
            assert list(particle_states.shape) == [batch_size, trajlen, num_particles, 3]
            assert list(particle_weights.shape) == [batch_size, trajlen, num_particles]
            assert list(true_states.shape) == [batch_size, trajlen, 3]

            # compute loss
            loss_dict = pfnet_loss.compute_loss(particle_states, particle_weights, true_states,
                                                arg_params.trav_map_resolution)

            # we have squared differences along the trajectory
            mse = np.mean(loss_dict['coords'])
            mse_list.append(mse)

            # log mse (in meters)
            print(f'eps:{eps_idx} mean mse: {mse}')
            tf.summary.scalar('eps_mean_rmse', np.sqrt(mse), step=eps_idx)
            tf.summary.scalar('eps_final_rmse', np.sqrt(loss_dict['coords'][0][-1]), step=eps_idx)

            # localization is successfull if the rmse error is below 1m for the last 25% of the trajectory
            successful = np.all(loss_dict['coords'][-trajlen // 4:] < 1.0 ** 2)  # below 1 meter
            success_list.append(successful)

            if arg_params.store_results:
                # store results as video
                arg_params.out_folder = os.path.join(arg_params.root_dir, f'output')
                Path(arg_params.out_folder).mkdir(parents=True, exist_ok=True)
                store_results(eps_idx, floor_map, org_map_shape, particle_states, particle_weights, true_states, env, arg_params)

        # report results
        init_mean_rmse = np.mean(np.sqrt(init_mse_list)) * 100
        total_init_rmse = np.sqrt(np.mean(init_mse_list)) * 100
        print(f'Initial Mean RMSE (average RMSE per trajectory) = {init_mean_rmse:03.3f} cm')
        print(f'Overall Initial RMSE (reported value) = {total_init_rmse:03.3f} cm')

        mean_rmse = np.mean(np.sqrt(mse_list)) * 100
        total_rmse = np.sqrt(np.mean(mse_list)) * 100
        mean_success = np.mean(np.array(success_list, 'i')) * 100
        print(f'Mean RMSE (average RMSE per trajectory) = {mean_rmse:03.3f} cm')
        print(f'Overall RMSE (reported value) = {total_rmse:03.3f} cm')
        print(f'Success rate = {mean_success:03.3f} %')

    # close gym env
    env.close()


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
    arg_params.init_env_pfnet = True

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

        for _ in range(arg_params.loop):
            # take action and get new observation
            obs, reward, done, _ = env.step(action)
        env.render('human')

    env.close()

if __name__ == '__main__':
    parsed_params = parse_args()
    pfnet_test(parsed_params)

    # rt_pfnet_test(parsed_params)
