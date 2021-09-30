#!/usr/bin/env python3

import argparse
import glob
import numpy as np
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
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
        '--epochs',
        type=int,
        default=1,
        help='Number of epochs for training'
    )
    arg_parser.add_argument(
        '--num_train_samples',
        type=int,
        default=1,
        help='Total number of samples to use for training. Total training samples will be num_train_samples=num_train_batches*batch_size'
    )
    arg_parser.add_argument(
        '--num_eval_samples',
        type=int,
        default=1,
        help='Total number of samples to use for evaluation. Total evaluation samples will be num_eval_samples=num_eval_batches*batch_size'
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
        '--pfnet_loadpath',
        type=str,
        default=os.path.join(
            os.path.dirname(os.path.realpath(__file__)),
            'pfnetwork/checkpoints/pfnet_igibson_data/checkpoint_87_5.830',
            'pfnet_checkpoint'),
        help='Load a previously trained pfnet model from a checkpoint file.'
    )
    arg_parser.add_argument(
        '--learning_rate',
        type=float,
        default=2.5e-3,
        help='Learning rate for training'
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
        help='Use gpu no. to train/eval'
    )
    arg_parser.add_argument(
        '--multiple_gpus',
        type=str,
        default='false',
        help='Use multiple GPUs'
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

    # For the igibson maps, originally each pixel represents 0.01m, and the center of the image correspond to (0,0)
    params.map_pixel_in_meters = 0.01
    # in igibson we work with rescaled 0.01m to 0.1m maps to sample robot poses
    params.trav_map_resolution = 0.1

    # post-processing
    params.num_train_batches = params.num_train_samples//params.batch_size
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

    if params.resample not in ['false', 'true'] or \
        params.multiple_gpus not in ['false', 'true']:
        raise ValueError
    else:
        params.resample = (params.resample == 'true')
        params.multiple_gpus = (params.multiple_gpus == 'true')

    # use RNN as stateful/non-stateful
    params.stateful = False
    params.return_state = True

    # HACK:
    params.loop = 6
    params.use_tf_function = False
    params.init_env_pfnet = False
    params.store_results = True
    params.num_clusters = 10

    params.env_mode = 'headless'
    if params.multiple_gpus:
        os.environ['CUDA_VISIBLE_DEVICES'] = '0,1'
    else:
        os.environ['CUDA_VISIBLE_DEVICES'] = str(params.device_idx)
    os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'

    # set random seeds
    random.seed(params.seed)
    np.random.seed(params.seed)
    tf.random.set_seed(params.seed)

    return params


def pfnet_train(arg_params):
    """
    A simple train for particle filter network

    :param arg_params:
        parsed command-line arguments
    :return:
    """
    root_dir = os.path.expanduser(arg_params.root_dir)
    train_dir = os.path.join(root_dir, 'train')
    eval_dir = os.path.join(root_dir, 'eval')

    # training data
    filenames = list(glob.glob(os.path.join(arg_params.tfrecordpath, 'train', '*.tfrecord')))
    train_ds = datautils.get_dataflow(filenames, arg_params.batch_size, arg_params.s_buffer_size, is_training=True)
    print(f'train data: {filenames}')

    # evaluation data
    filenames = list(glob.glob(os.path.join(arg_params.tfrecordpath, 'eval', '*.tfrecord')))
    eval_ds = datautils.get_dataflow(filenames, arg_params.batch_size, arg_params.s_buffer_size, is_training=True)
    print(f'eval data: {filenames}')

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
    arg_params.trajlen = env.config.get('max_step', 500)//arg_params.loop
    assert arg_params.trav_map_resolution == env.trav_map_resolution

    # create particle filter net model
    if arg_params.multiple_gpus:
        strategy = tf.distribute.MirroredStrategy()
        with strategy.scope():
            pfnet_model = pfnet.pfnet_model(arg_params)
    else:
        pfnet_model = pfnet.pfnet_model(arg_params)
    print("=====> Created pf model ")

    # load model from checkpoint file
    if arg_params.pfnet_loadpath:
        pfnet_model.load_weights(arg_params.pfnet_loadpath)
        print("=====> Loaded pf model from: " + arg_params.pfnet_loadpath)

    # Adam optimizer.
    optimizer = tf.optimizers.Adam(learning_rate=arg_params.learning_rate)

    # Define metrics
    train_loss = keras.metrics.Mean('train_loss', dtype=tf.float32)
    eval_loss = keras.metrics.Mean('eval_loss', dtype=tf.float32)

    # Logging
    summaries_flush_secs=10
    train_summary_writer = tf.compat.v2.summary.create_file_writer(
        train_dir, flush_millis=summaries_flush_secs * 1000)
    eval_summary_writer = tf.compat.v2.summary.create_file_writer(
        eval_dir, flush_millis=summaries_flush_secs * 1000)

    trajlen = arg_params.trajlen
    batch_size = arg_params.batch_size
    num_particles = arg_params.num_particles
    print(arg_params)

    # Recommended: wrap to tf.graph for better performance
    @tf.function
    def train_step(model_input, true_states):
        """ Run one training step """

        # enable auto-differentiation
        with tf.GradientTape() as tape:
            # forward pass
            output, state = pfnet_model(model_input, training=True)
            particle_states, particle_weights = output

            # sanity check
            assert list(particle_states.shape) == [batch_size, trajlen, num_particles, 3]
            assert list(particle_weights.shape) == [batch_size, trajlen, num_particles]
            assert list(true_states.shape) == [batch_size, trajlen, 3]

            # compute loss
            loss_dict = pfnet_loss.compute_mse_loss(particle_states, particle_weights, true_states,
                                                arg_params.trav_map_resolution)
            loss_pred = loss_dict['pred']

        # compute gradients of the trainable variables with respect to the loss
        gradients = tape.gradient(loss_pred, pfnet_model.trainable_weights)
        gradients = list(zip(gradients, pfnet_model.trainable_weights))

        # run one step of gradient descent
        optimizer.apply_gradients(gradients)
        train_loss(loss_pred)  # overall trajectory loss

    # Recommended: wrap to tf.graph for better performance
    @tf.function
    def eval_step(model_input, true_states):
        """ Run one evaluation step """

        # forward pass
        output, state = pfnet_model(model_input, training=False)
        particle_states, particle_weights = output

        # sanity check
        assert list(particle_states.shape) == [batch_size, trajlen, num_particles, 3]
        assert list(particle_weights.shape) == [batch_size, trajlen, num_particles]
        assert list(true_states.shape) == [batch_size, trajlen, 3]

        # compute loss
        loss_dict = pfnet_loss.compute_mse_loss(particle_states, particle_weights, true_states,
                                            arg_params.trav_map_resolution)
        loss_pred = loss_dict['pred']

        eval_loss(loss_pred)  # overall trajectory loss

    # repeat for a fixed number of epochs train and eval loops
    for epoch in tqdm(range(arg_params.epochs)):

        #------------------------#
        # run training over all training samples in an epoch
        train_itr = train_ds.as_numpy_iterator()
        for idx in range(arg_params.num_train_batches):

            parsed_record = next(train_itr)
            batch_sample = datautils.transform_raw_record(env, parsed_record, arg_params)

            observation = tf.convert_to_tensor(batch_sample['observation'], dtype=tf.float32)
            odometry = tf.convert_to_tensor(batch_sample['odometry'], dtype=tf.float32)
            true_states = tf.convert_to_tensor(batch_sample['true_states'], dtype=tf.float32)
            floor_map = tf.convert_to_tensor(batch_sample['floor_map'], dtype=tf.float32)
            init_particles = tf.convert_to_tensor(batch_sample['init_particles'], dtype=tf.float32)
            init_particle_weights = tf.constant(np.log(1.0 / float(num_particles)),
                                                shape=(batch_size, num_particles), dtype=tf.float32)

            # start trajectory with initial particles and weights
            state = [init_particles, init_particle_weights, floor_map]

            # if stateful: reset RNN s.t. initial_state is set to initial particles and weights
            # if non-stateful: pass the state explicity every step
            if arg_params.stateful:
                pfnet_model.layers[-1].reset_states(state)  # RNN layer

            pf_input = [observation, odometry]
            model_input = (pf_input, state)

            train_step(model_input, true_states)

        # log epoch training stats
        with train_summary_writer.as_default():
            tf.summary.scalar('loss', train_loss.result(), step=epoch)

        # Save the weights
        print("=====> saving trained model ")
        pfnet_model.save_weights(
            os.path.join(
                train_dir,
                f'chks/checkpoint_{epoch}_{train_loss.result():03.3f}/pfnet_checkpoint'
            )
        )

        #------------------------#
        # run evaluation over all eval samples in an epoch
        eval_itr = eval_ds.as_numpy_iterator()
        for idx in range(arg_params.num_eval_batches):

            parsed_record = next(eval_itr)
            batch_sample = datautils.transform_raw_record(env, parsed_record, arg_params)

            observation = tf.convert_to_tensor(batch_sample['observation'], dtype=tf.float32)
            odometry = tf.convert_to_tensor(batch_sample['odometry'], dtype=tf.float32)
            true_states = tf.convert_to_tensor(batch_sample['true_states'], dtype=tf.float32)
            floor_map = tf.convert_to_tensor(batch_sample['floor_map'], dtype=tf.float32)
            init_particles = tf.convert_to_tensor(batch_sample['init_particles'], dtype=tf.float32)
            init_particle_weights = tf.constant(np.log(1.0 / float(num_particles)),
                                                shape=(batch_size, num_particles), dtype=tf.float32)

            # start trajectory with initial particles and weights
            state = [init_particles, init_particle_weights, floor_map]

            # if stateful: reset RNN s.t. initial_state is set to initial particles and weights
            # if non-stateful: pass the state explicity every step
            if arg_params.stateful:
                pfnet_model.layers[-1].reset_states(state)  # RNN layer

            pf_input = [observation, odometry]
            model_input = (pf_input, state)

            eval_step(model_input, true_states)

        # log epoch evaluation stats
        with eval_summary_writer.as_default():
            tf.summary.scalar('loss', eval_loss.result(), step=epoch)

        # Save the weights
        print("=====> saving evaluation model ")
        pfnet_model.save_weights(
            os.path.join(
                eval_dir,
                f'chks/checkpoint_{epoch}_{eval_loss.result():03.3f}/pfnet_checkpoint'
            )
        )

        print(f'Epoch {epoch}, train loss: {train_loss.result():03.3f}, eval loss: {eval_loss.result():03.3f}')

        # Reset the metrics at the start of the next epoch
        train_loss.reset_states()
        eval_loss.reset_states()

    print('training finished')


if __name__ == '__main__':
    parsed_params = parse_args()
    pfnet_train(parsed_params)
