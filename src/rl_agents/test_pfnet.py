#!/usr/bin/env python3

import argparse
import numpy as np
import os
import random
import tensorflow as tf

from pfnetwork.pfnet import pfnet_model
from environments import suite_gibson


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
            'pfnetwork/checkpoints/checkpoint_87_5.830',
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
        '--trajlen',
        type=int,
        default=1,
        help='Total length of trajectory'
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
            'turtlebot_point_nav.yaml'
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

    if params.resample not in ['false', 'true']:
        raise ValueError
    else:
        params.resample = (params.resample == 'true')

    # use RNN as stateful/non-stateful
    params.stateful = False
    params.return_state = True

    # HACK: hardcoded values for floor map/obstacle map
    params.global_map_size = [1000, 1000, 1]
    params.window_scaler = 8.0

    params.is_localize_env = True

    # set random seeds
    random.seed(params.seed)
    np.random.seed(params.seed)
    tf.random.set_seed(params.seed)

    return params


def pfnet_test(arg_params):
    """
    A simple test for particle filter network

    :param arg_params:
        parsed command-line arguments
    :return:
    """

    # create particle filter net model
    pfnet = pfnet_model(arg_params)
    print("=====> Created pf model ")

    # load model from checkpoint file
    if arg_params.pfnet_loadpath:
        pfnet.load_weights(arg_params.pfnet_loadpath)
        print("=====> Loaded pf model from: " + arg_params.pfnet_loadpath)

        # create igibson env
        env = suite_gibson.load(
            config_file=arg_params.config_file,
            model_id=None,
            env_mode='headless',
            is_localize_env=arg_params.is_localize_env,
            action_timestep=arg_params.action_timestep,
            physics_timestep=arg_params.physics_timestep,
            device_idx=arg_params.device_idx,
        )
        env.reset()


if __name__ == '__main__':
    parsed_params = parse_args()
    pfnet_test(parsed_params)
