#!/usr/bin/env python3

import os
import argparse
import numpy as np
import tensorflow as tf

np.set_printoptions(precision=3, suppress=True)

def parse_args():
    """
    parse command line arguments

    :return dict: dictionary of parameters
    """

    argparser = argparse.ArgumentParser()

    # training data
    argparser.add_argument('--trainfiles', nargs='*', default=['./house3d_data/eval/valid.tfrecords'], help='Data file(s) for training (tfrecord).')
    argparser.add_argument('--evalfiles', nargs='*', default=['./house3d_data/eval/valid.tfrecords'], help='Data file(s) for validation or evaluation (tfrecord).')
    argparser.add_argument('--testfiles', nargs='*', default=['./house3d_data/eval/valid.tfrecords'], help='Data file(s) for testing (tfrecord).')

    # input configuration
    argparser.add_argument('--map_pixel_in_meters', type=float, default=0.02, help='The width (and height) of a pixel of the map in meters. Defaults to 0.02 for House3D data.')

    argparser.add_argument('--init_particles_distr', type=str, default='tracking', help='Distribution of initial particles. Possible values: tracking / one-room.')
    argparser.add_argument('--init_particles_std', nargs='*', default=["0.3", "0.523599"], help='Standard deviations for generated initial particles for tracking distribution. Values: translation std (meters), rotation std (radians)')
    argparser.add_argument('--trajlen', type=int, default=24, help='Length of trajectories.')

    # PF configuration
    argparser.add_argument('--num_particles', type=int, default=30, help='Number of particles in Particle Filter.')
    argparser.add_argument('--transition_std', nargs='*', default=["0.0", "0.0"], help='Standard deviations for transition model. Values: translation std (meters), rotation std (radians)')
    argparser.add_argument('--resample', type=str, default='false', help='Resample particles in Particle Filter. Possible values: true / false.')
    argparser.add_argument('--alpha_resample_ratio', type=float, default=1.0, help='Trade-off parameter for soft-resampling in PF-net. Only effective if resample == true. Assumes values 0.0 < alpha <= 1.0. Alpha equal to 1.0 corresponds to hard-resampling.')

    # training configuration
    argparser.add_argument('--batch_size', type=int, default=24, help='Minibatch size for training.')
    argparser.add_argument('--learningrate', type=float, default=0.0025, help='Initial learning rate for training.')
    argparser.add_argument('--epochs', type=int, default=1, help='Number of epochs for training.')

    argparser.add_argument('--load', type=str, default='', help='Load a previously trained model from a checkpoint file.')
    argparser.add_argument('--seed', type=int, default='42', help='Fix the random seed of numpy and tensorflow.')
    argparser.add_argument('--logpath', type=str, default='./log/', help='Specify path for logs.')
    argparser.add_argument('--gpu_num', type=int, default='0', help='use gpu no. to train')

    params = argparser.parse_args()

    # convert multi-input fileds to numpy arrays
    params.transition_std = np.array(params.transition_std, np.float32)
    params.init_particles_std = np.array(params.init_particles_std, np.float32)

    params.transition_std[0] = params.transition_std[0] / params.map_pixel_in_meters  # convert meters to pixels
    params.init_particles_std[0] = params.init_particles_std[0] / params.map_pixel_in_meters  # convert meters to pixels

    # build initial covariance matrix of particles, in pixels and radians
    particle_std = params.init_particles_std.copy()
    particle_std2 = np.square(particle_std)  # variance
    params.init_particles_cov = np.diag(particle_std2[(0, 0, 1), ])

    # fix seed
    np.random.seed(params.seed)
    tf.random.set_seed(params.seed)

    # use RNN as stateful/non-stateful
    params.stateful = False
    params.return_state = True

    #HACK hardcode fix padding for map
    params.global_map_size = np.array([4000, 4000, 1])
    params.window_scaler = 8.0

    # filter out info and warning messages
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

    # convert boolean fields
    if params.resample not in ['false', 'true']:
        raise ValurError
    else:
        params.resample = (params.resample == 'true')

    gpus = tf.config.experimental.list_physical_devices('GPU')
    assert params.gpu_num < len(gpus)
    if gpus:
        # restrict TF to only use the first GPU
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            tf.config.experimental.set_visible_devices(gpus[params.gpu_num], 'GPU')
            logical_gpus = tf.config.experimental.list_logical_devices('GPU')
        except RuntimeError as e:
            # visible devices must be set before GPUs have been initialized
            print(e)

    return params

if __name__ == '__main__':
    parse_args()
