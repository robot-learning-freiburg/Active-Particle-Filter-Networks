#!/usr/bin/env python3

from absl import app
from absl import flags
from absl import logging
import argparse
import numpy as np
import os
import random
import tensorflow as tf

# import custom tf_agents
from environments.env_utils import datautils
from environments.envs.localize_env import LocalizeGibsonEnv

# define testing parameters
flags.DEFINE_string(
    name='filename',
    default='./test.tfrecord',
    help='The tf record.'
)
flags.DEFINE_integer(
    name='num_records',
    default=10,
    help='The number of episode data.'
)
flags.DEFINE_integer(
    name='seed',
    default=100,
    help='Fix the random seed'
)
flags.DEFINE_string(
    name='agent',
    default='random',
    help='Agent Behavior'
)

# define igibson env parameters
flags.DEFINE_string(
    name='config_file',
    default=os.path.join(
        os.path.dirname(os.path.realpath(__file__)),
        'configs',
        'turtlebot_random_nav.yaml'
    ),
    help='Config file for the experiment'
)
flags.DEFINE_string(
    name='scene_id',
    default=None,
    help='Environment scene'
)
flags.DEFINE_string(
    name='env_mode',
    default='headless',
    help='Environment render mode'
)
flags.DEFINE_string(
    name='obs_mode',
    default='rgb-depth',
    help='Observation input type. Possible values: rgb / depth / rgb-depth / occupancy_grid.'
)
flags.DEFINE_list(
    name='custom_output',
    default=['rgb_obs', 'depth_obs', 'floor_map', 'kmeans_cluster'],
    help='A comma-separated list of env observation types.'
)
flags.DEFINE_float(
    name='action_timestep',
    default=1.0 / 10.0,
    help='Action time step for the simulator'
)
flags.DEFINE_float(
    name='physics_timestep',
    default=1.0 / 40.0,
    help='Physics time step for the simulator'
)
flags.DEFINE_integer(
    name='gpu_num',
    default=0,
    help='GPU id for graphics/computation'
)
flags.DEFINE_boolean(
    name='is_discrete',
    default=False,
    help='Whether to use discrete/continuous actions'
)
flags.DEFINE_float(
    name='velocity',
    default=1.0,
    help='Velocity of Robot'
)
flags.DEFINE_integer(
    name='max_step',
    default=10,
    help='The maimum number of episode steps.'
)

# define pfNet env parameters
flags.DEFINE_boolean(
    name='init_env_pfnet',
    default=False,
    help='Whether to initialize particle filter net'
)

FLAGS = flags.FLAGS


def collect_data(env, params, filename='./test.tfrecord', num_records=10):
    """
    Run the gym environment and collect the required stats
    :param env: igibson env instance
    :param params: parsed parameters
    :param filename: tf record file name
    :param num_records: number of records(episodes) to collect
    :return dict: episode stats data containing:
        odometry, true poses, observation, particles, particles weights, floor map
    """

    with tf.io.TFRecordWriter(filename) as writer:
        for i in range(num_records):
            print(f'episode: {i}')
            episode_data = datautils.gather_episode_stats(env, params, sample_particles=False)
            record = datautils.serialize_tf_record(episode_data)
            writer.write(record)

    print(f'Collected successfully in {filename}')

    # sanity check
    ds = datautils.get_dataflow([filename], batch_size=1, s_buffer_size=100, is_training=False)
    data_itr = ds.as_numpy_iterator()
    for idx in range(num_records):
        parsed_record = next(data_itr)
        batch_sample = datautils.transform_raw_record(env, parsed_record, params)

def main(_):
    logging.set_verbosity(logging.INFO)
    tf.compat.v1.enable_v2_behavior()
    # tf.debugging.enable_check_numerics()  # error out inf or NaN

    os.environ['CUDA_VISIBLE_DEVICES'] = str(FLAGS.gpu_num)
    os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'

    # set random seeds
    random.seed(FLAGS.seed)
    np.random.seed(FLAGS.seed)
    tf.random.set_seed(FLAGS.seed)

    if os.path.exists(FLAGS.filename):
        print('File Already Exists !!!')
        return

    env = LocalizeGibsonEnv(
        config_file=FLAGS.config_file,
        scene_id=FLAGS.scene_id,
        mode=FLAGS.env_mode,
        use_tf_function=True,
        init_pfnet=FLAGS.init_env_pfnet,
        action_timestep=FLAGS.action_timestep,
        physics_timestep=FLAGS.physics_timestep,
        device_idx=FLAGS.gpu_num
    )
    # HACK: override value from config file
    FLAGS.max_step = env.config.get('max_step', 500)
    FLAGS.is_discrete = env.config.get("is_discrete", False)
    FLAGS.velocity = env.config.get("velocity", 1.0)

    print('==================================================')
    for k, v in FLAGS.flag_values_dict().items():
        print(k, v)
    print('==================================================')

    argparser = argparse.ArgumentParser()
    params = argparser.parse_args([])

    params.loop = 6
    params.agent = FLAGS.agent
    params.trajlen = FLAGS.max_step//params.loop
    params.max_lin_vel = env.config.get("linear_velocity", 0.5)
    params.max_ang_vel = env.config.get("angular_velocity", np.pi/2)
    params.global_map_size = np.array([4000, 4000, 1])
    params.obs_mode = FLAGS.obs_mode
    params.batch_size = 1
    params.num_particles = 10
    params.init_particles_distr = 'gaussian'
    particle_std = np.array([0.3, 0.523599])
    particle_std2 = np.square(particle_std)  # variance
    params.init_particles_cov = np.diag(particle_std2[(0, 0, 1), ])
    params.particles_range = 100

    # compute observation channel dim
    if params.obs_mode == 'rgb-depth':
        params.obs_ch = 4
    elif params.obs_mode == 'rgb':
        params.obs_ch = 3
    elif params.obs_mode == 'depth' or params.obs_mode == 'occupancy_grid':
        params.obs_ch = 1
    else:
        raise ValueError

    print(params)
    collect_data(env, params, FLAGS.filename, FLAGS.num_records)

    # test_ds = get_dataflow([FLAGS.filename])
    # itr = test_ds.as_numpy_iterator()
    # parsed_record = next(itr)
    # data_sample = transform_raw_record(parsed_record)
    # print(data_sample['actions'])


if __name__ == '__main__':
    app.run(main)
