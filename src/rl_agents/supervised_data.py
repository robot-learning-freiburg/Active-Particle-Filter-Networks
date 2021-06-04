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
    name='env_mode',
    default='headless',
    help='Environment render mode'
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
    name='use_pfnet',
    default=False,
    help='Whether to use particle filter net'
)

FLAGS = flags.FLAGS


def collect_data(env, params, filename='./test.tfrecord', num_records=10):
    """
    Run the gym environment and collect the required stats
    :param params: parsed parameters
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
        scene_id=None,
        mode=FLAGS.env_mode,
        use_tf_function=True,
        use_pfnet=FLAGS.use_pfnet,
        action_timestep=FLAGS.action_timestep,
        physics_timestep=FLAGS.physics_timestep,
        device_idx=FLAGS.gpu_num
    )
    FLAGS.max_step = env.config.get('max_step', 500)
    FLAGS.is_discrete = env.config.get("is_discrete", False)
    FLAGS.velocity = env.config.get("velocity", 1.0)

    print('==================================================')
    for k, v in FLAGS.flag_values_dict().items():
        print(k, v)
    print('==================================================')

    argparser = argparse.ArgumentParser()
    params = argparser.parse_args([])

    params.agent = FLAGS.agent
    params.trajlen = FLAGS.max_step
    params.global_map_size = [1000, 1000, 1]


    collect_data(env, params, FLAGS.filename, FLAGS.num_records)

    # test_ds = get_dataflow([FLAGS.filename])
    # itr = test_ds.as_numpy_iterator()
    # parsed_record = next(itr)
    # data_sample = transform_raw_record(parsed_record)
    # print(data_sample['actions'])


if __name__ == '__main__':
    app.run(main)
