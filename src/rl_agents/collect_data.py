#!/usr/bin/env python3

from absl import app
from absl import flags
from absl import logging
import numpy as np
import os
import random
import tensorflow as tf

# import custom tf_agents
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
    name='max_step',
    default=10,
    help='The maimum number of episode steps.'
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
        'turtlebot_point_nav.yaml'
    ),
    help='Config file for the experiment'
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

# define pfNet env parameters
flags.DEFINE_boolean(
    name='use_pfnet',
    default=False,
    help='Whether to use particle filter net'
)

FLAGS = flags.FLAGS


def serialize_tf_record(episode_data):
    """
    Serialize complete episode data as tf record
    """

    actions = episode_data['actions']
    record = {
        'actions': tf.train.Feature(
            float_list=tf.train.FloatList(
                value=actions.flatten()
            )
        ),
        'actions_shape': tf.train.Feature(
            int64_list=tf.train.Int64List(
                value=actions.shape
            )
        ),
    }

    return tf.train.Example(
        features=tf.train.Features(feature=record)
    ).SerializeToString()


def deserialize_tf_record(raw_record):
    """
    De-Serialize tf record containing complete episode data
    """

    tfrecord_format = {
        'actions': tf.io.FixedLenSequenceFeature(
            (),
            dtype=tf.float32,
            allow_missing=True
        ),
        'actions_shape': tf.io.FixedLenSequenceFeature(
            (),
            dtype=tf.int64,
            allow_missing=True
        )
    }

    return tf.io.parse_single_example(raw_record, tfrecord_format)


def get_discrete_action():
    """
    Get manual keyboard action
    :return int: discrete action for moving forward/backward/left/right
    """
    key = input('Enter Key: ')
    # default stay still
    action = 4
    if key == 'w':
        action = 0  # forward
    elif key == 's':
        action = 1  # backward
    elif key == 'd':
        action = 2  # right
    elif key == 'a':
        action = 3  # left
    return action


def get_continuous_action():
    pass


def get_manual_action():
    if FLAGS.is_discrete:
        return get_discrete_action()
    else:
        return get_continuous_action()


def gather_episode_stats(env):
    """
    Step through igibson environment and collect the required stats
    """

    env.reset()

    actions = []
    for _ in range(FLAGS.max_step):
        if FLAGS.agent == 'manual':
            action = get_manual_action()
        else:
            # default random action
            action = env.action_space.sample()
        actions.append(action)

        # take action and get new observation
        obs, reward, done, _ = env.step(action)

    episode_data = {'actions': np.stack(actions)}

    return episode_data


def collect_data(env):
    """
    """

    with tf.io.TFRecordWriter(FLAGS.filename) as writer:
        for i in range(FLAGS.num_records):
            print(f'episode: {i}')

            episode_data = gather_episode_stats(env)
            record = serialize_tf_record(episode_data)
            writer.write(record)

    print(f'Collected successfully in {FLAGS.filename}')


def transform_raw_record(parsed_record):
    trans_record = {'actions': parsed_record['actions'].reshape([] + list(parsed_record['actions_shape']))}

    return trans_record


def get_dataflow(filenames):
    """
    Custom dataset for TF record
    """
    ds = tf.data.TFRecordDataset(filenames)
    ds = ds.map(deserialize_tf_record, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    return ds


def main(_):
    logging.set_verbosity(logging.INFO)
    tf.compat.v1.enable_v2_behavior()
    # tf.debugging.enable_check_numerics()  # error out inf or NaN

    os.environ['CUDA_VISIBLE_DEVICES'] = str(FLAGS.gpu_num)
    os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'

    print('==================================================')
    for k, v in FLAGS.flag_values_dict().items():
        print(k, v)
    print('==================================================')

    # set random seeds
    random.seed(FLAGS.seed)
    np.random.seed(FLAGS.seed)
    tf.random.set_seed(FLAGS.seed)

    if not os.path.isfile(FLAGS.filename):
        env = LocalizeGibsonEnv(
            config_file=FLAGS.config_file,
            scene_id=None,
            mode='gui',
            use_tf_function=True,
            action_timestep=FLAGS.action_timestep,
            physics_timestep=FLAGS.physics_timestep,
            device_idx=FLAGS.gpu_num
        )
        FLAGS.max_step = env.config.get('max_step', 500)
        FLAGS.is_discrete = env.config.get("is_discrete", False),

        collect_data(env)

    test_ds = get_dataflow([FLAGS.filename])
    itr = test_ds.as_numpy_iterator()
    parsed_record = next(itr)
    data_sample = transform_raw_record(parsed_record)
    print(data_sample['actions'])


if __name__ == '__main__':
    app.run(main)
