#!/usr/bin/env python3

from absl import logging
import argparse
import os
import tensorflow as tf

# import custom tf_agents
from tf_agents.train import learner
from tf_agents.train.utils import strategy_utils


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
        '--replay_buffer_capacity',
        type=int,
        default=1000,
        help='Replay buffer capacity'
    )
    arg_parser.add_argument(
        '--sequence_length',
        type=int,
        default=2,
        help='Consecutive sequence length'
    )
    arg_parser.add_argument(
        '--stride_length',
        type=int,
        default=1,
        help='Sliding window stride'
    )
    arg_parser.add_argument(
        '--batch_size',
        type=int,
        default=8,
        help='Batch size for each training step'
    )
    arg_parser.add_argument(
        '--initial_collect_steps',
        type=int,
        default=100,
        help='Number of steps to collect at the beginning of training using random policy'
    )
    arg_parser.add_argument(
        '--num_eval_episodes',
        type=int,
        default=5,
        help='Number of episodes to run evaluation'
    )
    arg_parser.add_argument(
        '--num_iterations',
        type=int,
        default=3000,
        help='Total number train/eval iterations to perform'
    )
    arg_parser.add_argument(
        '--eval_interval',
        type=int,
        default=500,
        help='Run evaluation every eval_interval train steps'
    )
    arg_parser.add_argument(
        '--log_interval',
        type=int,
        default=100,
        help='Log loss every log_interval train steps'
    )
    arg_parser.add_argument(
        '--policy_save_interval',
        type=int,
        default=500,
        help='Save policies every policy_save_interval train steps'
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
    arg_parser.add_argument(
        '--gpu_num',
        type=int,
        default=0,
        help='GPU id for graphics/computation'
    )

    # parse parameters
    params = arg_parser.parse_args()

    # post-processing
    params.root_dir = './test_output'
    params.is_localize_env = False
    params.summary_interval = 1000
    params.use_tf_function = True

    os.environ['CUDA_VISIBLE_DEVICES'] = str(params.gpu_num)
    os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'

    logging.set_verbosity(logging.INFO)
    tf.compat.v1.enable_v2_behavior()
    tf.debugging.enable_check_numerics()  # error out inf or NaN

    return params


def train_eval(arg_params):
    """
    A simple train and eval for PPO agent

    :param arg_params:
        parsed command-line arguments
    :return:
    """

    """
    initialize distribution strategy
        use_gpu=False means use tf.distribute.get_strategy() which uses CPU
        use_gpu=True mean use tf.distribute.MirroredStrategy() which uses all GPUs that are visible
    """
    strategy = strategy_utils.get_strategy(tpu=False, use_gpu=True)

    train_dir = os.path.join(
        arg_params.root_dir,
        learner.TRAIN_DIR
    )
    eval_dir = os.path.join(
        arg_params.root_dir,
        'eval'
    )
    policy_dir = os.path.join(
        arg_params.root_dir,
        'policy'
    )

    with strategy.scope():
        # create or get global step tensor
        global_step = tf.compat.v1.train.get_or_create_global_step()

    # create ppo agent


if __name__ == '__main__':
    parsed_params = parse_args()
    train_eval(parsed_params)
