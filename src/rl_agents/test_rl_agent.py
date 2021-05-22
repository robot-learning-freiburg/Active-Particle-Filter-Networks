#!/usr/bin/env python3

from absl import app
from absl import flags
from absl import logging
import numpy as np
import os
import random
import tensorflow as tf

# import custom tf_agents
from environments import suite_gibson
from tf_agents.policies import random_py_policy
from tf_agents.train import actor


flags.DEFINE_string(
    name='root_dir',
    default='./test_output',
    help='Root directory for writing logs/summaries/checkpoints.'
)

# define training parameters
flags.DEFINE_integer(
    name='initial_collect_steps',
    default=100,
    help='Number of steps to collect at the beginning of training using random policy'
)
flags.DEFINE_boolean(
    name='use_tf_function',
    default=True,
    help='Whether to use graph/eager mode execution'
)
flags.DEFINE_integer(
    name='seed',
    default=100,
    help='Fix the random seed'
)

# define igibson env parameters
flags.DEFINE_boolean(
    name='is_localize_env',
    default=True,
    help='Whether to use navigation/localization env'
)
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

FLAGS = flags.FLAGS

def test_agent(arg_params):
    """
    """

    tf.profiler.experimental.start(logdir='./log_dir')

    # create sac agent
    logging.info('Creating SAC Agent')

    # create igibson env
    py_env = suite_gibson.load(
        config_file=arg_params.config_file,
        model_id=None,
        env_mode='headless',
        use_tf_function=arg_params.use_tf_function,
        is_localize_env=arg_params.is_localize_env,
        action_timestep=arg_params.action_timestep,
        physics_timestep=arg_params.physics_timestep,
        device_idx=arg_params.gpu_num,
    )

    # create random policy
    random_policy = random_py_policy.RandomPyPolicy(
        time_step_spec=py_env.time_step_spec(),
        action_spec=py_env.action_spec()
    )

    # use random policy to collect initial experiences to seed the replay buffer
    initial_collect_actor = actor.Actor(
        env=py_env,
        policy=random_policy,
        train_step=0,
        steps_per_run=arg_params.initial_collect_steps,
        observers=[],
        metrics=None,
    )
    logging.info('Initializing replay buffer by collecting experience for %d steps '
                 'with a random policy.', arg_params.initial_collect_steps)
    initial_collect_actor.run()

    # time_step = py_env.reset()
    # while not time_step.is_last():
    #     py_env.render('human')
    #     action_step = random_policy.action(time_step)
    #     time_step = py_env.step(action_step.action)
    #     print(time_step.reward)
    # py_env.close()

    logging.info('Test Done')
    tf.profiler.experimental.stop()


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

    test_agent(FLAGS)

if __name__ == '__main__':
    app.run(main)
