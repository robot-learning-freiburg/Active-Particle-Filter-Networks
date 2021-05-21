#!/usr/bin/env python3

import numpy as np
import os
import random
import tensorflow as tf

# import custom tf_agents
from environments import suite_gibson
from tf_agents.policies import random_py_policy
from tf_agents.train import actor


def main():
    config_file = os.path.join(
        os.path.dirname(os.path.realpath(__file__)),
        'configs',
        'turtlebot_point_nav.yaml'
    )
    is_localize_env = True
    use_tf_function = True
    action_timestep = 1.0 / 10.0
    physics_timestep = 1.0 / 40.0
    device_idx = 0
    initial_collect_steps = 100

    # create igibson env
    py_env = suite_gibson.load(
        config_file=config_file,
        model_id=None,
        env_mode='headless',
        use_tf_function=use_tf_function,
        is_localize_env=is_localize_env,
        action_timestep=action_timestep,
        physics_timestep=physics_timestep,
        device_idx=device_idx,
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
        steps_per_run=initial_collect_steps,
        observers=[],
        metrics=None,
    )
    initial_collect_actor.run()

    # time_step = py_env.reset()
    # while not time_step.is_last():
    #     py_env.render('human')
    #     action_step = random_policy.action(time_step)
    #     time_step = py_env.step(action_step.action)
    #     print(time_step.reward)
    # py_env.close()

    print('done')


if __name__ == '__main__':
    # set gpu
    gpu_num = 0
    os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_num)
    os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'

    # set random seeds
    seed = 100
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)

    logdir = './test_output'
    tf.profiler.experimental.start(logdir=logdir)
    main()
    tf.profiler.experimental.stop()
