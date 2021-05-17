#!/usr/bin/env python3

import os

# import custom tf_agents
from environments import suite_gibson
from tf_agents.policies import random_py_policy


def main():
    config_file = os.path.join(
        os.path.dirname(os.path.realpath(__file__)),
        'configs',
        'turtlebot_point_nav.yaml'
    )
    is_localize_env = True
    action_timestep = 1.0 / 10.0
    physics_timestep = 1.0 / 40.0
    device_idx = 0

    # create igibson env
    py_env = suite_gibson.load(
        config_file=config_file,
        model_id=None,
        env_mode='headless',
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

    time_step = py_env.reset()
    while not time_step.is_last():
        py_env.render('human')
        action_step = random_policy.action(time_step)
        time_step = py_env.step(action_step.action)
        print(time_step.reward)
    py_env.close()

    print('done')


if __name__ == '__main__':
    main()
