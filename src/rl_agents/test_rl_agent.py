#!/usr/bin/env python3

import os

from environments import suite_gibson

config_file = os.path.join(
    os.path.dirname(os.path.realpath(__file__)),
    'configs',
    'turtlebot_point_nav.yaml'
)
is_localize_env=True
action_timestep=1.0 / 10.0
physics_timestep=1.0 / 40.0
device_idx=0

# create igibson env
env = suite_gibson.load(
    config_file=config_file,
    model_id=None,
    env_mode='headless',
    is_localize_env=is_localize_env,
    action_timestep=action_timestep,
    physics_timestep=physics_timestep,
    device_idx=device_idx,
)
env.reset()