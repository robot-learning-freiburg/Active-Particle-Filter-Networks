#!/usr/bin/env python3
from environments import suite_gibson
import os

tf_env = suite_gibson.load(
    config_file=os.path.join('./configs', 'turtlebot_point_nav.yaml'),
    model_id=None,
    env_mode='headless',
    device_idx=0,
)
print('observation_spec', tf_env.time_step_spec().observation)
print('action_spec', tf_env.action_spec())
