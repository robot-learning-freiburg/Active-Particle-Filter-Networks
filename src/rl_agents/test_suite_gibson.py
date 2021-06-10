#!/usr/bin/env python3
import os
import numpy as np

from environments import suite_gibson
from tf_agents.policies import random_py_policy

tf_env = suite_gibson.load(
    config_file=os.path.join('./configs', 'turtlebot_random_nav.yaml'),
    model_id='Adrian',
    env_mode='gui',
    use_tf_function=True,
    init_pfnet=False,
    is_localize_env=False,
    device_idx=0,
)
print('observation_spec', tf_env.time_step_spec().observation)
print('action_spec', tf_env.action_spec())

random_policy = random_py_policy.RandomPyPolicy(
        time_step_spec=tf_env.time_step_spec(),
        action_spec=tf_env.action_spec()
)

for _ in range(10):
    time_step = tf_env.reset()
    while not time_step.is_last():
        action_step = random_policy.action(time_step)
        action = np.random.choice(5, 1, p=[0.7, 0.0, 0.15, 0.15, 0.0])
        time_step = tf_env.step(action)
tf_env.close()
