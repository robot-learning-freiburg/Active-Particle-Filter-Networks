#!/usr/bin/env python3
from environments import suite_gibson
import os
from tf_agents.policies import random_py_policy

tf_env = suite_gibson.load(
    config_file=os.path.join('./configs', 'turtlebot_point_nav.yaml'),
    model_id='Rs',
    env_mode='gui',
    device_idx=0,
)
print('observation_spec', tf_env.time_step_spec().observation)
print('action_spec', tf_env.action_spec())

random_policy = random_py_policy.RandomPyPolicy(
        time_step_spec=tf_env.time_step_spec(),
        action_spec=tf_env.action_spec()
)

time_step = tf_env.reset()
for _ in range(100):
    action_step = random_policy.action(time_step)
    time_step = tf_env.step(action_step.action)
tf_env.close()
