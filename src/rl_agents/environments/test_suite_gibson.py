#!/usr/bin/env python3
from environments import suite_gibson
import os

suite_gibson.load(
    config_file=os.path.join('../configs', 'turtlebot_point_nav.yaml'),
    model_id=None,
    env_mode='headless',
    device_idx=0,
)
