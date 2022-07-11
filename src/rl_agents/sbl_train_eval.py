#!/usr/bin/env python3


from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from multiprocessing.sharedctypes import Value

import os

from absl import flags
from absl import logging
from pathlib import Path
import tensorflow as tf
import numpy as np
import wandb
from stable_baselines3 import SAC
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import CallbackList

from pfnetwork.arguments import parse_common_args
from pfnetwork.train import WANDB_PROJECT, init_pfnet_model
from custom_agents.stable_baselines_utils import create_env, CustomCombinedExtractor3, MyWandbCallback, get_run_name, \
    get_logdir, MetricsCallback
from supervised_data import get_scene_ids


def make_sbl_env(rank, seed, params):
    def _init():
        pfnet_model = init_pfnet_model(params, is_igibson=True)
        env = create_env(params, pfnet_model=pfnet_model)

        env = Monitor(env)

        env.seed(seed + rank)

        return env

    set_random_seed(seed)
    return _init


def main(params, test_scenes=None):
    tf.compat.v1.enable_v2_behavior()
    logging.set_verbosity(logging.INFO)

    if params.rl_architecture == 1:
        conv_2d_layer_params = [(32, (8, 8), 4), (64, (4, 4), 2), (64, (3, 3), 2)]
        encoder_fc_layers = [512, 512]
        actor_fc_layers = [512, 512]
    elif params.rl_architecture == 2:
        conv_2d_layer_params = [(32, (3, 3), 2), (64, (3, 3), 2), (64, (3, 3), 2), (64, (3, 3), 2)]
        encoder_fc_layers = [512]
        actor_fc_layers = [512, 512]
    elif params.rl_architecture == 3:
        conv_2d_layer_params = [(32, (3, 3), 2), (64, (3, 3), 2), (64, (3, 3), 1), (64, (2, 2), 1)]
        encoder_fc_layers = [1024]
        actor_fc_layers = [512, 512]
    else:
        raise Value(params.rl_architecture)

    if params.num_parallel_environments > 1:
        env = SubprocVecEnv(
            [make_sbl_env(rank=i, seed=params.seed, params=params) for i in range(params.num_parallel_environments)])
    else:
        env = make_sbl_env(rank=0, seed=params.seed, params=params)()

    eval_env = None

    features_extractor_kwargs = dict(conv_2d_layer_params=conv_2d_layer_params,
                                     encoder_fc_layers=encoder_fc_layers)
    policy_kwargs = dict(features_extractor_class=CustomCombinedExtractor3,
                         features_extractor_kwargs=features_extractor_kwargs,
                         net_arch=actor_fc_layers)

    model = SAC("MultiInputPolicy",
                env,
                verbose=1,
                policy_kwargs=policy_kwargs,
                buffer_size=params.replay_buffer_capacity,
                gamma=params.gamma,
                learning_rate=params.actor_learning_rate,
                batch_size=params.rl_batch_size,
                seed=params.seed,
                learning_starts=params.initial_collect_steps,
                train_freq=params.collect_steps_per_iteration,
                ent_coef=params.ent_coef,
                tensorboard_log=os.path.join(params.root_dir, 'train'))
    if params.num_iterations:
        cb = CallbackList([MetricsCallback(),
                           MyWandbCallback(model_save_path=Path(params.root_dir) / 'train' / 'ckpts',
                                           model_save_freq=params.eval_interval)])
        model.learn(total_timesteps=params.num_iterations,
                    log_interval=4,
                    eval_freq=params.eval_interval,
                    n_eval_episodes=params.num_eval_episodes,
                    callback=cb,
                    eval_env=eval_env)
        model.save("sac_rl_agent")


if __name__ == '__main__':
    params = parse_common_args('igibson', add_rl_args=True)
    params.agent = 'rl'
    # run_name = Path(params.root_dir).name

    if params.scene_id == "all":
        train_scenes, test_scenes = get_scene_ids(params.global_map_size)
        params.scene_id = train_scenes
    else:
        assert False, "Sure you want to train on a single scene?"

    run_name = get_run_name(params)
    params.root_dir = str(get_logdir(run_name))

    run = wandb.init(config=params, name=run_name, project=WANDB_PROJECT, sync_tensorboard=True,
                     mode='disabled' if params.debug else 'online')

    main(params, test_scenes=test_scenes)
