#!/usr/bin/env python3

from absl import logging
import argparse
import numpy as np
import os
import random
import tensorflow as tf

# import custom tf_agents
from custom_agents.sac_rl_agent import SACAgent
from custom_agents.replay_buffer import ReverbReplayBuffer
from environments import suite_gibson
from tf_agents.train import actor
from tf_agents.train import learner
from tf_agents.train import triggers
from tf_agents.train.utils import strategy_utils
from tf_agents.utils import common


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
    arg_parser.add_argument(
        '--seed',
        type=int,
        default=100,
        help='Fix the random seed'
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

    # set random seeds
    random.seed(params.seed)
    np.random.seed(params.seed)
    tf.random.set_seed(params.seed)

    os.environ['CUDA_VISIBLE_DEVICES'] = str(params.gpu_num)
    os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'

    logging.set_verbosity(logging.INFO)
    tf.compat.v1.enable_v2_behavior()
    tf.debugging.enable_check_numerics()  # error out inf or NaN

    print(params)
    return params


def get_eval_metrics(eval_actor):
    """
    Run the actor to generate metrics

    :param eval_actor:
        evaluation actor
    :return: dict
        dictionary of eval metrics
    """

    eval_actor.run()
    results = {}
    for metric in eval_actor.metrics:
        results[metric.name] = metric.result()

    return results


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
    tf.profiler.experimental.start(logdir=arg_params.root_dir)

    with strategy.scope():
        # create or get global step tensor
        global_step = tf.compat.v1.train.get_or_create_global_step()

    # create sac agent
    sac_agent = SACAgent(
        root_dir=arg_params.root_dir,
        env_load_fn=lambda model_id, mode, use_tf_function, device_idx: suite_gibson.load(
            config_file=arg_params.config_file,
            model_id=model_id,
            env_mode=mode,
            use_tf_function=use_tf_function,
            is_localize_env=arg_params.is_localize_env,
            action_timestep=arg_params.action_timestep,
            physics_timestep=arg_params.physics_timestep,
            device_idx=device_idx,
        ),
        train_step_counter=global_step,
        strategy=strategy,
        gpu=arg_params.gpu_num,
        use_tf_function=arg_params.use_tf_function
    )
    tf_agent = sac_agent.tf_agent
    collect_env = sac_agent.train_py_env
    eval_env = sac_agent.eval_py_env
    random_policy = sac_agent.random_policy
    collect_policy = sac_agent.collect_policy
    eval_policy = sac_agent.eval_policy

    # instantiate reverb replay buffer
    rb = ReverbReplayBuffer(
        table_name='uniform_table',
        replay_buffer_capacity=arg_params.replay_buffer_capacity
    )

    # generate tf dataset from replay buffer
    dataset = rb.get_dataset(
        collect_data_spec=tf_agent.collect_data_spec,
        sequence_length=arg_params.sequence_length,
        batch_size=arg_params.batch_size,
    )
    experience_dataset_fn = lambda: dataset

    # instantiate replay buffer traj observer
    rb_traj_observer = rb.get_rb_traj_observer(
        sequence_length=arg_params.sequence_length,
        stride_length=arg_params.stride_length,
    )

    # Metrics
    train_metrics = actor.collect_metrics(
        buffer_size=10,
    )
    eval_metrics = actor.eval_metrics(
        buffer_size=arg_params.num_eval_episodes,
    )

    # use random policy to collect initial experiences to seed the replay buffer
    initial_collect_actor = actor.Actor(
        env=collect_env,
        policy=random_policy,
        train_step=global_step,
        steps_per_run=arg_params.initial_collect_steps,
        observers=[rb_traj_observer],
        metrics=train_metrics,
    )
    logging.info('Initializing replay buffer by collecting experience for %d steps '
                 'with a random policy.', arg_params.initial_collect_steps)
    initial_collect_actor.run()

    # use collect policy to gather more experiences during training
    collect_actor = actor.Actor(
        env=collect_env,
        policy=collect_policy,
        train_step=global_step,
        steps_per_run=1,
        observers=[rb_traj_observer],
        metrics=train_metrics,
        summary_dir=train_dir,
        name='train',
    )

    # use eval policy to evaluate during training
    eval_actor = actor.Actor(
        env=eval_env,
        policy=eval_policy,
        train_step=global_step,
        episodes_per_run=arg_params.num_eval_episodes,
        observers=None,
        metrics=eval_metrics,
        summary_dir=eval_dir,
        summary_interval=arg_params.eval_interval,
        name='eval',
    )

    # policy checkpoint trigger
    policy_checkpointer = common.Checkpointer(
        ckpt_dir=policy_dir,
        policy=tf_agent.policy,
        global_step=global_step,
    )

    # HACK: there is problem with triggers.PolicySavedModelTrigger
    # instantiate agent learner with triggers
    learning_triggers = [
        triggers.StepPerSecondLogTrigger(
            train_step=global_step,
            interval=1000
        ),
    ]
    agent_learner = learner.Learner(
        root_dir=arg_params.root_dir,
        train_step=global_step,
        agent=tf_agent,
        experience_dataset_fn=experience_dataset_fn,
        # triggers=learning_triggers,
        strategy=strategy,
    )

    logging.info('====> Starting training')

    # reset the train step
    tf_agent.train_step_counter.assign(0)

    returns = []
    for _ in range(arg_params.num_iterations):
        # training
        collect_actor.run()
        loss_info = agent_learner.run(
            iterations=1
        )

        step = agent_learner.train_step_numpy

        # evaluation
        if step % arg_params.eval_interval == 0:
            metrics = get_eval_metrics(eval_actor)
            returns.append(metrics["AverageReturn"])

            # eval_actor.log_metrics()
            eval_results = ', '.join('{} = {:.6f}'.format(name, result) for name, result in metrics.items())
            logging.info('step = %d: %s', step, eval_results)

        # logging
        if step % arg_params.log_interval == 0:
            # collect_actor.log_metrics()
            logging.info('step = %d: loss = %f', step, loss_info.loss.numpy())

        # save policy
        if step % arg_params.policy_save_interval == 0:
            policy_checkpointer.save(global_step=step)

    # close replay buffer
    rb.close()

    tf.profiler.experimental.stop()


if __name__ == '__main__':
    parsed_params = parse_args()
    train_eval(parsed_params)
