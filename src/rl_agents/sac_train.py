#!/usr/bin/env python3

# reference: https://www.tensorflow.org/agents/tutorials/7_SAC_minitaur_tutorial

from absl import logging
from datetime import datetime
import matplotlib.pyplot as plt
import numpy as np
import os
import random
import reverb
import tempfile
import tensorflow as tf
from tf_agents.agents.ddpg import critic_network
from tf_agents.agents.sac import sac_agent
from tf_agents.agents.sac import tanh_normal_projection_network
# from tf_agents.environments import suite_gibson
from environments import suite_gibson
from tf_agents.metrics import py_metrics
from tf_agents.networks import actor_distribution_network
from tf_agents.networks import normal_projection_network
from tf_agents.networks.utils import mlp_layers
from tf_agents.policies import greedy_policy
from tf_agents.policies import policy_saver
from tf_agents.policies import py_tf_eager_policy
from tf_agents.policies import random_py_policy
from tf_agents.replay_buffers import reverb_replay_buffer
from tf_agents.replay_buffers import reverb_utils
from tf_agents.train import actor
from tf_agents.train import learner
from tf_agents.train import triggers
from tf_agents.train.utils import spec_utils
from tf_agents.train.utils import strategy_utils
from tf_agents.train.utils import train_utils

logging.set_verbosity(logging.INFO)

# error out inf or NaN
tf.debugging.enable_check_numerics()

def normal_projection_net(action_spec,
                          init_action_stddev=0.35,
                          init_means_output_factor=0.1):
    del init_action_stddev
    return normal_projection_network.NormalProjectionNetwork(
        action_spec,
        mean_transform=None,
        state_dependent_std=True,
        init_means_output_factor=init_means_output_factor,
        std_transform=sac_agent.std_clip_transform,
        scale_distribution=True)

#### Hyperparameters ####
batch_size = 64
num_iterations = 150000
initial_collect_steps = 500
collect_steps_per_iteration = 1
replay_buffer_capacity = 150000

log_interval = 100
eval_interval = 10000
num_eval_episodes = 10
policy_save_interval = 10000

critic_learning_rate = 3e-4
actor_learning_rate = 3e-4
alpha_learning_rate = 3e-4
target_update_tau = 0.005
target_update_period = 1
td_errors_loss_fn=tf.math.squared_difference
gamma = 0.99
reward_scale_factor = 1.0
gradient_clipping=None
debug_summaries=False
summarize_grads_and_vars=False

conv_1d_layer_params = [(32, 8, 4), (64, 4, 2), (64, 3, 1)]
conv_2d_layer_params = [(32, (8, 8), 4), (64, (4, 4), 2), (64, (3, 3), 2)]
encoder_fc_layers = [256]
actor_fc_layers = [256]
critic_obs_fc_layers = [256]
critic_action_fc_layers = [256]
critic_joint_fc_layers = [256]
device_idx = 0
seed = 100

#### Setup ####

# set random seeds
random.seed(seed)
np.random.seed(seed)
tf.random.set_seed(seed)

# rootdir = tempfile.gettempdir()
rootdir = os.path.join('./runs', datetime.now().strftime('%Y-%m-%d_%H-%M-%S'))

gpus = tf.config.experimental.list_physical_devices('GPU')
assert device_idx < len(gpus)
if gpus:
    # restrict TF to only use the first GPU
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        tf.config.experimental.set_visible_devices(gpus[device_idx], 'GPU')
        logical_gpus = tf.config.experimental.list_logical_devices('GPU')
    except RuntimeError as e:
        # visible devices must be set before GPUs have been initialized
        print(e)

#### Environment ####

config_file = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'configs', 'turtlebot_point_nav.yaml')
model_id = None
mode = 'headless'
is_localize_env = False
action_timestep = 1.0 / 10.0
physics_timestep = 1.0 / 40.0

collect_env = suite_gibson.load(
    config_file=config_file,
    model_id=model_id,
    env_mode=mode,
    is_localize_env=is_localize_env,
    action_timestep=action_timestep,
    physics_timestep=physics_timestep,
    device_idx=device_idx,
)
eval_env = suite_gibson.load(
    config_file=config_file,
    model_id=model_id,
    env_mode=mode,
    is_localize_env=is_localize_env,
    action_timestep=action_timestep,
    physics_timestep=physics_timestep,
    device_idx=device_idx,
)
#eval_env = collect_env

#### Distribution Strategy ####

use_gpu = True
strategy = strategy_utils.get_strategy(tpu=False, use_gpu=use_gpu)

#### Agent ####
observation_spec, action_spec, time_step_spec = (spec_utils.get_tensor_specs(collect_env))
logging.info('Observation Spec = {0}'.format(observation_spec))
logging.info('Action Spec = {0}'.format(action_spec))

with strategy.scope():

    glorot_uniform_initializer = tf.compat.v1.keras.initializers.glorot_uniform()
    preprocessing_layers = {}
    if 'rgb_obs' in observation_spec:
        preprocessing_layers['rgb_obs'] = tf.keras.Sequential(mlp_layers(
            conv_1d_layer_params=None,
            conv_2d_layer_params=conv_2d_layer_params,
            fc_layer_params=encoder_fc_layers,
            kernel_initializer=glorot_uniform_initializer,
        ))

    if 'depth' in observation_spec:
        preprocessing_layers['depth'] = tf.keras.Sequential(mlp_layers(
            conv_1d_layer_params=None,
            conv_2d_layer_params=conv_2d_layer_params,
            fc_layer_params=encoder_fc_layers,
            kernel_initializer=glorot_uniform_initializer,
        ))

    if 'scan' in observation_spec:
        preprocessing_layers['scan'] = tf.keras.Sequential(mlp_layers(
            conv_1d_layer_params=conv_1d_layer_params,
            conv_2d_layer_params=None,
            fc_layer_params=encoder_fc_layers,
            kernel_initializer=glorot_uniform_initializer,
        ))

    if 'task_obs' in observation_spec:
        preprocessing_layers['task_obs'] = tf.keras.Sequential(mlp_layers(
            conv_1d_layer_params=None,
            conv_2d_layer_params=None,
            fc_layer_params=encoder_fc_layers,
            kernel_initializer=glorot_uniform_initializer,
        ))

    if len(preprocessing_layers) <= 1:
        preprocessing_combiner = None
    else:
        preprocessing_combiner = tf.keras.layers.Concatenate(axis=-1)

with strategy.scope():
    critic_net = critic_network.CriticNetwork(
        (observation_spec, action_spec),
        preprocessing_layers=preprocessing_layers,
        preprocessing_combiner=preprocessing_combiner,
        observation_fc_layer_params=critic_obs_fc_layers,
        action_fc_layer_params=critic_action_fc_layers,
        joint_fc_layer_params=critic_joint_fc_layers,
        kernel_initializer=glorot_uniform_initializer
    )

with strategy.scope():
    actor_net = actor_distribution_network.ActorDistributionNetwork(
        observation_spec,
        action_spec,
        preprocessing_layers=preprocessing_layers,
        preprocessing_combiner=preprocessing_combiner,
        fc_layer_params=actor_fc_layers,
        continuous_projection_net=normal_projection_net,
        # continuous_projection_net=(tanh_normal_projection_network.TanhNormalProjectionNetwork),
        kernel_initializer=glorot_uniform_initializer
    )

with strategy.scope():
    train_step = train_utils.create_train_step()

    tf_agent = sac_agent.SacAgent(
        time_step_spec,
        action_spec,
        actor_network=actor_net,
        critic_network=critic_net,
        actor_optimizer=tf.compat.v1.train.AdamOptimizer(
            learning_rate=actor_learning_rate),
        critic_optimizer=tf.compat.v1.train.AdamOptimizer(
            learning_rate=critic_learning_rate),
        alpha_optimizer=tf.compat.v1.train.AdamOptimizer(
            learning_rate=alpha_learning_rate),
        target_update_tau=target_update_tau,
        target_update_period=target_update_period,
        td_errors_loss_fn=td_errors_loss_fn,
        gamma=gamma,
        reward_scale_factor=reward_scale_factor,
        gradient_clipping=gradient_clipping,
        debug_summaries=debug_summaries,
        summarize_grads_and_vars=summarize_grads_and_vars,
        train_step_counter=train_step)
    tf_agent.initialize()

#### Replay Buffer ####

table_name = 'uniform_table'
table = reverb.Table(
    table_name,
    max_size=replay_buffer_capacity,
    sampler=reverb.selectors.Uniform(),
    remover=reverb.selectors.Fifo(),
    rate_limiter=reverb.rate_limiters.MinSize(1))

reverb_server = reverb.Server([table])

reverb_replay = reverb_replay_buffer.ReverbReplayBuffer(
    tf_agent.collect_data_spec,
    sequence_length=2,
    table_name=table_name,
    local_server=reverb_server)

dataset = reverb_replay.as_dataset(
      sample_batch_size=batch_size, num_steps=2).prefetch(50)
experience_dataset_fn = lambda: dataset

#### Policies ####

tf_eval_policy = tf_agent.policy
eval_policy = py_tf_eager_policy.PyTFEagerPolicy(
  tf_eval_policy, use_tf_function=True)

tf_collect_policy = tf_agent.collect_policy
collect_policy = py_tf_eager_policy.PyTFEagerPolicy(
  tf_collect_policy, use_tf_function=True)

random_policy = random_py_policy.RandomPyPolicy(
  collect_env.time_step_spec(), collect_env.action_spec())

#### Actors ####

rb_observer = reverb_utils.ReverbAddTrajectoryObserver(
  reverb_replay.py_client,
  table_name,
  sequence_length=2,
  stride_length=1)

initial_collect_actor = actor.Actor(
  collect_env,
  random_policy,
  train_step,
  steps_per_run=initial_collect_steps,
  observers=[rb_observer])
initial_collect_actor.run()

env_step_metric = py_metrics.EnvironmentSteps()
collect_actor = actor.Actor(
  collect_env,
  collect_policy,
  train_step,
  steps_per_run=1,
  metrics=actor.collect_metrics(10),
  summary_dir=os.path.join(rootdir, learner.TRAIN_DIR),
  observers=[rb_observer, env_step_metric])

eval_actor = actor.Actor(
  eval_env,
  eval_policy,
  train_step,
  episodes_per_run=num_eval_episodes,
  metrics=actor.eval_metrics(num_eval_episodes),
  summary_dir=os.path.join(rootdir, 'eval'),
)

#### Learners ####

saved_model_dir = os.path.join(rootdir, learner.POLICY_SAVED_MODEL_DIR)

# Triggers to save the agent's policy checkpoints.
learning_triggers = [
    # triggers.PolicySavedModelTrigger(
    #     saved_model_dir,
    #     tf_agent,
    #     train_step,
    #     interval=policy_save_interval),
    triggers.StepPerSecondLogTrigger(train_step, interval=1000),
]

agent_learner = learner.Learner(
  rootdir,
  train_step,
  tf_agent,
  experience_dataset_fn,
  triggers=learning_triggers)

#### Metrics and Evaluation ####

def get_eval_metrics():
  eval_actor.run()
  results = {}
  for metric in eval_actor.metrics:
    results[metric.name] = metric.result()
  return results

def log_eval_metrics(step, metrics):
  eval_results = (', ').join(
      '{} = {:.6f}'.format(name, result) for name, result in metrics.items())
  logging.info('step = {0}: {1}'.format(step, eval_results))

#### Training the agent ####

tf_policy_saver = policy_saver.PolicySaver(tf_agent.policy)

# Reset the train step
tf_agent.train_step_counter.assign(0)

# Evaluate the agent's policy once before training.
avg_return = get_eval_metrics()["AverageReturn"]
returns = [avg_return]

for _ in range(num_iterations):
  # Training.
  collect_actor.run()
  loss_info = agent_learner.run(iterations=1)

  # Evaluating.
  step = agent_learner.train_step_numpy

  if eval_interval and step % eval_interval == 0:
    metrics = get_eval_metrics()
    log_eval_metrics(step, metrics)
    returns.append(metrics["AverageReturn"])

  if log_interval and step % log_interval == 0:
    logging.info('step = {0}: loss = {1}'.format(step, loss_info.loss.numpy()))

rb_observer.close()
reverb_server.stop()

policy_dir = os.path.join(rootdir, 'output')
tf_policy_saver.save(policy_dir)

steps = range(0, num_iterations + 1, eval_interval)
plt.plot(steps, returns)
plt.ylabel('Average Return')
plt.xlabel('Step')
plt.ylim()
plt.savefig(os.path.join(policy_dir,'/average_return.png'))

logging.info('training finished')
