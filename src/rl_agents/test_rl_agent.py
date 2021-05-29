#!/usr/bin/env python3

from absl import app
from absl import flags
from absl import logging
import numpy as np
import os
import random
import tensorflow as tf

# import custom tf_agents
from tf_agents.agents.ddpg import critic_network
from tf_agents.agents.sac import sac_agent
from environments import suite_gibson
from tf_agents.environments import tf_py_environment
from tf_agents.eval import metric_utils
from tf_agents.metrics import tf_metrics
from tf_agents.networks import actor_distribution_network
from tf_agents.networks import normal_projection_network
from tf_agents.networks.utils import mlp_layers
from tf_agents.policies import greedy_policy
from tf_agents.policies import random_tf_policy
from tf_agents.utils import common

# define testing parameters
flags.DEFINE_string(
    name='root_dir',
    default='./test_output',
    help='Root directory for pretrained agent logs/summaries/checkpoints.'
)
flags.DEFINE_integer(
    name='num_eval_episodes',
    default=10,
    help='The number of episodes to run eval on.'
)
flags.DEFINE_integer(
    name='seed',
    default=100,
    help='Fix the random seed'
)
flags.DEFINE_string(
    name='agent',
    default='random',
    help='Agent Behavior'
)

# define igibson env parameters
flags.DEFINE_boolean(
    name='is_localize_env',
    default=True,
    help='Whether to use navigation/localization env'
)
flags.DEFINE_string(
    name='config_file',
    default=os.path.join(
        os.path.dirname(os.path.realpath(__file__)),
        'configs',
        'turtlebot_point_nav.yaml'
    ),
    help='Config file for the experiment'
)
flags.DEFINE_float(
    name='action_timestep',
    default=1.0 / 10.0,
    help='Action time step for the simulator'
)
flags.DEFINE_float(
    name='physics_timestep',
    default=1.0 / 40.0,
    help='Physics time step for the simulator'
)
flags.DEFINE_integer(
    name='gpu_num',
    default=0,
    help='GPU id for graphics/computation'
)

# define pfNet env parameters
flags.DEFINE_string(
    name='init_particles_distr',
    default='gaussian',
    help='Distribution of initial particles. Possible values: gaussian / uniform.'
)
flags.DEFINE_list(
    name='init_particles_std',
    default=[30, 0.523599],
    help='Standard deviations for generated initial particles for tracking distribution. '
         'Values: translation std (meters), rotation std (radians)'
)
flags.DEFINE_integer(
    name='num_particles',
    default=500,
    help='Number of particles in Particle Filter.'
)
flags.DEFINE_boolean(
    name='resample',
    default=True,
    help='Resample particles in Particle Filter. Possible values: true / false.'
)
flags.DEFINE_float(
    name='alpha_resample_ratio',
    default=0.5,
    help='Trade-off parameter for soft-resampling in PF-net. '
         'Only effective if resample == true. Assumes values 0.0 < alpha <= 1.0. '
         'Alpha equal to 1.0 corresponds to hard-resampling.'
)
flags.DEFINE_list(
    name='transition_std',
    default=[0.0, 0.0],
    help='Standard deviations for transition model. Values: translation std (meters), rotation std (radians)'
)
flags.DEFINE_string(
    name='pfnet_load',
    default=os.path.join(
        os.path.dirname(os.path.realpath(__file__)),
        'pfnetwork/checkpoints',
        'checkpoint_87_5.830/pfnet_checkpoint'
    ),
    help='Load a previously trained pfnet model from a checkpoint file.'
)
flags.DEFINE_boolean(
    name='use_plot',
    default=False,
    help='Enable Plotting of particles on env map. Possible values: true / false.'
)
flags.DEFINE_boolean(
    name='store_plot',
    default=False,
    help='Store the plots sequence as video. Possible values: true / false.'
)

FLAGS = flags.FLAGS


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


def test_agent(arg_params):
    """
    """

    root_dir = os.path.expanduser(arg_params.root_dir)
    train_dir = os.path.join(arg_params.root_dir, 'train')

    conv_1d_layer_params = [(32, 8, 4), (64, 4, 2), (64, 3, 1)]
    conv_2d_layer_params = [(32, (8, 8), 4), (64, (4, 4), 2), (64, (3, 3), 2)]
    encoder_fc_layers = [256]
    actor_fc_layers = [256]
    critic_obs_fc_layers = [256]
    critic_action_fc_layers = [256]
    critic_joint_fc_layers = [256]
    gamma = 0.99
    actor_learning_rate = 3e-4
    critic_learning_rate = 3e-4
    alpha_learning_rate = 3e-4
    target_update_tau = 0.005
    target_update_period = 1
    td_errors_loss_fn = tf.math.squared_difference
    reward_scale_factor = 1.0
    gradient_clipping = None
    train_checkpoint_interval = 10000
    policy_checkpoint_interval = 10000
    rb_checkpoint_interval = 50000
    log_interval = 100
    summary_interval = 1000
    summaries_flush_secs = 10
    debug_summaries = False
    summarize_grads_and_vars = False
    eval_metrics_callback = None
    eval_deterministic = False

    env_load_fn = lambda model_id, mode, use_tf_function, device_idx: suite_gibson.load(
        config_file=arg_params.config_file,
        model_id=model_id,
        env_mode=mode,
        use_tf_function=use_tf_function,
        is_localize_env=arg_params.is_localize_env,
        action_timestep=arg_params.action_timestep,
        physics_timestep=arg_params.physics_timestep,
        device_idx=device_idx,
    )

    # HACK: use same env for train and eval
    tf_py_env = env_load_fn(None, 'headless', True, arg_params.gpu_num)
    tf_env = tf_py_environment.TFPyEnvironment(tf_py_env)
    eval_tf_env = tf_env

    time_step_spec = tf_env.time_step_spec()
    observation_spec = time_step_spec.observation
    action_spec = tf_env.action_spec()
    logging.info('\n Observation specs: %s \n Action specs: %s', observation_spec, action_spec)

    global_step = tf.compat.v1.train.get_or_create_global_step()
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

    actor_net = actor_distribution_network.ActorDistributionNetwork(
        observation_spec,
        action_spec,
        preprocessing_layers=preprocessing_layers,
        preprocessing_combiner=preprocessing_combiner,
        fc_layer_params=actor_fc_layers,
        continuous_projection_net=normal_projection_net,
        # tanh_normal_projection_network.TanhNormalProjectionNetwork,
        kernel_initializer=glorot_uniform_initializer
    )
    critic_net = critic_network.CriticNetwork(
        (observation_spec, action_spec),
        preprocessing_layers=preprocessing_layers,
        preprocessing_combiner=preprocessing_combiner,
        observation_fc_layer_params=critic_obs_fc_layers,
        action_fc_layer_params=critic_action_fc_layers,
        joint_fc_layer_params=critic_joint_fc_layers,
        kernel_initializer=glorot_uniform_initializer
    )

    logging.info('Creating SAC Agent')
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
        train_step_counter=global_step)
    tf_agent.initialize()

    if eval_deterministic:
        eval_policy = greedy_policy.GreedyPolicy(tf_agent.policy)
    else:
        eval_policy = tf_agent.policy

    random_policy = random_tf_policy.RandomTFPolicy(
        time_step_spec=tf_env.time_step_spec(),
        action_spec=tf_env.action_spec()
    )

    eval_metrics = [
        tf_metrics.AverageReturnMetric(buffer_size=arg_params.num_eval_episodes),
        tf_metrics.AverageEpisodeLengthMetric(buffer_size=arg_params.num_eval_episodes)
    ]
    train_metrics = [
        tf_metrics.NumberOfEpisodes(),
        tf_metrics.EnvironmentSteps(),
        tf_metrics.AverageReturnMetric(
            buffer_size=100, batch_size=tf_env.batch_size),
        tf_metrics.AverageEpisodeLengthMetric(
            buffer_size=100, batch_size=tf_env.batch_size),
    ]

    train_checkpointer = common.Checkpointer(
        ckpt_dir=train_dir,
        agent=tf_agent,
        global_step=global_step,
        metrics=metric_utils.MetricsGroup(train_metrics, 'train_metrics'))
    policy_checkpointer = common.Checkpointer(
        ckpt_dir=os.path.join(train_dir, 'policy'),
        policy=eval_policy,
        global_step=global_step)

    train_checkpointer.initialize_or_restore()

    if arg_params.agent == 'sac_agent':
        policy = eval_policy
        log_dir = os.path.join(arg_params.root_dir, 'sac_agent')
    else:
        policy = random_policy
        log_dir = os.path.join(arg_params.root_dir, 'rnd_agent')

    test_summary_writer = tf.summary.create_file_writer(log_dir)
    with test_summary_writer.as_default():
        step = 0
        time_step = tf_env.reset()
        while not time_step.is_last():
            tf_env.render('human')
            action_step = policy.action(time_step)
            time_step = tf_env.step(action_step.action)
            tf.summary.scalar('mse_reward', time_step.reward[0], step=step)
            step += 1
        tf_env.close()

    logging.info('Test Done')


def main(_):
    logging.set_verbosity(logging.INFO)
    tf.compat.v1.enable_v2_behavior()
    # tf.debugging.enable_check_numerics()  # error out inf or NaN

    os.environ['CUDA_VISIBLE_DEVICES'] = str(FLAGS.gpu_num)
    os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'

    print('==================================================')
    for k, v in FLAGS.flag_values_dict().items():
        print(k, v)
    print('==================================================')

    # set random seeds
    random.seed(FLAGS.seed)
    np.random.seed(FLAGS.seed)
    tf.random.set_seed(FLAGS.seed)

    test_agent(FLAGS)


if __name__ == '__main__':
    app.run(main)
