#!/usr/bin/env python3

from absl import logging

import os
import tensorflow as tf

# import custom tf_agents
from tf_agents.agents.ddpg import critic_network
from tf_agents.agents.sac import sac_agent
from tf_agents.agents.sac import tanh_normal_projection_network
from tf_agents.environments import py_environment
from tf_agents.networks import actor_distribution_network
from tf_agents.networks import normal_projection_network
from tf_agents.policies import py_tf_eager_policy
from tf_agents.policies import random_py_policy
from tf_agents.networks.utils import mlp_layers
from tf_agents.train.utils import spec_utils


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


class SACAgent(object):
    """
    Custom SAC Agent implementation for iGibsonEnv
    """

    def __init__(self,
                 root_dir,
                 env_load_fn,
                 train_step_counter,
                 strategy,
                 gpu=0,
                 use_tf_function=True,
                 # train
                 actor_learning_rate=3e-4,
                 critic_learning_rate=3e-4,
                 alpha_learning_rate=3e-4,
                 target_update_tau=0.005,
                 target_update_period=1,
                 td_errors_loss_fn=tf.math.squared_difference,
                 gamma=0.99,
                 reward_scale_factor=1.0,
                 gradient_clipping=None,
                 debug_summaries=False,
                 summarize_grads_and_vars=False,
                 ):
        """
        Initialize

        :param root_dir: str
            root directory to store the results/logs
        :param env_load_fn: lambda function
            function to create igibson env
        :param train_step_counter: tensor
            counter to track training
        :param strategy: tf.distribute.Strategy
            distribution strategy
        :param gpu: int
            gpu device number
        :param use_tf_function: bool
            whether or not wrap function into tf graph
        :param actor_learning_rate: float
            learning rate of optimizer used for actor
        :param critic_learning_rate: float
            learning rate of optimizer used for critic
        :param alpha_learning_rate: float
            learning rate of optimizer used for agent
        :param target_update_tau: float
            factor for soft update
        :param target_update_period: int
            period for soft update
        :param td_errors_loss_fn:
            function to compute element wise loss
        :param gamma:
            reward discount factor
        :param reward_scale_factor:
            scale factor for reward
        :param gradient_clipping:
            to clip gradients
        :param debug_summaries: bool
            whether or not to gather debug summaries
        :param summarize_grads_and_vars: bool
            whether or not to write gradient summaries
        """

        self.root_dir = os.path.expanduser(root_dir)

        # create train and eval environments
        self.train_py_env = env_load_fn(None, 'headless', gpu)
        # eval_py_env = env_load_fn(None, 'headless', gpu)
        self.eval_py_env = self.train_py_env

        # tf_agents actor currently only support PyEnvironment
        assert isinstance(self.train_py_env, py_environment.PyEnvironment)
        assert isinstance(self.eval_py_env, py_environment.PyEnvironment)

        # get environment specs
        observation_spec, action_spec, time_step_spec = spec_utils.get_tensor_specs(
            env=self.train_py_env
        )
        logging.info('\n Observation specs: %s \n Action specs: %s', observation_spec, action_spec)

        with strategy.scope():
            self.__glorot_uniform_initializer = tf.compat.v1.keras.initializers.glorot_uniform()
            self.__conv_1d_layer_params = [(32, 8, 4), (64, 4, 2), (64, 3, 1)]
            self.__conv_2d_layer_params = [(32, (8, 8), 4), (64, (4, 4), 2), (64, (3, 3), 2)]
            self.__encoder_fc_layers = [256]
            self.__actor_fc_layers = [256]
            self.__critic_obs_fc_layers = [256]
            self.__critic_action_fc_layers = [256]
            self.__critic_joint_fc_layers = [256]

        with strategy.scope():
            # instantiate preprocessing_layers and preprocessing_combiner
            preprocessing_layers, preprocessing_combiner = self.instantiate_preprocessing_layers(
                observation_spec
            )

        with strategy.scope():
            # instantiate actor network to sample action from distribution conditioned on current observation
            actor_net = actor_distribution_network.ActorDistributionNetwork(
                input_tensor_spec=observation_spec,
                output_tensor_spec=action_spec,
                preprocessing_layers=preprocessing_layers,
                preprocessing_combiner=preprocessing_combiner,
                fc_layer_params=self.__actor_fc_layers,
                # continuous_projection_net=normal_projection_net,
                continuous_projection_net=tanh_normal_projection_network.TanhNormalProjectionNetwork,
                kernel_initializer=self.__glorot_uniform_initializer,
            )

        with strategy.scope():
            # instantiate value network to estimate V(s)
            critic_net = critic_network.CriticNetwork(
                input_tensor_spec=(
                    observation_spec,
                    action_spec
                ),
                preprocessing_layers=preprocessing_layers,
                preprocessing_combiner=preprocessing_combiner,
                observation_fc_layer_params=self.__critic_obs_fc_layers,
                action_fc_layer_params=self.__critic_action_fc_layers,
                joint_fc_layer_params=self.__critic_joint_fc_layers,
                kernel_initializer=self.__glorot_uniform_initializer,
            )

        with strategy.scope():
            # instantiate the sac agent
            tf_agent = sac_agent.SacAgent(
                time_step_spec=time_step_spec,
                action_spec=action_spec,
                critic_network=critic_net,
                actor_network=actor_net,
                actor_optimizer=tf.compat.v1.train.AdamOptimizer(
                    learning_rate=actor_learning_rate
                ),
                critic_optimizer=tf.compat.v1.train.AdamOptimizer(
                    learning_rate=critic_learning_rate
                ),
                alpha_optimizer=tf.compat.v1.train.AdamOptimizer(
                    learning_rate=alpha_learning_rate
                ),
                target_update_tau=target_update_tau,
                target_update_period=target_update_period,
                td_errors_loss_fn=td_errors_loss_fn,
                gamma=gamma,
                reward_scale_factor=reward_scale_factor,
                gradient_clipping=gradient_clipping,
                debug_summaries=debug_summaries,
                summarize_grads_and_vars=summarize_grads_and_vars,
                train_step_counter=train_step_counter,
            )
            tf_agent.initialize()

        # instantiate agent policies
        self.tf_agent = tf_agent
        self.eval_policy = py_tf_eager_policy.PyTFEagerPolicy(
            policy=tf_agent.policy,
            use_tf_function=use_tf_function
        )
        self.collect_policy = py_tf_eager_policy.PyTFEagerPolicy(
            policy=tf_agent.collect_policy,
            use_tf_function=use_tf_function
        )
        self.random_policy = random_py_policy.RandomPyPolicy(
            time_step_spec=self.train_py_env.time_step_spec(),
            action_spec=self.train_py_env.action_spec()
        )

    def instantiate_preprocessing_layers(self,
                                         observation_spec
                                         ):
        """

        :param observation_spec: OrderedDict
            environment observation specs of [task_obs, rgb_obs, depth, scan, ..]
        :return: dict(tf.keras.Sequential), tf.keras.layers.concatenate
            tuple of preprocessing_layers and preprocessing_combiner
        """

        preprocessing_combiner = None
        preprocessing_layers = {}
        if 'rgb_obs' in observation_spec:
            preprocessing_layers['rgb_obs'] = tf.keras.Sequential(
                layers=mlp_layers(
                    conv_1d_layer_params=None,
                    conv_2d_layer_params=self.__conv_2d_layer_params,
                    fc_layer_params=self.__encoder_fc_layers,
                    kernel_initializer=self.__glorot_uniform_initializer,
                )
            )
        if 'task_obs' in observation_spec:
            preprocessing_layers['task_obs'] = tf.keras.Sequential(
                layers=mlp_layers(
                    conv_1d_layer_params=None,
                    conv_2d_layer_params=None,
                    fc_layer_params=self.__encoder_fc_layers,
                    kernel_initializer=self.__glorot_uniform_initializer
                )
            )

        ## TODO: add support for more observation types, if required

        if len(preprocessing_layers) > 1:
            preprocessing_combiner = tf.keras.layers.Concatenate(axis=-1)

        return preprocessing_layers, preprocessing_combiner
