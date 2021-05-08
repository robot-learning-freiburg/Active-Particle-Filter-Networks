#!/usr/bin/env python3

from absl import logging

import os
import tensorflow as tf

# import custom tf_agents
from tf_agents.agents.ppo import ppo_clip_agent
from tf_agents.environments import py_environment
from tf_agents.networks import actor_distribution_network
from tf_agents.networks import value_network
from tf_agents.networks.utils import mlp_layers
from tf_agents.train.utils import spec_utils


class PPOAgent(object):
    """
    Custom PPO Agent implementation for iGibsonEnv
    """

    def __init__(self,
                 root_dir,
                 env_load_fn,
                 train_step_counter,
                 strategy,
                 gpu=0,
                 # train
                 num_epochs=25,
                 learning_rate=1e-3,
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
            distribution stratergy
        :param gpu: int
            gpu device number
        :param num_epochs: int
            number of epochs for computing policy updates
        :param learning_rate: float
            learning rate of optimizer used for agent
        :param debug_summaries: bool
            whether or not to gather debug summaries
        :param summarize_grads_and_vars: bool
            whether or not to write gradient summaries
        """

        self.root_dir = os.path.expanduser(root_dir)

        # create train and eval environments
        train_py_env = env_load_fn(None, 'headless', gpu)
        # eval_py_env = env_load_fn(None, 'headless', gpu)
        eval_py_env = train_py_env

        # tf_agents actor currently only support PyEnvironment
        assert isinstance(train_py_env, py_environment.PyEnvironment)
        assert isinstance(eval_py_env, py_environment.PyEnvironment)

        # get environment specs
        observation_spec, action_spec, time_step_spec = spec_utils.get_tensor_specs(train_py_env)
        logging.info('\n Observation specs: %s \n Action specs: %s', observation_spec, action_spec)

        with strategy.scope():
            self.__glorot_uniform_initializer = tf.compat.v1.keras.initializers.glorot_uniform()
            self.__conv_1d_layer_params = [(32, 8, 4), (64, 4, 2), (64, 3, 1)]
            self.__conv_2d_layer_params = [(32, (8, 8), 4), (64, (4, 4), 2), (64, (3, 3), 2)]
            self.__encoder_fc_layers = [256]
            self.__actor_fc_layers = [256]
            self.__value_fc_layers = [256]

        with strategy.scope():
            # construct preprocessing_layers and preprocessing_combiner
            preprocessing_layers, preprocessing_combiner = self.construct_preprocessing_layers(
                observation_spec
            )

        with strategy.scope():
            # construct actor network to sample action from distribution conditioned on current observation
            actor_net = actor_distribution_network.ActorDistributionNetwork(
                input_tensor_spec=observation_spec,
                output_tensor_spec=action_spec,
                preprocessing_layers=preprocessing_layers,
                preprocessing_combiner=preprocessing_combiner,
                fc_layer_params=self.__actor_fc_layers,
                activation_fn=tf.keras.activations.tanh,
            )

        with strategy.scope():
            # construct value network to estimate V(s)
            value_net = value_network.ValueNetwork(
                input_tensor_spec=observation_spec,
                preprocessing_layers=preprocessing_layers,
                preprocessing_combiner=preprocessing_combiner,
                fc_layer_params=self.__value_fc_layers,
                activation_fn=tf.keras.activations.tanh,
            )

        with strategy.scope():
            # instantiate the ppo agent
            tf_agent = ppo_clip_agent.PPOClipAgent(
                time_step_spec=time_step_spec,
                action_spec=action_spec,
                optimizer=tf.compat.v1.train.AdamOptimizer(
                    learning_rate=learning_rate
                ),
                action_net=actor_net,
                value_net=value_net,
                importance_ratio_clipping=0.2,
                normalize_observations=False,
                normalize_rewards=False,
                use_gae=True,
                num_epochs=num_epochs,
                debug_summaries=debug_summaries,
                summarize_grads_and_vars=summarize_grads_and_vars,
                train_step_counter=train_step_counter,
            )
            tf_agent.initialize()


    def construct_preprocessing_layers(self, observation_spec):
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
