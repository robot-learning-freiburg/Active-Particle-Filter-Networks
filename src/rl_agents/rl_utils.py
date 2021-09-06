#!/usr/bin/env python3

"""
reference:
    https://github.com/tensorflow/agents/blob/master/tf_agents/metrics/tf_metrics.py#L158
"""

import gin
import tensorflow as tf  # pylint: disable=g-explicit-tensorflow-version-import
import numpy as np

# custom tf_agents
from tf_agents.metrics import tf_metric
from tf_agents.metrics import tf_metrics
from tf_agents.utils import common
from tf_agents.trajectories import trajectory

@gin.configurable(module='tf_agents')
class AverageStepPositionErrorMetric(tf_metric.TFStepMetric):
    """Metric to compute the average of mean over episode length's position error."""

    def __init__(self,
            name='AverageStepPositionError',
            prefix='Metrics',
            dtype=tf.float32,
            batch_size=1,
            buffer_size=10):
        super(AverageStepPositionErrorMetric, self).__init__(name=name, prefix=prefix)
        self._buffer = tf_metrics.TFDeque(buffer_size, dtype)
        self._dtype = dtype
        self._coords_error_accumulator = common.create_variable(
            initial_value=0, dtype=dtype, shape=(batch_size, ), name='Accumulator')
        self.environment_steps = common.create_variable(
            initial_value=0, dtype=dtype, shape=(batch_size, ), name='environment_steps')

    @common.function(autograph=True)
    def call(self, trajectory_tuple):
        traj = trajectory.from_transition(trajectory_tuple[0], trajectory_tuple[1], trajectory_tuple[2])
        coords = trajectory_tuple[3]['coords'] if 'coords' in trajectory_tuple[3] else [0.0]

        # Zero out batch indices where a new episode is starting.
        self._coords_error_accumulator.assign(
            tf.where(traj.is_first(), tf.zeros_like(self._coords_error_accumulator),
                self._coords_error_accumulator))
        self.environment_steps.assign(
            tf.where(traj.is_first(), tf.zeros_like(self.environment_steps),
                self.environment_steps))

        # Update accumulator with received pose mse.
        self._coords_error_accumulator.assign_add(coords)

        # increment step when not final step
        self.environment_steps.assign_add(tf.cast(~traj.is_boundary(), self._dtype))

        # Add mean over episode length's position error to buffer.
        last_episode_indices = tf.squeeze(tf.where(traj.is_last()), axis=-1)
        for indx in last_episode_indices:
            self._buffer.add(self._coords_error_accumulator[indx]/self.environment_steps[indx])

        return traj

    def result(self):
        return self._buffer.mean()

    @common.function
    def reset(self):
        self._buffer.clear()
        self._coords_error_accumulator.assign(tf.zeros_like(self._coords_error_accumulator))

@gin.configurable(module='tf_agents')
class AverageStepOrientationErrorMetric(tf_metric.TFStepMetric):
    """Metric to compute the average of mean over episode length's orientation error."""

    def __init__(self,
            name='AverageStepOrientationError',
            prefix='Metrics',
            dtype=tf.float32,
            batch_size=1,
            buffer_size=10):
        super(AverageStepOrientationErrorMetric, self).__init__(name=name, prefix=prefix)
        self._buffer = tf_metrics.TFDeque(buffer_size, dtype)
        self._dtype = dtype
        self._orient_error_accumulator = common.create_variable(
            initial_value=0, dtype=dtype, shape=(batch_size, ), name='Accumulator')
        self.environment_steps = common.create_variable(
            initial_value=0, dtype=dtype, shape=(batch_size, ), name='environment_steps')

    @common.function(autograph=True)
    def call(self, trajectory_tuple):
        traj = trajectory.from_transition(trajectory_tuple[0], trajectory_tuple[1], trajectory_tuple[2])
        coords = trajectory_tuple[3]['orient'] if 'orient' in trajectory_tuple[3] else [0.0]

        # Zero out batch indices where a new episode is starting.
        self._orient_error_accumulator.assign(
            tf.where(traj.is_first(), tf.zeros_like(self._orient_error_accumulator),
                self._orient_error_accumulator))
        self.environment_steps.assign(
            tf.where(traj.is_first(), tf.zeros_like(self.environment_steps),
                self.environment_steps))

        # Update accumulator with received pose mse.
        self._orient_error_accumulator.assign_add(coords)

        # increment step when not final step
        self.environment_steps.assign_add(tf.cast(~traj.is_boundary(), self._dtype))

        # Add mean over episode length's orientation error to buffer.
        last_episode_indices = tf.squeeze(tf.where(traj.is_last()), axis=-1)
        for indx in last_episode_indices:
            self._buffer.add(self._orient_error_accumulator[indx]/self.environment_steps[indx])

        return traj

    def result(self):
        return self._buffer.mean()

    @common.function
    def reset(self):
        self._buffer.clear()
        self._orient_error_accumulator.assign(tf.zeros_like(self._orient_error_accumulator))

@gin.configurable(module='tf_agents')
class AverageStepCollisionPenalityMetric(tf_metric.TFStepMetric):
    """Metric to compute the average of mean over episode length's collision penality."""

    def __init__(self,
            name='AverageStepCollisionPenality',
            prefix='Metrics',
            dtype=tf.float64,
            batch_size=1,
            buffer_size=10):

        super(AverageStepCollisionPenalityMetric, self).__init__(name=name, prefix=prefix)
        self._buffer = tf_metrics.TFDeque(buffer_size, dtype)
        self._dtype = dtype
        self._collision_penality_accumulator = common.create_variable(
            initial_value=0, dtype=dtype, shape=(batch_size, ), name='Accumulator')
        self.environment_steps = common.create_variable(
            initial_value=0, dtype=dtype, shape=(batch_size, ), name='environment_steps')

    @common.function(autograph=True)
    def call(self, trajectory_tuple):
        traj = trajectory.from_transition(trajectory_tuple[0], trajectory_tuple[1], trajectory_tuple[2])
        collision_penality = trajectory_tuple[3]['collision_penality'] if 'collision_penality' in trajectory_tuple[3] else [0.0]

        # Zero out batch indices where a new episode is starting.
        self._collision_penality_accumulator.assign(
            tf.where(traj.is_first(), tf.zeros_like(self._collision_penality_accumulator),
                self._collision_penality_accumulator))
        self.environment_steps.assign(
            tf.where(traj.is_first(), tf.zeros_like(self.environment_steps),
                self.environment_steps))

        # Update accumulator with received pose mse.
        self._collision_penality_accumulator.assign_add(collision_penality)

        # increment step when not final step
        self.environment_steps.assign_add(tf.cast(~traj.is_boundary(), self._dtype))

        # Add mean over episode length's collision penality to buffer.
        last_episode_indices = tf.squeeze(tf.where(traj.is_last()), axis=-1)
        for indx in last_episode_indices:
            self._buffer.add(-self._collision_penality_accumulator[indx]/self.environment_steps[indx])

        return traj

    def result(self):
        return self._buffer.mean()

    @common.function
    def reset(self):
        self._buffer.clear()
        self._collision_penality_accumulator.assign(tf.zeros_like(self._collision_penality_accumulator))
