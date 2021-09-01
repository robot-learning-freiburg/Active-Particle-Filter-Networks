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

@gin.configurable(module='tf_agents')
class AveragePoseMSEMetrix(tf_metric.TFStepMetric):
  """Metric to compute the average return."""

  def __init__(self,
               name='AveragePoseMSE',
               prefix='Metrics',
               dtype=tf.float32,
               batch_size=1,
               buffer_size=10):
    super(AveragePoseMSE, self).__init__(name=name, prefix=prefix)
    self._buffer = tf_metrics.TFDeque(buffer_size, dtype)
    self._dtype = dtype
    self._pose_mse_accumulator = common.create_variable(
        initial_value=0, dtype=dtype, shape=(batch_size,), name='Accumulator')

  @common.function(autograph=True)
  def call(self, trajectory):
    # Zero out batch indices where a new episode is starting.
    self._pose_mse_accumulator.assign(
        tf.where(trajectory.is_first(), tf.zeros_like(self._pose_mse_accumulator),
                 self._pose_mse_accumulator))

    # Update accumulator with received pose mse.
    self._pose_mse_accumulator.assign_add(trajectory.reward)

    # Add final returns to buffer.
    last_episode_indices = tf.squeeze(tf.where(trajectory.is_last()), axis=-1)
    for indx in last_episode_indices:
      self._buffer.add(self._pose_mse_accumulator[indx])

    return trajectory

  def result(self):
    return self._buffer.mean()

  @common.function
  def reset(self):
    self._buffer.clear()
    self._pose_mse_accumulator.assign(tf.zeros_like(self._pose_mse_accumulator))

@gin.configurable(module='tf_agents')
class AverageCollisionMetric(tf_metric.TFStepMetric):
  """Metric to compute the average collision."""

  def __init__(self,
               name='AverageCollision',
               prefix='Metrics',
               dtype=tf.float32,
               batch_size=1,
               buffer_size=10):
    super(AveragePoseMSE, self).__init__(name=name, prefix=prefix)
    self._buffer = tf_metrics.TFDeque(buffer_size, dtype)
    self._dtype = dtype
    self._collision_accumulator = common.create_variable(
        initial_value=0, dtype=dtype, shape=(batch_size,), name='Accumulator')

  @common.function(autograph=True)
  def call(self, trajectory):
    # Zero out batch indices where a new episode is starting.
    self._collision_accumulator.assign(
        tf.where(trajectory.is_first(), tf.zeros_like(self._collision_accumulator),
                 self._collision_accumulator))

    # Update accumulator with collision penality.
    self._collision_accumulator.assign_add(trajectory.reward)

    # Add final returns to buffer.
    last_episode_indices = tf.squeeze(tf.where(trajectory.is_last()), axis=-1)
    for indx in last_episode_indices:
      self._buffer.add(self._collision_accumulator[indx])

    return trajectory

  def result(self):
    return self._buffer.mean()

  @common.function
  def reset(self):
    self._buffer.clear()
    self._pose_mse_accumulator.assign(tf.zeros_like(self._pose_mse_accumulator))
