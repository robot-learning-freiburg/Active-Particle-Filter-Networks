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
class AveragePoseMSEMetric(tf_metric.TFStepMetric):
  """Metric to compute the average pose mse."""

  def __init__(self,
               name='AveragePoseMSE',
               prefix='Metrics',
               dtype=tf.float32,
               batch_size=1,
               buffer_size=10):
    super(AveragePoseMSEMetric, self).__init__(name=name, prefix=prefix)
    self._buffer = tf_metrics.TFDeque(buffer_size, dtype)
    self._dtype = dtype
    self._pose_mse_accumulator = common.create_variable(
        initial_value=0, dtype=dtype, shape=(batch_size,), name='Accumulator')

  @common.function(autograph=True)
  def call(self, trajectory_tuple):
    traj = trajectory.from_transition(trajectory_tuple[0], trajectory_tuple[1], trajectory_tuple[2])
    pose_mse = trajectory_tuple[3]['pose_mse'] if 'pose_mse' in trajectory_tuple[3] else [0.0]

    # Zero out batch indices where a new episode is starting.
    self._pose_mse_accumulator.assign(
        tf.where(traj.is_first(), tf.zeros_like(self._pose_mse_accumulator),
                 self._pose_mse_accumulator))

    # Update accumulator with received pose mse.
    self._pose_mse_accumulator.assign_add(pose_mse)

    # Add final returns to buffer.
    last_episode_indices = tf.squeeze(tf.where(traj.is_last()), axis=-1)
    for indx in last_episode_indices:
      self._buffer.add(self._pose_mse_accumulator[indx])

    return traj

  def result(self):
    return self._buffer.mean()

  @common.function
  def reset(self):
    self._buffer.clear()
    self._pose_mse_accumulator.assign(tf.zeros_like(self._pose_mse_accumulator))

@gin.configurable(module='tf_agents')
class AverageCollisionPenalityMetric(tf_metric.TFStepMetric):
  """Metric to compute the average collision penality."""

  def __init__(self,
               name='AverageCollisionPenality',
               prefix='Metrics',
               dtype=tf.float64,
               batch_size=1,
               buffer_size=10):
    super(AverageCollisionPenalityMetric, self).__init__(name=name, prefix=prefix)
    self._buffer = tf_metrics.TFDeque(buffer_size, dtype)
    self._dtype = dtype
    self._collision_penality_accumulator = common.create_variable(
        initial_value=0, dtype=dtype, shape=(batch_size,), name='Accumulator')

  @common.function(autograph=True)
  def call(self, trajectory_tuple):
    traj = trajectory.from_transition(trajectory_tuple[0], trajectory_tuple[1], trajectory_tuple[2])
    collision_penality = trajectory_tuple[3]['collision_penality'] if 'collision_penality' in trajectory_tuple[3] else [0.0]

    # Zero out batch indices where a new episode is starting.
    self._collision_penality_accumulator.assign(
        tf.where(traj.is_first(), tf.zeros_like(self._collision_penality_accumulator),
                 self._collision_penality_accumulator))

    # Update accumulator with received pose mse.
    self._collision_penality_accumulator.assign_add(collision_penality)

    # Add final returns to buffer.
    last_episode_indices = tf.squeeze(tf.where(traj.is_last()), axis=-1)
    for indx in last_episode_indices:
      self._buffer.add(self._collision_penality_accumulator[indx])

    return traj

  def result(self):
    return self._buffer.mean()

  @common.function
  def reset(self):
    self._buffer.clear()
    self._collision_penality_accumulator.assign(tf.zeros_like(self._collision_penality_accumulator))
