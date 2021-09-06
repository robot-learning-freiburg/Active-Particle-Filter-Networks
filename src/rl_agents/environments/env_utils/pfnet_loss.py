#!/usr/bin/env python3

import numpy as np
import tensorflow as tf

def compute_loss(particle_states, particle_weights, true_states, trav_map_resolution):
    """

    :param particle_states: particle states after observation update but before motion update (batch, trajlen, k, 3)
    :param particle_weights: particle likelihoods in the log space (unnormalized) (batch, trajlen, k)
    :param true_states: true state of robot (batch, trajlen, 3)
    :param float trav_map_resolution: The map rescale factor for iGibsonEnv

    :return dict: total loss and coordinate loss (in meters)
    """

    # assert particle_states.ndim == 4 and particle_weights.ndim == 3 and true_states.ndim == 3
    lin_weights = tf.nn.softmax(particle_weights, axis=-1)

    true_coords = true_states[:, :, :2]
    mean_coords = tf.math.reduce_sum(tf.math.multiply(
                        particle_states[:, :, :, :2], lin_weights[:, :, :, None]
                    ), axis=2)
    coords_diffs = mean_coords - true_coords

    # iGibsonEnv.scene.map_to_world()
    coords_diffs = coords_diffs * trav_map_resolution

    # coordinates loss component: (x-x')^2 + (y-y')^2
    loss_coords = tf.math.reduce_sum(tf.math.square(coords_diffs), axis=2)

    true_orients = true_states[:, :, 2]
    orient_diffs = particle_states[:, :, :, 2] - true_orients[:, :, None]

    # normalize between [-pi, +pi]
    orient_diffs = tf.math.floormod(orient_diffs + np.pi, 2*np.pi) - np.pi

    # orintation loss component: (sum_k[(theta_k-theta')*weight_k] )^2
    loss_orient = tf.math.square(tf.math.reduce_sum(orient_diffs * lin_weights, axis=2))

    # combine translational and orientation losses
    loss_combined = loss_coords + 0.36 * loss_orient
    loss_pred = tf.math.reduce_mean(loss_combined)

    loss = {}
    loss['pred'] = loss_pred      # float
    loss['coords'] = loss_coords    # [batch_size, trajlen]
    loss['orient'] = loss_orient    # [batch_size, trajlen]

    return loss
