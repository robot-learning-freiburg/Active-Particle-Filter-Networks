#!/usr/bin/env python3

# MIT License

# Copyright (c) 2018 Peter Karkus, AdaCompNUS

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

# This code has been adapted from https://github.com/AdaCompNUS/pfnet

import numpy as np
import tensorflow as tf
from tensorflow import keras
import pybullet as p

try:
    from .architecture import networks
    from .architecture.spatial_transformer import transformer
except:
    from architecture import networks
    from architecture.spatial_transformer import transformer

from environments.env_utils.datautils import get_random_particles


def datautils_norm_angle(angle):
    """
    Normalize the angle to [-pi, pi]
    :param float angle: input angle to be normalized
    :return float: normalized angle
    """
    quaternion = p.getQuaternionFromEuler(np.array([0, 0, angle]))
    euler = p.getEulerFromQuaternion(quaternion)
    return euler[2]


def norm_angle(angle):
    return tf.math.floormod(angle + np.pi, 2 * np.pi) - np.pi


class PFCell(keras.layers.AbstractRNNCell):
    """
    PF-Net custom implementation for localization with RNN interface
    Implements the particle set update: observation, tramsition models and soft-resampling

    Cell inputs: observation, odometry
    Cell states: particle_states, particle_weights
    Cell outputs: particle_states, particle_weights (updated)
    """

    def __init__(self, params, is_igibson: bool, **kwargs):
        """
        :param params: parsed arguments
        """
        self.params = params

        self.states_shape = (self.params.batch_size, self.params.num_particles, 3)
        self.weights_shape = (self.params.batch_size, self.params.num_particles)
        self.map_shape = (self.params.batch_size, *self.params.global_map_size)
        super(PFCell, self).__init__(**kwargs)

        self.is_igibson = is_igibson

        # models
        if self.params.likelihood_model == 'learned':
            self.obs_model = networks.obs_encoder(obs_shape=[56, 56, params.obs_ch])
            self.map_model = networks.map_encoder(map_shape=[28, 28, 1])
            self.joint_matrix_model = networks.map_obs_encoder()
            self.joint_vector_model = networks.likelihood_estimator()
        elif self.params.likelihood_model == 'scan_correlation':
            raise NotImplementedError()
        else:
            raise ValueError()

    @staticmethod
    def reset(robot_pose_pixel, env, params):
        # get random particles and weights based on init distribution conditions
        particles = tf.cast(tf.convert_to_tensor(
            get_random_particles(
                params.num_particles,
                params.init_particles_distr,
                tf.expand_dims(tf.convert_to_tensor(robot_pose_pixel, dtype=tf.float32), axis=0),
                env.trav_map,
                params.init_particles_cov,
                params.particles_range)), dtype=tf.float32)
        particle_weights = tf.constant(np.log(1.0 / float(params.num_particles)),
                                       shape=(params.batch_size, params.num_particles),
                                       dtype=tf.float32)
        return particles, particle_weights

    @staticmethod
    def compute_mse_loss(particles, particle_weights, true_states, trav_map_resolution):
        """
        Compute Mean Square Error (MSE) between ground truth pose and particles

        :param particle_states: particle states after observation update but before motion update (batch, trajlen, k, 3)
        :param particle_weights: particle likelihoods in the log space (unnormalized) (batch, trajlen, k)
        :param true_states: true state of robot (batch, trajlen, 3)
        :param float trav_map_resolution: The map rescale factor for iGibsonEnv

        :return dict: total loss and coordinate loss (in meters)
        """
        est_pose = PFCell.get_est_pose(particles=particles, particle_weights=particle_weights)

        pose_diffs = est_pose - true_states

        coords_diffs = pose_diffs[..., :2] * trav_map_resolution
        # coordinates loss component: (x-x')^2 + (y-y')^2
        loss_coords = tf.math.reduce_sum(tf.math.square(coords_diffs), axis=-1)

        orient_diffs = particles[..., 2] - true_states[..., 2][..., None]
        # normalize between [-pi, +pi]
        orient_diffs = tf.math.floormod(orient_diffs + np.pi, 2 * np.pi) - np.pi
        # orientation loss component: (sum_k[(theta_k-theta')*weight_k] )^2
        lin_weights = tf.nn.softmax(particle_weights, axis=-1)
        loss_orient = tf.square(tf.reduce_sum(orient_diffs * lin_weights, axis=-1))

        loss_combined = loss_coords + 0.36 * loss_orient

        loss = {}
        loss['pred'] = loss_combined  # [batch_size, trajlen]
        loss['coords'] = loss_coords  # [batch_size, trajlen]
        loss['orient'] = loss_orient  # [batch_size, trajlen]

        return loss

    @staticmethod
    def bounding_box(img, robot_pose=None, lmt=100):
        """
        Bounding box of non-zeros in an array.

        :param img: numpy array
        :param robot_pose: numpy array of robot pose
        :param lmt: integer representing width/length of bounding box in pixels

        :return (int, int, int, int): bounding box indices top_row, bottom_row, left_column, right_column
        """
        rows = np.any(img, axis=1)
        cols = np.any(img, axis=0)
        rmin, rmax = np.where(rows)[0][[0, -1]]
        cmin, cmax = np.where(cols)[0][[0, -1]]

        if robot_pose is not None:
            # futher constraint the bounding box
            x, y, _ = robot_pose

            rmin = np.rint(y - lmt) if (y - lmt) > rmin else rmin
            rmax = np.rint(y + lmt) if (y + lmt) < rmax else rmax
            cmin = np.rint(x - lmt) if (x - lmt) > cmin else cmin
            cmax = np.rint(x + lmt) if (x + lmt) < cmax else cmax

        return rmin, rmax, cmin, cmax,

    @staticmethod
    def get_est_pose(particles, particle_weights):
        """
        Compute estimate pose from particle and particle_weights (== weighted mean)
        """
        lin_weights = tf.nn.softmax(particle_weights, axis=-1)
        est_pose_xy = tf.math.reduce_sum(tf.math.multiply(particles[..., :2], lin_weights[..., None]), axis=-2)

        particle_theta_normed = norm_angle(particles[..., 2])
        est_pose_theta = tf.math.reduce_sum(tf.math.multiply(particle_theta_normed, lin_weights), axis=-1,
                                            keepdims=True)

        return tf.concat([est_pose_xy, est_pose_theta], axis=-1)

    @staticmethod
    def get_likelihood_map(particles, particle_weights, floor_map):
        """
        Construct Belief map/Likelihood map of particles & particle_weights from particle filter's current state

        :return likelihood_map_ext: [H, W, 4] map where each pixel position corresponds particle's position
            channel 0: floor_map of the environment
            channel 1: particle's weights
            channel 2, 3: particle's orientiation sine and cosine components
        """
        particles = particles[0].cpu().numpy()
        lin_weights = tf.nn.softmax(particle_weights, axis=-1)[0].cpu().numpy()  # normalize weights
        likelihood_map_ext = np.zeros(list(floor_map.shape)[:2] + [4])  # [H, W, 4]

        # update obstacle map channel
        num_particles = particle_weights.shape[-1]
        likelihood_map_ext[:, :, 0] = np.squeeze(floor_map).copy() > 0  # clip to 0 or 1

        xy_clipped = np.clip(particles[..., :2], (0, 0),
                             (likelihood_map_ext.shape[0] - 1, likelihood_map_ext.shape[1] - 1)).astype(int)

        for idx in range(num_particles):
            x, y = xy_clipped[idx]
            orn = particles[idx, 2]
            orn = datautils_norm_angle(orn)
            wt = lin_weights[idx]

            likelihood_map_ext[x, y, 1] += wt
            likelihood_map_ext[x, y, 2] += wt * np.cos(orn)
            likelihood_map_ext[x, y, 3] += wt * np.sin(orn)

        return likelihood_map_ext

    @property
    def state_size(self):
        """
        Size(s) of state(s) used by this cell
        :return tuple(TensorShapes): shape of particle_states, particle_weights
        """
        return [tf.TensorShape(self.states_shape[1:]), tf.TensorShape(self.weights_shape[1:]),
                tf.TensorShape(self.map_shape[1:])]

    @property
    def output_size(self):
        """
        Size(s) of output(s) produced by this cell
        :return tuple(TensorShapes): shape of particle_states, particle_weights
        """
        return [tf.TensorShape(self.states_shape[1:]), tf.TensorShape(self.weights_shape[1:]),
                tf.TensorShape(self.map_shape[1:])]

    def call(self, input, state):
        """
        Implements a particle update
        :param input: observation (batch, 56, 56, ch), odometry (batch, 3), global_map (batch, H, W, 1)
            observation is the sensor reading at time t,
            odometry is the relative motion from time t to t+1,
            global map of environment
        :param state: particle_states (batch, k, 3), particle_weights (batch, k)
            weights are assumed to be in log space and unnormalized
        :return output: particle_states and particle_weights after the observation update.
            (but before the transition update)
        :return state: updated particle_states and particle_weights.
            (but after both observation and transition updates)
        """
        particles, particle_weights, global_map = state
        observation, odometry = input

        if self.is_igibson:
            # motion update: odometry is motion from t-1 to t
            particles = self.transition_model(particles, odometry)

        # observation update
        if self.params.likelihood_model == 'learned':
            lik = self.observation_update(global_map, particles, observation)
        else:
            raise NotImplementedError()
        particle_weights = particle_weights + lik  # unnormalized

        # resample
        if self.params.resample:
            particles, particle_weights = self.resample(particles, particle_weights,
                                                        alpha=self.params.alpha_resample_ratio)

        # construct output before motion update
        output = [particles, particle_weights]

        # motion update which affect the particle state input at next step
        if not self.is_igibson:
            particles = self.transition_model(particles, odometry)

        state = [particles, particle_weights, global_map]
        return output, state

    @tf.function(jit_compile=True)
    def observation_update(self, global_map, particle_states, observation):
        """
        Implements a discriminative observation model for localization
        The model transforms global map to local maps for each particle,
        where a local map is a local view from state defined by the particle.
        :param global_map: global map input (batch, None, None, ch)
            assumes range[0, 2] were 0: occupied and 2: free space
        :param particle_states: particle states before observation update (batch, k, 3)
        :param observation: image observation (batch, 56, 56, ch)
        :return (batch, k): particle likelihoods in the log space (unnormalized)
        """
        if self.params.obs_mode == "occupancy_grid":
            # robot is looking to the right, but should be looking up
            observation = tf.image.rot90(observation, 1)

        batch_size, num_particles = particle_states.shape.as_list()[:2]

        # transform global maps to local maps
        # TODO: only set agent_at_bottom true if using [rgb, d], not for lidar?
        local_maps = PFCell.transform_maps(global_map, particle_states, (28, 28), self.params.window_scaler,
                                           agent_at_bottom=True, flip_map=self.is_igibson)

        # rescale from [0, 2] to [-1, 1]    -> optional
        local_maps = -(local_maps - 1)

        # flatten batch and particle dimensions
        local_maps = tf.reshape(local_maps, [batch_size * num_particles] + local_maps.shape.as_list()[2:])

        # get features from local maps
        map_features = self.map_model(local_maps)

        # get features from observation
        obs_features = self.obs_model(observation)

        # tile observation features
        obs_features = tf.tile(tf.expand_dims(obs_features, axis=1), [1, num_particles, 1, 1, 1])
        obs_features = tf.reshape(obs_features, [batch_size * num_particles] + obs_features.shape.as_list()[2:])

        # sanity check
        assert obs_features.shape.as_list()[:-1] == map_features.shape.as_list()[:-1]

        # merge map and observation features
        joint_features = tf.concat([map_features, obs_features], axis=-1)
        joint_features = self.joint_matrix_model(joint_features)

        # reshape to a vector
        joint_features = tf.reshape(joint_features, [batch_size * num_particles, -1])
        lik = self.joint_vector_model(joint_features)
        lik = tf.reshape(lik, [batch_size, num_particles])

        return lik

    @tf.function(jit_compile=True)
    def resample(self, particle_states, particle_weights, alpha):
        """
        Implements soft-resampling of particles
        :param particle_states: particle states (batch, k, 3)
        :param particle_weights: unnormalized particle weights in log space (batch, k)
        :param alpha: trade-off parameter for soft-resampling
            alpha == 1, corresponds to standard hard-resampling
            alpha == 0, corresponds to sampling particles uniformly ignoring weights
        :return (batch, k, 3) (batch, k): resampled particle states and particle weights
        """

        assert 0.0 < alpha <= 1.0
        batch_size, num_particles = particle_states.shape.as_list()[:2]

        # normalize weights
        particle_weights = particle_weights - tf.math.reduce_logsumexp(particle_weights, axis=-1, keepdims=True)

        # sample uniform weights
        uniform_weights = tf.constant(np.log(1.0 / float(num_particles)), shape=(batch_size, num_particles),
                                      dtype=tf.float32)

        # build sample distribution q(s) and update particle weights
        if alpha < 1.0:
            # soft-resampling
            q_weights = tf.stack([particle_weights + np.log(alpha),
                                  uniform_weights + np.log(1.0 - alpha)],
                                 axis=-1)
            q_weights = tf.math.reduce_logsumexp(q_weights, axis=-1, keepdims=False)
            q_weights = q_weights - tf.reduce_logsumexp(q_weights, axis=-1, keepdims=True)  # normalized

            particle_weights = particle_weights - q_weights  # unnormalized
        else:
            # hard-resampling -> produces zero gradients
            q_weights = particle_weights
            particle_weights = uniform_weights

        # sample particle indices according to q(s)
        indices = tf.random.categorical(q_weights, num_particles, dtype=tf.int32)  # shape: (bs, k)

        # index into particles
        helper = tf.range(0, batch_size * num_particles, delta=num_particles, dtype=tf.int32)  # (batch, )
        indices = indices + tf.expand_dims(helper, axis=1)

        particle_states = tf.reshape(particle_states, (batch_size * num_particles, 3))
        particle_states = tf.gather(particle_states, indices=indices, axis=0)  # (bs, k, 3)

        particle_weights = tf.reshape(particle_weights, (batch_size * num_particles,))
        particle_weights = tf.gather(particle_weights, indices=indices, axis=0)  # (bs, k)

        return particle_states, particle_weights

    @tf.function(jit_compile=True)
    def transition_model(self, particle_states, odometry):
        """
        Implements a stochastic transition model for localization
        :param particle_states: particle states before motion update (batch, k, 3)
        :param odometry: odometry reading - relative motion in robot coordinate frame (batch, 3)
        :return (batch, k, 3): particle states updated with the odometry and optionally transition noise
        """

        translation_std = self.params.transition_std[0]  # in pixels
        rotation_std = self.params.transition_std[1]  # in radians

        part_x, part_y, part_th = tf.unstack(particle_states, axis=-1, num=3)  # (bs, k, 3)

        # non-noisy odometry
        odometry = tf.expand_dims(odometry, axis=1)  # (batch_size, 1, 3)
        odom_x, odom_y, odom_th = tf.unstack(odometry, axis=-1, num=3)

        # sample noisy orientation
        noise_th = tf.random.normal(part_th.get_shape(), mean=0.0, stddev=1.0) * rotation_std

        # add orientation noise before translation
        part_th = part_th + noise_th

        # non-noisy translation and rotation
        cos_th = tf.cos(part_th)
        sin_th = tf.sin(part_th)
        delta_x = cos_th * odom_x - sin_th * odom_y
        delta_y = sin_th * odom_x + cos_th * odom_y
        delta_th = odom_th

        # sample noisy translation
        delta_x = delta_x + tf.random.normal(delta_x.get_shape(), mean=0.0, stddev=1.0) * translation_std
        delta_y = delta_y + tf.random.normal(delta_y.get_shape(), mean=0.0, stddev=1.0) * translation_std

        return tf.stack([part_x + delta_x, part_y + delta_y, part_th + delta_th], axis=-1)  # (bs, k, 3)

    @staticmethod
    @tf.function(jit_compile=True)
    def transform_maps(global_map, particle_states, local_map_size, window_scaler=None, agent_at_bottom: bool = True,
                       flip_map: bool = False):
        """
        Implements global to local map transformation
        :param global_map: global map input (batch, None, None, ch)
        :param particle_states: particle states that define local view for transformation (batch, k, 3)
        :param local_map_size: size of output local maps (height, width)
        :param window_scaler: global map will be down-scaled by some int factor
        :return (batch, k, local_map_size[0], local_map_size[1], ch): each local map shows different transformation
            of global map corresponding to particle state
        """

        # flatten batch and particle
        batch_size, num_particles = particle_states.shape.as_list()[:2]
        total_samples = batch_size * num_particles
        flat_states = tf.reshape(particle_states, [total_samples, 3])

        # NOTE: For igibson, first value indexes the y axis, second the x axis
        if flip_map:
            flat_states = tf.gather(flat_states, [1, 0, 2], axis=-1)

        # define variables
        # TODO: could resize the map before doing the affine transform, instead of doing it at the same time

        input_shape = tf.shape(global_map)
        global_height = tf.cast(input_shape[1], tf.float32)
        global_width = tf.cast(input_shape[2], tf.float32)
        height_inverse = 1.0 / global_height
        width_inverse = 1.0 / global_width
        zero = tf.constant(0, dtype=tf.float32, shape=(total_samples,))
        one = tf.constant(1, dtype=tf.float32, shape=(total_samples,))

        # normalize orientations and precompute cos and sin functions
        theta = -flat_states[:, 2] - 0.5 * np.pi
        costheta = tf.cos(theta)
        sintheta = tf.sin(theta)

        # construct affine transformation matrix step-by-step

        # 1: translate the global map s.t. the center is at the particle state
        translate_x = (flat_states[:, 0] * width_inverse * 2.0) - 1.0
        translate_y = (flat_states[:, 1] * height_inverse * 2.0) - 1.0

        transm1 = tf.stack((one, zero, translate_x, zero, one, translate_y, zero, zero, one), axis=1)
        transm1 = tf.reshape(transm1, [total_samples, 3, 3])

        # 2: rotate map s.t the orientation matches that of the particles
        rotm = tf.stack((costheta, sintheta, zero, -sintheta, costheta, zero, zero, zero, one), axis=1)
        rotm = tf.reshape(rotm, [total_samples, 3, 3])

        # 3: scale down the map
        if window_scaler is not None:
            scale_x = tf.fill((total_samples,), float(local_map_size[1] * window_scaler) * width_inverse)
            scale_y = tf.fill((total_samples,), float(local_map_size[0] * window_scaler) * height_inverse)
        else:
            # identity
            scale_x = one
            scale_y = one

        scalem = tf.stack((scale_x, zero, zero, zero, scale_y, zero, zero, zero, one), axis=1)
        scalem = tf.reshape(scalem, [total_samples, 3, 3])

        # finally chain all traformation matrices into single one
        transform_m = tf.matmul(tf.matmul(transm1, rotm), scalem)
        # 4: translate the local map s.t. the particle defines the bottom mid_point instead of the center
        if agent_at_bottom:
            translate_y2 = tf.constant(-1.0, dtype=tf.float32, shape=(total_samples,))

            transm2 = tf.stack((one, zero, zero, zero, one, translate_y2, zero, zero, one), axis=1)
            transm2 = tf.reshape(transm2, [total_samples, 3, 3])

            transform_m = tf.matmul(transform_m, transm2)

        # reshape to format expected by spatial transform network
        transform_m = tf.reshape(transform_m[:, :2], [batch_size, num_particles, 6])

        # iterate over num_particles to transform image using spatial transform network
        def transform_batch(U, thetas, out_size):
            num_batch, num_transforms = map(int, thetas.get_shape().as_list()[:2])
            indices = [[i] * num_transforms for i in range(num_batch)]
            input_repeated = tf.gather(U, tf.reshape(indices, [-1]))
            return transformer(input_repeated, thetas, out_size)

        # t = time.time()
        if batch_size == 1 and global_map.shape[2] <= 100:
            # assert batch_size == 1, (batch_size, "Just haven't tested it for batches yet, might already work though")
            # with tf.device('CPU'):
            local_maps_new = transform_batch(U=global_map, thetas=transform_m, out_size=local_map_size)
            local_maps = local_maps_new[tf.newaxis]
        else:
            # create mini batches to not run out of vram
            # min_batch_size = 1
            # lmaps = []
            # for b in np.arange(0, batch_size, min_batch_size):
            #     # print(b, b+min_batch_size, global_map[b:b+min_batch_size].shape)
            #     lmaps.append(transform_batch(U=global_map[b:b+min_batch_size], thetas=transform_m[b:b+min_batch_size], out_size=local_map_size))
            # local_maps = tf.concat(lmaps, 0)
            # print(f"ccccccccccc {(time.time() - t) / 60:.3f}")

            # t = time.time()
            local_maps = tf.stack([
                transformer(global_map, transform_m[:, i], local_map_size) for i in range(num_particles)
            ], axis=1)
            # print(f"zzzzzzzzzzzzzzz {(time.time() - t) / 60:.3f}")

        # reshape if any information has lost in spatial transform network
        local_maps = tf.reshape(local_maps, [batch_size, num_particles, local_map_size[0], local_map_size[1],
                                             global_map.shape.as_list()[-1]])

        # NOTE: flip to have the same alignment as the other modalities
        if flip_map:
            local_maps = tf.experimental.numpy.flip(local_maps, -2)

        return local_maps  # (batch_size, num_particles, 28, 28, 1)


def pfnet_model(params, is_igibson: bool):
    if hasattr(params, 'obs_ch'):
        obs_ch = params.obs_ch
    else:
        obs_ch = params.obs_ch = 3

    if params.likelihood_model == "scan_correlation":
        sz = 128
    else:
        sz = 56

    observation = keras.Input(shape=[params.trajlen, sz, sz, obs_ch],
                              batch_size=params.batch_size)  # (bs, T, 56, 56, C)
    odometry = keras.Input(shape=[params.trajlen, 3], batch_size=params.batch_size)  # (bs, T, 3)

    global_map = keras.Input(shape=params.global_map_size, batch_size=params.batch_size)  # (bs, H, W, 1)
    particle_states = keras.Input(shape=[params.num_particles, 3], batch_size=params.batch_size)  # (bs, k, 3)
    particle_weights = keras.Input(shape=[params.num_particles], batch_size=params.batch_size)  # (bs, k)

    cell = PFCell(params, is_igibson=is_igibson)
    rnn = keras.layers.RNN(cell, return_sequences=True, return_state=params.return_state, stateful=False)

    state = [particle_states, particle_weights, global_map]
    input = (observation, odometry)

    x = rnn(inputs=input, initial_state=state)
    output, out_state = x[:2], x[2:]

    return keras.Model(
        inputs=([observation, odometry], state),
        outputs=([output, out_state])
    )
