#!/usr/bin/env python3

import tensorflow as tf
from tensorflow import keras

##### helper fuctions for constructing layers ####
def conv2_layer(
    filters, kernel_size,
    activation=None, padding='same',
    strides=(1, 1), dilation_rate=(1, 1),
    data_format='channels_last', use_bias=False):

    # initializer = tf.random_normal_initializer(0., 0.02)
    initializer = keras.initializers.VarianceScaling()
    regularizer = keras.regularizers.L2(1.0)

    result = keras.layers.Conv2D(
                filters, kernel_size, strides, padding, data_format,
                dilation_rate, activation=activation, use_bias=use_bias,
                kernel_initializer=initializer, kernel_regularizer=regularizer
    )

    return result

def locallyconn2_layer(
    filters, kernel_size,
    activation=None, padding='same',
    strides=(1, 1), data_format='channels_last', use_bias=False):

    initializer = keras.initializers.VarianceScaling()
    regularizer = keras.regularizers.L2(1.0)

    result = keras.layers.LocallyConnected2D(
                filters, kernel_size, strides, padding, data_format,
                activation=activation, use_bias=use_bias,
                kernel_initializer=initializer, kernel_regularizer=regularizer
    )

    return result

def dense_layer(units, activation=None, use_bias=True):

    initializer = keras.initializers.VarianceScaling()
    regularizer = keras.regularizers.L2(1.0)

    result = keras.layers.Dense(
                units, activation=activation, use_bias=use_bias,
                kernel_regularizer=regularizer
    )

    return result

##### custom fuctions for constructing models ####
def map_encoder(map_shape=[28, 28, 1]):

    local_maps = keras.Input(shape=map_shape, name="local_maps")   # (bs*np, 28, 28, 1)
    assert local_maps.get_shape().as_list()[1:3] == [28, 28]
    x = local_maps

    conv_stack = [
        conv2_layer(24, 3, use_bias=True)(x),   # (bs*np, 28, 28, 24)
        conv2_layer(16, 5, use_bias=True)(x),   # (bs*np, 28, 28, 16)
        conv2_layer(8, 7, use_bias=True)(x),    # (bs*np, 28, 28, 8)
        conv2_layer(8, 7, dilation_rate=(2, 2), use_bias=True)(x),  # (bs*np, 28, 28, 8)
        conv2_layer(8, 7, dilation_rate=(3, 3), use_bias=True)(x),  # (bs*np, 28, 28, 8)
    ]
    x = tf.concat(conv_stack, axis=-1)  # (bs*np, 28, 28, 64)

    x = keras.layers.LayerNormalization(axis=-1)(x)
    x = keras.layers.ReLU()(x)
    assert x.get_shape().as_list()[1:4] == [28, 28, 64]
    x = keras.layers.MaxPool2D(
            pool_size=(3, 3), strides=(2, 2), padding='same',
            data_format='channels_last')(x) # (bs*np, 14, 14, 64)

    conv_stack = [
        conv2_layer(4, 3, use_bias=True)(x),   # (bs*np, 14, 14, 4)
        conv2_layer(4, 5, use_bias=True)(x),   # (bs*np, 14, 14, 4)
    ]
    x = tf.concat(conv_stack, axis=-1)  # (bs*np, 14, 14, 8)

    x = keras.layers.LayerNormalization(axis=-1)(x)
    x = keras.layers.ReLU()(x)
    assert x.get_shape().as_list()[1:4] == [14, 14, 8]

    return keras.Model(inputs=local_maps, outputs=x, name="map_encoder")

def obs_encoder(obs_shape=[56, 56, 3]):

    observations = keras.Input(shape=obs_shape, name="observations")   # (bs, 56, 56, 3)
    assert observations.get_shape().as_list()[1:3] == [56, 56]
    x = observations

    conv_stack = [
        conv2_layer(128, 3, use_bias=True)(x),  # (bs, 56, 56, 128)
        conv2_layer(128, 5, use_bias=True)(x),  # (bs, 56, 56, 128)
        conv2_layer(64, 5, dilation_rate=(2, 2), use_bias=True)(x), # (bs, 56, 56, 64)
        conv2_layer(64, 5, dilation_rate=(4, 4), use_bias=True)(x), # (bs, 56, 56, 64)
    ]
    x = tf.concat(conv_stack, axis=-1)  # (bs, 56, 56, 384)

    x = keras.layers.MaxPool2D(
            pool_size=(3, 3), strides=(2, 2), padding='same',
            data_format='channels_last')(x) # (bs, 28, 28, 384)
    x = keras.layers.LayerNormalization(axis=-1)(x)
    x = keras.layers.ReLU()(x)
    assert x.get_shape().as_list()[1:4] == [28, 28, 384]

    x = conv2_layer(16, 3, use_bias=True)(x)    # (bs, 28, 28, 16)

    x = keras.layers.MaxPool2D(
            pool_size=(3, 3), strides=(2, 2), padding='same',
            data_format='channels_last')(x) # (bs, 14, 14, 16)
    x = keras.layers.LayerNormalization(axis=-1)(x)
    x = keras.layers.ReLU()(x)
    assert x.get_shape().as_list()[1:4] == [14, 14, 16]

    return keras.Model(inputs=observations, outputs=x, name="obs_encoder")

def map_obs_encoder():

    joint_matrix = keras.Input(shape=[14, 14, 24], name="map_obs_features")   # (bs*np, 14, 14, 24)
    assert joint_matrix.get_shape().as_list()[1:4] == [14, 14, 24]
    x = joint_matrix

    # pad manually to match different kernel sizes
    x_pad1 = tf.pad(x, paddings=tf.constant([[0, 0], [1, 1,], [1, 1], [0, 0]]), constant_values=0)  # (bs*np, 16, 16, 24)

    conv_stack = [
        locallyconn2_layer(8, 3, activation='relu', padding='valid', use_bias=True)(x),  # (bs*np, 12, 12, 8)
        locallyconn2_layer(8, 5, activation='relu', padding='valid', use_bias=True)(x_pad1),  # (bs*np, 12, 12, 8)
    ]
    x = tf.concat(conv_stack, axis=-1)   # (bs*np, 12, 12, 16)

    x = keras.layers.MaxPool2D(
            pool_size=(3, 3), strides=(2, 2), padding='valid',
            data_format='channels_last')(x) # (bs, 5, 5, 16)
    assert x.get_shape().as_list()[1:4] == [5, 5, 16]

    return keras.Model(inputs=joint_matrix, outputs=x, name="map_obs_encoder")

def likelihood_estimator():

    joint_vector = keras.Input(shape=[400], name="map_obs_joint_features")   # (bs*np, 5 * 5 * 16)
    assert joint_vector.get_shape().as_list()[1] == 400
    x = joint_vector

    x = dense_layer(1, use_bias=True)(x)   # (bs*np, 1)
    assert x.get_shape().as_list()[1] == 1

    return keras.Model(inputs=joint_vector, outputs=x, name="likelihood_estimator")
