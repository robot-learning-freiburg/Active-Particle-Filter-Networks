#!/usr/bin/env python3

import os
import pfnet
import numpy as np
from tqdm import tqdm
import tensorflow as tf
from tensorflow import keras
from datetime import datetime
import preprocess, arguments, pfnet_loss

def run_training(params):
    """
    run training with the parsed arguments
    """

    root_dir = os.path.expanduser(params.root_dir)
    train_dir = os.path.join(root_dir, 'train')
    eval_dir = os.path.join(root_dir, 'eval')

    batch_size = params.batch_size
    num_particles = params.num_particles
    trajlen = params.trajlen
    num_train_batches = params.num_train_samples // batch_size
    num_eval_batches = params.num_eval_samples // batch_size

    # training data
    train_ds = preprocess.get_dataflow(params.trainfiles, params.batch_size, params.s_buffer_size, is_training=True)

    # evaluation data
    eval_ds = preprocess.get_dataflow(params.evalfiles, params.batch_size, params.s_buffer_size, is_training=True)

    # pf model
    model = pfnet.pfnet_model(params)

    # load model from checkpoint file
    if params.load:
        print("=====> Loading model from " + params.load)
        model.load_weights(params.load)

    # Adam optimizer.
    optimizer = tf.optimizers.Adam(learning_rate=params.learningrate)

    # Define metrics
    train_loss = keras.metrics.Mean('train_loss', dtype=tf.float32)
    eval_loss = keras.metrics.Mean('eval_loss', dtype=tf.float32)

    # Logging
    summaries_flush_secs=10
    train_summary_writer = tf.compat.v2.summary.create_file_writer(
        train_dir, flush_millis=summaries_flush_secs * 1000)
    eval_summary_writer = tf.compat.v2.summary.create_file_writer(
        eval_dir, flush_millis=summaries_flush_secs * 1000)
    print(params)

    # Recommended: wrap to tf.graph for better performance
    @tf.function
    def train_step(model_input, true_states):
        """ Run one training step """

        # enable auto-differentiation
        with tf.GradientTape() as tape:
            # forward pass
            output, state = model(model_input, training=True)

            # compute loss
            particle_states, particle_weights = output
            loss_dict = pfnet_loss.compute_loss(particle_states, particle_weights, true_states, params.map_pixel_in_meters)
            loss_pred = loss_dict['pred']

        # compute gradients of the trainable variables with respect to the loss
        gradients = tape.gradient(loss_pred, model.trainable_weights)
        gradients = list(zip(gradients, model.trainable_weights))

        # run one step of gradient descent
        optimizer.apply_gradients(gradients)
        train_loss(loss_pred)  # overall trajectory loss

    # Recommended: wrap to tf.graph for better performance
    @tf.function
    def eval_step(model_input, true_states):
        """ Run one evaluation step """
        # forward pass
        output, state = model(model_input, training=False)

        # compute loss
        particle_states, particle_weights = output
        loss_dict = pfnet_loss.compute_loss(particle_states, particle_weights, true_states, params.map_pixel_in_meters)
        loss_pred = loss_dict['pred']

        eval_loss(loss_pred)  # overall trajectory loss


    # repeat for a fixed number of epochs
    for epoch in range(params.epochs):
        train_itr = train_ds.as_numpy_iterator()
        # run training over all training samples in an epoch
        for train_idx in tqdm(range(num_train_batches)):
            raw_train_record = next(train_itr)
            data_sample = preprocess.transform_raw_record(raw_train_record, params)

            observation = tf.convert_to_tensor(data_sample['observation'], dtype=tf.float32)
            odometry = tf.convert_to_tensor(data_sample['odometry'], dtype=tf.float32)
            true_states = tf.convert_to_tensor(data_sample['true_states'], dtype=tf.float32)
            global_map = tf.convert_to_tensor(data_sample['global_map'], dtype=tf.float32)
            init_particles = tf.convert_to_tensor(data_sample['init_particles'], dtype=tf.float32)
            init_particle_weights = tf.constant(np.log(1.0/float(num_particles)),
                                        shape=(batch_size, num_particles), dtype=tf.float32)

            # start trajectory with initial particles and weights
            state = [init_particles, init_particle_weights, global_map]

            # if stateful: reset RNN s.t. initial_state is set to initial particles and weights
            # if non-stateful: pass the state explicity every step
            if params.stateful:
                model.layers[-1].reset_states(state)    # RNN layer

            # run training over trajectory
            input = [observation, odometry]
            model_input = (input, state)

            train_step(model_input, true_states)

        # log epoch training stats
        with train_summary_writer.as_default():
            tf.summary.scalar('loss', train_loss.result(), step=epoch)

        # Save the weights
        print("=====> saving trained model ")
        model.save_weights(
            os.path.join(
                train_dir,
                f'chks/checkpoint_{epoch}_{train_loss.result():03.3f}/pfnet_checkpoint'
            )
        )

        if params.run_evaluation:
            eval_itr = eval_ds.as_numpy_iterator()
            # run evaluation over all eval samples in an epoch
            for eval_idx in tqdm(range(num_eval_batches)):
                raw_eval_record = next(eval_itr)
                data_sample = preprocess.transform_raw_record(raw_eval_record, params)

                observation = tf.convert_to_tensor(data_sample['observation'], dtype=tf.float32)
                odometry = tf.convert_to_tensor(data_sample['odometry'], dtype=tf.float32)
                true_states = tf.convert_to_tensor(data_sample['true_states'], dtype=tf.float32)
                global_map = tf.convert_to_tensor(data_sample['global_map'], dtype=tf.float32)
                init_particles = tf.convert_to_tensor(data_sample['init_particles'], dtype=tf.float32)
                init_particle_weights = tf.constant(np.log(1.0/float(num_particles)),
                                            shape=(batch_size, num_particles), dtype=tf.float32)

                # start trajectory with initial particles and weights
                state = [init_particles, init_particle_weights, global_map]

                # if stateful: reset RNN s.t. initial_state is set to initial particles and weights
                # if non-stateful: pass the state explicity every step
                if params.stateful:
                    model.layers[-1].reset_states(state)    # RNN layer

                # run evaluation over trajectory
                input = [observation, odometry]
                model_input = (input, state)

                eval_step(model_input, true_states)

            # log epoch evaluation stats
            with eval_summary_writer.as_default():
                tf.summary.scalar('loss', eval_loss.result(), step=epoch)

            # Save the weights
            print("=====> saving evaluation model ")
            model.save_weights(
                os.path.join(
                    eval_dir,
                    f'chks/checkpoint_{epoch}_{eval_loss.result():03.3f}/pfnet_checkpoint'
                )
            )

        print(f'Epoch {epoch}, train loss: {train_loss.result():03.3f}, eval loss: {eval_loss.result():03.3f}')

        # Reset the metrics at the start of the next epoch
        train_loss.reset_states()
        eval_loss.reset_states()

    print('training finished')

if __name__ == '__main__':
    params = arguments.parse_args()

    current_time = datetime.now().strftime("%Y%m%d-%H%M%S")
    params.root_dir = os.path.join(params.logpath, f'run_{current_time}')

    params.run_evaluation = True
    params.s_buffer_size = 500

    run_training(params)
