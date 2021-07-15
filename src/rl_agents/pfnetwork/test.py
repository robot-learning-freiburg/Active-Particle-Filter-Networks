#!/usr/bin/env python3

import os
import pfnet
import numpy as np
from tqdm import tqdm
import tensorflow as tf
from tensorflow import keras
from datetime import datetime
import preprocess, arguments, pfnet_loss

def run_testing(params):
    """
    run testing with the parsed arguments
    """

    root_dir = os.path.expanduser(params.root_dir)
    test_dir = os.path.join(root_dir, 'test')

    batch_size = params.batch_size
    num_particles = params.num_particles
    trajlen = params.trajlen
    num_test_batches = params.num_test_samples // batch_size

    # testing data
    test_ds = preprocess.get_dataflow(params.testfiles, params.batch_size, is_training=False)

    # pf model
    model = pfnet.pfnet_model(params)

    # load model from checkpoint file
    if params.load:
        print("=====> Loading model from " + params.load)
        model.load_weights(params.load)

    model = tf.function(model)
    print("=====> wrapped pfnet in tf.graph")

    # Logging
    summaries_flush_secs=10
    test_summary_writer = tf.compat.v2.summary.create_file_writer(
        test_dir, flush_millis=summaries_flush_secs * 1000)
    print(params)

    with test_summary_writer.as_default():
        # run testing over all test samples in an epoch
        mse_list = []
        init_mse_list = []
        success_list = []
        test_itr = test_ds.as_numpy_iterator()
        for eps_idx in tqdm(range(num_test_batches)):
            raw_eval_record = next(test_itr)
            data_sample = preprocess.transform_raw_record(raw_eval_record, params)

            observation = tf.convert_to_tensor(data_sample['observation'], dtype=tf.float32)
            odometry = tf.convert_to_tensor(data_sample['odometry'], dtype=tf.float32)
            true_states = tf.convert_to_tensor(data_sample['true_states'], dtype=tf.float32)
            global_map = tf.convert_to_tensor(data_sample['global_map'], dtype=tf.float32)
            init_particles = tf.convert_to_tensor(data_sample['init_particles'], dtype=tf.float32)
            init_particle_weights = tf.constant(np.log(1.0/float(num_particles)),
                                        shape=(batch_size, num_particles), dtype=tf.float32)
            # compute initial loss
            init_loss_dict = pfnet_loss.compute_loss(tf.expand_dims(init_particles, axis=1),
                            tf.expand_dims(init_particle_weights, axis=1),
                            tf.expand_dims(true_states[:, 0], axis=1),
                            params.map_pixel_in_meters)
            # we have squared differences along the trajectory
            init_mse = np.mean(init_loss_dict['coords'])
            init_mse_list.append(init_mse)

            # start trajectory with initial particles and weights
            state = [init_particles, init_particle_weights, global_map]

            # if stateful: reset RNN s.t. initial_state is set to initial particles and weights
            # if non-stateful: pass the state explicity every step
            if params.stateful:
                model.layers[-1].reset_states(state)    # RNN layer

            input = [observation, odometry]
            model_input = (input, state)

            # forward pass
            output, state = model(model_input, training=False)

            # compute loss
            particle_states, particle_weights = output
            loss_dict = pfnet_loss.compute_loss(particle_states, particle_weights, true_states, params.map_pixel_in_meters)

            # we have squared differences along the trajectory
            mse = np.mean(loss_dict['coords'])
            mse_list.append(mse)

            # log mse (in meters)
            print(f'eps:{eps_idx} mean mse: {mse}')
            tf.summary.scalar('eps_mean_rmse', np.sqrt(mse), step=eps_idx)
            tf.summary.scalar('eps_final_rmse', np.sqrt(loss_dict['coords'][0][-1]), step=eps_idx)

            # localization is successfull if the rmse error is below 1m for the last 25% of the trajectory
            successful = np.all(loss_dict['coords'][-trajlen // 4:] < 1.0 ** 2)  # below 1 meter
            success_list.append(successful)

        # report results
        init_mean_rmse = np.mean(np.sqrt(init_mse_list)) * 100
        total_init_rmse = np.sqrt(np.mean(init_mse_list)) * 100
        print(f'Initial Mean RMSE (average RMSE per trajectory) = {init_mean_rmse:03.3f} cm')
        print(f'Overall Initial RMSE (reported value) = {total_init_rmse:03.3f} cm')

        mean_rmse = np.mean(np.sqrt(mse_list)) * 100
        total_rmse = np.sqrt(np.mean(mse_list)) * 100
        mean_success = np.mean(np.array(success_list, 'i')) * 100
        print(f'Mean RMSE (average RMSE per trajectory) = {mean_rmse:03.3f} cm')
        print(f'Overall RMSE (reported value) = {total_rmse:03.3f} cm')
        print(f'Success rate = {mean_success:03.3f} %')

    print('evaluation finished')

if __name__ == '__main__':
    params = arguments.parse_args()

    current_time = datetime.now().strftime("%Y%m%d-%H%M%S")
    params.root_dir = os.path.join(params.logpath, f'run_{current_time}')

    run_testing(params)
