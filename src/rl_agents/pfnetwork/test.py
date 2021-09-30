#!/usr/bin/env python3

import os
import cv2
import pfnet
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg
import render
from pathlib import Path
import numpy as np
from tqdm import tqdm
import tensorflow as tf
from tensorflow import keras
from datetime import datetime
import preprocess, arguments, pfnet_loss

def store_results(eps_idx, floor_map, org_map_shape, particle_states, particle_weights, true_states, params):
    """
    Store results as video
    """
    trajlen = params.trajlen
    b_idx = 0

    fig = plt.figure(figsize=(7, 7))
    plt_ax = fig.add_subplot(111)
    canvas = FigureCanvasAgg(fig)

    lin_weights = tf.nn.softmax(particle_weights, axis=-1)
    est_states = tf.math.reduce_sum(tf.math.multiply(
        particle_states[:, :, :, :], lin_weights[:, :, :, None]
    ), axis=2)

    # normalize between [-pi, +pi]
    part_x, part_y, part_th = tf.unstack(est_states, axis=-1, num=3)  # (k, 3)
    part_th = tf.math.floormod(part_th + np.pi, 2 * np.pi) - np.pi
    est_states = tf.stack([part_x, part_y, part_th], axis=-1)

    # plot map
    floor_map = floor_map[b_idx].numpy()  # [H, W, 1]
    pad_map_shape = floor_map.shape
    o_map_shape = org_map_shape[b_idx]

    # HACK:
    plt_ax.set_yticks(np.arange(0, pad_map_shape[0], pad_map_shape[0]//10))
    plt_ax.set_xticks(np.arange(0, pad_map_shape[1], pad_map_shape[1]//10))

    map_plt = render.draw_floor_map(floor_map, o_map_shape, plt_ax, None)

    images = []
    gt_plt = {
        'robot_position': None,
        'robot_heading': None,
    }
    est_plt = {
        'robot_position': None,
        'robot_heading': None,
        'particles': None,
    }
    for traj in range(trajlen):
        true_state = true_states[:, traj, :]
        est_state = est_states[:, traj, :]
        particle_state = particle_states[:, traj, :, :]
        lin_weight = lin_weights[:, traj, :]
        particle_weight = particle_weights[:, traj, :]

        # plot true robot pose
        position_plt, heading_plt = gt_plt['robot_position'], gt_plt['robot_heading']
        gt_plt['robot_position'], gt_plt['robot_heading'] = render.draw_robot_pose(
            true_state[b_idx], '#7B241C', pad_map_shape, plt_ax,
            position_plt, heading_plt, plt_path=True)

        # plot est robot pose
        position_plt, heading_plt = est_plt['robot_position'], est_plt['robot_heading']
        est_plt['robot_position'], est_plt['robot_heading'] = render.draw_robot_pose(
            est_state[b_idx], '#515A5A', pad_map_shape, plt_ax,
            position_plt, heading_plt, plt_path=False)

        # plot est pose particles
        particles_plt = est_plt['particles']
        est_plt['particles'] = render.draw_particles_pose(
            particle_state[b_idx], particle_weight[b_idx],
            pad_map_shape, particles_plt)

        plt_ax.legend([gt_plt['robot_position'], est_plt['robot_position']], ["gt_pose", "est_pose"])

        canvas.draw()
        img = np.array(canvas.renderer._renderer)
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        images.append(img)

    # gt_pose_mts = np.array([*env.scene.map_to_world(true_state[b_idx][:2]), true_state[b_idx][2]])
    # est_pose_mts = np.array([*env.scene.map_to_world(est_state[b_idx][:2]), est_state[b_idx][2]])
    # print(f'{eps_idx} End True Pose: {gt_pose_mts}, End Estimated Pose: {est_pose_mts} in mts')
    print(f'{eps_idx} End True Pose: {true_state[b_idx]}, End Estimated Pose: {est_state[b_idx]} in px')

    size = (images[0].shape[0], images[0].shape[1])
    out = cv2.VideoWriter(
            os.path.join(params.out_folder, f'result_{eps_idx}.avi'),
            cv2.VideoWriter_fourcc(*'XVID'), 30, size)

    for i in range(len(images)):
        out.write(images[i])
        # cv2.imwrite(params.out_folder + f'result_img_{i}.png', images[i])
    out.release()


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
            org_map_shapes = data_sample['org_map_shapes']
            init_particles = tf.convert_to_tensor(data_sample['init_particles'], dtype=tf.float32)
            init_particle_weights = tf.constant(np.log(1.0/float(num_particles)),
                                        shape=(batch_size, num_particles), dtype=tf.float32)
            # compute initial loss
            init_loss_dict = pfnet_loss.compute_mse_loss(tf.expand_dims(init_particles, axis=1),
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
            loss_dict = pfnet_loss.compute_mse_loss(particle_states, particle_weights, true_states, params.map_pixel_in_meters)

            # we have squared differences along the trajectory
            mse = np.mean(loss_dict['coords'])
            mse_list.append(mse)

            # log mse (in meters)
            print(f'eps:{eps_idx} mean mse: {mse}')
            tf.summary.scalar('eps_mean_rmse', np.sqrt(mse), step=eps_idx)
            tf.summary.scalar('eps_final_rmse', np.sqrt(loss_dict['coords'][0][-1]), step=eps_idx)

            # localization is successfull if the rmse error is below 1m for the last 25% of the trajectory
            assert list(loss_dict['coords'].shape) == [batch_size, trajlen]
            successful = np.all(loss_dict['coords'][:, -trajlen // 4:] < 1.0 ** 2)  # below 1 meter
            success_list.append(successful)

            if params.store_results:
                # store results as video
                params.out_folder = os.path.join(params.root_dir, f'output')
                Path(params.out_folder).mkdir(parents=True, exist_ok=True)
                store_results(eps_idx, global_map, org_map_shapes, particle_states, particle_weights, true_states, params)

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
    params.store_results = True

    run_testing(params)
