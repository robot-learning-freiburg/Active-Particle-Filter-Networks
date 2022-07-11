#!/usr/bin/env python3

import os
import numpy as np
from tqdm import tqdm
import tensorflow as tf
from tensorflow import keras
from datetime import datetime
import wandb
from pfnetwork import pfnet, preprocess, arguments

WANDB_PROJECT = "pfnet"


def calc_success(loss_dict):
    """localization is successfull if the rmse error is below 1m for the last 25% of the trajectory"""
    assert len(loss_dict['coords'].shape) == 2, loss_dict['coords'].shape
    traj_len = loss_dict['coords'].shape[1]
    last_x = traj_len // 4
    # loss_dict['coords'] is the squared error
    return np.all(tf.sqrt(loss_dict['coords'][:, -last_x:]) < 1.0, 1)  # below 1 meter


def calc_metrics(loss_dict, prefix: str) -> dict:
    metrics = {f"{prefix}/{k}": tf.reduce_mean(tf.cast(v, tf.float32)) for k, v in loss_dict.items()}
    metrics[f"{prefix}/success"] = tf.reduce_mean(tf.cast(calc_success(loss_dict), tf.float32))

    metrics[f"{prefix}/rmse_mean"] = tf.reduce_mean(tf.sqrt(loss_dict['coords']))
    metrics[f"{prefix}/rmse_final"] = tf.reduce_mean(tf.sqrt(loss_dict['coords'][..., -1]))

    metrics[f"{prefix}/rmse_orient_mean"] = tf.reduce_mean(tf.sqrt(loss_dict['orient']))
    metrics[f"{prefix}/rmse_orient_final"] = tf.reduce_mean(tf.sqrt(loss_dict['orient'][..., -1]))

    metrics[f"{prefix}/pred_final"] = tf.reduce_mean(loss_dict['pred'][..., -1])

    return metrics


def stack_loss_dicts(loss_dicts: list, axis: int, concat: bool = False):
    stacked_loss_dicts = {}
    for k in loss_dicts[0].keys():
        if concat:
            stacked_loss_dicts[k] = tf.concat([d[k] for d in loss_dicts], axis)
        else:
            stacked_loss_dicts[k] = tf.stack([d[k] for d in loss_dicts], axis)
    return stacked_loss_dicts


def init_pfnet_model(params, is_igibson: bool):
    if params.multiple_gpus:
        strategy = tf.distribute.MirroredStrategy()
        with strategy.scope():
            pfnet_model = pfnet.pfnet_model(params, is_igibson=is_igibson)
    else:
        pfnet_model = pfnet.pfnet_model(params, is_igibson=is_igibson)

    # load model from checkpoint file
    if params.pfnet_loadpath:
        pfnet_model.load_weights(params.pfnet_loadpath)
        print("=====> loaded pf model checkpoint " + params.pfnet_loadpath)

    if params.use_tf_function:
        print("=====> wrapped pfnet in tf.graph")
        pfnet_model = tf.function(pfnet_model)

    return pfnet_model


# Recommended: wrap to tf.graph for better performance
@tf.function
def train_step(data, model, optimizer, train_loss, map_pixel_in_meters: float):
    """ Run one training step """

    state = [data['init_particles'], data['init_particle_weights'], data['global_map']]
    input = [data['observation'], data['odometry']]

    # enable auto-differentiation
    with tf.GradientTape() as tape:
        # forward pass over
        output, state = model((input, state), training=True)
        loss_dict = pfnet.PFCell.compute_mse_loss(particles=output[0], particle_weights=output[1],
                                                  true_states=data['true_states'],
                                                  trav_map_resolution=map_pixel_in_meters)

        loss_pred = tf.reduce_mean(loss_dict['pred'])

    # compute gradients of the trainable variables with respect to the loss
    gradients = tape.gradient(loss_pred, model.trainable_variables)
    gradients = list(zip(gradients, model.trainable_variables))

    # run one step of gradient descent
    optimizer.apply_gradients(gradients)
    train_loss(loss_pred)  # overall trajectory loss

    return loss_dict, output, state


def vis_output(env, output, state, data):
    env.env_plots = env._get_empty_env_plots()
    env.reset_variables()
    env.reset()

    b = 0
    for t in range(data['observation'].shape[1]):
        env.render(particles=output[0][b:b + 1, t],
                   particle_weights=output[1][b:b + 1, t],
                   floor_map=state[2][b],
                   gt_pose=data['true_states'][b, t],
                   observation=data['observation'][b, t],
                   current_step=t)


# Recommended: wrap to tf.graph for better performance
@tf.function
def eval_step(data, model, eval_loss, map_pixel_in_meters: float):
    """ Run one evaluation step """
    # forward pass
    state = [data['init_particles'], data['init_particle_weights'], data['global_map']]
    input = [data['observation'], data['odometry']]
    output, state = model((input, state), training=False)
    loss_dict = pfnet.PFCell.compute_mse_loss(particles=output[0], particle_weights=output[1],
                                              true_states=data['true_states'],
                                              trav_map_resolution=map_pixel_in_meters)

    loss_pred = tf.reduce_mean(loss_dict['pred'])
    eval_loss(loss_pred)  # overall trajectory loss

    return loss_dict, output, state


def prepare_data(raw_data_record, params):
    data_sample = preprocess.transform_raw_record(raw_data_record, params)

    processed = {}
    for k in ['observation', 'odometry', 'true_states', 'global_map', 'init_particles']:
        processed[k] = tf.convert_to_tensor(data_sample[k], dtype=tf.float32)
    processed['init_particle_weights'] = tf.constant(np.log(1.0 / float(params.num_particles)),
                                                     shape=(params.batch_size, params.num_particles),
                                                     dtype=tf.float32)
    return processed


def run_training(params):
    """
    run training with the parsed arguments
    """

    root_dir = os.path.expanduser(params.root_dir)
    train_dir = os.path.join(root_dir, 'train')

    # data
    train_ds = preprocess.get_dataflow(params.trainfiles, params.batch_size, params.s_buffer_size, is_training=True)
    eval_ds = preprocess.get_dataflow(params.evalfiles, params.batch_size, params.s_buffer_size, is_training=True)
    test_ds = preprocess.get_dataflow(params.testfiles, params.batch_size, params.s_buffer_size, is_training=False)

    # pf model
    model = init_pfnet_model(params, is_igibson=False)

    # load model from checkpoint file
    if params.pfnet_loadpath:
        print("=====> Loading model from " + params.pfnet_loadpath)
        model.load_weights(params.pfnet_loadpath)

    # Adam optimizer.
    optimizer = tf.optimizers.Adam(learning_rate=params.learning_rate)

    # Define metrics
    train_loss = keras.metrics.Mean('train_loss', dtype=tf.float32)
    eval_loss = keras.metrics.Mean('eval_loss', dtype=tf.float32)

    print(params)

    # repeat for a fixed number of epochs
    for epoch in range(params.epochs):
        train_itr = train_ds.as_numpy_iterator()
        train_loss_dicts = []
        # run training over all training samples in an epoch
        for train_idx in tqdm(range(params.num_train_batches), desc=f"Epoch {epoch}/{params.epochs}"):
            raw_train_record = next(train_itr)
            processed_data = prepare_data(raw_train_record, params=params)

            # model.reset_supervised(processed_data['init_particles'], processed_data['init_particle_weights'])
            train_loss_dict, train_output, train_state = train_step(data=processed_data, model=model,
                                                                    optimizer=optimizer, train_loss=train_loss,
                                                                    map_pixel_in_meters=params.map_pixel_in_meters)
            train_loss_dicts.append(train_loss_dict)

        if params.run_evaluation:
            eval_itr = eval_ds.as_numpy_iterator()
            eval_loss_dicts = []
            # run evaluation over all eval samples in an epoch
            for eval_idx in tqdm(range(params.num_eval_batches), desc=f"Epoch {epoch}/{params.epochs}"):
                raw_eval_record = next(eval_itr)
                processed_data = prepare_data(raw_eval_record, params=params)

                eval_loss_dict, eval_output, eval_state = eval_step(data=processed_data,
                                                                    model=model,
                                                                    eval_loss=eval_loss,
                                                                    map_pixel_in_meters=params.map_pixel_in_meters)
                eval_loss_dicts.append(eval_loss_dict)

        if epoch % 5 == 0:
            print("=====> saving trained model ")
            model.save_weights(
                os.path.join(train_dir, f'chks/checkpoint_{epoch}_{eval_loss.result():03.3f}/pfnet_checkpoint'))
            # tf.saved_model.save(model, os.path.join(train_dir, f'chks/checkpoint_{epoch}_{train_loss.result():03.3f}/pfnet_checkpoint'))

        train_loss_dicts = stack_loss_dicts(train_loss_dicts, 0, concat=True)
        eval_loss_dicts = stack_loss_dicts(eval_loss_dicts, 0, concat=True)
        train_metrics = calc_metrics(train_loss_dicts, prefix='train')
        eval_metrics = calc_metrics(eval_loss_dicts, prefix='eval')
        wandb.log(dict(train_metrics, **eval_metrics), step=epoch)

        print(f'Epoch {epoch}, train loss: {train_loss.result():03.3f}, eval loss: {eval_loss.result():03.3f}')

        # Reset the metrics at the start of the next epoch
        train_loss.reset_states()
        eval_loss.reset_states()

    # test set evaluation
    test_itr = test_ds.as_numpy_iterator()
    test_loss_dicts = []
    # run evaluation over all eval samples in an epoch
    for eval_idx in range(params.num_test_batches):
        raw_eval_record = next(test_itr)
        processed_data = prepare_data(raw_eval_record, params=params)

        test_loss_dict = eval_step(data=processed_data, model=model, eval_loss=eval_loss)
        test_loss_dicts.append(test_loss_dict)

    test_loss_dicts = stack_loss_dicts(test_loss_dicts, 0, concat=True)
    test_metrics = calc_metrics(test_loss_dicts, prefix='test')
    wandb.log(test_metrics)

    print('training finished')


if __name__ == '__main__':
    params = arguments.parse_common_args('house3d')

    current_time = datetime.now().strftime("%Y%m%d-%H%M%S")
    run_name = f'run_{current_time}'
    params.root_dir = os.path.join(params.logpath, run_name)

    run = wandb.init(config=params, name=run_name, project=WANDB_PROJECT, sync_tensorboard=True)

    params.run_evaluation = True

    run_training(params)
