#!/usr/bin/env python3

import cv2
import numpy as np
import pybullet as p
import tensorflow as tf
import os
from PIL import Image

from igibson.utils.assets_utils import get_scene_path

ORIG_IGIBSON_MAP_RESOLUTION = 0.01


def get_floor_map(scene_id, floor_num, trav_map_resolution, trav_map_erosion, pad_map_size):
    """
    Get the scene floor map (traversability map + obstacle map)

    :param str: scene id
    :param int: task floor number
    :return ndarray: floor map of current scene (H, W, 1)
    """
    # NOTE: these values might be hardcoded in a place, so if changing the task config, also change the hardcoded values!
    assert trav_map_resolution == 0.1, trav_map_resolution
    assert trav_map_erosion == 2, trav_map_erosion

    obstacle_map = np.array(Image.open(
        os.path.join(get_scene_path(scene_id), f'floor_{floor_num}.png'))
    )

    trav_map = np.array(Image.open(
        os.path.join(get_scene_path(scene_id), f'floor_trav_{floor_num}.png'))
    )

    # 0: free / unexplored / outside map, 1: obstacle
    # NOTE: shouldn't mather that outside the map is not unexplored, because this will always be unexplored in the lidar scan and so will always be masked out
    occupancy_map = np.zeros_like(trav_map)
    # occupancy_map[trav_map == 0] = 2
    occupancy_map[obstacle_map == 0] = 1

    height, width = trav_map.shape
    resize = (int(width * ORIG_IGIBSON_MAP_RESOLUTION / trav_map_resolution),
              int(height * ORIG_IGIBSON_MAP_RESOLUTION / trav_map_resolution))
    occupancy_map_small = cv2.resize(occupancy_map.astype(float), resize, interpolation=cv2.INTER_AREA)
    occupancy_map_small = occupancy_map_small[:, :, np.newaxis]
    # plt.imshow(occupancy_map_small > 0.1);
    # plt.show()

    trav_map[obstacle_map == 0] = 0

    # use same rescaling as in iGibsonEnv
    # height, width = trav_map.shape
    # resize = int(height * self.map_pixel_in_meters / self.trav_map_resolution)
    trav_map = cv2.resize(trav_map, resize)
    trav_map_erosion = trav_map_erosion
    trav_map = cv2.erode(trav_map, np.ones((trav_map_erosion, trav_map_erosion)))
    # 1: traversible
    trav_map[trav_map < 255] = 0
    trav_map[trav_map == 255] = 1
    trav_map = trav_map[:, :, np.newaxis]

    # right zero-pad floor/obstacle map
    if pad_map_size is not None:
        trav_map = pad_images(trav_map, pad_map_size)
        occupancy_map_small = pad_images(occupancy_map_small, pad_map_size)

    return occupancy_map_small, occupancy_map_small.shape, trav_map


def get_random_points_map(npoints, trav_map, true_mask=None):
    """
    Sample a random point on the given floor number. If not given, sample a random floor number.

    :param floor: floor number
    :return floor: floor number
    :return point: randomly sampled point in [x, y, z]
    """
    trav_map = trav_map.copy()
    if true_mask is not None:
        trav_map *= true_mask

    trav_space = np.where(trav_map == 1)
    idx = np.random.randint(0, high=trav_space[0].shape[0], size=npoints)
    orn = np.random.uniform(0, np.pi * 2, size=npoints)

    xy_map = np.stack([trav_space[0][idx], trav_space[1][idx], orn], 1)

    return xy_map


def get_random_particles(num_particles, particles_distr, robot_pose, trav_map, particles_cov, particles_range=100):
    """
    Sample random particles based on the scene

    :param particles_distr: string type of distribution, possible value: [gaussian, uniform]
    :param robot_pose: ndarray indicating the robot pose ([batch_size], 3) in pixel space
        if None, random particle poses are sampled using unifrom distribution
        otherwise, sampled using gaussian distribution around the robot_pose
    :param particles_cov: for tracking Gaussian covariance matrix (3, 3)
    :param num_particles: integer indicating the number of random particles per batch
    :param scene_map: floor map to sample valid random particles
    :param particles_range: limit particles range in pixels centered from robot_pose for uniform distribution

    :return ndarray: random particle poses  (batch_size, num_particles, 3) in pixel space
    """

    assert list(robot_pose.shape[1:]) == [3], f'{robot_pose.shape}'
    assert list(particles_cov.shape) == [3, 3], f'{particles_cov.shape}'

    particles = []
    batches = robot_pose.shape[0]
    if particles_distr == 'uniform':
        # iterate per batch_size
        for b_idx in range(batches):
            # sample_i = 0
            # b_particles = []

            # sample offset from the Gaussian ground truth
            center = np.random.multivariate_normal(mean=robot_pose[b_idx], cov=particles_cov)

            # NOTE: cv2 expects [x, y] order like matplotlib
            mask = np.zeros_like(trav_map)
            cv2.rectangle(mask,
                          (int(center[1] - particles_range), int(center[0] - particles_range)),
                          (int(center[1] + particles_range), int(center[0] + particles_range)),
                          1, -1)
            b_particles = get_random_points_map(npoints=num_particles, trav_map=trav_map, true_mask=mask)

            particles.append(b_particles)
    elif particles_distr == 'gaussian':
        # iterate per batch_size
        for b_idx in range(batches):
            # sample offset from the Gaussian ground truth
            center = np.random.multivariate_normal(mean=robot_pose[b_idx], cov=particles_cov)

            # sample particles from the Gaussian, centered around the offset
            particles.append(np.random.multivariate_normal(mean=center, cov=particles_cov, size=num_particles))
    else:
        raise ValueError(particles_distr)

    particles = np.stack(particles)  # [batch_size, num_particles, 3]
    return particles


def normalize(angle):
    """
    Normalize the angle to [-pi, pi]
    :param float angle: input angle to be normalized
    :return float: normalized angle
    """
    quaternion = p.getQuaternionFromEuler(np.array([0, 0, angle]))
    euler = p.getEulerFromQuaternion(quaternion)
    return euler[2]


def calc_odometry(old_pose, new_pose):
    """
    Calculate the odometry between two poses
    :param ndarray old_pose: pose1 (x, y, theta)
    :param ndarray new_pose: pose2 (x, y, theta)
    :return ndarray: odometry (odom_x, odom_y, odom_th)
    """
    x1, y1, th1 = old_pose
    x2, y2, th2 = new_pose

    abs_x = (x2 - x1)
    abs_y = (y2 - y1)

    th1 = normalize(th1)
    sin = np.sin(th1)
    cos = np.cos(th1)

    th2 = normalize(th2)
    odom_th = normalize(th2 - th1)
    odom_x = cos * abs_x + sin * abs_y
    odom_y = cos * abs_y - sin * abs_x

    odometry = np.array([odom_x, odom_y, odom_th])
    return odometry


def get_discrete_action(max_lin_vel, max_ang_vel):
    """
    Get manual keyboard action
    :return int: discrete action for moving forward/backward/left/right
    """
    key = input('Enter Key: [wsda, nothing]')
    # default stay still
    if key == 'w':
        # forward
        action = np.array([max_lin_vel, 0.])
    elif key == 's':
        # backward
        action = np.array([-max_lin_vel, 0.])
    elif key == 'd':
        # right
        action = np.array([0., -max_ang_vel])
    elif key == 'a':
        # left
        action = np.array([0., max_ang_vel])
    else:
        # do nothing
        action = np.array([0., 0.])

    return action


def obstacle_avoidance(state, max_lin_vel, max_ang_vel):
    """
    Choose action by avoiding obstances which highest preference to move forward
    """
    assert list(state.shape) == [4]
    left, left_front, right_front, right = state  # obstacle (not)present area

    if not left_front and not right_front:
        # move forward
        action = np.array([max_lin_vel, 0.])
    elif not left or not left_front:
        # turn left
        action = np.array([0., max_ang_vel])
    elif not right or not right_front:
        # turn right
        action = np.array([0., -max_ang_vel])
    else:
        # backward
        action = np.array([-max_lin_vel, np.random.uniform(low=-max_ang_vel, high=max_ang_vel)])

    return action


def goal_nav_agent(env, current_pose_pixel, max_lin_vel, max_ang_vel):
    path, dist = env.scene.get_shortest_path(env.task.floor_num, env.map_to_world(current_pose_pixel[:2]),
                                             env.task.target_pos[:2], entire_path=True)
    # rel_path = []
    # for p in path:
    #     rel_path.append(env.task.global_to_local(env, [p[0], p[1], 0]))
    wp = 1 if len(path) > 1 else 0
    rel_subgoal = env.task.global_to_local(env, [path[wp][0], path[wp][1], 0])
    # lin_vel = params.max_lin_vel * rel_path[0] / np.linalg.norm(rel_path[0])
    lin_vel = max_lin_vel
    angle = np.math.atan2(rel_subgoal[1], rel_subgoal[0])
    angular_vel = np.clip(angle, -max_ang_vel, max_ang_vel)
    return lin_vel, angular_vel


def select_action(agent: str, obs, params, env, old_pose):
    if agent == 'manual_agent':
        action = get_discrete_action(env.config["linear_velocity"], env.config["angular_velocity"])
    elif agent == 'rnd_agent':
        action = env.action_space.sample()
    elif agent == 'avoid_agent':
        action = obstacle_avoidance(obs['obstacle_obs'], env.config["linear_velocity"], env.config["angular_velocity"])
    elif agent == 'goalnav_agent':
        action = goal_nav_agent(env=env, current_pose_pixel=old_pose, max_lin_vel=env.config["linear_velocity"],
                                max_ang_vel=env.config["angular_velocity"])
    elif agent == "turn_agent":
        action = np.array([0., env.config["angular_velocity"]])
    else:
        raise ValueError(agent)
    return action


def gather_episode_stats(env, params, sample_particles=False):
    """
    Run the gym environment and collect the required stats
    :param env: igibson env instance
    :param params: parsed parameters
    :param sample_particles: whether or not to sample particles
    :return dict: episode stats data containing:
        odometry, true poses, observation, particles, particles weights, floor map
    """

    agent = params.agent
    trajlen = params.trajlen
    # max_lin_vel = params.max_lin_vel
    # max_ang_vel = params.max_ang_vel

    assert agent in ['manual_agent', 'avoid_agent', 'rnd_agent', 'goalnav_agent']

    odometry = []
    true_poses = []
    rgb_observation = []
    depth_observation = []
    occupancy_grid_observation = []

    obs = env.reset()  # observations are not processed

    # # process [0, 1] ->[0, 255] -> [-1, +1] range
    # rgb = process_raw_image(obs['rgb_obs']*255, resize=(56, 56))
    rgb_observation.append(obs['rgb_obs'])
    #
    # # process [0, 1] ->[0, 100] -> [-1, +1] range
    # depth = process_raw_image(obs['depth_obs']*100, resize=(56, 56))
    depth_observation.append(obs['depth_obs'])
    #
    # # process [0, 0.5, 1]
    # occupancy_grid = np.atleast_3d(decode_image(obs['occupancy_grid'], resize=(56, 56)).astype(np.float32))
    occupancy_grid_observation.append(obs['occupancy_grid'])

    scene_id = env.config.get('scene_id')
    floor_num = env.task.floor_num
    # floor_map, _, trav_map = env.get_floor_map()  # already processed
    floor_map = env.floor_map
    trav_map = env.trav_map
    # trav_map, _ = env.get_obstacle_map()  # already processed
    assert list(floor_map.shape) == list(trav_map.shape)

    old_pose = env.get_robot_pose(env.robots[0].calc_state())
    assert list(old_pose.shape) == [3]
    true_poses.append(old_pose)

    def _plot_poses(floor_map, poses, invert):
        import matplotlib.pyplot as plt
        s = floor_map.copy()
        for pose in poses:
            if invert:
                y, x = pose[..., 0], pose[..., 1]
            else:
                x, y = pose[..., 0], pose[..., 1]
            s[int(np.rint(x)), int(np.rint(y))] = 2
        plt.imshow(s);
        plt.show()

    odom = calc_odometry(old_pose, old_pose)
    assert list(odom.shape) == [3]
    odometry.append(odom)

    for _ in range(trajlen - 1):
        action = select_action(agent=agent, params=params, obs=obs, env=env, old_pose=old_pose)
        obs, reward, done, _ = env.step(action)

        rgb_observation.append(obs['rgb_obs'])
        depth_observation.append(obs['depth_obs'])
        occupancy_grid_observation.append(obs['occupancy_grid'])

        # get new robot state after taking action
        new_pose = env.get_robot_pose(env.robots[0].calc_state())
        assert list(new_pose.shape) == [3]
        true_poses.append(new_pose)

        # calculate actual odometry b/w old pose and new pose
        odom = calc_odometry(old_pose, new_pose)
        assert list(odom.shape) == [3]
        odometry.append(odom)
        old_pose = new_pose

        if env.pf_params.use_plot:
            target_pixel = np.concatenate([env.world_to_map(env.task.target_pos[:2]), [env.task.target_pos[2]]])
            env.render(gt_pose=new_pose, est_pose=target_pixel)

    if sample_particles:
        raise NotImplementedError()

    episode_data = {
        'scene_id': scene_id,  # str
        'floor_num': floor_num,  # int
        'floor_map': floor_map,  # (height, width, 1)
        'trav_map': trav_map,  # (height, width, 1)
        'odometry': np.stack(odometry),  # (trajlen, 3)
        'true_states': np.stack(true_poses),  # (trajlen, 3)
        'rgb_observation': np.stack(rgb_observation),  # (trajlen, height, width, 3)
        'depth_observation': np.stack(depth_observation),  # (trajlen, height, width, 1)
        'occupancy_grid': np.stack(occupancy_grid_observation),  # (trajlen, height, width, 1)
    }

    return episode_data


def serialize_tf_record(episode_data):
    """
    Serialize episode data (state, odometry, observation, global map) as tf record
    :param dict episode_data: episode data
    :return tf.train.Example: serialized tf record
    """
    rgb_observation = episode_data['rgb_observation']
    depth_observation = episode_data['depth_observation']
    occupancy_grid = episode_data['occupancy_grid']
    states = episode_data['true_states']
    odometry = episode_data['odometry']
    scene_id = episode_data['scene_id']
    floor_num = episode_data['floor_num']

    record = {
        'state': tf.train.Feature(float_list=tf.train.FloatList(value=states.flatten())),
        'state_shape': tf.train.Feature(int64_list=tf.train.Int64List(value=states.shape)),
        'odometry': tf.train.Feature(float_list=tf.train.FloatList(value=odometry.flatten())),
        'odometry_shape': tf.train.Feature(int64_list=tf.train.Int64List(value=odometry.shape)),
        'rgb_observation': tf.train.Feature(float_list=tf.train.FloatList(value=rgb_observation.flatten())),
        'rgb_observation_shape': tf.train.Feature(int64_list=tf.train.Int64List(value=rgb_observation.shape)),
        'depth_observation': tf.train.Feature(float_list=tf.train.FloatList(value=depth_observation.flatten())),
        'depth_observation_shape': tf.train.Feature(int64_list=tf.train.Int64List(value=depth_observation.shape)),
        'occupancy_grid': tf.train.Feature(float_list=tf.train.FloatList(value=occupancy_grid.flatten())),
        'occupancy_grid_shape': tf.train.Feature(int64_list=tf.train.Int64List(value=occupancy_grid.shape)),
        'scene_id': tf.train.Feature(bytes_list=tf.train.BytesList(value=[scene_id.encode('utf-8')])),
        'floor_num': tf.train.Feature(int64_list=tf.train.Int64List(value=[floor_num])),
    }

    return tf.train.Example(features=tf.train.Features(feature=record)).SerializeToString()


def deserialize_tf_record(raw_record):
    """
    Serialize episode tf record (state, odometry, observation, global map)
    :param tf.train.Example raw_record: serialized tf record
    :return tf.io.parse_single_example: de-serialized tf record
    """
    tfrecord_format = {
        'state': tf.io.FixedLenSequenceFeature((), dtype=tf.float32, allow_missing=True),
        'state_shape': tf.io.FixedLenSequenceFeature((), dtype=tf.int64, allow_missing=True),
        'odometry': tf.io.FixedLenSequenceFeature((), dtype=tf.float32, allow_missing=True),
        'odometry_shape': tf.io.FixedLenSequenceFeature((), dtype=tf.int64, allow_missing=True),
        'rgb_observation': tf.io.FixedLenSequenceFeature((), dtype=tf.int64, allow_missing=True),
        'rgb_observation_shape': tf.io.FixedLenSequenceFeature((), dtype=tf.int64, allow_missing=True),
        'depth_observation': tf.io.FixedLenSequenceFeature((), dtype=tf.int64, allow_missing=True),
        'depth_observation_shape': tf.io.FixedLenSequenceFeature((), dtype=tf.int64, allow_missing=True),
        'occupancy_grid': tf.io.FixedLenSequenceFeature((), dtype=tf.int64, allow_missing=True),
        'occupancy_grid_shape': tf.io.FixedLenSequenceFeature((), dtype=tf.int64, allow_missing=True),
        'scene_id': tf.io.FixedLenSequenceFeature((), dtype=tf.string, allow_missing=True),
        'floor_num': tf.io.FixedLenSequenceFeature((), dtype=tf.int64, allow_missing=True),
    }

    features_tensor = tf.io.parse_single_example(raw_record, tfrecord_format)
    return features_tensor


def deserialize_tf_record_igibson(raw_record):
    """
    Serialize episode tf record (state, odometry, observation, global map)
    :param tf.train.Example raw_record: serialized tf record
    :return tf.io.parse_single_example: de-serialized tf record
    """
    tfrecord_format = {
        'state': tf.io.FixedLenSequenceFeature((), dtype=tf.float32, allow_missing=True),
        'state_shape': tf.io.FixedLenSequenceFeature((), dtype=tf.int64, allow_missing=True),
        'odometry': tf.io.FixedLenSequenceFeature((), dtype=tf.float32, allow_missing=True),
        'odometry_shape': tf.io.FixedLenSequenceFeature((), dtype=tf.int64, allow_missing=True),
        'rgb_observation': tf.io.FixedLenSequenceFeature((), dtype=tf.float32, allow_missing=True),
        'rgb_observation_shape': tf.io.FixedLenSequenceFeature((), dtype=tf.int64, allow_missing=True),
        'depth_observation': tf.io.FixedLenSequenceFeature((), dtype=tf.float32, allow_missing=True),
        'depth_observation_shape': tf.io.FixedLenSequenceFeature((), dtype=tf.int64, allow_missing=True),
        'occupancy_grid': tf.io.FixedLenSequenceFeature((), dtype=tf.float32, allow_missing=True),
        'occupancy_grid_shape': tf.io.FixedLenSequenceFeature((), dtype=tf.int64, allow_missing=True),
        'scene_id': tf.io.FixedLenSequenceFeature((), dtype=tf.string, allow_missing=True),
        'floor_num': tf.io.FixedLenSequenceFeature((), dtype=tf.int64, allow_missing=True),
    }

    features_tensor = tf.io.parse_single_example(raw_record, tfrecord_format)
    return features_tensor


def get_dataflow(filenames, batch_size, s_buffer_size=100, is_training=False, is_igibson: bool = False):
    """
    Custom dataset for TF record
    """
    ds = tf.data.TFRecordDataset(filenames)
    if is_training:
        ds = ds.shuffle(s_buffer_size, reshuffle_each_iteration=True)

    if is_igibson:
        fn = deserialize_tf_record_igibson
    else:
        fn = deserialize_tf_record
    ds = ds.map(fn, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    ds = ds.batch(batch_size, drop_remainder=True).prefetch(tf.data.experimental.AUTOTUNE)

    return ds


def pad_images(images, new_shape):
    """
    Center padded gray scale images
    :param images: input gray images as a png (H, W, 1)
    :param new_shape: output padded image shape (new_H, new_W, 1)
    :return ndarray: padded gray images as a png (new_H, new_W, 1)
    """
    H, W, C = images.shape
    new_H, new_W, new_C = new_shape
    assert new_H >= H and new_W >= W and new_C >= C
    if new_H == H and new_W == W and new_C == C:
        return images

    top = 0
    bottom = new_H - H - top
    left = 0
    right = new_W - W - left

    padded_images = np.pad(images,
                           pad_width=[(top, bottom), (left, right), (0, 0)],
                           mode='constant',
                           constant_values=0)

    return padded_images


def transform_raw_record(parsed_record, params):
    """
    process de-serialized tfrecords data
    :param env: igibson env instance
    :param parsed_record: de-serialized tfrecord data
    :param params: parsed parameters
    :return dict: processed data containing: true_states, odometries, observations, global map, initial particles
    """
    trans_record = {}

    obs_mode = params.obs_mode
    trajlen = params.trajlen
    batch_size = params.batch_size
    pad_map_size = params.global_map_size
    num_particles = params.num_particles
    particles_cov = params.init_particles_cov

    # perform required rescale to [-1, 1]
    if obs_mode == 'rgb-depth':
        rgb_observation = parsed_record['rgb_observation'].reshape(
            [batch_size] + list(parsed_record['rgb_observation_shape'][0]))[:, :trajlen]
        assert np.min(rgb_observation) >= 0. and np.max(rgb_observation) <= 1.
        depth_observation = parsed_record['depth_observation'].reshape(
            [batch_size] + list(parsed_record['depth_observation_shape'][0]))[:, :trajlen]
        assert np.min(depth_observation) >= 0. and np.max(depth_observation) <= 1.
        trans_record['observation'] = np.concatenate([rgb_observation, depth_observation, ], axis=-1)
    elif obs_mode == 'depth':
        depth_observation = parsed_record['depth_observation'].reshape(
            [batch_size] + list(parsed_record['depth_observation_shape'][0]))[:, :trajlen]
        assert np.min(depth_observation) >= 0. and np.max(depth_observation) <= 1.
        trans_record['observation'] = depth_observation
    elif obs_mode == 'rgb':
        rgb_observation = parsed_record['rgb_observation'].reshape(
            [batch_size] + list(parsed_record['rgb_observation_shape'][0]))[:, :trajlen]
        assert np.min(rgb_observation) >= 0. and np.max(rgb_observation) <= 1.
        trans_record['observation'] = rgb_observation
    elif obs_mode == 'occupancy_grid':
        occupancy_grid_observation = parsed_record['occupancy_grid'].reshape(
            [batch_size] + list(parsed_record['occupancy_grid_shape'][0]))[:, :trajlen]
        assert np.min(occupancy_grid_observation) >= 0. and np.max(occupancy_grid_observation) <= 1.
        if params.likelihood_model == 'learned':
            resized = []
            for batch in range(occupancy_grid_observation.shape[0]):
                for img in range(occupancy_grid_observation.shape[1]):
                    resized.append(cv2.resize(occupancy_grid_observation[batch, img], (56, 56)))
            occupancy_grid_observation = np.reshape(np.stack(resized, 0), (
            occupancy_grid_observation.shape[0], occupancy_grid_observation.shape[1], 56, 56, 1))
        trans_record['observation'] = occupancy_grid_observation.astype(float)  # [0, 0.5, 1]
    else:
        raise ValueError(obs_mode)

    trans_record['odometry'] = parsed_record['odometry'].reshape(
        [batch_size] + list(parsed_record['odometry_shape'][0]))[:, :trajlen]
    trans_record['true_states'] = parsed_record['state'].reshape([batch_size] + list(parsed_record['state_shape'][0]))[
                                  :, :trajlen]

    # get floor and obstance map from environment instance for the scene
    trans_record['org_map_shape'] = []
    trans_record['trav_map'] = []
    trans_record['global_map'] = []
    trans_record['init_particles'] = []
    for b_idx in range(batch_size):
        # iterate per batch_size
        if parsed_record['scene_id'].size > 0:
            scene_id = parsed_record['scene_id'][b_idx][0].decode('utf-8')
            floor_num = parsed_record['floor_num'][b_idx][0]
        else:
            scene_id = None
            floor_num = None

        # right zero-pad floor/obstacle map
        floor_map, org_map_shape, trav_map = get_floor_map(scene_id=scene_id,
                                                           floor_num=floor_num,
                                                           pad_map_size=pad_map_size,
                                                           trav_map_erosion=2,
                                                           trav_map_resolution=0.1)

        # sample random particles using gt_pose at start of trajectory
        gt_first_pose = np.expand_dims(trans_record['true_states'][b_idx, 0, :], axis=0)
        init_particles = get_random_particles(num_particles=num_particles,
                                              particles_distr=params.init_particles_distr,
                                              robot_pose=gt_first_pose,
                                              trav_map=trav_map,
                                              particles_cov=particles_cov,
                                              particles_range=params.particles_range)

        trans_record['org_map_shape'].append(org_map_shape)
        trans_record['init_particles'].append(init_particles)
        trans_record['trav_map'].append(trav_map)
        trans_record['global_map'].append(floor_map)
    trans_record['org_map_shape'] = np.stack(trans_record['org_map_shape'], axis=0)  # [batch_size, 3]
    trans_record['trav_map'] = np.stack(trans_record['trav_map'], axis=0)  # [batch_size, H, W, C]
    trans_record['global_map'] = np.stack(trans_record['global_map'], axis=0)  # [batch_size, H, W, C]
    trans_record['init_particles'] = np.concatenate(trans_record['init_particles'], axis=0)  # [batch_size, N, 3]

    # sanity check
    assert list(trans_record['odometry'].shape) == [batch_size, trajlen, 3], f'{trans_record["odometry"].shape}'
    assert list(trans_record['true_states'].shape) == [batch_size, trajlen, 3], f'{trans_record["true_states"].shape}'
    assert list(trans_record['observation'].shape) in [[batch_size, trajlen, 56, 56, params.obs_ch],
                                                       [batch_size, trajlen, 128, 128,
                                                        params.obs_ch]], f'{trans_record["observation"].shape}'
    assert list(trans_record['init_particles'].shape) == [batch_size, num_particles,
                                                          3], f'{trans_record["init_particles"].shape}'
    assert list(trans_record['global_map'].shape) == [batch_size, *pad_map_size], f'{trans_record["global_map"].shape}'
    assert list(trans_record['trav_map'].shape) == [batch_size, *pad_map_size], f'{trans_record["trav_map"].shape}'

    trans_record['init_particle_weights'] = tf.constant(np.log(1.0 / float(params.num_particles)),
                                                        shape=(params.batch_size, params.num_particles),
                                                        dtype=tf.float32)

    return trans_record
