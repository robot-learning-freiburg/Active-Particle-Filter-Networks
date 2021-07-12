#!/usr/bin/env python3

import cv2
import numpy as np
import pybullet as p
import tensorflow as tf


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


def calc_velocity_commands(old_pose, new_pose, dt=0.1):
    """
    Calculate the velocity model command between two poses
    :param ndarray old_pose: pose1 (x, y, theta)
    :param ndarray new_pose: pose2 (x, y, theta)
    :param float dt: time interval
    :return ndarray: velocity command (linear_vel, angular_vel, final_rotation)
    """

    x1, y1, th1 = old_pose
    x2, y2, th2 = new_pose

    if x1==x2 and y1==y2:
        # only angular motion
        linear_velocity = 0
        angular_velocity = 0
    elif x1!=x2 and np.tan(th1) == np.tan( (y1-y2)/(x1-x2) ):
        # only linear motion
        linear_velocity = (x2-x1)/dt
        angular_velocity = 0
    else:
        # both linear + angular motion
        mu = 0.5 * ( ((x1-x2)*np.cos(th1) + (y1-y2)*np.sin(th1))
                 / ((y1-y2)*np.cos(th1) - (x1-x2)*np.sin(th1)) )
        x_c = (x1+x2) * 0.5 + mu * (y1-y2)
        y_c = (y1+y2) * 0.5 - mu * (x1-x2)
        r_c = np.sqrt( (x1-x_c)**2 + (y1-y_c)**2 )
        delta_th = np.arctan2(y2-y_c, x2-x_c) - np.arctan2(y1-y_c, x1-x_c)

        angular_velocity = delta_th/dt
        # HACK: to handle unambiguous postive/negative quadrants
        if np.arctan2(y1-y_c, x1-x_c) < 0:
            linear_velocity = angular_velocity * r_c
        else:
            linear_velocity = -angular_velocity * r_c

    final_rotation = (th2-th1)/dt - angular_velocity
    return np.array([linear_velocity, angular_velocity, final_rotation])


def sample_motion_odometry(old_pose, odometry):
    """
    Sample new pose based on give pose and odometry
    :param ndarray old_pose: given pose (x, y, theta)
    :param ndarray odometry: given odometry (odom_x, odom_y, odom_th)
    :return ndarray: new pose (x, y, theta)
    """
    x1, y1, th1 = old_pose
    odom_x, odom_y, odom_th = odometry

    th1 = normalize(th1)
    sin = np.sin(th1)
    cos = np.cos(th1)

    x2 = x1 + (cos * odom_x - sin * odom_y)
    y2 = y1 + (sin * odom_x + cos * odom_y)
    th2 = normalize(th1 + odom_th)

    new_pose = np.array([x2, y2, th2])
    return new_pose


def sample_motion_velocity(old_pose, velocity, dt=0.1):
    """
    Sample new pose based on give pose and velocity commands
    :param ndarray old_pose: given pose (x, y, theta)
    :param ndarray velocity: velocity model (linear_vel, angular_vel, final_rotation)
    :param float dt: time interval
    :return ndarray: new pose (x, y, theta)
    """
    x1, y1, th1 = old_pose
    linear_vel, angular_vel, final_rotation = velocity

    if angular_vel == 0:
        x2 = x1 + linear_vel*dt
        y2 = y1
    else:
        r = linear_vel/angular_vel
        x2 = x1 - r*np.sin(th1) + r*np.sin(th1 + angular_vel*dt)
        y2 = y1 + r*np.cos(th1) - r*np.cos(th1 + angular_vel*dt)
    th2 = th1 + angular_vel*dt + final_rotation*dt

    new_pose = np.array([x2, y2, th2])
    return new_pose


def decode_image(img, resize=None):
    """
    Decode image
    :param img: image encoded as a png in a string
    :param resize: tuple of width, height, new size of image (optional)
    :return np.ndarray: image (k, H, W, 1)
    """
    # TODO
    # img = cv2.imdecode(img, -1)
    if resize is not None:
        img = cv2.resize(img, resize)
    return img


def process_raw_map(image):
    """
    Decode and normalize image
    :param image: floor map image as ndarray (H, W)
    :return np.ndarray: image (H, W, 1)
        white: empty space, black: occupied space
    """

    assert np.min(image)>=0. and np.max(image)>=1. and np.max(image)<=255.
    image = normalize_map(np.atleast_3d(image.astype(np.float32)))
    assert np.min(image)>=0. and np.max(image)<=2.

    return image


def normalize_map(x):
    """
    Normalize map input
    :param x: map input (H, W, ch)
    :return np.ndarray: normalized map (H, W, ch)
    """
    # rescale to [0, 2], later zero padding will produce equivalent obstacle
    return x * (2.0 / 255.0)


def normalize_observation(x):
    """
    Normalize observation input: an rgb image or a depth image
    :param x: observation input (56, 56, ch)
    :return np.ndarray: normalized observation (56, 56, ch)
    """
    # resale to [-1, 1]
    if x.ndim == 2 or x.shape[2] == 1:  # depth
        return x * (2.0 / 100.0) - 1.0
    else:  # rgb
        return x * (2.0 / 255.0) - 1.0


def denormalize_observation(x):
    """
    Denormalize observation input to store efficiently
    :param x: observation input (B, 56, 56, ch)
    :return np.ndarray: denormalized observation (B, 56, 56, ch)
    """
    # resale to [0, 255]
    if x.ndim == 2 or x.shape[-1] == 1:  # depth
        x = (x + 1.0) * (100.0 / 2.0)
    else:  # rgb
        x = (x + 1.0) * (255.0 / 2.0)
    return x.astype(np.int32)


def process_raw_image(image, resize=(56, 56)):
    """
    Decode and normalize image
    :param image: image encoded as a png (H, W, ch)
    :param resize: resize image (new_H, new_W)
    :return np.ndarray: images (new_H, new_W, ch) normalized for training
    """

    # assert np.min(image)>=0. and np.max(image)>=1. and np.max(image)<=255.
    image = decode_image(image, resize)
    image = normalize_observation(np.atleast_3d(image.astype(np.float32)))
    assert np.min(image)>=-1. and np.max(image)<=1.

    return image


def get_discrete_action():
    """
    Get manual keyboard action
    :return int: discrete action for moving forward/backward/left/right
    """
    key = input('Enter Key: ')
    # default stay still
    action = 4
    if key == 'w':
        action = 0  # forward
    elif key == 's':
        action = 1  # backward
    elif key == 'd':
        action = 2  # right
    elif key == 'a':
        action = 3  # left
    return action


def transform_position(position, map_shape, map_pixel_in_meters):
    """
    Transform position from 2D co-ordinate space to pixel space
    :param ndarray position: [x, y] in co-ordinate space
    :param tuple map_shape: [height, width, channel] of the map the co-ordinated need to be transformed
    :param float map_pixel_in_meters: The width (and height) of a pixel of the map in meters
    :return ndarray: position [x, y] in pixel space of map
    """
    x, y = position
    height, width, channel = map_shape

    x = (x / map_pixel_in_meters) + width / 2
    y = (y / map_pixel_in_meters) + height / 2

    return np.array([x, y])


def inv_transform_pose(pose, map_shape, map_pixel_in_meters):
    """
    Transform pose from pixel space to 2D co-ordinate space
    :param ndarray pose: [x, y, theta] in pixel space of map
    :param tuple map_shape: [height, width, channel] of the map the co-ordinated need to be transformed
    :param float map_pixel_in_meters: The width (and height) of a pixel of the map in meters
    :return ndarray: pose [x, y, theta] in co-ordinate space
    """
    x, y, theta = pose
    height, width, channel = map_shape

    x = (x - width / 2) * map_pixel_in_meters
    y = (y - height / 2) * map_pixel_in_meters

    return np.array([x, y, theta])


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

    odometry = []
    true_poses = []
    rgb_observation = []
    depth_observation = []

    obs = env.reset()  # already processed
    rgb_observation.append(obs[0])
    depth_observation.append(obs[1])
    left_free, front_free, right_free = obs[2] # obstacle (not)present

    scene_id = env.config.get('scene_id')
    floor_num = env.task.floor_num
    floor_map = env.get_floor_map()  # already processed
    obstacle_map = env.get_obstacle_map()  # already processed
    assert list(floor_map.shape) == list(obstacle_map.shape)

    old_pose = env.get_robot_pose(env.robots[0].calc_state(), floor_map.shape)
    assert list(old_pose.shape) == [3]
    true_poses.append(old_pose)

    for _ in range(trajlen - 1):
        if agent == 'manual':
            action = get_discrete_action()
        else:
            if front_free:
                # move forward
                action = 0
            elif left_free:
                # turn left
                action = 3
            elif right_free:
                # turn right
                action = 2
            else:
                # uniform random [front, back, left, right, do_nothing]
                action = np.random.choice(5, p=[0.25, 0.25, 0.25, 0.25, 0.0])

        # take action and get new observation
        obs, reward, done, _ = env.step(action)
        assert list(obs[0].shape) == [56, 56, 3]
        assert list(obs[1].shape) == [56, 56, 1]
        rgb_observation.append(obs[0])
        depth_observation.append(obs[1])

        # get new robot state after taking action
        new_pose = env.get_robot_pose(env.robots[0].calc_state(), floor_map.shape)
        assert list(new_pose.shape) == [3]
        true_poses.append(new_pose)

        # calculate actual odometry b/w old pose and new pose
        odom = calc_odometry(old_pose, new_pose)
        assert list(odom.shape) == [3]
        odometry.append(odom)
        old_pose = new_pose

    # end of episode
    odom = calc_odometry(old_pose, new_pose)
    odometry.append(odom)

    if sample_particles:
        num_particles = params.num_particles
        particles_cov = params.init_particles_cov
        particles_distr = params.init_particles_distr
        # sample random particles and corresponding weights
        init_particles = env.get_random_particles(num_particles, particles_distr, true_poses[0], particles_cov).squeeze(
            axis=0)
        init_particle_weights = np.full((num_particles,), (1. / num_particles))
        assert list(init_particles.shape) == [num_particles, 3]
        assert list(init_particle_weights.shape) == [num_particles]

    else:
        init_particles = None
        init_particle_weights = None

    episode_data = {
        'scene_id': scene_id, # str
        'floor_num': floor_num, # int
        'floor_map': floor_map,  # (height, width, 1)
        'obstacle_map': obstacle_map,  # (height, width, 1)
        'odometry': np.stack(odometry),  # (trajlen, 3)
        'true_states': np.stack(true_poses),  # (trajlen, 3)
        'rgb_observation': np.stack(rgb_observation),  # (trajlen, height, width, 3)
        'depth_observation': np.stack(depth_observation),  # (trajlen, height, width, 1)
        'init_particles': init_particles,  # (num_particles, 3)
        'init_particle_weights': init_particle_weights,  # (num_particles,)
    }

    return episode_data


def get_batch_data(env, params):
    """
    Gather batch of episode stats
    :param env: igibson env instance
    :param params: parsed parameters
    :return dict: episode stats data containing:
        odometry, true poses, observation, particles, particles weights, floor map
    """

    trajlen = params.trajlen
    batch_size = params.batch_size
    map_size = params.global_map_size
    num_particles = params.num_particles

    odometry = []
    floor_map = []
    obstacle_map = []
    observation = []
    true_states = []
    init_particles = []
    init_particle_weights = []

    for _ in range(batch_size):
        episode_data = gather_episode_stats(env, params, sample_particles=True)

        odometry.append(episode_data['odometry'])
        floor_map.append(episode_data['floor_map'])
        obstacle_map.append(episode_data['obstacle_map'])
        true_states.append(episode_data['true_states'])
        observation.append(episode_data['observation'])
        init_particles.append(episode_data['init_particles'])
        init_particle_weights.append(episode_data['init_particle_weights'])

    batch_data = {}
    batch_data['odometry'] = np.stack(odometry)
    batch_data['floor_map'] = np.stack(floor_map)
    batch_data['obstacle_map'] = np.stack(obstacle_map)
    batch_data['true_states'] = np.stack(true_states)
    batch_data['observation'] = np.stack(observation)
    batch_data['init_particles'] = np.stack(init_particles)
    batch_data['init_particle_weights'] = np.stack(init_particle_weights)

    # sanity check
    assert list(batch_data['odometry'].shape) == [batch_size, trajlen, 3]
    assert list(batch_data['true_states'].shape) == [batch_size, trajlen, 3]
    assert list(batch_data['observation'].shape) == [batch_size, trajlen, 56, 56, 3]
    assert list(batch_data['init_particles'].shape) == [batch_size, num_particles, 3]
    assert list(batch_data['init_particle_weights'].shape) == [batch_size, num_particles]
    assert list(batch_data['floor_map'].shape) == [batch_size, map_size[0], map_size[1], map_size[2]]
    assert list(batch_data['obstacle_map'].shape) == [batch_size, map_size[0], map_size[1], map_size[2]]

    return batch_data


def serialize_tf_record(episode_data):
    """
    Serialize episode data (state, odometry, observation, global map) as tf record
    :param dict episode_data: episode data
    :return tf.train.Example: serialized tf record
    """
    # HACK: rescale to [0, 255] and [0, 100]
    rgb_observation = denormalize_observation(episode_data['rgb_observation'])
    depth_observation = denormalize_observation(episode_data['depth_observation'])
    states = episode_data['true_states']
    odometry = episode_data['odometry']
    scene_id = episode_data['scene_id']
    floor_num = episode_data['floor_num']
    # floor_map = episode_data['floor_map']
    # obstacle_map = episode_data['obstacle_map']
    # init_particles = episode_data['init_particles']
    # init_particle_weights = episode_data['init_particle_weights']

    record = {
        'state': tf.train.Feature(float_list=tf.train.FloatList(value=states.flatten())),
        'state_shape': tf.train.Feature(int64_list=tf.train.Int64List(value=states.shape)),
        'odometry': tf.train.Feature(float_list=tf.train.FloatList(value=odometry.flatten())),
        'odometry_shape': tf.train.Feature(int64_list=tf.train.Int64List(value=odometry.shape)),
        'rgb_observation': tf.train.Feature(int64_list=tf.train.Int64List(value=rgb_observation.flatten())),
        'rgb_observation_shape': tf.train.Feature(int64_list=tf.train.Int64List(value=rgb_observation.shape)),
        'depth_observation': tf.train.Feature(int64_list=tf.train.Int64List(value=depth_observation.flatten())),
        'depth_observation_shape': tf.train.Feature(int64_list=tf.train.Int64List(value=depth_observation.shape)),
        'scene_id': tf.train.Feature(bytes_list=tf.train.BytesList(value=[scene_id.encode('utf-8')])),
        'floor_num': tf.train.Feature(int64_list=tf.train.Int64List(value=[floor_num])),
        # 'floor_map': tf.train.Feature(float_list=tf.train.FloatList(value=floor_map.flatten())),
        # 'floor_map_shape': tf.train.Feature(int64_list=tf.train.Int64List(value=floor_map.shape)),
        # 'obstacle_map': tf.train.Feature(float_list=tf.train.FloatList(value=obstacle_map.flatten())),
        # 'obstacle_map_shape': tf.train.Feature(int64_list=tf.train.Int64List(value=obstacle_map.shape)),
        # 'init_particles': tf.train.Feature(float_list=tf.train.FloatList(value=init_particles.flatten())),
        # 'init_particles_shape': tf.train.Feature(int64_list=tf.train.Int64List(value=init_particles.shape)),
        # 'init_particle_weights': tf.train.Feature(float_list=tf.train.FloatList(value=init_particle_weights.flatten())),
        # 'init_particle_weights_shape': tf.train.Feature(int64_list=tf.train.Int64List(value=init_particle_weights.shape)),
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
        'scene_id': tf.io.FixedLenSequenceFeature((), dtype=tf.string, allow_missing=True),
        'floor_num': tf.io.FixedLenSequenceFeature((), dtype=tf.int64, allow_missing=True),
        # 'floor_map': tf.io.FixedLenSequenceFeature((), dtype=tf.float32, allow_missing=True),
        # 'floor_map_shape': tf.io.FixedLenSequenceFeature((), dtype=tf.int64, allow_missing=True),
        # 'obstacle_map': tf.io.FixedLenSequenceFeature((), dtype=tf.float32, allow_missing=True),
        # 'obstacle_map_shape': tf.io.FixedLenSequenceFeature((), dtype=tf.int64, allow_missing=True),
        # 'init_particles': tf.io.FixedLenSequenceFeature((), dtype=tf.float32, allow_missing=True),
        # 'init_particles_shape': tf.io.FixedLenSequenceFeature((), dtype=tf.int64, allow_missing=True),
        # 'init_particle_weights': tf.io.FixedLenSequenceFeature((), dtype=tf.float32, allow_missing=True),
        # 'init_particle_weights_shape': tf.io.FixedLenSequenceFeature((), dtype=tf.int64, allow_missing=True),
    }

    features_tensor = tf.io.parse_single_example(raw_record, tfrecord_format)
    return features_tensor


def get_dataflow(filenames, batch_size, s_buffer_size=100, is_training=False):
    """
    Custom dataset for TF record
    """
    ds = tf.data.TFRecordDataset(filenames)
    if is_training:
        ds = ds.shuffle(s_buffer_size, reshuffle_each_iteration=True)
    ds = ds.map(deserialize_tf_record, num_parallel_calls=tf.data.experimental.AUTOTUNE)
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
    assert new_H>=H and new_W>=W and new_C>=C
    if new_H==H and new_W==W and new_C==C:
        return images

    top = 0
    bottom = new_H-H-top
    left = 0
    right = new_W-W-left

    padded_images = np.pad(
                        images,
                        pad_width=[(top, bottom),(left, right),(0, 0)],
                        mode='constant',
                        constant_values=0)

    return padded_images


def transform_raw_record(env, parsed_record, params):
    """
    process de-serialized tfrecords data
    :param env: igibson env instance
    :param parsed_record: de-serialized tfrecord data
    :param params: parsed parameters
    :return dict: processed data containing: true_states, odometries, observations, global map, initial particles
    """
    trans_record = {}

    obs_ch = params.obs_ch
    obs_mode = params.obs_mode
    trajlen = params.trajlen
    batch_size = params.batch_size
    pad_map_size = params.global_map_size
    num_particles = params.num_particles
    particles_cov = params.init_particles_cov
    particles_distr = params.init_particles_distr

    # Required rescale to [-1, 1]
    assert obs_mode in ['rgb', 'depth', 'rgb-depth']
    if obs_mode == 'rgb-depth':
        rgb_observation = parsed_record['rgb_observation'].reshape(
            [batch_size] + list(parsed_record['rgb_observation_shape'][0]))[:, :trajlen]
        assert np.min(rgb_observation)>=0. and np.max(rgb_observation)<=255.
        depth_observation = parsed_record['depth_observation'].reshape(
            [batch_size] + list(parsed_record['depth_observation_shape'][0]))[:, :trajlen]
        assert np.min(depth_observation)>=0. and np.max(depth_observation)<=100.
        trans_record['observation'] = np.concatenate([
            normalize_observation(rgb_observation.astype(np.float32)),
            normalize_observation(depth_observation.astype(np.float32)),
        ], axis=-1)
    elif obs_mode == 'depth':
        depth_observation = parsed_record['depth_observation'].reshape(
            [batch_size] + list(parsed_record['depth_observation_shape'][0]))[:, :trajlen]
        assert np.min(depth_observation)>=0. and np.max(depth_observation)<=100.
        trans_record['observation'] = normalize_observation(depth_observation.astype(np.float32))
    else:
        rgb_observation = parsed_record['rgb_observation'].reshape(
            [batch_size] + list(parsed_record['rgb_observation_shape'][0]))[:, :trajlen]
        assert np.min(rgb_observation)>=0. and np.max(rgb_observation)<=255.
        trans_record['observation'] = normalize_observation(rgb_observation.astype(np.float32))

    trans_record['odometry'] = parsed_record['odometry'].reshape(
        [batch_size] + list(parsed_record['odometry_shape'][0]))[:, :trajlen]
    trans_record['true_states'] = parsed_record['state'].reshape(
        [batch_size] + list(parsed_record['state_shape'][0]))[:, :trajlen]

    # HACK: get floor and obstance map from environment instance for the scene
    trans_record['org_map_shape'] = []
    trans_record['obstacle_map'] = []
    trans_record['floor_map'] = []
    trans_record['init_particles'] = []
    for b_idx in range(batch_size):
        # iterate per batch_size
        if parsed_record['scene_id'].size > 0:
            scene_id = parsed_record['scene_id'][b_idx][0].decode('utf-8')
            floor_num = parsed_record['floor_num'][b_idx][0]
        else:
            scene_id = None
            floor_num = None

        obstacle_map = env.get_obstacle_map(scene_id, floor_num)
        floor_map = env.get_floor_map(scene_id, floor_num)
        org_map_shape = obstacle_map.shape

        # sample random particles using gt_pose at start of trajectory
        gt_first_pose = np.expand_dims(trans_record['true_states'][b_idx, 0, :], axis=0)
        init_particles = env.get_random_particles(num_particles, particles_distr,
                                        gt_first_pose, floor_map, particles_cov)

        # HACK: center zero-pad floor/obstacle map
        obstacle_map = pad_images(obstacle_map, pad_map_size)
        floor_map = pad_images(floor_map, pad_map_size)

        trans_record['org_map_shape'].append(org_map_shape)
        trans_record['init_particles'].append(init_particles)
        trans_record['obstacle_map'].append(obstacle_map)
        trans_record['floor_map'].append(floor_map)
    trans_record['org_map_shape'] = np.stack(trans_record['org_map_shape'], axis=0)  # [batch_size, 3]
    trans_record['obstacle_map'] = np.stack(trans_record['obstacle_map'], axis=0)  # [batch_size, H, W, C]
    trans_record['floor_map'] = np.stack(trans_record['floor_map'], axis=0)  # [batch_size, H, W, C]
    trans_record['init_particles'] = np.concatenate(trans_record['init_particles'], axis=0)   # [batch_size, N, 3]

    # sanity check
    assert list(trans_record['odometry'].shape) == [batch_size, trajlen, 3]
    assert list(trans_record['true_states'].shape) == [batch_size, trajlen, 3]
    assert list(trans_record['observation'].shape) == [batch_size, trajlen, 56, 56, obs_ch]
    assert list(trans_record['init_particles'].shape) == [batch_size, num_particles, 3]
    assert list(trans_record['floor_map'].shape) == [batch_size, *pad_map_size]
    assert list(trans_record['obstacle_map'].shape) == [batch_size, *pad_map_size]

    return trans_record
