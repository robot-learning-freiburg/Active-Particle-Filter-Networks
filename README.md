# deep-activate-localization

#### Verified Setup On:
* Operating System: Ubuntu 20.04.2 LTS
* Nvidia-Driver Version: 460.73.01
* CUDA Version: 11.2

#### Wiki Page:
For more details to train/test particle filter or rl agent [refer](https://github.com/suresh-guttikonda/deep-activate-localization/wiki).

#### Installation:
- Option1 : Installation from source (https://github.com/suresh-guttikonda/deep-activate-localization#steps-to-setup-project)
- Option2 : Installation from singularity image (https://github.com/suresh-guttikonda/deep-activate-localization#steps-to-create-singularity-image-to-setup-project) Note: modifying locobot urdf can be tricky in this case.

#### Useful References:
- Install Cuda Without Root [link](https://stackoverflow.com/questions/39379792/install-cuda-without-root)
   ```
      sh cuda_10.1.105_418.39_linux.run --silent --toolkit --toolkitpath=$HOME/tkit
   ```
- Install Cudnn in Ubuntu [link](https://askubuntu.com/questions/1230645/when-is-cuda-gonna-be-released-for-ubuntu-20-04)


#### Steps to setup project:
0. Install required Nvidia Driver + CUDA + CUDNN for the system. Also refer igibson documentation for system requirements [link](http://svl.stanford.edu/igibson/docs/installation.html).
1. Install virtual environment/package management platform like anaconda/[miniconda](https://docs.conda.io/en/latest/miniconda.html) or python virtualenv. Following assumes anaconda is installed.
2. Create conda environment
   ```
      $ conda create -y -n igibson python=3.7
      $ source activate igibson
    ```
3. Setup iGibson2.0 environment (forked repository). For latest installation procedure always refer to official doc [link](http://svl.stanford.edu/igibson/docs/installation.html)
   ```
      <root_folder>$ git clone --branch master https://github.com/suresh-guttikonda/iGibson.git --recursive
      <root_folder>$ cd iGibson
      <root_folder>/iGibson$ pip3 install -e .
   ```
4. Test igibson installation is successful
   ```
      <root_folder>/iGibson$ python
      >>> import igibson
   ```
5. Download required igibson's assets (robot's urdf, demo apartment, and others). For more datasets [refer](http://svl.stanford.edu/igibson/docs/dataset.html)
   ```
      <root_folder>/iGibson$ python -m igibson.utils.assets_utils --download_assets
      <root_folder>/iGibson$ python -m igibson.utils.assets_utils --download_demo_data
   ```
6. For locobot robot, we modify urdf file by adding additional lidar sensor 'scan_link' link and joint to 'base_link'. For reference, turtlebot's urdf file has the sensor by default.
   ```
      <root_folder>/iGibson$ vi igibson/data/assets/models/locobot/locobot.urdf
    
    <link name="scan_link">
      <inertial>
         <mass value="0"/>
         <origin xyz="0 0 0"/>
         <inertia ixx="0.0001" ixy="0.0" ixz="0.0" iyy="0.0001" iyz="0.0" izz="0.0001"/>
      </inertial>
      <visual>
         <geometry>
            <cylinder length="0.02" radius="0.01"/>
         </geometry>
         <origin rpy="0 0 0" xyz="0.001 0 0.05199"/>
      </visual>
    </link>
    <joint name="scan_joint" type="fixed">
      <parent link="base_link"/>
      <child link="scan_link"/>
      <origin rpy="0 0 0" xyz="0.10 0.0 0.46"/>
      <axis xyz="0 0 1"/>
    </joint>
   ```
7. Install additional packages
   ```
      <root_folder>$ pip install --upgrade pip
      <root_folder>$ pip install tensorflow==2.6.0
      <root_folder>$ pip install -U numpy==1.21.1
      <root_folder>$ pip install -U scikit-learn
      <root_folder>$ conda install -c anaconda cudnn=7.6.5
   ```
8. Test tensorflow installation is successful
   ```
      <root_folder>$ python
      >>> import tensorflow as tf
      >>> tf.config.list_logical_devices()
   ```
9. Install TF Agents (forked repository)
   ```
      <root_folder>/iGibson$ cd ..
      <root_folder>$ git clone --branch cvpr21_challenge_tf2.4 https://github.com/suresh-guttikonda/agents/ --recursive
      <root_folder>$ cd agents
      <root_folder>/agents$ pip3 install -e .
   ```
10. Get the Active Localization code + pretrained checkpoints
   ```
      <root_folder>/agents$ cd ..
      <root_folder>$ git clone https://github.com/suresh-guttikonda/deep-activate-localization.git --recursive
   ```
11. Test pretrained particle filter + igibson environment with obstacle avoidance agent, result will be stored to test_output directory.
   ```
      <root_folder>$ cd deep-activate-localization/src/rl_agents
      <root_folder>/deep-activate-localization/src/rl_agents$ python -u test_pfnet.py \
         --pfnet_loadpath=./pfnetwork/checkpoints/pfnet_igibson_data/report/rs_rgbd/checkpoint_28_0.065/pfnet_checkpoint \
         --obs_mode='rgb-depth' \
         --custom_output 'rgb_obs' 'depth_obs' 'likelihood_map' 'obstacle_obs' \
         --scene_id='Rs' \
         --num_eval_episodes=1 \
         --agent='avoid_agent' \
         --init_particles_distr='uniform' \
         --init_particles_std '0.2' '0.523599' \
         --particles_range=100 \
         --num_particles=250 \
         --transition_std '0.04' '0.0872665' \
         --resample=true \
         --alpha_resample_ratio=0.95 \
         --global_map_size '100' '100' '1' \
         --window_scaler=1.0 \
         --config_file=./configs/locobot_pfnet_nav.yaml \
         --device_idx=0 \
         --seed=15
   ```
12. Test pretrained SAC agent for igibson environment as follows (for global localization).
   ```
      <root_folder>$ cd deep-activate-localization/src/rl_agents
      <root_folder>/deep-activate-localization/src/rl_agents$ python -u test_rl_agent.py \
         --root_dir=checkpoints/train_rl_uniform_2.0_box_50 \
         --num_eval_episodes=10 \
         --use_tf_functions=False \
         --agent='sac_agent' \
         --eval_deterministic=True \
         --is_localize_env=True \
         --config_file=./configs/locobot_point_nav.yaml \
         --gpu_num=0 \
         --init_env_pfnet=True \
         --init_particles_distr='uniform' \
         --init_particles_std=0.2,0.523599 \
         --particles_range=200 \
         --num_particles=1000 \
         --resample=True \
         --alpha_resample_ratio=0.99 \
         --transition_std=0.04,0.0872665 \
         --obs_mode='rgb-depth' \
         --custom_output='rgb_obs','depth_obs','likelihood_map' \
         --num_clusters=10 \
         --global_map_size=100,100,1 \
         --window_scaler=1.0 \
         --pfnet_load=./pfnetwork/checkpoints/pfnet_igibson_data/report/rs_rgbd/checkpoint_28_0.065/pfnet_checkpoint \
         --use_plot=True \
         --store_plot=True \
         --seed=1198
    ```

#### Steps to install singularity:
1. Enable NeuroDebian repository by following instructions on [neuro-debian](http://neuro.debian.net/). Also refer [FAQ](http://neuro.debian.net/faq.html).
2. Install Singularity-Container as follows:
   ```
   $ sudo apt-get install -y singularity-container
   ```
3. For more details, refer: [singularity guide](https://sylabs.io/guides/3.7/user-guide/index.html)

#### Steps to create singularity image to setup project:
0. We assume corresponding nvidia-driver is already installed in host machine.
1. Bootstrap container definition file. 'sudo' ensure proper privileges get assigned.
    ```
    # command takes few minutes to completely build image
    <path_to_def_file>$ sudo singularity build tensorflow_latest-gpu.sif tensorflow_latest-gpu.def
    ```
2. After successful build, verify that packages such as tensorflow+gpu are installed correctly.\
    Note: --nv ensure cuda related packages are loaded
    ```
    <path_to_sif_file>$ sudo singularity shell --nv tensorflow_latest-gpu.sif
    
    # get driver details if above assumptions are satisfied
    ~> nvidia-smi
    
    # activate virtualenv
    ~> source /opt/venvs/py3-igibson/bin/activate
    
    ~> python
    >>> import tensorflow as tf
    >>> tf.__version__
    >>>> tf.config.list_physical_devices() => returns list of available devices
    ```
3. Run demo igibson gui example. Following should open window with robot otherwise we see and error 'ERROR: Unable to initialize EGL'
    ```
    # Note: don't run with sudo once image is created
    <path_to_def_file>$ singularity shell --nv -B /usr/share/glvnd tensorflow_latest-gpu.sif
    
    ~> source /opt/venvs/py3-igibson/bin/activate
    
    # launch igibson gui example
    (py3-igibson)~> python -m gibson2.examples.demo.env_example
    ```
4. To Bind/Mount host machine directories within container use --bind [ref](https://sylabs.io/guides/3.0/user-guide/bind_paths_and_mounts.html)
    ```
    <path_to_sif_file>$ singularity shell --nv --bind /usr/share/glvnd,./src:/mnt/src tensorflow_latest-gpu.sif
    ```
