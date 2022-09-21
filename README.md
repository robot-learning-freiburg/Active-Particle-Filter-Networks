# Active Particle Filter Networks

Repository providing the source code for the paper "Active Particle Filter Networks: Efficient Active Localization in Continuous Action Spaces and Large Maps", see the [project website](http://apfn.cs.uni-freiburg.de).
Please cite the paper as follows:

```
   @article{honerkamp2021active,
            title={Active Particle Filter Networks: Efficient Active Localization in Continuous Action Spaces and Large Maps},
            author={Honerkamp, Daniel and Guttikonda, Suresh and Valada, Abhinav},
            journal={arXiv preprint arXiv:2209.09646},
            year={2022},
   }
```

The PF-net module in this repository is based on https://github.com/AdaCompNUS/pfnet and implemented in tensorflow. The RL agents are implemented in pytorch based on stable-baselines3.

## Installation

1. Create conda environment
   ```
      conda env create -f environment.yaml
      conda activate active_localization
      export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$CONDA_PREFIX/lib/
   ```
2. Setup iGibson2.0 environment (forked repository). For latest installation procedure always refer to official doc [link](http://svl.stanford.edu/igibson/docs/installation.html)
   ```
      git clone --branch master https://github.com/suresh-guttikonda/iGibson.git --recursive
      cd iGibson
      pip3 install -e .
   ```
3. Test igibson installation is successful
   ```
      python
      >>> import igibson
   ```
4. Download required igibson's assets (robot's urdf, demo apartment, and others). For more datasets [refer](http://svl.stanford.edu/igibson/docs/dataset.html)
   ```
      <root_folder>/iGibson$ python -m igibson.utils.assets_utils --download_assets
      <root_folder>/iGibson$ python -m igibson.utils.assets_utils --download_demo_data
   ```
5. For locobot robot, we modify urdf file by adding additional lidar sensor 'scan_link' link and joint to 'base_link'. For reference, turtlebot's urdf file has the sensor by default.
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


6. Test tensorflow installation is successful
   ```
      python3 -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"

   ```
   if it doesn't detect the gpu, try:
   ```
   pip install --upgrade pip
   pip uninstall tensorflow
   pip install tensorflow==2.6.0
   pip install numpy==1.21.1
   ```
7. Test if pytorch detects the GPU:
   ```
   import torch
   torch.cuda.is_available()
   torch.Tensor([0]).to('cuda')
   ```

## Run
Please activate the conda environment and start the following commands from `src/rl_agents`.

Supervised data collection:
```
python -u supervised_data.py --device_idx 0 --agent goalnav_agent --num_records 1000 --custom_output rgb_obs depth_obs occupancy_grid task_obs obstacle_obs --global_map_size 1000 1000 1
```

Pretrain PF-net:
```
python -u train_pfnet.py --device_idx=0 --root_dir=logs/pfnet_below1000_lidar --tfrecordpath=/data/honerkam/pfnet_data/ --epochs=100 --obs_mode=occupancy_grid --num_train_samples=4000 --num_eval_samples=500 --batch_size=8 --pfnet_loadpath='' --learning_rate=5e-5 --init_particles_distr=gaussian --init_particles_std '0.3' '0.523599' --particles_range=100 --num_particles=30 --transition_std '0.' '0.' --resample=false --alpha_resample_ratio=0.5 --global_map_size 1000 1000 1 --seed=42
```

Train RL agent:
```
python -u sbl_train_eval.py --device_idx 0 --scene_id all --num_parallel_environments 7 --custom_output task_obs likelihood_map occupancy_grid depth_obs rgb_obs --obs_mode occupancy_grid --global_map_size 1000 1000 1 --replay_buffer_capacity 50000 --initial_collect_steps 0 --resample yes --alpha_resample_ratio 0.5 --num_particles 250 --eval_interval 500 --pfnet_loadpath /home/honerkam/repos/deep-activate-localization/src/rl_agents/logs/pfnet_below1000_lidar030/train_navagent_below1000/chks/checkpoint_95_0.157/pfnet_checkpoint --init_particles_distr uniform --particles_range 10 --trajlen 50 --rl_architecture 2
```

Eval RL agent:
```
python -u  eval_agents.py --device_idx 0 --custom_output rgb_obs depth_obs likelihood_map obstacle_obs occupancy_grid --obs_mode occupancy_grid --pfnet_loadpath /home/honerkam/repos/deep-activate-localization/src/rl_agents/logs/pfnet_below1000_lidar030/train_navagent_below1000/chks/checkpoint_95_0.157/pfnet_checkpoint --agent goalnav_agent --eval_only --use_plot --store_plot --num_eval_episodes 50 --scene_id all --global_map_size 1000 1000 1 --resume_id [enter wandb run id]
```
