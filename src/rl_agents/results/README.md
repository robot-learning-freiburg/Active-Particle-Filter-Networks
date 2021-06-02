# deep-activate-localization/src/rl_agents/results

<details close>
  <summary>April 28th - May 4th Summary</summary>

  ### Results
  *blue - training curve, orange - evaluation curve*
  |Experiement | Result     | AverageEpisodeLength      | AverageEpisodeReturn      |
  |------------|------------|------------|-------------|
  |task_obs only (2021-04-30)| [Navigate_Fixed_Goal - (parallel_py_env)](2021-04-30_12-19-33)|![Metrics_AverageEpisodeLength](2021-04-30_12-19-33/images/Metrics_AverageEpisodeLength.svg)|![Metrics_AverageReturn](2021-04-30_12-19-33/images/Metrics_AverageReturn.svg)| -> different train and eval envs with normal_projection_network.NormalProjectionNetwork
  |task_obs only (2021-04-30)| [Navigate_Fixed_Goal - (non-parallel_py_env-1)](2021-04-30_15-00-08)|![Metrics_AverageEpisodeLength](2021-04-30_15-00-08/images/Metrics_AverageEpisodeLength.svg)|![Metrics_AverageReturn](2021-04-30_15-00-08/images/Metrics_AverageReturn.svg)| -> different train and eval envs with normal_projection_network.NormalProjectionNetwork
  |task_obs only (2021-04-30)| [Navigate_Fixed_Goal - (non-parallel_py_env-2)](2021-04-30_20-07-39)|![Metrics_AverageEpisodeLength](2021-04-30_20-07-39/images/Metrics_AverageEpisodeLength.svg)|![Metrics_AverageReturn](2021-04-30_20-07-39/images/Metrics_AverageReturn.svg)| -> same train and eval envs with normal_projection_network.NormalProjectionNetwork
  |rgb_obs only (2021-05-03)| [Navigate_Fixed_Goal - (parallel_py_env)](2021-05-03_14-54-25)|![Metrics_AverageEpisodeLength](2021-05-03_14-54-25/images/Metrics_AverageEpisodeLength.svg)|![Metrics_AverageReturn](2021-05-03_14-54-25/images/Metrics_AverageReturn.svg)| -> different train and eval envs with tanh_normal_projection_network.TanhNormalProjectionNetwork
  |rgb_obs only (2021-05-04)| [Navigate_Fixed_Goal - (non-parallel_py_env)](2021-05-04_07-42-43)|![Metrics_AverageEpisodeLength](2021-05-04_07-42-43/images/Metrics_AverageEpisodeLength.svg)|![Metrics_AverageReturn](2021-05-04_07-42-43/images/Metrics_AverageReturn.svg)| -> different train and eval envs with tanh_normal_projection_network.TanhNormalProjectionNetwork sac_agent.py

</details>

<details close>
  <summary>May 5th - May 11th Summary</summary>

  ### Results
  *blue - training curve, orange - evaluation curve*
  |Experiement | Agent |  Result     | AverageEpisodeLength      | AverageEpisodeReturn      |
  |------------|-------|-----|------------|-------------|
  |task_obs only (2021-05-06)| SAC |[Navigate_Fixed_Goal - (non-parallel_py_env)](2021-05-06_10-06-32)|![Metrics_AverageEpisodeLength](2021-05-06_10-06-32/images/Metrics_AverageEpisodeLength.svg)|![Metrics_AverageReturn](2021-05-06_10-06-32/images/Metrics_AverageReturn.svg)| -> same train and eval envs with normal_projection_network.NormalProjectionNetwork train_eval.py
  |rgb_obs only (2021-05-07)| SAC |[Navigate_Fixed_Goal - (non-parallel_py_env)](2021-05-07_00-07-34)|![Metrics_AverageEpisodeLength](2021-05-07_00-07-34/images/Metrics_AverageEpisodeLength.svg)|![Metrics_AverageReturn](2021-05-07_00-07-34/images/Metrics_AverageReturn.svg)| -> same train and eval envs with tanh_normal_projection_network.TanhNormalProjectionNetwork train_eval.py
  |task_obs only (2021-05-12)| PPOClipAgent |[Navigate_Fixed_Goal - (non-parallel_py_env)](2021-05-12_12-46-55)|![Metrics_AverageEpisodeLength](2021-05-12_12-46-55/images/Metrics_AverageEpisodeLength.svg)|![Metrics_AverageReturn](2021-05-12_12-46-55/images/Metrics_AverageReturn.svg)| -> same train and eval envs with tanh activation non-mini batch training
  |rgb_obs only (2021-05-17)| PPOClipAgent |[Navigate_Fixed_Goal - (non-parallel_py_env)](2021-05-17_08-16-35)|![Metrics_AverageEpisodeLength](2021-05-17_08-16-35/images/Metrics_AverageEpisodeLength.svg)|![Metrics_AverageReturn](2021-05-17_08-16-35/images/Metrics_AverageReturn.svg)| -> same train and eval envs with tanh activation non-mini batch training

</details>


<details open>
  <summary>May 26th - May 2nd Summary</summary>

  ### Results
  *<span style="color:blue">blue</span> - random agent, <span style="color:red">red</span> - sac trained agent*

  |With Out Transition Noise |With Transition Noise |With Lower Initial Covariance |With More Particles |
  |------------|------------|------------|------------|
  |![Eval_WithOut_Noise](2021-05-29_11-11-00/images/eval_wo_noise.svg)|![Eval_With_Noise](2021-05-29_11-11-00/images/eval_w_noise.svg)|![Eval_Low_Init_Covariance](2021-05-29_11-11-00/images/eval_low_init_covariance.svg)|![Eval_More_Particles](2021-05-29_11-11-00/images/eval_more_particles.svg)|
  |transition_std: ['0', '0']<br/>init_particles_std: ['30', '0.523599']<br/>init_particles_distr: gaussian<br/>num_particles: 1000<br/>alpha_resample_ratio 0.8|transition_std: ['1', '0']<br/>init_particles_std ['30', '0.523599']<br/>init_particles_distr: gaussian<br/>num_particles: 1000<br/>alpha_resample_ratio 0.8|transition_std: ['0.2', '0']<br/>init_particles_std ['15', '0.523599']<br/>init_particles_distr: gaussian<br/>num_particles: 1500<br/>alpha_resample_ratio 0.8|transition_std: ['0.5', '0']<br/>init_particles_std ['30', '0.523599']<br/>init_particles_distr: gaussian<br/>num_particles: 2500<br/>alpha_resample_ratio 0.8|

</details>
