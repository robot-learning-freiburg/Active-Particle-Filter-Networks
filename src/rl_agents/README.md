# deep-activate-localization/src/rl_agents

#### Steps to train agent:
1. `python -u train_eval.py --root_dir 'train_output' --config_file './configs/turtlebot_point_nav.yaml'--num_iterations 3000 --initial_collect_steps 500 --collect_steps_per_iteration 1 --num_parallel_environments 1 --num_parallel_environments_eval 1 --replay_buffer_capacity 1000 --train_steps_per_iteration 1 --batch_size 16 --num_eval_episodes 10 --eval_interval 500 --gpu_c 0 --env_mode 'headless' --gpu_g 0`
