#!/bin/sh
env="MPE"
scenario="simple_spread"
num_landmarks=3
num_agents=3
algo="rmaddpg"
exp="debug"
seed=1


CUDA_VISIBLE_DEVICES=7 python train/train_mpe.py --env_name ${env} --algorithm_name ${algo} --experiment_name ${exp} --scenario_name ${scenario} --num_agents 3 --num_landmarks ${num_landmarks} --seed 10 --episode_length 25 --actor_train_interval_step 1 --tau 0.005 --lr 7e-4 --num_env_steps 3000000 --use_reward_normalization --sym_ipt 0.01 --mu 0.5 --sigma 0.5
CUDA_VISIBLE_DEVICES=7 python train/train_mpe.py --env_name ${env} --algorithm_name ${algo} --experiment_name ${exp} --scenario_name ${scenario} --num_agents 3 --num_landmarks ${num_landmarks} --seed 10 --episode_length 25 --actor_train_interval_step 1 --tau 0.005 --lr 7e-4 --num_env_steps 3000000 --use_reward_normalization --sym_ipt 0.05  --mu 0.5 --sigma 0.5
CUDA_VISIBLE_DEVICES=7 python train/train_mpe.py --env_name ${env} --algorithm_name ${algo} --experiment_name ${exp} --scenario_name ${scenario} --num_agents 3 --num_landmarks ${num_landmarks} --seed 10 --episode_length 25 --actor_train_interval_step 1 --tau 0.005 --lr 7e-4 --num_env_steps 3000000 --use_reward_normalization --sym_ipt 0.1 --mu 0. --sigma 0.5
CUDA_VISIBLE_DEVICES=7 python train/train_mpe.py --env_name ${env} --algorithm_name ${algo} --experiment_name ${exp} --scenario_name ${scenario} --num_agents 3 --num_landmarks ${num_landmarks} --seed 10 --episode_length 25 --actor_train_interval_step 1 --tau 0.005 --lr 7e-4 --num_env_steps 3000000 --use_reward_normalization --sym_ipt 0.2 --mu 0.5 --sigma 0.5
CUDA_VISIBLE_DEVICES=7 python train/train_mpe.py --env_name ${env} --algorithm_name ${algo} --experiment_name ${exp} --scenario_name ${scenario} --num_agents 3 --num_landmarks ${num_landmarks} --seed 10 --episode_length 25 --actor_train_interval_step 1 --tau 0.005 --lr 7e-4 --num_env_steps 3000000 --use_reward_normalization --sym_ipt 0.5 --mu 0.5 --sigma 0.5
