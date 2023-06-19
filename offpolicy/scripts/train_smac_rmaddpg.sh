#!/bin/sh
env="MPE"
scenario="simple_spread"
num_landmarks=3
num_agents=3
algo="rmaddpg"
exp="debug"
seed=2
echo "seed is ${seed}:"
CUDA_VISIBLE_DEVICES=2 python train/train_mpe.py --env_name ${env} --algorithm_name ${algo} --experiment_name ${exp} --scenario_name ${scenario} --num_agents ${num_agents} --num_landmarks ${num_landmarks} --seed ${seed} --episode_length 25 --actor_train_interval_step 1 --tau 0.005 --lr 7e-4 --num_env_steps 3000000 --use_reward_normalization   
echo "training is done!"

seed=3
echo "seed is ${seed}:"
CUDA_VISIBLE_DEVICES=2 python train/train_mpe.py --env_name ${env} --algorithm_name ${algo} --experiment_name ${exp} --scenario_name ${scenario} --num_agents ${num_agents} --num_landmarks ${num_landmarks} --seed ${seed} --episode_length 25 --actor_train_interval_step 1 --tau 0.005 --lr 7e-4 --num_env_steps 3000000 --use_reward_normalization   
echo "training is done!"

seed=4
echo "seed is ${seed}:"
CUDA_VISIBLE_DEVICES=2 python train/train_mpe.py --env_name ${env} --algorithm_name ${algo} --experiment_name ${exp} --scenario_name ${scenario} --num_agents ${num_agents} --num_landmarks ${num_landmarks} --seed ${seed} --episode_length 25 --actor_train_interval_step 1 --tau 0.005 --lr 7e-4 --num_env_steps 3000000 --use_reward_normalization   
echo "training is done!"

seed=5
echo "seed is ${seed}:"
CUDA_VISIBLE_DEVICES=2 python train/train_mpe.py --env_name ${env} --algorithm_name ${algo} --experiment_name ${exp} --scenario_name ${scenario} --num_agents ${num_agents} --num_landmarks ${num_landmarks} --seed ${seed} --episode_length 25 --actor_train_interval_step 1 --tau 0.005 --lr 7e-4 --num_env_steps 3000000 --use_reward_normalization   
echo "training is done!"