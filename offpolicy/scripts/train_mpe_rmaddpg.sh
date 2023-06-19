#!/bin/sh
env="MPE"
scenario="simple_spread"
num_landmarks=3
num_agents=3
algo="rmaddpg"
exp="debug"
seed_max=5


seed=10

echo "env is ${seed}"


CUDA_VISIBLE_DEVICES=7 python train/train_mpe.py --env_name ${env} --algorithm_name ${algo} --experiment_name ${exp} --scenario_name ${scenario} --num_agents 3 --num_landmarks ${num_landmarks} --seed 10 --episode_length 25 --actor_train_interval_step 1 --tau 0.005 --lr 7e-4 --num_env_steps 3000000 --use_reward_normalization  --data_aug --use_sym_loss

CUDA_VISIBLE_DEVICES=7 python train/train_mpe.py --env_name ${env} --algorithm_name ${algo} --experiment_name ${exp} --scenario_name ${scenario} --num_agents 3 --num_landmarks ${num_landmarks} --seed 10 --episode_length 25 --actor_train_interval_step 1 --tau 0.005 --lr 7e-4 --num_env_steps 3000000 --use_reward_normalization  --first_aug 0 --second_aug 0


