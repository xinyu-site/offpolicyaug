import numpy as np
import torch
import time

from offpolicy.runner.rnn.base_runner import RecRunner


class MPERunner(RecRunner):
    """Runner class for Multiagent Particle Envs (MPE). See parent class for more information."""

    def __init__(self, config):
        super(MPERunner, self).__init__(config)
        self.collecter = self.shared_collect_rollout if self.share_policy else self.separated_collect_rollout
        # fill replay buffer with random actions
        # fill replay buffer with random actions
        num_warmup_episodes = max((self.batch_size, self.args.num_random_episodes))
        self.warmup(num_warmup_episodes)
        self.start = time.time()
        self.log_clear()

    def eval(self):
        """Collect episodes to evaluate the policy."""
        self.trainer.prep_rollout()
        eval_infos = {}
        eval_infos['average_episode_rewards'] = []

        for _ in range(self.args.num_eval_episodes):
            env_info = self.collecter(explore=False, training_episode=False, warmup=False)
            for k, v in env_info.items():
                eval_infos[k].append(v)

        self.log_env(eval_infos, suffix="eval_")

    # for mpe-simple_spread and mpe-simple_reference  
    @torch.no_grad()
    def shared_collect_rollout(self, explore=True, training_episode=True, warmup=False):

        """
        Collect a rollout and store it in the buffer. All agents share a single policy.
        :param explore: (bool) whether to use an exploration strategy when collecting the episoide.
        :param training_episode: (bool) whether this episode is used for evaluation or training.
        :param warmup: (bool) whether this episode is being collected during warmup phase.

        :return env_info: (dict) contains information about the rollout (total rewards, etc).
        """
        env_info = {}
        # only 1 policy since all agents share weights
        p_id = "policy_0"
        policy = self.policies[p_id]

        env = self.env if training_episode or warmup else self.eval_env

        def rotation(theta):
            a = np.ones((2, 2))
            a[0, 0] = np.cos(theta)
            a[0, 1] = np.sin(theta)
            a[1, 0] = -np.sin(theta)
            a[1, 1] = np.cos(theta)

            return a

        def flip(axis):
            a = np.ones((2, 2))
            if (axis == 0):
                a[0, 0] = 1
                a[0, 1] = 0
                a[1, 0] = 0
                a[1, 1] = -1
            else:
                a[0, 0] = -1
                a[0, 1] = 0
                a[1, 0] = 0
                a[1, 1] = 1
            return a
    
        def generate_mask_array(repeats, n):
            arrays = np.ones((1, repeats, n), dtype=int)
            for i in range(repeats):
                zero_position=np.random.choice(n)
                arrays[0, i, zero_position] = 0
            return arrays


        def swap_dimensions(array, dim1, dim2):
            temp = np.copy(array[:, :, dim1:dim1+2])
            array[:, :, dim1:dim1+2] = array[:, :, dim2:dim2+2]
            array[:, :, dim2:dim2+2] = temp
            return array
        # flip x 0
        # [-1  0]
        # [ 0  1]
        # flp y 1
        # [ 1  0]
        # [ 0 -1]

        obs = env.reset()

        sym_obs = np.dot(obs.reshape(self.num_envs, self.num_agents, -1, 2), rotation(0.5 * np.pi)).reshape(self.num_envs, self.num_agents, -1)
        sym_obs180 = np.dot(obs.reshape(self.num_envs, self.num_agents, -1, 2), rotation(np.pi)).reshape(self.num_envs, self.num_agents, -1)
        sym_obs270 = np.dot(obs.reshape(self.num_envs, self.num_agents, -1, 2), rotation(1.5 * np.pi)).reshape(self.num_envs, self.num_agents, -1)
        sym_obs_flipx = np.dot(obs.reshape(self.num_envs, self.num_agents, -1, 2), flip(0)).reshape(self.num_envs, self.num_agents, -1)
        sym_obs_flipy = np.dot(obs.reshape(self.num_envs, self.num_agents, -1, 2), flip(1)).reshape(self.num_envs, self.num_agents, -1)

        #add gaussian noise for obs 
        gaussian = np.random.normal(0,0.1,obs.shape)
        obs_gaussian = obs + gaussian

        #add uniform noise for obs
        uniform = np.random.uniform(-0.1, 0.1, obs.shape)
        obs_uniform = obs + uniform
       
        #add dropout for obs
        mask = generate_mask_array(self.num_agents, obs.shape[-1])
        obs_dim_drop_out =  mask*obs
        
        #add swap for obs
        obs_swap_dim = obs.copy()
        obs_swap_dim = swap_dimensions(obs_swap_dim, 10, 12)     
        
        sym_obs_list = [sym_obs, sym_obs180, sym_obs270, sym_obs_flipx, sym_obs_flipy, obs_gaussian, obs_uniform, obs_dim_drop_out, obs_swap_dim]

        rnn_states_batch = np.zeros((self.num_envs * self.num_agents, self.hidden_size), dtype=np.float32)
        last_acts_batch = np.zeros((self.num_envs * self.num_agents, policy.output_dim), dtype=np.float32)

        # initialize variables to store episode information.
        episode_obs = {p_id: np.zeros((self.episode_length + 1, self.num_envs, self.num_agents, policy.obs_dim), dtype=np.float32) for p_id in self.policy_ids}
        episode_share_obs = {p_id: np.zeros((self.episode_length + 1, self.num_envs, self.num_agents, policy.central_obs_dim), dtype=np.float32) for p_id in self.policy_ids}
        episode_acts = {p_id: np.zeros((self.episode_length, self.num_envs, self.num_agents, policy.output_dim), dtype=np.float32) for p_id in self.policy_ids}
        episode_rewards = {p_id: np.zeros((self.episode_length, self.num_envs, self.num_agents, 1), dtype=np.float32) for p_id in self.policy_ids}
        episode_dones = {p_id: np.ones((self.episode_length, self.num_envs, self.num_agents, 1), dtype=np.float32) for p_id in self.policy_ids}
        episode_dones_env = {p_id: np.ones((self.episode_length, self.num_envs, 1), dtype=np.float32) for p_id in self.policy_ids}
        # 星际争霸中有用
        episode_avail_acts = {p_id: None for p_id in self.policy_ids}


        sy_episode_obs_list = []
        sy_episode_share_obs_list = []
        sy_episode_acts_list = []
        for i in range(self.num_first_ucb):
            sy_episode_obs_list.append({p_id: np.zeros((self.episode_length + 1, self.num_envs, self.num_agents, policy.obs_dim), dtype=np.float32) for p_id in self.policy_ids})
            sy_episode_share_obs_list.append({p_id: np.zeros((self.episode_length + 1, self.num_envs, self.num_agents, policy.central_obs_dim), dtype=np.float32) for p_id in self.policy_ids})
            sy_episode_acts_list.append({p_id: np.zeros((self.episode_length, self.num_envs, self.num_agents, policy.output_dim), dtype=np.float32) for p_id in self.policy_ids})

        keys = []
        for i in range(5):
            orig = np.zeros(5)
            orig[i] = 1
            keys.append(orig)

        mapping_ori = np.array([keys[0], keys[1], keys[2], keys[3], keys[4]])
        mapping = np.array([keys[0], keys[3], keys[4], keys[2], keys[1]])
        mapping180 = np.array([keys[0], keys[2], keys[1], keys[4], keys[3]])
        mapping270 = np.array([keys[0], keys[4], keys[3], keys[1], keys[2]])
        mapping_flipx = np.array([keys[0], keys[1], keys[2], keys[4], keys[3]])
        mapping_flipy = np.array([keys[0], keys[2], keys[1], keys[3], keys[4]])
        mapping_list = [mapping, mapping180, mapping270, mapping_flipx, mapping_flipy, mapping_ori, mapping_ori, mapping_ori, mapping_ori]

        def symmetry_action(origin_action, mapping):
            out = mapping[np.argmax(origin_action, axis=2)]
            return [x for x in out]

        t = 0
        while t < self.episode_length:
            share_obs = obs.reshape(self.num_envs, -1)
            sym_share_obs_list = []
            for i in range(self.num_first_ucb - 1):
                sym_share_obs_list.append(sym_obs_list[i].reshape(self.num_envs, -1))

            # group observations from parallel envs into one batch to process at once
            obs_batch = np.concatenate(obs)
            # get actions for all agents to step the env
            if warmup:
                # completely random actions in pre-training warmup phase
                acts_batch = policy.get_random_actions(obs_batch)
                # get new rnn hidden state
                _, rnn_states_batch, _ = policy.get_actions(obs_batch, last_acts_batch, rnn_states_batch)
            else:
                # get actions with exploration noise (eps-greedy/Gaussian)
                acts_batch, rnn_states_batch, _ = policy.get_actions(obs_batch, last_acts_batch, rnn_states_batch, t_env=self.total_env_steps, explore=explore)
            acts_batch = acts_batch if isinstance(acts_batch, np.ndarray) else acts_batch.cpu().detach().numpy()
            # update rnn hidden state
            rnn_states_batch = rnn_states_batch if isinstance(rnn_states_batch, np.ndarray) else rnn_states_batch.cpu().detach().numpy()
            last_acts_batch = acts_batch

            env_acts = np.split(acts_batch, self.num_envs)

            sym_env_acts_list = []
            for i in range(self.num_first_ucb - 1):
                sym_env_acts_list.append(symmetry_action(np.array(env_acts), mapping_list[i]))

            # env step and store the relevant episode information
            next_obs, rewards, dones, infos = env.step(env_acts)
            if training_episode:
                self.total_env_steps += self.num_envs

            dones_env = np.all(dones, axis=1)
            terminate_episodes = np.any(dones_env) or t == self.episode_length - 1
            # episode_share_obs = {p_id: np.zeros((self.episode_length + 1, 
            #                                      self.num_envs, 
            #                                      self.num_agents, 
            #                                      policy.central_obs_dim), dtype=np.float32) for p_id in self.policy_ids}
            episode_obs[p_id][t] = obs
            episode_share_obs[p_id][t] = share_obs
            episode_acts[p_id][t] = np.stack(env_acts)
            episode_rewards[p_id][t] = rewards
            episode_dones[p_id][t] = dones
            episode_dones_env[p_id][t] = dones_env

            for i in range(self.num_first_ucb - 1):
                sy_episode_obs_list[i][p_id][t] = sym_obs_list[i]
                sy_episode_share_obs_list[i][p_id][t] = sym_share_obs_list[i]
                sy_episode_acts_list[i][p_id][t] = np.stack(sym_env_acts_list[i])

            '''
            sy_episode_rewards[p_id][t] = rewards
            sy_episode_dones[p_id][t] = dones
            sy_episode_dones_env[p_id][t] = dones_env
            '''

            t += 1

            obs = next_obs
            sym_obs = np.dot(next_obs.reshape(self.num_envs, self.num_agents, -1, 2), rotation(0.5 * np.pi)).reshape(self.num_envs, self.num_agents, -1)
            sym_obs180 = np.dot(next_obs.reshape(self.num_envs, self.num_agents, -1, 2), rotation(np.pi)).reshape(self.num_envs, self.num_agents, -1)
            sym_obs270 = np.dot(next_obs.reshape(self.num_envs, self.num_agents, -1, 2), rotation(1.5 * np.pi)).reshape(self.num_envs, self.num_agents, -1)
            sym_obs_flipx = np.dot(next_obs.reshape(self.num_envs, self.num_agents, -1, 2), flip(0)).reshape(self.num_envs, self.num_agents, -1)
            sym_obs_flipy = np.dot(next_obs.reshape(self.num_envs, self.num_agents, -1, 2), flip(1)).reshape(self.num_envs, self.num_agents, -1)
            #add gaussian noise for obs 
            gaussian = np.random.normal(0,0.1,obs.shape)
            obs_gaussian = obs + gaussian

            #add uniform noise for obs
            uniform = np.random.uniform(-0.1, 0.1, obs.shape)
            obs_uniform = obs + uniform
        
            #add dropout for obs
            
            obs_dim_drop_out =  mask*obs
            
            #add swap for obs
            obs_swap_dim = obs.copy()
            obs_swap_dim = swap_dimensions(obs_swap_dim, 10, 12)     
            
            sym_obs_list = [sym_obs, sym_obs180, sym_obs270, sym_obs_flipx, sym_obs_flipy, obs_gaussian, obs_uniform, obs_dim_drop_out, obs_swap_dim]

            if terminate_episodes:
                break

        episode_obs[p_id][t] = obs
        episode_share_obs[p_id][t] = obs.reshape(self.num_envs, -1)

        # 插入mixup数据
        alpha = 0.5
        obs_temp = []
        share_obs_temp = []
        acts_temp = []
        for i in range(len(episode_obs[p_id]) - 1):
            obs_temp.append(alpha * episode_obs[p_id][i] + (1 - alpha) * episode_obs[p_id][i + 1])
            share_obs_temp.append(alpha * episode_share_obs[p_id][i] + (1 - alpha) * episode_share_obs[p_id][i + 1])
        obs_temp.append(episode_obs[p_id][-1])
        share_obs_temp.append(episode_share_obs[p_id][-1])

        for i in range(len(episode_obs[p_id]) - 2):
            acts_temp.append(alpha * episode_acts[p_id][i] + (1 - alpha) * episode_acts[p_id][i + 1])
        acts_temp.append(episode_acts[p_id][-1])
        sy_episode_obs_list[self.num_first_ucb - 1][p_id] = np.array(obs_temp)
        sy_episode_share_obs_list[self.num_first_ucb - 1][p_id] = np.array(share_obs_temp)
        sy_episode_acts_list[self.num_first_ucb - 1][p_id] = np.array(acts_temp)


        if explore:
            self.num_episodes_collected += self.num_envs
            # push all episodes collected in this rollout step to the buffer

            # print(episode_obs[p_id].shape)
            # print(episode_share_obs[p_id].shape)
            # print(episode_acts[p_id].shape)
            # print(episode_acts[p_id][0])


            self.buffer.insert(self.num_envs,
                               episode_obs,
                               episode_share_obs,
                               episode_acts,
                               episode_rewards,
                               episode_dones,
                               episode_dones_env,
                               episode_avail_acts)

            if self.args.data_aug and training_episode:
                self.log_ucb()
                self.selected_ucb.append((int(self.first_aug_id), int(self.second_aug_id)))
                for i in self.all_selections[self.first_aug_id][self.second_aug_id]:
                    self.buffer.insert(self.num_envs,
                                       sy_episode_obs_list[i],
                                       sy_episode_share_obs_list[i],
                                       sy_episode_acts_list[i],
                                       episode_rewards,
                                       episode_dones,
                                       episode_dones_env,
                                       episode_avail_acts)
        average_episode_rewards = np.mean(np.sum(episode_rewards[p_id], axis=0))
        env_info['average_episode_rewards'] = average_episode_rewards
        # if training_episode:
        #     self.origin_qval.append(average_episode_rewards)

        return env_info

    # for mpe-simple_speaker_listener
    @torch.no_grad()
    def separated_collect_rollout(self, explore=True, training_episode=True, warmup=False):
        """
        Collect a rollout and store it in the buffer. Each agent has its own policy.
        :param explore: (bool) whether to use an exploration strategy when collecting the episoide.
        :param training_episode: (bool) whether this episode is used for evaluation or training.
        :param warmup: (bool) whether this episode is being collected during warmup phase.

        :return env_info: (dict) contains information about the rollout (total rewards, etc).
        """
        env_info = {}
        env = self.env if training_episode or warmup else self.eval_env

        obs = env.reset()

        rnn_states = np.zeros((self.num_agents, self.num_envs, self.hidden_size), dtype=np.float32)

        last_acts = {p_id: np.zeros((self.num_envs, len(self.policy_agents[p_id]), self.policies[p_id].output_dim),
                                    dtype=np.float32) for p_id in self.policy_ids}
        episode_obs = {p_id: np.zeros(
            (self.episode_length + 1, self.num_envs, len(self.policy_agents[p_id]), self.policies[p_id].obs_dim),
            dtype=np.float32) for p_id in self.policy_ids}
        episode_share_obs = {p_id: np.zeros((self.episode_length + 1, self.num_envs, len(self.policy_agents[p_id]),
                                             self.policies[p_id].central_obs_dim), dtype=np.float32) for p_id in
                             self.policy_ids}
        episode_acts = {p_id: np.zeros(
            (self.episode_length, self.num_envs, len(self.policy_agents[p_id]), self.policies[p_id].output_dim),
            dtype=np.float32) for p_id in self.policy_ids}
        episode_rewards = {
            p_id: np.zeros((self.episode_length, self.num_envs, len(self.policy_agents[p_id]), 1), dtype=np.float32) for
            p_id in self.policy_ids}
        episode_dones = {
            p_id: np.ones((self.episode_length, self.num_envs, len(self.policy_agents[p_id]), 1), dtype=np.float32) for
            p_id in self.policy_ids}
        episode_dones_env = {p_id: np.ones((self.episode_length, self.num_envs, 1), dtype=np.float32) for p_id in
                             self.policy_ids}
        episode_avail_acts = {p_id: None for p_id in self.policy_ids}

        t = 0
        while t < self.episode_length:
            for agent_id, p_id in zip(self.agent_ids, self.policy_ids):
                policy = self.policies[p_id]
                agent_obs = np.stack(obs[:, agent_id])
                share_obs = np.concatenate([obs[0, i] for i in range(self.num_agents)]).reshape(self.num_envs,
                                                                                                -1).astype(np.float32)
                # get actions for all agents to step the env
                if warmup:
                    # completely random actions in pre-training warmup phase
                    # [parallel envs, agents, dim]
                    act = policy.get_random_actions(agent_obs)
                    # get new rnn hidden state
                    _, rnn_state, _ = policy.get_actions(agent_obs,
                                                         last_acts[p_id][:, 0],
                                                         rnn_states[agent_id])
                else:
                    # get actions with exploration noise (eps-greedy/Gaussian)
                    if self.algorithm_name == "rmasac":
                        act, rnn_state, _ = policy.get_actions(agent_obs,
                                                               last_acts[p_id],
                                                               rnn_states[agent_id],
                                                               sample=explore)
                    else:
                        act, rnn_state, _ = policy.get_actions(agent_obs,
                                                               last_acts[p_id].squeeze(axis=0),
                                                               rnn_states[agent_id],
                                                               t_env=self.total_env_steps,
                                                               explore=explore)
                # update rnn hidden state
                rnn_states[agent_id] = rnn_state if isinstance(rnn_state,
                                                               np.ndarray) else rnn_state.cpu().detach().numpy()
                last_acts[p_id] = np.expand_dims(act, axis=1) if isinstance(act, np.ndarray) else np.expand_dims(
                    act.cpu().detach().numpy(), axis=1)

                episode_obs[p_id][t] = agent_obs
                episode_share_obs[p_id][t] = share_obs
                episode_acts[p_id][t] = act

            env_acts = []
            for i in range(self.num_envs):
                env_act = []
                for p_id in self.policy_ids:
                    env_act.append(last_acts[p_id][i, 0])
                env_acts.append(env_act)

            # env step and store the relevant episode information
            next_obs, rewards, dones, infos = env.step(env_acts)

            dones_env = np.all(dones, axis=1)
            terminate_episodes = np.any(dones_env) or t == self.episode_length - 1
            if terminate_episodes:
                dones_env = np.ones_like(dones_env).astype(bool)

            for agent_id, p_id in zip(self.agent_ids, self.policy_ids):
                episode_rewards[p_id][t] = np.expand_dims(rewards[:, agent_id], axis=1)
                episode_dones[p_id][t] = np.expand_dims(dones[:, agent_id], axis=1)
                episode_dones_env[p_id][t] = dones_env

            obs = next_obs
            t += 1

            if training_episode:
                self.total_env_steps += self.num_envs

            if terminate_episodes:
                break

        for agent_id, p_id in zip(self.agent_ids, self.policy_ids):
            episode_obs[p_id][t] = np.stack(obs[:, agent_id])
            episode_share_obs[p_id][t] = np.concatenate([obs[0, i] for i in range(self.num_agents)]).reshape(
                self.num_envs,
                -1).astype(np.float32)

        if explore:
            self.num_episodes_collected += self.num_envs
            self.buffer.insert(self.num_envs, episode_obs, episode_share_obs, episode_acts, episode_rewards,
                               episode_dones, episode_dones_env, episode_avail_acts)

        average_episode_rewards = []
        for p_id in self.policy_ids:
            average_episode_rewards.append(np.mean(np.sum(episode_rewards[p_id], axis=0)))

        env_info['average_episode_rewards'] = np.mean(average_episode_rewards)

        return env_info

    def log(self):
        """See parent class."""
        end = time.time()
        print("\n Env {} Algo {} Exp {} runs total num timesteps {}/{}, FPS {}.\n"
              .format(self.args.scenario_name,
                      self.algorithm_name,
                      self.args.experiment_name,
                      self.total_env_steps,
                      self.num_env_steps,
                      int(self.total_env_steps / (end - self.start))))
        for p_id, train_info in zip(self.policy_ids, self.train_infos):
            self.log_train(p_id, train_info)

        self.log_env(self.env_infos)
        self.log_clear()

    def log_clear(self):
        """See parent class."""
        self.env_infos = {}

        self.env_infos['average_episode_rewards'] = []