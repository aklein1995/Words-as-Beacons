from abc import ABC, abstractmethod
import torch
from collections import deque
import numpy as np
import os
import csv
from tensorboardX import SummaryWriter

from torch_ac.utils.dictlist import DictList

def default_preprocess_obss(obss, device=None):
    return torch.tensor(obss, device=device)

class BaseAlgo(ABC):
    """The base class for RL algorithms"""

    def __init__(self, envs, eval_envs, eval_episodes,
                 acmodel, device, num_frames_per_proc, discount, lr,
                 gae_lambda, entropy_coef, value_loss_coef, max_grad_norm,
                 preprocess_obss, num_actions, use_subgoal, subgoal_type,
                 subgoal_reward_value, total_num_frames):
        """
        Initializes a `BaseAlgo` instance.

        Parameters:
        ----------
        envs : list
            a list of environments that will be run in parallel
        acmodel : torch.Module or tuple of torch.Module(s)
            the model(s); the separated_actor_critic parameter defines that
        num_frames_per_proc : int
            the number of frames collected by every process for an update
        discount : float
            the discount for future rewards
        lr : float
            the learning rate for optimizers
        gae_lambda : float
            the lambda coefficient in the GAE formula
            ([Schulman et al., 2015](https://arxiv.org/abs/1506.02438))
        entropy_coef : float
            the weight of the entropy cost in the final objective
        value_loss_coef : float
            the weight of the value loss in the final objective
        max_grad_norm : float
            gradient will be clipped to be at most this value
        recurrence : int
            the number of steps the gradient is propagated back in time
        preprocess_obss : function
            a function that takes observations returned by the environment
            and converts them into the format that the model can handle
        separated_networks: boolean
            set whether we are going to use a single AC neural network or
            two differents
        """
        
        self.acmodel = acmodel

        self.num_actions = num_actions
        self.env = envs
        self.env_max_steps = self.env[0].max_steps
        self.evaluation_envs = eval_envs
        self.evaluation_episodes = eval_episodes
        self.eval_writers = {}
        self.device = device
        self.num_frames_per_proc = num_frames_per_proc
        self.discount = discount
        self.lr = lr
        self.gae_lambda = gae_lambda
        self.entropy_coef = entropy_coef
        self.value_loss_coef = value_loss_coef
        self.max_grad_norm = max_grad_norm
        self.preprocess_obss = preprocess_obss or default_preprocess_obss
        self.use_subgoal = use_subgoal

        # Must be always 1, there is no support for multi-step propagation yet
        self.recurrence = 1

        # Configure model
        self.acmodel.to(self.device)
        self.acmodel.train()

        self.num_procs = len(envs)
        self.num_frames = self.num_frames_per_proc * self.num_procs
        self.total_num_frames = total_num_frames
        print('Total number of frames:', self.total_num_frames)

        # Initialize experience lists
        shape = (self.num_frames_per_proc, self.num_procs)

        self.obs = self.env.reset()
        self.obss = [None]*(shape[0])
        self.agent_position = [None]*(shape[0])

        self.mask = torch.ones(shape[1], device=self.device)
        self.masks = torch.zeros(*shape, device=self.device)
        self.actions = torch.zeros(*shape, device=self.device, dtype=torch.int)
        self.values = torch.zeros(*shape, device=self.device)
        self.advantages = torch.zeros(*shape, device=self.device)
        self.returns = torch.zeros(*shape, device=self.device)
        self.rewards = torch.zeros(*shape, device=self.device)
        self.log_probs = torch.zeros(*shape, device=self.device)
            # Subgoal related
        subgoal_lens = {
            "relative": 2,
            "representation": 3,
            "language": 384,
        }
        subgoal_len = subgoal_lens[subgoal_type]
        self.subgoals = torch.zeros((*shape, subgoal_len), device=self.device)
        self.subgoal_reward_value = subgoal_reward_value
        
        # Initialize LOGs values
        self.log_subgoal_return = torch.zeros(self.num_procs, device=self.device)
        self.log_episode_num_frames = torch.zeros(self.num_procs, device=self.device)

        self.log_done_counter = 0
        self.log_return = [0] * self.num_procs # Monitor total return for each whoel episode (updates after each episode)
        self.log_num_frames = [0] * self.num_procs
        self.log_success = [0] * self.num_procs
        self.episode_counter = 0
        self.frames_counter = 0

        self.last_100return = deque([0], maxlen=100)
        self.last_100originalreturn = deque([0], maxlen=100)
        
        print("Number of Frame per process: ", self.num_frames_per_proc)
        print("Number of processes: ", self.num_procs)
        print("Number of frames: ", self.num_frames)

    def collect_experiences(self):
        """Collects rollouts and computes advantages.

        Runs several environments concurrently. The next actions are computed
        in a batch mode for all environments at the same time. The rollouts
        and advantages from all environments are concatenated together.

        Returns
        -------
        exps : DictList
            Contains actions, rewards, advantages etc as attributes.
            Each attribute, e.g. `exps.reward` has a shape
            (self.num_frames_per_proc * num_envs, ...). k-th block
            of consecutive `self.num_frames_per_proc` frames contains
            data obtained from the k-th environment. Be careful not to mix
            data from different environments!
        logs : dict
            Useful stats about the training process, including the average
            reward, policy loss, value loss, etc.
        """

        for i in range(self.num_frames_per_proc):

            # update frame counter after each step
            self.frames_counter += self.num_procs

            preprocessed_obs = self.preprocess_obss(self.obs, device=self.device)
            with torch.no_grad():
                if self.use_subgoal:
                    subgoals = self.env.get_subgoals()
                    subgoals = np.array([np.array(subgoal, dtype=np.float32) for subgoal in subgoals])
                    subgoals = torch.tensor(np.array(subgoals)).to(self.device)
                    dist, value = self.acmodel(preprocessed_obs, subgoals)
                else:
                    dist, value = self.acmodel(preprocessed_obs)

            # Take action from the distribution
            action = dist.sample()

            # Step the environment
            obs, reward, done, info = self.env.step(action.cpu().numpy())
            agent_pos = [entry['agent_pos'] for entry in info]
            subgoal_completed = [entry['subgoal_completed'] for entry in info]
            subgoal_steps = [entry['subgoal_steps'] for entry in info]
            reward = list(reward)
            original_reward = torch.tensor(reward, device=self.device, dtype=torch.float)

            # If subgoal completed add reward
            for index, completed in enumerate(subgoal_completed):
                if completed:
                    reward[index] += self.subgoal_reward_value * (1 - (subgoal_steps[index]/self.env_max_steps))

            self.obss[i] = self.obs
            self.obs = obs
            self.agent_position[i] = agent_pos

            self.masks[i] = self.mask
            self.mask = 1 - torch.tensor(done, device=self.device, dtype=torch.float)
            
            self.actions[i] = action
            self.values[i] = value
            if self.use_subgoal: self.subgoals[i] = subgoals

            self.rewards[i] = torch.tensor(reward, device=self.device)
            self.log_probs[i] = dist.log_prob(action)

            self.log_subgoal_return += torch.tensor(reward, device=self.device, dtype=torch.float)
            self.log_episode_num_frames += torch.ones(self.num_procs, device=self.device)

            # Updates if the env is done
            for i, done_ in enumerate(done):
                if done_:
                    self.log_done_counter += 1
                    self.episode_counter += 1
                    self.log_return.append(self.log_subgoal_return[i].item())
                    self.log_num_frames.append(self.log_episode_num_frames[i].item())
                    self.last_100return.append(self.log_subgoal_return[i].item())
                    self.last_100originalreturn.append(original_reward[i].item())
                    if self.log_episode_num_frames[i].item() != self.env_max_steps:
                        self.log_success[i] = 1

            self.log_subgoal_return *= self.mask
            self.log_episode_num_frames *= self.mask

        # **********************************************************************
        # ROLLOUT COLLECTION FINISHED.
        # **********************************************************************

        preprocessed_obs = self.preprocess_obss(self.obs, device=self.device)

        with torch.no_grad():
            if self.use_subgoal:
                subgoals = self.env.get_subgoals()
                subgoals = np.array([np.array(subgoal, dtype=np.float32) for subgoal in subgoals])
                subgoals = torch.tensor(np.array(subgoals), dtype=torch.float32).to(self.device)
                _, next_value = self.acmodel(preprocessed_obs, subgoals)
            else:
                _, next_value = self.acmodel(preprocessed_obs)        


        # Compute rewards
        for i in reversed(range(self.num_frames_per_proc)):
            next_mask = self.masks[i + 1] if i < self.num_frames_per_proc - 1 else self.mask
            next_value = self.values[i + 1] if i < self.num_frames_per_proc - 1 else next_value
            next_advantage = self.advantages[i + 1] if i < self.num_frames_per_proc - 1 else 0

            delta = self.rewards[i] + self.discount * next_value * next_mask - self.values[i]
            self.advantages[i] = delta + self.discount * self.gae_lambda * next_advantage * next_mask

        self.returns = self.values + self.advantages
        
        # Get Experiences: the concatenation of the experience of each process.
        # - T is self.num_frames_per_proc
        # - P is self.num_procs
        # - D is the dimensionality

        exps = DictList()

        exps.obs = [self.obss[i][j]
                    for j in range(self.num_procs)
                    for i in range(self.num_frames_per_proc)]
        exps.obs = self.preprocess_obss(exps.obs, device=self.device)

        # For the tensors below, T x P -> P x T -> P * T
        exps.action = self.actions.transpose(0, 1).reshape(-1)
        exps.reward = self.rewards.transpose(0, 1).reshape(-1)
        exps.log_prob = self.log_probs.transpose(0, 1).reshape(-1)
        exps.value = self.actions.transpose(0, 1).reshape(-1)
        exps.advantage = self.advantages.transpose(0, 1).reshape(-1)
        exps.returnn = self.returns.transpose(0, 1).reshape(-1)

        exps.subgoals = self.subgoals.transpose(0, 1).reshape(-1, *self.subgoals.shape[2:])

        keep = max(self.log_done_counter, self.num_procs)

        logs = {
            "return_per_episode": self.log_return[-keep:],
            "num_frames_per_episode": self.log_num_frames[-keep:],
            "num_frames": self.num_frames,
            "episode_counter": self.episode_counter,
            "avg_return": np.mean(self.last_100return),
            "avg_original_return": np.mean(self.last_100originalreturn),
            "success_rate": np.sum(self.log_success) / self.num_procs,
        }

        # Reset for next experience collection
        self.log_done_counter = 0
        self.log_return = self.log_return[-self.num_procs:]
        self.log_num_frames = self.log_num_frames[-self.num_procs:]
        self.log_success = [0] * self.num_procs

        return exps, logs
    
    @abstractmethod
    def update_parameters(self):
        pass

    def evaluate(self, model_dir, num_frames):
        evaluation_dir = os.path.join(model_dir, "evaluation")
        os.makedirs(evaluation_dir, exist_ok=True)

        print("Evaluating...")
        for env in self.evaluation_envs:
            env_name = env.spec.id
            success_count = 0
            subgoals_completed_in_order = np.zeros(len(env.subgoal_generator.subgoals))
            subgoal_counts = np.zeros(len(env.subgoal_generator.subgoals))
            total_reward = 0.0
            total_original_reward = 0.0

            print(f"Evaluating env: {env_name}")

            if env_name not in self.eval_writers:
                env_log_dir = os.path.join(evaluation_dir, env_name)
                self.eval_writers[env_name] = SummaryWriter(log_dir=env_log_dir)

            writer = self.eval_writers[env_name]

            for episode in range(self.evaluation_episodes):
                obs = env.reset()
                done = False
                steps = 0

                while not done:
                    obs_tensor = self.preprocess_obss([obs], device=self.device)
                    with torch.no_grad():
                        if self.use_subgoal:
                            subgoal = env.get_current_subgoal()
                            subgoal_tensor = torch.tensor(np.array(subgoal)).to(self.device)
                            dist, value = self.acmodel(obs_tensor, subgoal_tensor)
                        else:
                            dist, value = self.acmodel(obs_tensor)
                    action = dist.sample()
                    obs, reward, done, info = env.step(action.cpu().numpy())
                    steps += 1
                    # 4. Original Reward
                    total_original_reward += reward

                    if info['subgoal_completed']:
                        reward += self.subgoal_reward_value * (1 - (info['subgoal_steps'] / env.max_steps))

                        # 2.2 Subgoal Success Rate Ordered
                        subgoals_completed_in_order[info['current_subgoal_index'] - 1] += 1
                    
                    # 3. Reward
                    total_reward += reward

                    if done and steps < env.max_steps:
                        # 1. Success Rate
                        success_count += 1
                for i, bool_subgoal_completion in enumerate(info['completed_subgoals']):
                    subgoal_counts[i] += bool_subgoal_completion

            success_rate = success_count / self.evaluation_episodes
            subgoal_success_rates = subgoal_counts / self.evaluation_episodes
            subgoals_in_order = subgoals_completed_in_order / self.evaluation_episodes
            total_reward /= self.evaluation_episodes
            total_original_reward /= self.evaluation_episodes

            writer.add_scalar("num_frames", num_frames, num_frames)
            writer.add_scalar("success_rate", success_rate, num_frames)
            writer.add_scalar("reward", total_reward, num_frames)
            writer.add_scalar("original_reward", total_original_reward, num_frames)

            for i, subgoal_rate in enumerate(subgoal_success_rates):
                writer.add_scalar(f"subgoal_{i}_success_rate", subgoal_rate, num_frames)

            for i, subgoal_rate in enumerate(subgoals_in_order):
                writer.add_scalar(f"subgoal_{i}_success_rate_order", subgoal_rate, num_frames)

            csv_file_path = os.path.join(evaluation_dir, f"{env_name}.csv")
            file_exists = os.path.isfile(csv_file_path)

            with open(csv_file_path, mode='a', newline='') as f:
                writer = csv.writer(f)

                if not file_exists:
                    headers = ["num_frames", "success_rate"] + \
                              [f"subgoal_{i}_success_rate" for i in range(len(subgoal_counts))] + \
                              [f"subgoal_{i}_success_rate_order" for i in range(len(subgoal_counts))] + \
                              ["reward", "original_reward"]
                    
                    writer.writerow(headers)

                row = (
                    [num_frames, success_rate] +
                    list(subgoal_success_rates) +
                    list(subgoals_in_order) +
                    [total_reward, total_original_reward]
                )
                writer.writerow(row)