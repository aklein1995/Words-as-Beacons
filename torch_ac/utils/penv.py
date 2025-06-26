import gym
import multiprocessing
from multiprocessing import Process
import json
import random
import itertools

class ParallelEnv():
    """A concurrent execution of environments in multiple processes."""

    def __init__(
            self, env_name, num_envs, use_subgoals=False,
            subgoal_file=None, subgoal_type="relative",
            subgoal_accuracy=[1], subgoal_mean_error=[0],
            subgoal_std_error=[0], n_random_subgoals=0,
            pretrain=False, pretrain_subgoal_distance=0,
            seed=42
    ):
        self.env_name = env_name if type(env_name) == list else [env_name]
        self.num_envs = num_envs
        self.use_subgoals = use_subgoals
        self.subgoal_file = subgoal_file
        self.subgoal_type = subgoal_type
        self.subgoal_accuracy = subgoal_accuracy
        self.subgoal_mean_error = subgoal_mean_error
        self.subgoal_std_error = subgoal_std_error
        self.nrandom_subgoals = n_random_subgoals
        self.pretrain = pretrain
        self.pretrain_subgoal_distance = pretrain_subgoal_distance

        self.envs = []
        self.pipes = []

        if subgoal_file:
            with open(subgoal_file, "r") as f:
                data = json.load(f)
            seeds = {entry['seed'] for entry in data}
            full_range = set(range(1000))
            self.exclude_seeds = full_range - seeds

        env_cycle = itertools.cycle(self.env_name)

        # First env to run in the parent process
        env_name = next(env_cycle)
        env = gym.make(
            env_name,
            seed=self.random_excluding_seed() if self.subgoal_file else seed,
            subgoal_file=subgoal_file,
            subgoal_type=subgoal_type,
            subgoal_accuracy=subgoal_accuracy,
            subgoal_mean_error=subgoal_mean_error,
            subgoal_std_error=subgoal_std_error,
            n_random_subgoals=n_random_subgoals,
            pretrain=pretrain,
            pretrain_subgoal_distance=pretrain_subgoal_distance,
        )
        if not subgoal_file:
            env.seed(seed)
        else:
            env.reset()
        self.envs.append(env)
        self.observation_space = env.observation_space
        self.action_space = env.action_space

        for i in range(1, num_envs):
            parent, child = multiprocessing.Pipe()
            # There is a pipe for each communication, but the parent is always the same 
            self.pipes.append(parent)
            env_name = next(env_cycle)
            env = gym.make(
                env_name,
                seed=self.random_excluding_seed() if self.subgoal_file else seed,
                subgoal_file=subgoal_file,
                subgoal_type=subgoal_type,
                subgoal_accuracy=subgoal_accuracy,
                subgoal_mean_error=subgoal_mean_error,
                subgoal_std_error=subgoal_std_error,
                n_random_subgoals=n_random_subgoals,
                pretrain=pretrain,
                pretrain_subgoal_distance=pretrain_subgoal_distance,
            )
            if not subgoal_file:
                env.seed(seed + i * 10000)
            else:
                env.reset()
            self.envs.append(env)
            process = Process(target=self.run_env, args=(child, self.envs[i]))
            process.daemon = True
            process.start()
            child.close()

    def __getitem__(self, idx):
        return self.envs[idx]

    def __len__(self):
        return len(self.envs)
    
    def random_excluding_seed(self):
        while True:
            num = random.randint(0, 999)
            if num not in self.exclude_seeds:
                return num

    def run_env(self, conn, env):
        while True:
            command, data = conn.recv()
            
            if command == "step":
                observation, reward, done, info = env.step(data)
                if done: 
                    if self.subgoal_file:
                        env.seed(self.random_excluding_seed())
                    observation = env.reset()
                info["agent_pos"] = env.agent_pos
                conn.send((observation, reward, done, info))
            elif command == "get_subgoal":
                subgoal = env.get_current_subgoal()
                conn.send(subgoal)
            elif command == "get_seed":
                seed = env.episode_seed
                conn.send(seed)
            elif command == "reset":
                if (self.subgoal_file):
                    env.seed(self.random_excluding_seed())
                observation = env.reset()
                conn.send(observation)
            else:
                raise NotImplementedError 
            
    def step(self, actions):
        results = []

        # Step env 1
        observation, reward, done, info = self.envs[0].step(actions[0])
        if done:
            if self.subgoal_file:
                self.envs[0].seed(self.random_excluding_seed())
                observation = self.envs[0].reset()
        info["agent_pos"] = self.envs[0].agent_pos
        results.append((observation, reward, done, info))

        # Step other envs
        for idx, action in enumerate(actions[1:]):
            self.pipes[idx].send(("step", action))

        # Get the results
        for pipe in self.pipes:
            results.append(pipe.recv())

        # Process results from the envs
        observations, rewards, dones, infos = zip(*results)

        return list(observations), list(rewards), list(dones), list(infos)
    
    def reset(self):
        observations = []

        if self.subgoal_file:
            self.envs[0].seed(self.random_excluding_seed())

        observation = self.envs[0].reset()
        observations.append(observation)

        for pipe in self.pipes:
            pipe.send(("reset", None))
        
        for pipe in self.pipes:
            observations.append(pipe.recv())

        return list(observations)
    
    def get_subgoals(self):
        if not self.use_subgoals:
            return None
        
        subgoals = []
        subgoal_0 = self.envs[0].get_current_subgoal()
        subgoals.append(subgoal_0)

        for pipe in self.pipes:
            pipe.send(("get_subgoal", None))

        for pipe in self.pipes:
            subgoals.append(pipe.recv())
        
        return list(subgoals)
    
    def render(self, mode='human'):
        return self.envs[0].render(mode)
    
    def get_seeds(self):
        seeds = []
        seeds.append(self.envs[0].episode_seed)

        for pipe in self.pipes:
            pipe.send(("get_seed", None))
        
        for pipe in self.pipes:
            seeds.append(pipe.recv())

        return list(seeds)