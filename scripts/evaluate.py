import argparse
import gym

import utils
import torch_ac

parser = argparse.ArgumentParser()
parser.add_argument("--model", required=True,
                    help="name of the trained model (REQUIRED)")
parser.add_argument("--env", default="MiniGrid-KeyCorridorS3R3-v0", nargs='+', required=True,
                    help="List of environments to evaluate on")
parser.add_argument("--seed", type=int, default=0,
                    help="random seed (default: 0)")
parser.add_argument("--episodes", type=int, default=1000,
                    help="number of episodes per environment to evaluate")
parser.add_argument("--argmax", action="store_true", default=False,
                    help="select the action with highest probability (default: False)")

parser.add_argument("--use-subgoal", type=bool, default=False, const=True, nargs='?',
                    help="use a subgoal based model to visualize")
parser.add_argument("--subgoal-file", default=None,
                    help="File with the subgoals to use in the model")
parser.add_argument("--subgoal-type", default="relative",
                    help="Type of subgoals to add to the input")
parser.add_argument("--subgoal-accuracy", type=float, nargs="+", default=[1],
                    help="Accuracy of the LLM model to predict the subgoal (default: 1)")
parser.add_argument("--subgoal-mean", type=float, nargs="+", default=[0],
                    help="Mean error in the subgoals of the LLM model (default:0)")
parser.add_argument("--subgoal-std", type=float, nargs="+", default=[0],
                    help="Std error in the subgoals of the LLM model (default:0)")

args = parser.parse_args()

utils.seed(args.seed)

args.env = utils.parse_env_name(args.env)
envs = []
for i in range(len(args.env)):
    envs.append(
        torch_ac.ParallelEnv(
            args.env[i], 1, args.use_subgoal,
            args.subgoal_file, args.subgoal_type,
            args.subgoal_accuracy,args.subgoal_mean,
            args.subgoal_std, args.seed
        )
    )

# Load agent
ACTION_SPACE = envs[0].envs[0].action_space.n
OBSERVATION_SPACE = envs[0].envs[0].observation_space
model_dir = utils.get_model_dir(args.model)
agent = utils.Agent(
    observation_space=OBSERVATION_SPACE, action_space=ACTION_SPACE,
    model_dir=model_dir, argmax=args.argmax, use_subgoal=args.use_subgoal,
    subgoal_type=args.subgoal_type)

success_rates = {}

for i, env in enumerate(envs):
    env_name = args.env[i]
    print(f"Evaluating sucess rate in environment {env_name}...")
    successes = 0

    for episode in range(args.episodes):
        print(f"Episode {episode} of {args.episodes}")
        obs = env.reset()
        done = False
        steps = 0

        while not done:
            subgoal = env.get_subgoals()[0]
            action = agent.get_action(obs, subgoal)
            obs, reward, done, info = env.step([action])
            obs, reward, done, info = obs[0], reward[0], done[0], info[0]
            steps += 1

            if done and steps < env.envs[0].max_steps:
                successes += 1

    success_rate = successes / args.episodes
    success_rates[env_name] = success_rate
    print(f"Environment: {env_name}, Success Rate: {success_rate:.2%}")

print("\nSummary of success rates per environment:")
for env_name, success_rate in success_rates.items():
    print(f"{env_name}: {success_rate:.2%}")