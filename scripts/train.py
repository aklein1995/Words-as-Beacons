import argparse
import datetime
import gym
import numpy as np
import sys
import tensorboardX
import time
import torch
import torch_ac
import utils

from model import ACModelRIDE

parser = argparse.ArgumentParser()

import setproctitle

setproctitle.setproctitle("python3 -m scripts.train")

## General paramters
# Logs related
parser.add_argument("--model", default=None,
                    help="name of the model")
parser.add_argument("--log-interval", type=int, default=1,
                    help="the number of updates between two logs (default: 1)")
parser.add_argument("--save-interval", type=int, default=10,
                    help="the number of updates between two saves (default: 10, 0 means no saving)")

# Environment related
parser.add_argument("--env", type=str, nargs="+", default="MiniGrid-KeyCorridorS3R3-v0",
                    help="name of the environment to train on (default: MiniGrid-KeyCorridorS3R3-v0)")
parser.add_argument("--eval-envs", default=None, nargs="+",
                    help="List of environments to evaluate the agent on.")
parser.add_argument("--eval-episodes", default=10, type=int,
                    help="Number of episodes to evaluate the agent on.")
parser.add_argument("--eval-interval", default=50, type=int,
                    help="Interval between each evaluation of the agent.")
parser.add_argument("--eval-nrandom-subgoals", default=0, type=int,
                    help="Added random subgoals in evaluation.")
parser.add_argument("--seed", type=int, default=1,
                    help="random seed (default: 1)")

# Subgoal related
parser.add_argument("--use-subgoal", type=bool, default=False, const=True,
                    nargs='?', help="run subgoal oriented training (default: 0)")
parser.add_argument("--subgoal-file", default=None,
                    help="File with the subgoals to use in the training")
parser.add_argument("--subgoal-type", default="relative",
                    help="Type of subgoals to add to the input, \n1. 'relative', each subgoal is passed as the position of it relative to the agent.\n2. 'absolute', each subgoal is passed as the absolute position of the subgoal.\n3. 'representation', it passed just the representation of the object it must modify. (obj.type, obj.color, obj.state)")
parser.add_argument("--subgoal-accuracy", type=float, nargs="+", default=[1],
                    help="Accuracy of the LLM model to predict the subgoal (default: 1)")
parser.add_argument("--subgoal-mean", type=float, nargs="+", default=[0],
                    help="Mean error in the subgoals of the LLM model (default:0)")
parser.add_argument("--subgoal-std", type=float, nargs="+", default=[0],
                    help="Std error in the subgoals of the LLM model (default:0)")
parser.add_argument("--nrandom-subgoals", type=int, default=0,
                    help="Intermediary random subgoals to help generalization (default: 0)")
parser.add_argument("--subgoal-reward-value", type=float, default=1.0,
                    help="Reward value for the subgoal completion")
parser.add_argument("--pretrain-subgoal-distance", type=int, default=0,
                    help="Distance to subgoals in the pretaining phase")
parser.add_argument("--pretrain", type=bool, default=False, const=True,
                    nargs='?', help="Doing pretrain (yes/no)")

# Algorithm related
parser.add_argument("--epochs", type=int, default=4,
                    help="number of epochs (default: 4)")
parser.add_argument("--batch-size", type=int, default=256,
                    help="batch size (default: 256)")
parser.add_argument("--nsteps", type=int, default=128,
                    help="number of steps per process before update (default: 128)")
parser.add_argument("--discount", type=float, default=0.99,
                    help="discount factor (default: 0.99)")
parser.add_argument("--lr", type=float, default=0.0001,
                    help="learning rate (default: 0.0001)")
parser.add_argument("--gae-lambda", type=float, default=0.95,
                    help="lambda coefficient in GAE formula (default: 0.95, 1 means no GAE)")
parser.add_argument("--entropy-coef", type=float, default=0.0005,
                    help="entropy term coefficient (default: 0.0005)")
parser.add_argument("--value-loss-coef", type=float, default=0.5,
                    help="value loss term coefficient (default: 0.5)")
parser.add_argument("--max-grad-norm", type=float, default=0.5,
                    help="maximum norm of gradient (default: 0.5)")
parser.add_argument("--optim-eps", type=float, default=1e-8,
                    help="Adam and RMSprop optimizer epsilon (default: 1e-8)")
parser.add_argument("--clip-eps", type=float, default=0.2,
                    help="clipping epsilon for PPO (default: 0.2)")

# Generic configuration
parser.add_argument("--procs", type=int, default=16,
                    help="number of processes in parallel (default: 16)")
parser.add_argument("--frames", type=int, default=int(3e7),
                    help="number of frames to train on (default: 3e7)")

# GPU/CPU config
parser.add_argument("--use-gpu", type=bool, default=False, const=True,
                    nargs='?', help="Use gpu (default: 0)")
parser.add_argument("--gpu-id", type=int, default=-1,
                    help="Choose the GPU to use (default: -1)")

args = parser.parse_args()

# ******************************************************************************
# Assertions to ensure consistency
# ******************************************************************************
assert (args.use_gpu == False) or (args.use_gpu and args.gpu_id != -1), \
    "If use_gpu is set to 1, then gpu_id must be set to a valid GPU"

# ******************************************************************************
# Set model and directory
# ******************************************************************************
date = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
default_model_name = f"{args.env}_{args.seed}_{date}"

model_name = args.model or default_model_name
model_dir = utils.get_model_dir(model_name)

# ******************************************************************************
# Load loggers and Tensorboard writer
# ******************************************************************************
txt_logger = utils.get_txt_logger(model_dir)
csv_file, csv_logger = utils.get_csv_logger(model_dir)
tb_writer = tensorboardX.SummaryWriter(model_dir)

# Log the command line and all the arguments passed to the script
txt_logger.info("{}\n".format(" ".join(sys.argv)))
txt_logger.info("{}\n".format(args))

# ******************************************************************************
# Set seed for all randomness sources
# ******************************************************************************
utils.seed(args.seed)

# ******************************************************************************
# Set device
# ******************************************************************************
device = torch.device("cuda:" + str(args.gpu_id) if args.use_gpu else "cpu")
txt_logger.info(f"Device: {device}\n")

# ******************************************************************************
# Load environment
# ******************************************************************************
args.env = utils.parse_env_name(args.env)
if args.eval_envs:
    args.eval_envs = utils.parse_env_name(args.eval_envs)

envs = torch_ac.ParallelEnv(args.env, args.procs, use_subgoals=args.use_subgoal,
                            subgoal_file=args.subgoal_file,
                            subgoal_type=args.subgoal_type, 
                            subgoal_accuracy=args.subgoal_accuracy,
                            subgoal_mean_error=args.subgoal_mean,
                            subgoal_std_error=args.subgoal_std,
                            n_random_subgoals=args.nrandom_subgoals,
                            pretrain=args.pretrain,
                            pretrain_subgoal_distance=args.pretrain_subgoal_distance,
                            seed=args.seed)

eval_envs = []
if args.eval_envs:
    for env_name in args.eval_envs:
        env = gym.make(
            env_name, subgoal_file=args.subgoal_file,
            subgoal_type=args.subgoal_type,
            subgoal_accuracy=args.subgoal_accuracy,
            subgoal_mean_error=args.subgoal_mean,
            subgoal_std_error=args.subgoal_std,
            n_random_subgoals=args.nrandom_subgoals,
            seed=args.seed
        )
        env.seed(args.seed)
        eval_envs.append(env)

txt_logger.info("Environments loaded\n")

ACTION_SPACE = envs[0].action_space.n
txt_logger.info(f"ACTION_SPACE: {ACTION_SPACE}\n")

# ******************************************************************************
# Load training status
# ******************************************************************************
try:
    status = utils.get_status(model_dir, device)
except OSError:
    status = {"num_frames": 0, "update": 0}
txt_logger.info("Training status loaded\n")

# ******************************************************************************
# Load observation proprocessor
# ******************************************************************************
obs_space, preprocess_obss = utils.get_obss_preprocessor(envs[0].observation_space)
if "vocab" in status:
    preprocess_obss.vocab.load_vocab(status["vocab"])
txt_logger.info("Observations preprocessor loaded")

# ******************************************************************************
# Load model
# ******************************************************************************
acmodel = ACModelRIDE(obs_space, ACTION_SPACE, args.use_subgoal, args.subgoal_type)
if "model_state" in status:
    acmodel.load_state_dict(status["model_state"])
acmodel.to(device)
txt_logger.info("Model loaded\n")
txt_logger.info("{}\n".format(acmodel))

total_params = sum(p.numel() for p in acmodel.parameters())
print("***PARAMS UNIQUE AC (RIDE): ", total_params)

# ******************************************************************************
# Load algorithm
# ******************************************************************************
algo = torch_ac.PPOAlgo(
    envs=envs, eval_envs=eval_envs, eval_episodes=args.eval_episodes,
    acmodel=acmodel, device=device, num_frames_per_proc=args.nsteps,
    discount=args.discount, lr=args.lr, gae_lambda=args.gae_lambda,
    entropy_coef=args.entropy_coef, value_loss_coef=args.value_loss_coef,
    max_grad_norm=args.max_grad_norm, optim_eps=args.optim_eps,
    clip_eps=args.clip_eps, epochs=args.epochs, batch_size=args.batch_size,
    preprocess_obss=preprocess_obss, num_actions=ACTION_SPACE,
    use_subgoal=args.use_subgoal, subgoal_type=args.subgoal_type,
    subgoal_reward_value=args.subgoal_reward_value, total_num_frames=args.frames)

# ******************************************************************************
# Load optimizer
# ******************************************************************************
if "optimizer_state" in status:
    algo.optimizer.load_state_dict(status["optimizer_state"])
    txt_logger.info("Optimizer loaded\n")

# ******************************************************************************
# Train model
# ******************************************************************************
num_frames = status["num_frames"]
update = status["update"]
start_time = time.time()

while num_frames < args.frames:
    # Update model parameters
    update_start_time = time.time()
    exps, logs1 = algo.collect_experiences()
    logs2 = algo.update_parameters(exps)

    if update % args.eval_interval == 0 and update != 0 and args.eval_envs:
        algo.evaluate(model_dir, num_frames)

    logs = {**logs1, **logs2}
    update_end_time = time.time()

    num_frames += logs["num_frames"]
    update += 1

    # Print logs
    if update % args.log_interval == 0:
        fps = logs["num_frames"] / (update_end_time - update_start_time)
        duration = int(time.time() - start_time)
        episodes = logs["episode_counter"]

        # extrinsic
        return_per_episode = utils.synthesize(logs["return_per_episode"])
        num_frames_per_episode = utils.synthesize(logs["num_frames_per_episode"])

        # general values
        header = ["update", "frames", "FPS", "duration","episodes"]
        data = [update, num_frames, fps, duration, episodes]
        only_txt = [update, num_frames, fps, duration, episodes]

        # returns
        header += ["rreturn_" + key for key in return_per_episode.keys()]
        data += return_per_episode.values()
        only_txt += [return_per_episode["mean"]]
        only_txt += [return_per_episode["std"]]

        header += ["avg_original_return"]
        data += [logs["avg_original_return"]]
        only_txt += [logs["avg_original_return"]]

        header += ["success_rate"]
        data += [logs["success_rate"]]
        only_txt += [logs["success_rate"]]

        # avg 100 episodes
        header += ["avg_return"]
        data += [logs["avg_return"]]
        header += ["entropy", "value", "policy_loss", "value_loss", "grad_norm"]
        data += [logs["entropy"], logs["value"], logs["policy_loss"], logs["value_loss"], logs["grad_norm"]]
        only_txt += [logs["entropy"], logs["value"], logs["policy_loss"], logs["value_loss"], logs["grad_norm"]]

        txt_logger.info(
            "U {:03} | F {:08} | FPS {:04.0f} | D {:03} | Eps {:03} |rR:uo {:.2f} {:.2f} | Or: {:.2f} | S% {:.2f} | H {:.3f} | V {: .3f} | pL {: .3f} | vL {: .3f} | Vp {: .3f}"
            .format(*only_txt))

        header += ["avg_num_frames_per_episode", "num_frames"]
        data += [np.mean(logs["num_frames_per_episode"]), logs["num_frames"]]

        if status["num_frames"] == 0:
            csv_logger.writerow(header)
        csv_logger.writerow(data)
        csv_file.flush()

        for field, value in zip(header, data):
            if field == "avg_return": field = "_avg_return"
            tb_writer.add_scalar(field, value, num_frames)

    if args.save_interval > 0 and update % args.save_interval == 0:
        acmodel_weights = acmodel.state_dict()
        optimizer_state = algo.optimizer.state_dict()
        status = {"num_frames": num_frames, "update": update,
                  "model_state": acmodel_weights, "optimizer_state": optimizer_state}
        
        if hasattr(preprocess_obss, "vocab"):
            status["vocab"] = preprocess_obss.vocab.vocab
        utils.save_status(status, model_dir, num_frames)
        txt_logger.info("Status saved!")