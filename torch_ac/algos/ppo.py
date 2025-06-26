import numpy
import torch
from torch import nn

from torch_ac.algos.base import BaseAlgo

class PPOAlgo(BaseAlgo):
    """The Proximal Policy Optimization algorithm
    ([Schulman et al., 2015](https://arxiv.org/abs/1707.06347))."""

    def __init__(self, envs, eval_envs=[], eval_episodes=10,
                 acmodel=None, device=None, num_frames_per_proc=None,
                 discount=0.99,lr=0.0001, gae_lambda=0.95, entropy_coef=0.01,
                 value_loss_coef=0.5, max_grad_norm=0.5, optim_eps=1e-5,
                 clip_eps=0.2, epochs=4, batch_size=256, preprocess_obss=None,
                 num_actions=7, use_subgoal=0, subgoal_type="relative",
                 subgoal_reward_value=1, total_num_frames=2e7):

        num_frames_per_proc = num_frames_per_proc or 128
        super().__init__(envs, eval_envs, eval_episodes,
                         acmodel, device, num_frames_per_proc, discount,
                         lr, gae_lambda, entropy_coef, value_loss_coef,
                         max_grad_norm, preprocess_obss, num_actions, use_subgoal,
                         subgoal_type, subgoal_reward_value, total_num_frames)

        self.clip_eps = clip_eps
        self.epochs = epochs
        self.batch_size = batch_size

        self.optimizer = torch.optim.Adam(self.acmodel.parameters(), lr, eps=optim_eps)
        self.batch_num = 0

        self.forward_mse = nn.MSELoss()

    def update_parameters(self, exps):
        # Collect experiences
        for _ in range(self.epochs):
            log_entropies = []
            log_values = []
            log_grad_norms = []
            log_grad_norms_critic = []
            log_policy_losses = []
            log_value_losses = []

            for inds in self._get_batches_starting_indexes():
                batch_entropy = 0
                batch_policy_loss = 0
                batch_value = 0
                batch_value_loss = 0
                batch_loss = 0

                for i in range(self.recurrence):
                    sb = exps[inds + i]

                    if self.use_subgoal:
                        dist, value = self.acmodel(sb.obs, sb.subgoals)
                    else:
                        dist, value = self.acmodel(sb.obs)

                    entropy = dist.entropy().mean()
                    ratio = torch.exp(dist.log_prob(sb.action) - sb.log_prob)
                    surr1 = ratio * sb.advantage
                    surr2 = torch.clamp(ratio, 1.0 - self.clip_eps, 1.0 + self.clip_eps) * sb.advantage
                    policy_loss = -torch.min(surr1, surr2).mean()
                    value_loss = self.forward_mse(value, sb.returnn)

                    # Compute total loss
                    loss = policy_loss - self.entropy_coef * entropy + self.value_loss_coef * value_loss

                    # Update batch values
                    # policy
                    batch_entropy += entropy.item()
                    batch_policy_loss += policy_loss.item()
                    # critic
                    batch_value += value.mean().item()
                    batch_value_loss += value_loss.item()
                    batch_loss += loss

                # Update batch values
                batch_entropy /= self.recurrence
                batch_policy_loss /= self.recurrence

                batch_value /= self.recurrence
                batch_value_loss /= self.recurrence
                batch_loss /= self.recurrence

                # Update
                self.optimizer.zero_grad()
                batch_loss.backward()

                torch.nn.utils.clip_grad_norm_(self.acmodel.parameters(), self.max_grad_norm)
                grad_norm = self.calculate_gradients(self.acmodel)
                grad_norm_critic = 0
                self.optimizer.step()

                log_entropies.append(batch_entropy)
                log_policy_losses.append(batch_policy_loss)

                log_values.append(batch_value)
                log_value_losses.append(batch_value_loss)

                log_grad_norms.append(grad_norm) # Monitor the actor
                log_grad_norms_critic.append(grad_norm_critic) # Monitor the critic

        logs = {
            "entropy": numpy.mean(log_entropies),
            "policy_loss": numpy.mean(log_policy_losses),
            "value": numpy.mean(log_values),
            "value_loss": numpy.mean(log_value_losses),
            "grad_norm": numpy.mean(log_grad_norms),
            "grad_norm_critic": numpy.mean(log_grad_norms_critic),
        }

        return logs
    
    def _get_batches_starting_indexes(self):
        """Gives, for each batch, the indexes of the observations given to
        the model and the experiences used to compute the loss at first.

        First, the indexes are the integers from 0 to `self.num_frames` with a
        step of 1. Then, the indexes are splited into the different batches.

        Returns
        -------
        batches_starting_indexes : list of list of int
            the indexes of the experiences to be used at first for each batch
        """

        indexes = numpy.arange(0, self.num_frames, self.recurrence)
        indexes = numpy.random.permutation(indexes)

        if self.batch_num % 2 == 1:
            indexes = indexes[(indexes + self.recurrence) % self.num_frames_per_proc != 0]
            indexes += self.recurrence // 2
        self.batch_num += 1 

        num_indexes = self.batch_size // self.recurrence
        batches_starting_indexes = [indexes[i:i+num_indexes] for i in range(0, len(indexes), num_indexes)]

        return batches_starting_indexes
    
    def calculate_gradients(self, model):
        """
        Given the current network with its graph, it calculates the gradients
        through each module and returns a sum of values

        By default, the network will have two-head of critic but we will propagate
        only through one or the both
        """
    
        grad_norm = 0
        for p in model.parameters():
            try:
                grad_norm += p.grad.data.norm(2).item() ** 2
            except AttributeError:
                continue
        grad_norm = grad_norm ** 0.5
    
        return grad_norm
