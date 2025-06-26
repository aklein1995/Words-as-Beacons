import torch
from torch import nn
from torch.nn import functional as F
from torch.distributions.categorical import Categorical

def init(module, weight_init, bias_init, gain=1):
    weight_init(module.weight.data, gain=gain)
    bias_init(module.bias.data)
    return module

class ACModelRIDE(nn.Module):
    def __init__(self, obs_space, action_space, use_subgoal=False, subgoal_type="relative"):
        super().__init__()

        self.use_subgoal = use_subgoal
        subgoal_lens = {
            "relative": 2,
            "representation": 3,
            "language": 384,
        }
        self.subgoal_len = subgoal_lens[subgoal_type]

        init_ = lambda m: init(m, nn.init.orthogonal_, lambda x: nn.init.
                            constant_(x, 0), nn.init.calculate_gain('relu'))

        # Define image embedding
        self.image_conv = nn.Sequential(
            init_(nn.Conv2d(in_channels=3, out_channels=32, kernel_size=(3,3), stride=(2,2), padding=1)),
            nn.ELU(),
            init_(nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(3,3), stride=(2,2), padding=1)),
            nn.ELU(),
            init_(nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(3,3), stride=(2,2), padding=1)),
            nn.ELU(),
        )
        self.image_embedding_size = 32

        self.fc = nn.Sequential(
            init_(nn.Linear(self.image_embedding_size+self.subgoal_len*self.use_subgoal, 256)),
            nn.ReLU()
        )

        init_ = lambda m: init(m, nn.init.orthogonal_,
            lambda x: nn.init.constant_(x, 0))
        
        # Define actor's model
        self.actor = nn.Sequential(
            init_(nn.Linear(256, action_space)),
        )

        # define critic's model
        self.critic = nn.Sequential(
            init_(nn.Linear(256, 1))
        )

        self.critic_int = nn.Sequential(
            init_(nn.Linear(256, 1))
        )

    def forward(self, obs, subgoals=[]):
        x = obs.image.transpose(1, 3).transpose(2, 3)
        x = self.image_conv(x)
        x = x.reshape(x.shape[0], -1)

        if self.use_subgoal:
            if not torch.is_tensor(subgoals):
                TypeError("Subgoals must be a tensor, got", type(subgoals), "instead.")

            if subgoals.dtype != torch.float32:
                subgoals = subgoals.to(torch.float32)

            if subgoals.dim() == 1 and subgoals.shape[0] == self.subgoal_len:
                subgoals = subgoals.unsqueeze(0)
            
            concat = torch.cat((x, subgoals), dim=1)
            embedding = self.fc(concat)
        else:
            embedding = self.fc(x)

        x = self.actor(embedding)
        dist = Categorical(logits=F.log_softmax(x, dim=1))

        x = self.critic(embedding)
        value = x.squeeze(1)

        return dist, value