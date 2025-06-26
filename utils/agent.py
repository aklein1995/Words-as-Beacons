import torch
import numpy as np

import utils
from model import *

class Agent:

    def __init__(
        self, observation_space, action_space, model_dir,
        argmax=False, use_subgoal=False, subgoal_type='relative',
        device='cpu'):
        
        obs_space, self.preprocess_obss = utils.get_obss_preprocessor(observation_space)
        self.action_space = action_space
        self.argmax = argmax
        self.use_subgoal = use_subgoal
        self.subgoal_type = subgoal_type

        self.device = device

        print("Model directory: ", model_dir)
        print("Action space: ", self.action_space)

        self.acmodel = ACModelRIDE(obs_space, action_space, use_subgoal=use_subgoal, subgoal_type=subgoal_type)
        self.acmodel.load_state_dict(utils.get_model_state(model_dir, device))
        self.acmodel.to(device)
        self.acmodel.eval()

        if hasattr(self.preprocess_obss, "vocab"):
            self.preprocess_obss.vocab.load_vocab(utils.get_vocab(model_dir, device))

    def get_actions(self, obss, subgoal=None):
        preprocessed_obss = self.preprocess_obss(obss, device=self.device)

        with torch.no_grad():
            if self.use_subgoal:
                subgoal = torch.tensor(np.array(subgoal)).unsqueeze(0).to(self.device)
                dist, _ = self.acmodel(preprocessed_obss, subgoal)

            else:
                dist, _ = self.acmodel(preprocessed_obss)
            
        if self.argmax:
            actions = dist.probs(1, keepdim=True)[1]
        else:
            actions = dist.sample()
        
        return actions.cpu().numpy()
    
    def get_action(self, obs, subgoal):
        return self.get_actions(obs, subgoal)[0]