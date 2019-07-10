import torch
import torch.nn as nn
import torch.nn.functional as F

class DQNNetwork(nn.Module):
    def __init__(self, state_size, action_size, seed,hidden_layers_size=[64, 32]):
        """Initialize parameters and build model.
        Params
        ======
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
            seed (int): Random seed
        """
        super(DQNNetwork, self).__init__()
        self.seed = torch.manual_seed(seed)
        
        self.hidden_layers = nn.ModuleList([nn.Linear(state_size, hidden_layers_size[0])])
        layer_sizes = zip(hidden_layers_size[:-1], hidden_layers_size[1:])
        self.hidden_layers.extend([nn.Linear(h1, h2) for h1, h2 in layer_sizes])  
        self.output = nn.Linear(hidden_layers_size[-1], action_size)


    def forward(self, state):
        x = state
        for layer in self.hidden_layers:
            x = F.relu(layer(x))
        x = self.output(x)
        
        return x
