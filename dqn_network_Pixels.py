import torch
import torch.nn as nn
import torch.nn.functional as F

class DQNNetwork(nn.Module):
    def __init__(self, input_channels, action_size, device, seed):
        """Initialize parameters and build model.
        Params
        ======
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
            seed (int): Random seed
        """
        super(DQNNetwork, self).__init__()
        self.seed = torch.manual_seed(seed)
        
        self.conv1 = nn.Conv2d(input_channels, 32, kernel_size=8, stride=4)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)
        self.bn3 = nn.BatchNorm2d(64)
        
        self.output = None


    def forward(self, state):
        x = state
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        
        x = x.reshape(x.size()[0], -1)
        
        if self.output is None:
            self.output = nn.Linear(x.size()[1], action_size).to(device)
         
        x = self.output(x)
        
        return x
