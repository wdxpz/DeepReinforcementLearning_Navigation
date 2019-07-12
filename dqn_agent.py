import numpy as np
import random
from collections import namedtuple, deque

from dqn_network import DQNNetwork

import torch
import torch.nn.functional as F
import torch.optim as optim

BUFFER_SIZE = int(1e5)  # replay buffer size
BATCH_SIZE = 64         # minibatch size
GAMMA = 0.99            # discount factor
TAU = 1e-3              # for soft update of target parameters
LR = 5e-4               # learning rate 
UPDATE_EVERY = 4        # how often to update the network

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class Agent():

    def __init__(self, state_size, action_size, seed, buffer_size = int(1e5), batch_size = 64, gamma = 0.99, tau = 1e-3, lr = 5e-4, hidden_layers_size=[64, 32], update_every = 4, update_target_very = 12):
        """Initialize an Agent object.
        
        Params
        ======
            state_size (int): dimension of each state
            action_size (int): dimension of each action
            seed (int): random seed
        """
        self.state_size = state_size
        self.action_size = action_size
        self.seed = random.seed(seed)
        
        self.buffer_size = buffer_size
        self.batch_size = batch_size
        self.gamma = gamma
        self.tau = tau
        self.lr = lr
        self.update_every = update_every
        self.update_target_very = update_target_very

        # Q-Network
        self.qnetwork_local = DQNNetwork(state_size, action_size, seed, hidden_layers_size=hidden_layers_size).to(device)
        self.qnetwork_target = DQNNetwork(state_size, action_size, seed, hidden_layers_size=hidden_layers_size).to(device)
        self.qnetwork_target.load_state_dict(self.qnetwork_local.state_dict())
        self.optimizer = optim.Adam(self.qnetwork_local.parameters(), lr=LR)

        # Replay memory
        self.memory = ReplayBuffer(action_size, buffer_size, batch_size, seed)
        # Initialize time step (for updating every UPDATE_EVERY steps)
        self.t_step = 0
        self.t_target_step = 0
    
    def step(self, state, action, reward, next_state, done):
        # Save experience in replay memory
        self.memory.add(state, action, reward, next_state, done)
        
        # If enough samples are available in memory, get random subset and learn
        if len(self.memory) > self.batch_size:      
            # Learn every UPDATE_EVERY time steps.
            self.t_step = self.t_step + 1
            
            if self.t_step % self.update_every == 0:
                experiences = self.memory.sample()
                self.learn(experiences, self.gamma)

            """
            a implementation of fixed Q-Targets
            """   
            if self.t_step % self.update_target_very == 0:
                    self.update_target_Q()

    def act(self, state, eps=0.):
        """Returns actions for given state as per current policy.
        
        Params
        ======
            state (array_like): current state
            eps (float): epsilon, for epsilon-greedy action selection
        """
        state = torch.from_numpy(state).float().unsqueeze(0).to(device)
        self.qnetwork_local.eval()
        with torch.no_grad():
            action_values = self.qnetwork_local(state)
        self.qnetwork_local.train()

        # Epsilon-greedy action selection
        if random.random() > eps:
            return np.argmax(action_values.cpu().data.numpy())
        else:
            return random.choice(np.arange(self.action_size))

    def learn(self, experiences, gamma, method='DDQN'):
        """Update value parameters using given batch of experience tuples.

        Params
        ======
            experiences (Tuple[torch.Tensor]): tuple of (s, a, r, s', done) tuples 
            gamma (float): discount factor
        """
        states, actions, rewards, next_states, dones = experiences #already to(device)

        ## TODO: compute and minimize the loss
        self.optimizer.zero_grad()
        
        if method=='DQN':
            target_values = self.qnetwork_target.forward(next_states)
            #Q Learning with the max q(next_state, a)
            accumulated_rewards = rewards.squeeze(1) + gamma * target_values.max(dim=1)[0]
        else:
            max_actions = self.qnetwork_local.forward(next_states)
            max_actions = max_actions.argmax(dim=1).unsqueeze(1)
            target_values = self.qnetwork_target.forward(next_states)
            evaluate_target_values = target_values.gather(1, max_actions)
            accumulated_rewards = rewards.squeeze(1) + gamma * evaluate_target_values.squeeze(1)
        
        
        # get the old q(current_state, action)    
        old_values = self.qnetwork_local.forward(states).gather(1, actions).squeeze(1)
        
        #detect done
        done_index = dones.argmax().item()
        if dones[done_index].item():
            accumulated_rewards[done_index] = 0.0 #should not be rewards[done_index], which is acturally -100

        loss = self.criterion(accumulated_rewards, old_values)
        loss.backward()
        self.optimizer.step()
        

        # ------------------- update target network ------------------- #
#         self.soft_update(self.qnetwork_local, self.qnetwork_target, TAU)  
    
    def criterion(self, accumulated_rewords, old_values):
        return (accumulated_rewords - old_values).pow(2).mean()
    
    def update_target_Q(self):
#         self.soft_update(self.qnetwork_local, self.qnetwork_target, self.tau) 
        self.qnetwork_target.load_state_dict(self.qnetwork_local.state_dict())

    def soft_update(self, local_model, target_model, tau):
        """Soft update model parameters.
        θ_target = τ*θ_local + (1 - τ)*θ_target

        Params
        ======
            local_model (PyTorch model): weights will be copied from
            target_model (PyTorch model): weights will be copied to
            tau (float): interpolation parameter 
        """
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(tau*local_param.data + (1.0-tau)*target_param.data)


class ReplayBuffer:
    """Fixed-size buffer to store experience tuples."""

    def __init__(self, action_size, buffer_size, batch_size, seed):
        """Initialize a ReplayBuffer object.

        Params
        ======
            action_size (int): dimension of each action
            buffer_size (int): maximum size of buffer
            batch_size (int): size of each training batch
            seed (int): random seed
        """
        self.action_size = action_size
        self.memory = deque(maxlen=buffer_size)  
        self.batch_size = batch_size
        self.experience = namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "done"])
        self.seed = random.seed(seed)
    
    def add(self, state, action, reward, next_state, done):
        """Add a new experience to memory."""
        e = self.experience(state, action, reward, next_state, done)
        self.memory.append(e)
    
    def sample(self):
        """Randomly sample a batch of experiences from memory."""
        experiences = random.sample(self.memory, k=self.batch_size)

        states = torch.from_numpy(np.vstack([e.state for e in experiences if e is not None])).float().to(device)
        actions = torch.from_numpy(np.vstack([e.action for e in experiences if e is not None])).long().to(device)
        rewards = torch.from_numpy(np.vstack([e.reward for e in experiences if e is not None])).float().to(device)
        next_states = torch.from_numpy(np.vstack([e.next_state for e in experiences if e is not None])).float().to(device)
        dones = torch.from_numpy(np.vstack([e.done for e in experiences if e is not None]).astype(np.uint8)).float().to(device)
  
        return (states, actions, rewards, next_states, dones)

    def __len__(self):
        """Return the current size of internal memory."""
        return len(self.memory)