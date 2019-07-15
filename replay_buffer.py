import random
import torch
import numpy as np
from collections import namedtuple, deque

from segment_tree import SumSegmentTree, MinSegmentTree

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

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
        self.buffer_size = buffer_size
        self.memory = deque(maxlen=buffer_size)  
        self.batch_size = batch_size
        self.experience = namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "done"])
        self.seed = random.seed(seed)
        self.ptr = 0
    
    def add(self, state, action, reward, next_state, done):
        """Add a new experience to memory."""
        e = self.experience(state, action, reward, next_state, done)
        if len(self.memory) < self.buffer_size:
            self.memory.append(e)
        else:
            self.memory[self.ptr] = e   
        self.ptr = (self.ptr+1) % self.buffer_size
    
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
    
class PrioritizedReplayBuffer(ReplayBuffer):
    def __init__(self, action_size, buffer_size, batch_size, seed, alpha=0.6):
        super(PrioritizedReplayBuffer, self).__init__(action_size, buffer_size, batch_size, seed)
        
        #capacity must be positive and a power of 2
        tree_capacity = 1
        while tree_capacity < self.buffer_size:
            tree_capacity *= 2
        self.sum_tree = SumSegmentTree(tree_capacity)
        self.min_tree = MinSegmentTree(tree_capacity)
        self.max_priority, self.tree_ptr = 1.0, 0
        self.alpha = alpha
        
    def add(self, state, action, reward, next_state, done):
        
        self.sum_tree[self.tree_ptr] = self.max_priority**self.alpha
        self.min_tree[self.tree_ptr] = self.max_priority**self.alpha
        super().add(state, action, reward, next_state, done)
        self.tree_ptr = (self.tree_ptr + 1) % self.buffer_size
        
#         if self.tree_ptr == self.buffer_size-1:
#             for i in range(0, self.buffer_size-1):
#                 self.sum_tree[i] = self.sum_tree[i+1] 
#                 self.min_tree[i] = self.min_tree[i+1]
#             self.sum_tree[self.tree_ptr] = self.max_priority**self.alpha
#             self.min_tree[self.tree_ptr] = self.max_priority**self.alpha
#         else:

#         
        
    def sample(self, beta=0.4):
        indices = self._sample_proportional()
        
        indices = [index for index in indices if index<len(self.memory)]
        states = torch.from_numpy(np.vstack([self.memory[index].state for index in indices])).float().to(device)
        actions = torch.from_numpy(np.vstack([self.memory[index].action for index in indices])).long().to(device)
        rewards = torch.from_numpy(np.vstack([self.memory[index].reward for index in indices])).float().to(device)
        next_states = torch.from_numpy(np.vstack([self.memory[index].next_state for index in indices])).float().to(device)
        dones = torch.from_numpy(np.vstack([self.memory[index].done for index in indices]).astype(np.uint8)).float().to(device)
        weights = torch.from_numpy(np.vstack([self._cal_weight(index, beta) for index in indices])).float().to(device)
         
        return (states, actions, rewards, next_states, dones, weights, indices)
        
    def update_priority(self, indices, loss_for_prior):
        for idx, priority in zip(indices, loss_for_prior):
            self.sum_tree[idx] = priority ** self.alpha
            self.min_tree[idx] = priority ** self.alpha
            
            self.max_priority = max(self.max_priority, priority)
        
    def _sample_proportional(self):
        indices = []
        p_total = self.sum_tree.sum() #sum(0, len(self.memory)-1)
        segment = p_total / self.batch_size
        
        for i in range(self.batch_size):
            start = segment * i
            end = start + segment
            upper = random.uniform(start, end)
            index = self.sum_tree.retrieve(upper)
            indices.append(index)
        return indices
    
    def _cal_weight(self, index, beta):
        sum_priority = self.sum_tree.sum()
        min_priority = self.min_tree.min()
        current_priority = self.sum_tree[index]
        
 
#         max_w = (len(self.memory) * (min_priority/sum_priority)) ** (-beta)
#         current_w = (len(self.memory) * (current_priority/sum_priority)) ** (-beta)
        
#         return current_w / max_w
        return (min_priority / current_priority) ** beta
                 
                 
        