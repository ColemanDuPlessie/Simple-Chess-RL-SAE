# Implementation of DQN taken from minimalRL at https://github.com/seungeunrho/minimalRL/blob/master/dqn.py

import gym
import collections
import random

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import envs # Adds RookCheckmate-v0 to gym

device = "cuda" if torch.cuda.is_available() else "cpu"

BOARD_SIZE = 5
ACTION_SPACE_SIZE = 8+(BOARD_SIZE-1)*4

#Hyperparameters
num_episodes  = 50000
learning_rate = 0.0005
gamma         = 0.98
buffer_limit  = 50000
batch_size    = 32

class ReplayBuffer():
    def __init__(self):
        self.buffer = collections.deque(maxlen=buffer_limit)
    
    def put(self, transition):
        self.buffer.append(transition)
    
    def sample(self, n):
        mini_batch = random.sample(self.buffer, n)
        s_lst, a_lst, r_lst, s_prime_lst, done_mask_lst = [], [], [], [], []
        
        for transition in mini_batch:
            s, a, r, s_prime, done_mask = transition
            s_lst.append(s)
            a_lst.append([a])
            r_lst.append([r])
            s_prime_lst.append(s_prime)
            done_mask_lst.append([done_mask])

        return torch.tensor(s_lst, dtype=torch.float).to(device), torch.tensor(a_lst).to(device), \
               torch.tensor(r_lst).to(device), torch.tensor(s_prime_lst, dtype=torch.float).to(device), \
               torch.tensor(done_mask_lst).to(device)
    
    def size(self):
        return len(self.buffer)

class CheckmateQnet(nn.Module):
    def __init__(self, action_size=ACTION_SPACE_SIZE, observation_size=6):
        super(CheckmateQnet, self).__init__()
        self.fc1 = nn.Linear(observation_size, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, action_size)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
    
    def get_activations(self, x):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x
    
    def activations_to_out(self, acts):
        return self.fc3(F.relu(acts))
      
    def sample_action(self, obs, epsilon):
        out = self.forward(obs)
        coin = random.random()
        if coin < epsilon:
            return random.randrange(ACTION_SPACE_SIZE)
        else : 
            return out.argmax().item()
            
    def load_pretrained(self, path: str = "model/pytorch_model.bin") -> None:
        self.load_state_dict(torch.load(path, map_location=device))
            
def train(q, q_target, memory, optimizer):
    for i in range(10):
        s,a,r,s_prime,done_mask = memory.sample(batch_size)

        q_out = q(s)
        q_a = q_out.gather(1,a)
        max_q_prime = q_target(s_prime).max(1)[0].unsqueeze(1)
        target = r + gamma * max_q_prime * done_mask
        loss = F.smooth_l1_loss(q_a, target)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

def main():
    env = gym.make('RookCheckmate-v0')
    q = CheckmateQnet().to(device)
    q_target = CheckmateQnet().to(device)
    q_target.load_state_dict(q.state_dict())
    memory = ReplayBuffer()

    print_interval = 20
    score = 0.0  
    optimizer = optim.Adam(q.parameters(), lr=learning_rate)

    for n_epi in range(num_episodes):
        epsilon = max(0.01, 0.08 - 0.01*(n_epi/200)) #Linear annealing from 8% to 1%
        obs, _ = env.reset()
        s = np.concatenate(tuple(obs.values()))
        done = False

        while not done:
            a = q.sample_action(torch.from_numpy(s).float().to(device), epsilon)     
            obs, r, done, truncated, info = env.step(a)
            s_prime = np.concatenate(tuple(obs.values()))
            done_mask = 0.0 if done else 1.0
            memory.put((s,a,r/100.0,s_prime, done_mask))
            s = s_prime

            score += r
            if done:
                break
            
        if memory.size()>2000:
            train(q, q_target, memory, optimizer)

        if n_epi%print_interval==0 and n_epi!=0:
            q_target.load_state_dict(q.state_dict())
            print("n_episode :{}, score : {:.1f}, n_buffer : {}, eps : {:.1f}%".format(
                                                            n_epi, score/print_interval, memory.size(), epsilon*100))
            score = 0.0
    env.close()
    
    torch.save(q, "trained_rook_qnet.pt")

if __name__ == '__main__':
    main()
