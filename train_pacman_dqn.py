# Implementation of DQN taken from minimalRL at https://github.com/seungeunrho/minimalRL/blob/master/dqn.py

import gym
import collections
import random

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

device = "cuda" if torch.cuda.is_available() else "cpu"

ACTION_SPACE_SIZE = 9

#Hyperparameters
num_episodes  = 20000
learning_rate = 0.0005
gamma         = 0.98
buffer_limit  = 50000
batch_size    = 256

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

        return torch.tensor(np.stack(s_lst), dtype=torch.float).to(device), torch.tensor(np.stack(a_lst)).to(device), \
               torch.tensor(np.stack(r_lst)).to(device), torch.tensor(np.stack(s_prime_lst), dtype=torch.float).to(device), \
               torch.tensor(np.stack(done_mask_lst)).to(device)
    
    def size(self):
        return len(self.buffer)

class AtariQnet(nn.Module):
    def __init__(self, action_size=ACTION_SPACE_SIZE, hidden_size=512):
        super(AtariQnet, self).__init__()
        # Input data is shape(3, 210, 160)
        self.conv1 = nn.Conv2d(3, 16, 9, stride=2) # Data is now shape(16, 101, 76)
        self.pool1 = nn.MaxPool2d(2, stride=2) # Data is now shape(16, 50, 37)
        self.conv2 = nn.Conv2d(16, 32, 9, stride=3) # Data is now shape(32, 14, 10).
        self.conv3 = nn.Conv2d(32, 64, 6, stride=2) # Data is now shape(64, 5, 3). Flatten to shape(960)
        self.fc1 = nn.Linear(960, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, action_size)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool1(x)
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = torch.flatten(x, -3) # Flatten all dims (except batch, if it exists) into one
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
    
    def get_activations(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool(x)
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = t.flatten(x, -3) # Flatten all dims (except batch, if it exists) into one
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
    print("Training...")
    for i in range(5):
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
    env = gym.make('ALE/MsPacman-v5', obs_type="rgb")# , max_episode_steps=1000)
    q = AtariQnet(hidden_size=512).to(device)
    q_target = AtariQnet(hidden_size=512).to(device)
    q_target.load_state_dict(q.state_dict())
    memory = ReplayBuffer()

    print_interval = 20
    score = 0.0  
    optimizer = optim.Adam(q.parameters(), lr=learning_rate)
    print("Beginning training!")
    for n_epi in range(num_episodes):
        if n_epi < 50:
            print(f"Starting episode {n_epi}")
        epsilon = max(0.01, 0.08 - 0.01*(14*n_epi/num_episodes)) #Linear annealing from 8% to 1% over the first half of training
        obs, _ = env.reset()
        s = np.transpose(obs, (2, 0, 1)) # Move from color-last to color-first
        done = False

        while not done:
            a = q.sample_action(torch.from_numpy(s).float().to(device), epsilon)
            obs, r, done, truncated, info = env.step(a)
            s_prime = np.transpose(obs, (2, 0, 1))
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
    
    torch.save(q, "pacman_qnet.pt")

if __name__ == '__main__':
    main()
