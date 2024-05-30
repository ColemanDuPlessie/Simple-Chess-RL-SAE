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

ONE_HOT_OBS_SPACE = True # TODO
OBS_SIZE = BOARD_SIZE**2*3 if ONE_HOT_OBS_SPACE else 6
BOARD_SIZE = 5
ACTION_SPACE_SIZE = 8+(BOARD_SIZE-1)*4

#Hyperparameters
is_equivariant = False
num_episodes  = 400000
learning_rate = 0.0005
gamma         = 0.98
buffer_limit  = 50000
batch_size    = 32
     
def make_equivariant(sample, board_size):
    best_sample = {"wRook" : (999, 999), "wKing" : (999, 999), "bKing" : (999, 999)}
    eq_code = None
    max_coord = board_size-1
    for vert_flip in (False, True):
        for horiz_flip in (False, True):
            for diag_flip in (False, True):
                curr_sample = sample.copy()
                for key in curr_sample.keys():
                    if vert_flip:
                        curr_sample[key] = (curr_sample[key][0], max_coord-curr_sample[key][1])
                    if horiz_flip:
                        curr_sample[key] = (max_coord-curr_sample[key][0], curr_sample[key][1])
                    if diag_flip:
                        curr_sample[key] = (curr_sample[key][1], curr_sample[key][0])
                for key in curr_sample.keys():
                    if curr_sample[key][0] < best_sample[key][0]:
                        best_sample = curr_sample
                        eq_code = (vert_flip, horiz_flip, diag_flip)
                        break
                    elif curr_sample[key][0] > best_sample[key][0]:
                        break
                    else:
                        if curr_sample[key][1] < best_sample[key][1]:
                            best_sample = curr_sample
                            eq_code = (vert_flip, horiz_flip, diag_flip)
                            break
                        elif curr_sample[key][1] > best_sample[key][1]:
                            break
    return best_sample, eq_code

def undo_equivariance(sample, eq_code, board_size):
    max_coord = BOARD_SIZE-1
    out = sample.copy()
    for key in out.keys():
        if eq_code[0]:
            out[key] = (out[key][0], max_coord-out[key][1])
        if eq_code[1]:
            out[key] = (max_coord-out[key][0], out[key][1])
        if eq_code[2]:
            out[key] = (out[key][1], out[key][2])
    return out

def undo_equivariance_action(action_code, eq_code):
    out = action_code
    if action_code < 8: # King move
        if eq_code[0]: # Vertical flip
            out = (8-out)%8
        if eq_code[1]: # Horizontal flip
            out = ((out//4)*4+(4-out%4))%8
        if eq_code[2]: # Diagonal flip
            out = ((9-out)%8+1)%8
    else: # Rook move
        rook_move_size = BOARD_SIZE-1
        dist = (out-8)%rook_move_size
        direction = (out-8)//rook_move_size
        if eq_code[0]: # Vertical flip
            if direction == 1: direction = 3
            elif direction == 3: direction = 1
        if eq_code[1]: # Horizontal flip
            if direction == 2: direction = 0
            elif direction == 0: direction = 2
        if eq_code[2]: # Diagonal flip
            if direction % 2 == 0: direction += 1
            else: direction -= 1
        out = 8+direction*rook_move_size+dist
    return out

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
    def __init__(self, action_size=ACTION_SPACE_SIZE, observation_size=6, hidden_size=128):
        super(CheckmateQnet, self).__init__()
        self.fc1 = nn.Linear(observation_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, action_size)

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
    env = gym.make('RookCheckmate-v0', random_opponent=False, one_hot_observation_space=ONE_HOT_OBS_SPACE)
    q = CheckmateQnet(hidden_size=512, observation_size=OBS_SIZE).to(device)
    q_target = CheckmateQnet(hidden_size=512, observation_size=OBS_SIZE).to(device)
    q_target.load_state_dict(q.state_dict())
    memory = ReplayBuffer()

    print_interval = 20
    score = 0.0  
    optimizer = optim.Adam(q.parameters(), lr=learning_rate)

    for n_epi in range(num_episodes):
        epsilon = max(0.01, 0.08 - 0.01*(n_epi/200)) #Linear annealing from 8% to 1%
        obs, _ = env.reset()
        if is_equivariant:
            obs, eq_code = make_equivariant(obs, BOARD_SIZE)
        s = np.concatenate(tuple(obs.values())) # This looks sketchy, but .values() is ordered based on order in which keys were created (which is always the same in my implementation of the environment) in python versions >= 3.6, so it's okay.
        done = False

        while not done:
            a = q.sample_action(torch.from_numpy(s).float().to(device), epsilon)
            if is_equivariant:
                a = undo_equivariance_action(a, eq_code)
            obs, r, done, truncated, info = env.step(a)
            if is_equivariant:
                obs, eq_code = make_equivariant(obs, BOARD_SIZE)
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
    
    torch.save(q, "smarter_trained_rook_qnet.pt")

if __name__ == '__main__':
    main()
