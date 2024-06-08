import os

import numpy as np

import gym

import torch as t

from autoencoder import QNetAutoencoder
from train_dqn import CheckmateQnet, device
from visualize_autoencoder import gen_all_board_states

QNET_PATH = "smarter_trained_rook_qnet.pt"

num_episodes = 400000
resampling_points = [100000, 200000, 300000]
resampling_prep_duration = 30000
resampling_prep_points = [point-resampling_prep_duration for point in resampling_points]

LEARNING_RATE = 0.001
SPARSITY_TERM = 0.000000025
PRETRAINED_HIDDEN_SIZE = 512
HIDDEN_SIZE = 4096
BATCH_SIZE = 2048

def train_one_epoch(autoencoder, optimizer, data):
    optimizer.zero_grad()
    loss, out = autoencoder(data)
    loss.backward()
    optimizer.step()
    return loss, out
            

def main():
    env = gym.make('RookCheckmate-v0', random_opponent=False, one_hot_observation_space=True)
    q = t.load(QNET_PATH, map_location=device)
    
    autoencoder = QNetAutoencoder(PRETRAINED_HIDDEN_SIZE, HIDDEN_SIZE, loss_sparsity_term = SPARSITY_TERM).to(device)

    print_interval = 20
    score = 0.0  
    optimizer = t.optim.Adam(autoencoder.parameters(), lr=LEARNING_RATE)
    
    activations = []
    losses = []
    batch_num = 0

    for n_epi in range(num_episodes):
        obs, _ = env.reset()
        s = np.concatenate(tuple(obs.values()))
        done = False
        
        if n_epi in resampling_prep_points:
            autoencoder.prepare_for_resampling()
        
        if n_epi in resampling_points:
            autoencoder.resample(gen_all_board_states(), verbose=True)
            optimizer = t.optim.Adam(autoencoder.parameters(), lr=LEARNING_RATE) # TODO This doesn't seem like the most idiomatic way to reset the optimizer, but maybe it is... (c.f. https://discuss.pytorch.org/t/reset-optimizer-stats/78516/2)

        while not done:
            s_tensor = t.from_numpy(s).float().to(device)
            a = q(s_tensor).argmax().item()
            activations.append(q.get_activations(s_tensor))
            
            obs, r, done, truncated, info = env.step(a)
            s = np.concatenate(tuple(obs.values()))
            done_mask = 0.0 if done else 1.0
            score += r
            
            if len(activations) == BATCH_SIZE:
    	        loss, out = train_one_epoch(autoencoder, optimizer, t.stack(activations, dim=0))
    	        activations = []
    	        losses.append(loss.detach().item())
    	        print(f"Completed batch {batch_num} with loss {loss.detach().item()}.")
    	        batch_num += 1

            if done:
                break

        if n_epi%print_interval==0 and n_epi!=0:
            print("n_episode :{}, score : {:.1f}".format(n_epi, score/print_interval))
            score = 0.0
    env.close()
    t.save(autoencoder.state_dict(), "4096_neuron_trained_autoencoder.pt")

if __name__ == "__main__":
    main()
    print("Program terminated successfully!")
