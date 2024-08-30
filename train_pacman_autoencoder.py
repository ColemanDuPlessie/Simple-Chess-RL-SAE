import os
import argparse

from itertools import permutations
import numpy as np

import gym

import torch as t

from autoencoder import QNetAutoencoder
from train_pacman_dqn import AtariQnet, device

def gen_all_board_state_tensors(board_size=5, pieces=3):
    """
    Note that the only sanity-checking this
    does is to make sure two pieces don't occupy the same space.
    """
    spaces = [np.array((i, j)) for i in range(board_size) for j in range(board_size)]
    ans = permutations(spaces, pieces)
    return t.stack([t.from_numpy(convert_to_one_hot(np.concatenate(pos), board_size)).float().to(device) for pos in ans])

DEFAULT_QNET_PATH = "pacman_qnet.pt"
DEFAULT_OUT_PATH = "trained_models/pacman/first_autoencoder.pt"

num_episodes = 20000
resampling_points = [] # TODO [5000, 10000, 15000]
resampling_prep_duration = 1500
resampling_prep_points = [point-resampling_prep_duration for point in resampling_points]
init_transpose = True

LEARNING_RATE = 0.001

SPARSITY_TERM = 0 # 0.0000000005
DEFAULT_TOPK_K = 20 # TODO

DEFAULT_PRETRAINED_HIDDEN_SIZE = 512
DEFAULT_HIDDEN_SIZE = 2048
BATCH_SIZE = 2048

def train_one_epoch(autoencoder, optimizer, data):
    optimizer.zero_grad()
    loss, out = autoencoder(data)
    loss.backward()
    optimizer.step()
    return loss, out
            

def main(in_path=DEFAULT_QNET_PATH, out_path=DEFAULT_OUT_PATH, topk_k = None, pretrained_hidden_size=DEFAULT_PRETRAINED_HIDDEN_SIZE, hidden_size=DEFAULT_HIDDEN_SIZE):
    env = gym.make('ALE/MsPacman-v5')
    q = t.load(in_path, map_location=device)
    
    use_topk = topk_k > 0
    autoencoder = QNetAutoencoder(pretrained_hidden_size, hidden_size, loss_sparsity_term = SPARSITY_TERM, topk_activation = use_topk, k = topk_k, init_decoder_transpose=init_transpose).to(device)

    print_interval = 20
    score = 0.0  
    optimizer = t.optim.Adam(autoencoder.parameters(), lr=LEARNING_RATE)
    
    activations = []
    losses = []
    batch_num = 0

    for n_epi in range(num_episodes):
        obs, _ = env.reset()
        s = np.transpose(obs, (2, 0, 1))
        done = False
        
        if n_epi in resampling_prep_points:
            autoencoder.prepare_for_resampling()
        
        if n_epi in resampling_points: # TODO
            autoencoder.resample(gen_all_board_state_tensors(), preprocessing=q.get_activations, verbose=True)
            optimizer = t.optim.Adam(autoencoder.parameters(), lr=LEARNING_RATE) # TODO This doesn't seem like the most idiomatic way to reset the optimizer, but maybe it is... (c.f. https://discuss.pytorch.org/t/reset-optimizer-stats/78516/2)

        while not done:
            s_tensor = t.from_numpy(s).float().to(device)
            a = q(s_tensor).argmax().item()
            activations.append(q.get_activations(s_tensor))
            
            obs, r, done, truncated, info = env.step(a)
            s = np.transpose(obs, (2, 0, 1))
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
    t.save(autoencoder.state_dict(), out_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-o", "--outfile", default=DEFAULT_OUT_PATH, help="Out filename to save autoencoder to")
    parser.add_argument("-i", "--infile", default=DEFAULT_QNET_PATH, help="In filename to read qnet from")
    parser.add_argument("-a", "--activation", default=DEFAULT_PRETRAINED_HIDDEN_SIZE, help="Size of activation space of qnet", type=int)
    parser.add_argument("-f", "--feature", default=DEFAULT_HIDDEN_SIZE, help="Size of feature space of trained autoencoder", type=int)
    parser.add_argument("-k", "--topk", default=DEFAULT_TOPK_K, help="Number of simultaneously active autoencoder neurons (use k=0 to use relu instead)", type=int)
    args = parser.parse_args()
    print(args)
    main(in_path=args.infile, out_path=args.outfile, topk_k=args.topk, pretrained_hidden_size=args.activation, hidden_size=args.feature)
    print("Program terminated successfully!")
