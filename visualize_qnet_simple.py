import envs
import numpy as np
import gym
import torch
from train_dqn import CheckmateQnet

device = "cuda" if torch.cuda.is_available() else "cpu"

QNET_PATH = "128_neuron_trained_rook_qnet.pt"
            
num_steps = 100

def main():
    e = gym.make("RookCheckmate-v0", render_mode="human", random_opponent=False, one_hot_observation_space=True)
    q = torch.load(QNET_PATH, map_location=device)
    q.eval()
    o, i = e.reset()
    
    for step in range(num_steps):
        o, r, t, t2, i = e.step(q(torch.from_numpy(np.concatenate(tuple(o.values()))).float()).argmax().item())
        if t or t2:
            o, i = e.reset()

if __name__ == "__main__":
    main()
