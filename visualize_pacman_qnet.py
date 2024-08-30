import numpy as np
import gym
import torch
from train_pacman_dqn import AtariQnet

device = "cuda" if torch.cuda.is_available() else "cpu"

QNET_PATH = "pacman_qnet.pt"
            
num_steps = 10000

def main():
    e = gym.make("ALE/MsPacman-v5", render_mode="human")
    q = torch.load(QNET_PATH, map_location=device)
    q.eval()
    o, i = e.reset()
    
    for step in range(num_steps):
        o, r, t, t2, i = e.step(q(torch.from_numpy(np.transpose(o, (2, 0, 1))).float()).argmax().item())
        if t or t2:
            o, i = e.reset()

if __name__ == "__main__":
    main()
