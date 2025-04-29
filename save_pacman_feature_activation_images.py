import os
import gym
import torch
from autoencoder import QNetAutoencoder
from train_pacman_dqn import AtariQnet
from tqdm import tqdm

device = "cuda" if torch.cuda.is_available() else "cpu"

QNET_PATH = "dqns/dqn9.pt"
AUTOENCODER_PATH = "dqns/autoencoder9.pt"
HIGHLIGHT_PATH = "dqns/highlights9/"

LOADED_IGNORED_STEPS = 66

def load_highlight_histories(feat):
    return torch.load(HIGHLIGHT_PATH+f"neuron_{feat}_activations.pt", map_location=device)

def get_highlight_game(feat, num): # num ranges from 0 to 24, where 0 is the highest-activating gamestate, 1 is the second-highest, etc.
    return load_highlight_histories(feat)[num].squeeze()

def save_game_png(env, feat, num, path):
    steps = get_highlight_game(feat, num)
    env.reset()
    for i in range(LOADED_IGNORED_STEPS):
        env.step(0)
    for step in steps:
        env.step(int(step.item()))
    env.env.ale.saveScreenPNG(path)

def main():
    e = gym.make("ALE/MsPacman-v5", repeat_action_probability=0.0)
    q = torch.load(QNET_PATH, map_location=device)
    q.eval()
    o, i = e.reset()
    
    for feat_num in tqdm(range(2048)):
        if os.path.isfile(HIGHLIGHT_PATH+f"neuron_{feat_num}_activations.pt"):
            if not os.path.isfile(f"imgs/feat_{feat_num}/0.png"):
                os.mkdir(f"imgs/feat_{feat_num}/")
            try:
                for act_num in range(25):
                    save_game_png(e, feat_num, act_num, f"imgs/feat_{feat_num}/{act_num}.png")
            except Exception as error:
                print("\n\n\nERROR: " + str(error))
    
    save_game_png(e, 0, 0, f"test.png")
    
    #for step in range(num_steps):
    #    o, r, t, t2, i = e.step(q(torch.from_numpy(np.transpose(o, (2, 0, 1))).float()).argmax().item())
    #    if t or t2:
    #        o, i = e.reset()

if __name__ == "__main__":
    main()
