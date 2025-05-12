import os
import numpy as np
import gym
import torch
from autoencoder import QNetAutoencoder
from train_pacman_dqn import AtariQnet
from tqdm import tqdm
from visualize_pacman_autoencoder import ControlPanel
import tkinter as tk
from PIL import ImageGrab
import time

root = tk.Tk()

device = "cuda" if torch.cuda.is_available() else "cpu"

QNET_PATH = "dqns/dqn9.pt"
AUTOENCODER_PATH = "dqns/autoencoder9.pt"
HIGHLIGHT_PATH = "dqns/highlights9/"

LOADED_IGNORED_STEPS = 66

PRETRAINED_HIDDEN_SIZE = 512
HIDDEN_SIZE = 2048
TOPK_ACT = True
K = 50

layers_skipped = 2

def load_highlight_histories(feat):
    return torch.load(HIGHLIGHT_PATH+f"neuron_{feat}_activations.pt", map_location=device)

def get_highlight_game(feat, num): # num ranges from 0 to 24, where 0 is the highest-activating gamestate, 1 is the second-highest, etc.
    return load_highlight_histories(feat)[num].squeeze()

def save_game_png(env, feat, num, path, q=None):
    steps = get_highlight_game(feat, num)
    env.reset()
    for i in range(LOADED_IGNORED_STEPS):
        env.step(0)
    for step in steps:
        o, a, b, c, d = env.step(int(step.item()))
    if q is not None:
        move = get_dqn_predicted_move(q, o).argmax().item()
        o, a, b, c, d = env.step(int(move))
    env.env.ale.saveScreenPNG(path)

def get_autoencoder_predicted_move(q, a, obs, ablated=-1):
    activation = q.get_activations_early(torch.from_numpy(np.transpose(obs, (2, 0, 1))).float(), layers_skipped)
    features = a.get_features(activation)
    post_act_func_feats = a.activation_func(features)
    if ablated >= 0:
        post_act_func_feats[ablated] = 0
    activation = a.out_layer(post_act_func_feats)
    out = q.early_activations_to_out(activation, layers_skipped)
    return out

def get_dqn_predicted_move(q, obs):
    return q(torch.from_numpy(np.transpose(obs, (2, 0, 1))).float())

MIN_COLOR = np.array((255, 0, 0))
ZERO_COLOR = np.array((127, 127, 127))
MAX_COLOR = np.array((0, 0, 255))

DIRECTIONS = {
    0: (0, 0),
    1: (0, -1),
    2: (1, 0),
    3: (-1, 0),
    4: (0, 1),
    5: (1, -1),
    6: (-1, -1),
    7: (1, 1),
    8: (-1, 1)
}

frame = tk.Frame(root)
canv = tk.Canvas(frame, width=200, height=200) # , bg="#aabbcc")
canv.pack()
frame.pack()

print("Generating canvas...")
root.update_idletasks()
root.update()
time.sleep(1)

def ints_to_hex(r, g, b):
    r = min(r, 255)
    g = min(g, 255)
    b = min(b, 255)
    return "#" + hex(int(r)*16**4+int(g)*16**2+int(b))[2:].zfill(6)

def save_canv(filepath):
    x = root.winfo_rootx()+canv.winfo_x()
    y = root.winfo_rooty()+canv.winfo_y()
    x1 = x+canv.winfo_width()
    y1 = y+canv.winfo_height()
    ImageGrab.grab().crop((x, y, x1, y1)).save(filepath)

def _draw_square(coords, color, scale, canv):
    return canv.create_rectangle(coords[0]-scale/2, coords[1]-scale/2, coords[0]+scale/2, coords[1]+scale/2, fill=color)

def draw_canvas(weights, highlight=False):
    max_str = torch.max(weights).item()
    min_str = torch.min(weights).item()
    weights = (weights-min_str)/(max_str-min_str)*2-1
    if max_str == min_str:
        weights = torch.Tensor([0]*9)
    normed_strs = [i.item() for i in weights]
    colors = [ZERO_COLOR*(1-abs(strength))+MIN_COLOR*abs(strength) if strength < 0 else ZERO_COLOR*(1-abs(strength))+MAX_COLOR*abs(strength) for strength in normed_strs]
    colors = [ints_to_hex(*color) for color in colors]
    
    canv.delete("all")
    chosen = torch.argmax(weights) if highlight else -1
    for i, color in enumerate(colors):
        _draw_square((100+DIRECTIONS[i][0]*50, 100+DIRECTIONS[i][1]*50), color, 50, canv)
        if i == chosen:
            canv.create_oval(80+DIRECTIONS[i][0]*50, 80+DIRECTIONS[i][1]*50, 120+DIRECTIONS[i][0]*50, 120+DIRECTIONS[i][1]*50, fill="#00ff00")
    root.update_idletasks()
    root.update()

def save_feat_act_png(env, feat, num, path, q, a, game_state_path=None):
    steps = get_highlight_game(feat, num)
    env.reset()
    for i in range(LOADED_IGNORED_STEPS):
        env.step(0)
    for step in steps:
        o, r, t, t2, i = env.step(int(step.item()))
    move = get_dqn_predicted_move(q, o).argmax().item()
    o, r, t, t2, i = env.step(int(move))
    q_acts = get_dqn_predicted_move(q, o)
    a_acts = get_autoencoder_predicted_move(q, a, o)
    ablated_acts = get_autoencoder_predicted_move(q, a, o, feat)
    ablated_a_diff = ablated_acts-a_acts
    ablated_q_diff = ablated_acts-q_acts
    draw_canvas(ablated_a_diff)
    if game_state_path is not None:
        env.env.ale.saveScreenPNG(game_state_path)
    time.sleep(0.1)
    save_canv(path)

def main():
    e = gym.make("ALE/MsPacman-v5", repeat_action_probability=0.0)
    q = torch.load(QNET_PATH, map_location=device)
    a = QNetAutoencoder(PRETRAINED_HIDDEN_SIZE, HIDDEN_SIZE, topk_activation=TOPK_ACT, k=K, preencoder_bias=1).to(device)
    a.load_pretrained(AUTOENCODER_PATH)
    a.eval()
    q.eval()
    o, i = e.reset()
    
    for feat_num in tqdm(range(1853, 2048)):
        if os.path.isfile(HIGHLIGHT_PATH+f"neuron_{feat_num}_activations.pt"):
            try:
                if not os.path.isfile(f"imgs/feat_{feat_num}/0.png"):
                    os.mkdir(f"imgs/feat_{feat_num}/")
                for act_num in range(25):
                    # save_game_png(e, feat_num, act_num, f"imgs/feat_{feat_num}/{act_num}.png", q)
                    save_feat_act_png(e, feat_num, act_num, f"imgs/feat_{feat_num}/act_{act_num}.png", q, a, game_state_path=f"imgs/feat_{feat_num}/{act_num}.png")
            except Exception as error:
                pass
                print("\n\n\nERROR: " + str(error))
    
    save_game_png(e, 0, 0, f"test.png")
    
    #for step in range(num_steps):
    #    o, r, t, t2, i = e.step(q(torch.from_numpy(np.transpose(o, (2, 0, 1))).float()).argmax().item())
    #    if t or t2:
    #        o, i = e.reset()

if __name__ == "__main__":
    main()
