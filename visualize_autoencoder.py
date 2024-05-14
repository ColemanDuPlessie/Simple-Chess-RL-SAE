import os
from itertools import permutations
from math import log, inf
import tkinter as tk

import numpy as np

import gym

import torch as t

from autoencoder import QNetAutoencoder
from train_dqn import CheckmateQnet, device

device = "cuda" if t.cuda.is_available() else "cpu"

root = tk.Tk()

QNET_PATH = "trained_rook_qnet.pt"
AUTOENCODER_PATH = "trained_autoencoder.pt"
            
num_episodes = 25000        
            
LEARNING_RATE = 0.001
REGULARIZATION_VALUE = 0.0001
PRETRAINED_HIDDEN_SIZE = 128
HIDDEN_SIZE = 1024
BATCH_SIZE = 1024

def ints_to_hex(r, g, b):
    return "#" + hex(int(r)*16**4+int(g)*16**2+int(b))[2:]

class FeatureExplorer:
    def __init__(self, tk_root, activations, ablations):
        self.root = tk_root
        self.board_size = ablations[0].shape[0]//4-2 # Note that this is 1 less than the actual board size: it's the maximum number of spaces a rook can possibly move in a straight line
        self.frame = tk.Frame(tk_root)
        self.feature_input = tk.Text(self.frame, width=5, height=1)
        self.go_button = tk.Button(self.frame, text="View feature", width=10, command=self.view_feature)
        self.canvas = tk.Canvas(self.frame, width=800, height=500, bg="#ffeedd")
        self.feature_input.pack()
        self.go_button.pack()
        self.canvas.pack()
        self.frame.pack()
        
        self.feat_acts = activations
        self.ablations = ablations
        self.max_ablation_magnitude = max([t.abs(ablation).max().item() for ablation in self.ablations])
        print(self.max_ablation_magnitude)
        self.MIN_ABLATION = np.array((255, 0, 0))
        self.ZERO_ABLATION = np.array((127, 127, 127))
        self.MAX_ABLATION = np.array((0, 0, 255))
        self.SQUARE_SIZE = 40
        self.KING_COORDS = [(50, 0), (50, 50), (0, 50), (-50, 50),
                            (-50, 0), (-50, -50), (0, -50), (50, -50)]
        self.ROOK_COORDS = [(50, 0), (0, 50), (-50, 0), (0, -50)]
    
    def _draw_square(self, coords, color):
        return self.canvas.create_rectangle(coords[0]-self.SQUARE_SIZE/2, coords[1]-self.SQUARE_SIZE/2, coords[0]+self.SQUARE_SIZE/2, coords[1]+self.SQUARE_SIZE/2, fill=color)
    
    def _draw_ablations(self, feat_num):
        ablation = self.ablations[feat_num]
        ablation_strs = [inf if square.item() == 0 else log(abs(square.item()/self.max_ablation_magnitude))*-square.item()/abs(square.item()) for square in ablation]
        print(ablation_strs) # TODO the code for determining color here is terrible
        ablation_colors = [self.ZERO_ABLATION*max(1+1/strength, 0)-self.MIN_ABLATION*min(1/strength, 1) if strength < 0 else self.ZERO_ABLATION*max(1-1/strength, 0)+self.MAX_ABLATION*min(1/strength, 1) for strength in ablation_strs]
        self.canvas.delete("all")
        self._draw_square((200, 250), "#ffffff")
        self._draw_square((600, 250), "#ffffff")
        for i, color in enumerate(ablation_colors):
            if i < 8:
                self._draw_square((200+self.KING_COORDS[i][0], 250+self.KING_COORDS[i][1]), ints_to_hex(*color))
            else:
                move = self.ROOK_COORDS[(i-8)//self.board_size]
                dist = (i-8)%self.board_size
                self._draw_square((600+move[0]*dist, 250+move[1]*dist), ints_to_hex(*color))
            
    def view_feature(self):
        feat_num = int(self.feature_input.get("1.0", "end-1c").split()[0])
        self._draw_ablations(feat_num)

def gen_all_board_states(board_size=5, pieces=3):
    """
    Note that the only sanity-checking this
    does is to make sure two pieces don't occupy the same space.
    """
    spaces = [np.array((i, j)) for i in range(board_size) for j in range(board_size)]
    ans = permutations(spaces, pieces)
    return [np.concatenate(pos) for pos in ans]

def main():
    q = t.load(QNET_PATH, map_location=device)
    autoencoder = QNetAutoencoder(PRETRAINED_HIDDEN_SIZE, HIDDEN_SIZE).to(device)
    autoencoder.load_pretrained(AUTOENCODER_PATH)
    
    q.eval()
    autoencoder.eval()
    
    board_states = gen_all_board_states(5, 3) # 13,800 unique states, counting rotations and reflections
    
    feature_activations = []
    losses = []
    batch_num = 0

    for n_state in range(len(board_states)):
        s = board_states[n_state]
        s_tensor = t.from_numpy(s).float().to(device)
        feature_activations.append(autoencoder.get_features(q.get_activations(s_tensor)))
    
    feature_tensor = t.stack(feature_activations, dim=0)
    mean_activation = t.mean(feature_tensor, dim=0)
    
    ablation_effects = []
    
    pre_ablation = q.activations_to_out(autoencoder.features_to_out(mean_activation))
    for feat_idx in range(HIDDEN_SIZE):
        ablated_activation = mean_activation.clone()
        ablated_activation[feat_idx] = 0
        post_ablation = q.activations_to_out(autoencoder.features_to_out(ablated_activation))
        ablation_effect = post_ablation-pre_ablation
        ablation_effects.append(ablation_effect)
    
    print(f"Data successfully generated! Input a feature # (up to but not including {HIDDEN_SIZE}) to see a visualization of that feature.")
    f = FeatureExplorer(root, feature_activations, ablation_effects)
    root.mainloop()
        

if __name__ == "__main__":
    main()
    print("Program terminated successfully!")
