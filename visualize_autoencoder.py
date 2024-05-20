import os
from itertools import permutations
from math import log, inf
import tkinter as tk

import numpy as np

import gym

import torch as t

from autoencoder import QNetAutoencoder
from train_dqn import CheckmateQnet

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
    r = min(r, 255)
    g = min(g, 255)
    b = min(b, 255)
    return "#" + hex(int(r)*16**4+int(g)*16**2+int(b))[2:].zfill(6)

class FeatureExplorer:
    def __init__(self, tk_root, activations, ablations, q=None, autoencoder=None):
        self.root = tk_root
        self.board_size = ablations[0].shape[0]//4-2 # Note that this is 1 less than the actual board size: it's the maximum number of spaces a rook can possibly move in a straight line
        self.frame = tk.Frame(tk_root)
        self.feature_input = tk.Text(self.frame, width=5, height=1)
        self.go_button = tk.Button(self.frame, text="View feature", width=10, command=self.view_feature)
        self.custom_ablation_button = tk.Button(self.frame, text="View specific ablation", width=20, command=self.view_ablation)
        self.run_button = tk.Button(self.frame, text="Run full model", width=15, command=self.view_move)
        self.play_canvas = tk.Canvas(self.frame, width=250, height=250, bg="#aabbcc")
        self.canvas = tk.Canvas(self.frame, width=160, height=100, bg="#ffeedd")
        
        self.play_canvas.bind("<Button-1>", self._play_board_clicked)
        
        self.feature_input.pack()
        self.go_button.pack()
        self.custom_ablation_button.pack()
        self.canvas.pack()
        self.play_canvas.pack()
        self.run_button.pack()
        self.frame.pack()
        
        self.PLAY_CANVAS_SIZE = 50
        self.rook_pos = (0, 0)
        self.wking_pos = (0, 4)
        self.bking_pos = (4, 4)
        self.play_space_selected = None
        
        self.q = q
        self.autoencoder = autoencoder
        
        self.feat_acts = activations
        self.ablations = ablations
        self.max_ablation_magnitude = max([t.abs(ablation).max().item() for ablation in self.ablations])
        print(self.max_ablation_magnitude)
        self.MIN_ABLATION = np.array((255, 0, 0))
        self.ZERO_ABLATION = np.array((127, 127, 127))
        self.MAX_ABLATION = np.array((0, 0, 255))
        self.SQUARE_SIZE = 10
        self.KING_COORDS = [(1, 0), (1, 1), (0, 1), (-1, 1),
                            (-1, 0), (-1, -1), (0, -1), (1, -1)]
        self.ROOK_COORDS = [(1, 0), (0, 1), (-1, 0), (0, -1)]
        
        self._draw_play_board()
    
    def _play_board_clicked(self, event):
        location = (event.x // self.PLAY_CANVAS_SIZE, event.y // self.PLAY_CANVAS_SIZE)
        if location in (self.rook_pos, self.wking_pos, self.bking_pos):
            self.play_space_selected = location
        elif self.play_space_selected is not None:
            if self.rook_pos == self.play_space_selected:
                self.rook_pos = location
            elif self.wking_pos == self.play_space_selected:
                self.wking_pos = location
            elif self.bking_pos == self.play_space_selected:
                self.bking_pos = location
            self.play_space_selected = None
        self._draw_play_board()
    
    def _draw_play_board(self):
        self.play_canvas.delete("all")
        if self.play_space_selected is not None:
            self.play_canvas.create_oval((self.play_space_selected[0]-0.25)*self.PLAY_CANVAS_SIZE, (self.play_space_selected[1]-0.25)*self.PLAY_CANVAS_SIZE, (self.play_space_selected[0]+1.25)*self.PLAY_CANVAS_SIZE, (self.play_space_selected[1]+1.25)*self.PLAY_CANVAS_SIZE, fill="#88ff88")
        self._draw_square(((self.rook_pos[0]+0.5)*self.PLAY_CANVAS_SIZE, (self.rook_pos[1]+0.5)*self.PLAY_CANVAS_SIZE), "#ffffff", self.PLAY_CANVAS_SIZE, self.play_canvas)
        self.play_canvas.create_oval(self.wking_pos[0]*self.PLAY_CANVAS_SIZE, self.wking_pos[1]*self.PLAY_CANVAS_SIZE, (self.wking_pos[0]+1)*self.PLAY_CANVAS_SIZE, (self.wking_pos[1]+1)*self.PLAY_CANVAS_SIZE, fill="#ffffff")
        self.play_canvas.create_oval(self.bking_pos[0]*self.PLAY_CANVAS_SIZE, self.bking_pos[1]*self.PLAY_CANVAS_SIZE, (self.bking_pos[0]+1)*self.PLAY_CANVAS_SIZE, (self.bking_pos[1]+1)*self.PLAY_CANVAS_SIZE, fill="#000000")
    
    def _draw_square(self, coords, color, scale, canv):
        return canv.create_rectangle(coords[0]-scale/2, coords[1]-scale/2, coords[0]+scale/2, coords[1]+scale/2, fill=color)
    
    def _draw_ablations(self, ablation, scale, canv):
        ablation_strs = [inf if square.item() == 0 else log(abs(square.item()/self.max_ablation_magnitude))*-square.item()/abs(square.item()) for square in ablation]
        print(ablation_strs) # TODO the code for determining color here is terrible
        ablation_colors = [self.ZERO_ABLATION*max(1+1/strength, 0)-self.MIN_ABLATION*min(1/strength, 1) if strength < 0 else self.ZERO_ABLATION*max(1-1/strength, 0)+self.MAX_ABLATION*min(1/strength, 1) for strength in ablation_strs]
        self.canvas.delete("all")
        self._draw_square((4*scale, 5*scale), "#ffffff", scale, canv)
        self._draw_square((11*scale, 5*scale), "#ffffff", scale, canv)
        for i, color in enumerate(ablation_colors):
            if i < 8:
                self._draw_square((scale*(4+self.KING_COORDS[i][0]), scale*(5+self.KING_COORDS[i][1])), ints_to_hex(*color), scale, canv)
            else:
                move = self.ROOK_COORDS[(i-8)//self.board_size]
                dist = (i-8)%self.board_size+1
                self._draw_square((scale*(11+move[0]*dist), scale*(5+move[1]*dist)), ints_to_hex(*color), scale, canv)
            
    def view_feature(self):
        feat_num = int(self.feature_input.get("1.0", "end-1c").split()[0])
        self._draw_ablations(self.ablations[feat_num], scale=self.SQUARE_SIZE, canv=self.canvas)
    
    def _get_play_board_state(self):
        return np.array((*self.rook_pos, *self.wking_pos, *self.bking_pos))
    
    def view_move(self):
        board_state = self._get_play_board_state()
        suggestion = self.q(t.from_numpy(board_state).float().to(device))
        self._draw_ablations(suggestion, scale=self.SQUARE_SIZE, canv=self.canvas)
    
    def view_ablation(self):
        feat_num = int(self.feature_input.get("1.0", "end-1c").split()[0])
        board_state = self._get_play_board_state()
        activation = self.q.get_activations(t.from_numpy(board_state).float().to(device))
        features = self.autoencoder.get_features(activation)
        ablation = gen_ablations(features, self.q, self.autoencoder)
        self._draw_ablations(ablation[feat_num], scale=self.SQUARE_SIZE, canv=self.canvas)

def gen_all_board_states(board_size=5, pieces=3):
    """
    Note that the only sanity-checking this
    does is to make sure two pieces don't occupy the same space.
    """
    spaces = [np.array((i, j)) for i in range(board_size) for j in range(board_size)]
    ans = permutations(spaces, pieces)
    return [np.concatenate(pos) for pos in ans]

def gen_feat_acts(q, autoencoder):
    board_states = gen_all_board_states(5, 3) # 13,800 unique states, counting rotations and reflections
    
    feature_activations = []
    losses = []
    batch_num = 0

    for n_state in range(len(board_states)):
        s = board_states[n_state]
        s_tensor = t.from_numpy(s).float().to(device)
        feature_activations.append(autoencoder.get_features(q.get_activations(s_tensor)))
        
    return board_states, feature_activations

def gen_ablations(activation, q, autoencoder):
    ablation_effects = []
    pre_ablation = q.activations_to_out(autoencoder.features_to_out(activation))
    for feat_idx in range(HIDDEN_SIZE):
        ablated_activation = activation.clone()
        ablated_activation[feat_idx] = 0
        post_ablation = q.activations_to_out(autoencoder.features_to_out(ablated_activation))
        ablation_effect = post_ablation-pre_ablation
        ablation_effects.append(ablation_effect)
    
    return ablation_effects

def main():
    q = t.load(QNET_PATH, map_location=device)
    autoencoder = QNetAutoencoder(PRETRAINED_HIDDEN_SIZE, HIDDEN_SIZE).to(device)
    autoencoder.load_pretrained(AUTOENCODER_PATH)
    
    q.eval()
    autoencoder.eval()
    
    board_states, feature_activations = gen_feat_acts(q, autoencoder)
    
    feature_tensor = t.stack(feature_activations, dim=0)
    mean_activation = t.mean(feature_tensor, dim=0)
    
    ablation_effects = gen_ablations(mean_activation, q, autoencoder)
    
    print(f"Data successfully generated! Input a feature # (up to but not including {HIDDEN_SIZE}) to see a visualization of that feature.")
    f = FeatureExplorer(root, feature_activations, ablation_effects, q, autoencoder)
    root.mainloop()
        

if __name__ == "__main__":
    main()
    print("Program terminated successfully!")
