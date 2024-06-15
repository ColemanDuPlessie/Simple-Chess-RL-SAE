import os
from itertools import permutations
from math import log, inf
import tkinter as tk
import matplotlib.pyplot as plt

import numpy as np

import gym

import torch as t

from autoencoder import QNetAutoencoder
from train_dqn import CheckmateQnet

device = "cuda" if t.cuda.is_available() else "cpu"

root = tk.Tk()

BOARD_SIZE = 5
ONE_HOT = True

QNET_PATH = "smarter_trained_rook_qnet.pt"
AUTOENCODER_PATH = "4096_neuron_trained_autoencoder.pt"
            
num_episodes = 25000        
            
LEARNING_RATE = 0.001
REGULARIZATION_VALUE = 0.0001
PRETRAINED_HIDDEN_SIZE = 512
HIDDEN_SIZE = 4096
BATCH_SIZE = 1024

def convert_to_one_hot(sample):
    out = [np.zeros(BOARD_SIZE**2) for i in range(3)]
    for idx in range(len(out)):
        out[idx][sample[idx*2]+sample[idx*2+1]*BOARD_SIZE] = 1
    return np.concatenate(out)

def ints_to_hex(r, g, b):
    r = min(r, 255)
    g = min(g, 255)
    b = min(b, 255)
    return "#" + hex(int(r)*16**4+int(g)*16**2+int(b))[2:].zfill(6)

class FeatureExplorer:
    def __init__(self, tk_root, activations, ablations, q=None, autoencoder=None):
        self.root = tk_root
        self.board_size = BOARD_SIZE-1 # Note that this is 1 less than the actual board size: it's the maximum number of spaces a rook can possibly move in a straight line
        self.frame = tk.Frame(tk_root)
        self.feature_input = tk.Text(self.frame, width=5, height=1)
        self.go_button = tk.Button(self.frame, text="View feature", width=10, command=self.view_feature)
        self.custom_ablation_button = tk.Button(self.frame, text="View ablation (delta)", width=20, command=self.view_ablation)
        self.ablated_run_button = tk.Button(self.frame, text="View ablation (result)", width=20, command=self.view_move_without_ablation)
        self.run_button = tk.Button(self.frame, text="Run full model", width=15, command=self.view_move)
        self.run_autoencoder_button = tk.Button(self.frame, text="Run full model (with autoencoder)", width=30, command=self.view_move_with_autoencoder)
        self.play_canvas = tk.Canvas(self.frame, width=250, height=250, bg="#aabbcc")
        self.canvas = tk.Canvas(self.frame, width=160, height=100, bg="#ffeedd")
        self.mini_canvs = [tk.Canvas(self.frame, width=120, height=120, bg="#bbddbb") for i in range(10)]
        self.canv_labels = [tk.Label(self.frame, text="0.0") for i in range(10)]
        self.label = tk.Label(self.frame, text="Page (of boards):")
        self.page_input = tk.Text(self.frame, width=5, height=1)
        self.page_input.insert(tk.END, "1")
        self.max_activation_button = tk.Button(self.frame, text="Get boards that maximally activate this feature", width=45, command=self.view_activation_boards)
        
        self.play_canvas.bind("<Button-1>", self._play_board_clicked)
        
        self.feature_input.grid(row=0, column=1, columnspan=3)
        self.go_button.grid(row=1, column=0)
        self.custom_ablation_button.grid(row=1, column=1, columnspan=2)
        self.ablated_run_button.grid(row=1, column=3, columnspan=2)
        self.canvas.grid(row=2, column=1, columnspan=3)
        self.play_canvas.grid(row=3, column=1, columnspan=3)
        self.run_button.grid(row=4, column=0, columnspan=2)
        self.run_autoencoder_button.grid(row=4, column=2, columnspan=3)
        for idx, canv in enumerate(self.mini_canvs):
            canv.grid(row=5+2*(idx//5), column=idx%5)
            self.canv_labels[idx].grid(row=6+2*(idx//5), column=idx%5)
        self.label.grid(row=9, column=0)
        self.page_input.grid(row=9, column=1)
        self.max_activation_button.grid(row=9, column=2, columnspan=3)
        self.frame.pack()
        
        self.current_feature_inspecting = -1
        self.current_feature_act_order = None
        
        self.PLAY_CANVAS_SIZE = 50
        self.MINI_CANVAS_SIZE = 24
        self.rook_pos = (0, 0)
        self.wking_pos = (0, 4)
        self.bking_pos = (4, 4)
        self.play_space_selected = None
        
        self.q = q
        self.autoencoder = autoencoder
        
        self.board_states = activations[0]
        self.feat_acts = activations[1]
        self.ablations = ablations
        self.max_ablation_magnitude = max([t.abs(ablation).max().item() for ablation in self.ablations]) if self.ablations is not None else 1
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
    
    def _draw_mini_canvas(self, canv, rook, wking, bking):
        canv.delete("all")
        self._draw_square(((rook[0]+0.5)*self.MINI_CANVAS_SIZE, (rook[1]+0.5)*self.MINI_CANVAS_SIZE), "#ffffff", self.MINI_CANVAS_SIZE, canv)
        canv.create_oval(wking[0]*self.MINI_CANVAS_SIZE, wking[1]*self.MINI_CANVAS_SIZE, (wking[0]+1)*self.MINI_CANVAS_SIZE, (wking[1]+1)*self.MINI_CANVAS_SIZE, fill="#ffffff")
        canv.create_oval(bking[0]*self.MINI_CANVAS_SIZE, bking[1]*self.MINI_CANVAS_SIZE, (bking[0]+1)*self.MINI_CANVAS_SIZE, (bking[1]+1)*self.MINI_CANVAS_SIZE, fill="#000000")
    
    def _draw_play_board(self):
        self.play_canvas.delete("all")
        if self.play_space_selected is not None:
            self.play_canvas.create_oval((self.play_space_selected[0]-0.25)*self.PLAY_CANVAS_SIZE, (self.play_space_selected[1]-0.25)*self.PLAY_CANVAS_SIZE, (self.play_space_selected[0]+1.25)*self.PLAY_CANVAS_SIZE, (self.play_space_selected[1]+1.25)*self.PLAY_CANVAS_SIZE, fill="#88ff88")
        self._draw_square(((self.rook_pos[0]+0.5)*self.PLAY_CANVAS_SIZE, (self.rook_pos[1]+0.5)*self.PLAY_CANVAS_SIZE), "#ffffff", self.PLAY_CANVAS_SIZE, self.play_canvas)
        self.play_canvas.create_oval(self.wking_pos[0]*self.PLAY_CANVAS_SIZE, self.wking_pos[1]*self.PLAY_CANVAS_SIZE, (self.wking_pos[0]+1)*self.PLAY_CANVAS_SIZE, (self.wking_pos[1]+1)*self.PLAY_CANVAS_SIZE, fill="#ffffff")
        self.play_canvas.create_oval(self.bking_pos[0]*self.PLAY_CANVAS_SIZE, self.bking_pos[1]*self.PLAY_CANVAS_SIZE, (self.bking_pos[0]+1)*self.PLAY_CANVAS_SIZE, (self.bking_pos[1]+1)*self.PLAY_CANVAS_SIZE, fill="#000000")
    
    def _draw_square(self, coords, color, scale, canv):
        return canv.create_rectangle(coords[0]-scale/2, coords[1]-scale/2, coords[0]+scale/2, coords[1]+scale/2, fill=color)
    
    def _draw_ablations(self, ablation, scale, canv, highlight=False):
        max_str = max([abs(square.item()) for square in ablation])
        if max_str == 0: max_str = 1 # This can only occur if all strengths are 0 (i.e. the ReLU is not active on this input), and is only here to avoid an error from dividing by zero.
        ablation_strs = [square.item()/max_str for square in ablation]
        print(ablation_strs) # TODO the code for determining color here is terrible
        ablation_colors = [self.ZERO_ABLATION*(1-abs(strength))+self.MIN_ABLATION*abs(strength) if strength < 0 else self.ZERO_ABLATION*(1-abs(strength))+self.MAX_ABLATION*abs(strength) for strength in ablation_strs]
        if highlight:
            ablation_colors[ablation_strs.index(max(ablation_strs))] = (0, 255, 0)
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
        print("This feature is not currently implemented as it uses too much RAM for large models. However, it wasn't useful anyway.")
        # feat_num = int(self.feature_input.get("1.0", "end-1c").split()[0])
        # self._draw_ablations(self.ablations[feat_num], scale=self.SQUARE_SIZE, canv=self.canvas)
    
    def _get_play_board_state(self):
        if ONE_HOT:
            return convert_to_one_hot(np.array((*self.rook_pos, *self.wking_pos, *self.bking_pos))) # TODO
        else:
            return np.array((*self.rook_pos, *self.wking_pos, *self.bking_pos))
    
    def view_move(self):
        board_state = self._get_play_board_state()
        suggestion = self.q(t.from_numpy(board_state).float().to(device))
        self._draw_ablations(suggestion, scale=self.SQUARE_SIZE, canv=self.canvas, highlight=True)
    
    def view_move_with_autoencoder(self):
        board_state = self._get_play_board_state()
        activation = self.q.get_activations(t.from_numpy(board_state).float().to(device))
        features = self.autoencoder.get_features(activation)
        for i in range(len(features)):
            if features[i] > 0:
                print(f"Feature {i} is active with value {features[i]}!")
        activation = self.autoencoder.features_to_out(features)
        out = self.q.activations_to_out(activation)
        self._draw_ablations(out, scale=self.SQUARE_SIZE, canv=self.canvas, highlight=True)
    
    def view_move_without_ablation(self):
        feat_num = int(self.feature_input.get("1.0", "end-1c").split()[0])
        board_state = self._get_play_board_state()
        activation = self.q.get_activations(t.from_numpy(board_state).float().to(device))
        features = self.autoencoder.get_features(activation)
        features[feat_num] = 0
        activation = self.autoencoder.features_to_out(features)
        out = self.q.activations_to_out(activation)
        self._draw_ablations(out, scale=self.SQUARE_SIZE, canv=self.canvas, highlight=True)
    
    def view_ablation(self):
        feat_num = int(self.feature_input.get("1.0", "end-1c").split()[0])
        board_state = self._get_play_board_state()
        activation = self.q.get_activations(t.from_numpy(board_state).float().to(device))
        features = self.autoencoder.get_features(activation)
        ablation = gen_one_ablation(features, self.q, self.autoencoder, feat_num)
        self._draw_ablations(ablation, scale=self.SQUARE_SIZE, canv=self.canvas)
    
    def view_activation_boards(self):
        feat_num = int(self.feature_input.get("1.0", "end-1c").split()[0])
        if feat_num != self.current_feature_inspecting:
            self.current_feature_act_order = sorted(range(len(self.feat_acts)), key=lambda f: self.feat_acts[f][feat_num], reverse=True)
            self.current_feature_inspecting = feat_num
        page_num = int(self.page_input.get("1.0", "end-1c").split()[0])
        for idx, canv in enumerate(self.mini_canvs):
            board_idx = idx+(page_num-1)*10
            board = self.board_states[self.current_feature_act_order[board_idx]]
            self._draw_mini_canvas(canv, board[:2], board[2:4], board[4:])
            self.canv_labels[idx]['text'] = f"{self.feat_acts[self.current_feature_act_order[board_idx]][feat_num].item():.8f}"

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
        if ONE_HOT: s = convert_to_one_hot(s)
        s_tensor = t.from_numpy(s).float().to(device)
        feature_activations.append(autoencoder.get_features(q.get_activations(s_tensor)))
        
    return board_states, feature_activations

def gen_one_ablation(activation, q, autoencoder, feat_idx):
    pre = q.activations_to_out(autoencoder.features_to_out(activation))
    ablated = activation.clone()
    ablated[feat_idx] = 0
    post = q.activations_to_out(autoencoder.features_to_out(ablated))
    return pre - post

def gen_ablations(activation, q, autoencoder):
    ablation_effects = []
    pre_ablation = q.activations_to_out(autoencoder.features_to_out(activation))
    for feat_idx in range(HIDDEN_SIZE):
        ablated_activation = activation.clone()
        ablated_activation[feat_idx] = 0
        post_ablation = q.activations_to_out(autoencoder.features_to_out(ablated_activation))
        ablation_effect = pre_ablation-post_ablation
        ablation_effects.append(ablation_effect)
    
    return ablation_effects

def graph_hist_feat_acts(feat_acts):
    pass # TODO

def main():
    q = t.load(QNET_PATH, map_location=device)
    autoencoder = QNetAutoencoder(PRETRAINED_HIDDEN_SIZE, HIDDEN_SIZE).to(device)
    autoencoder.load_pretrained(AUTOENCODER_PATH)
    
    q.eval()
    autoencoder.eval()
    
    print("Loaded model, now generating activations...")
    
    board_states, feature_activations = gen_feat_acts(q, autoencoder)
    
    print("Generated activations, now generating generic ablations...")
    
    feature_tensor = t.stack(feature_activations, dim=0)
    mean_activation = t.mean(feature_tensor, dim=0)
    
    print("Skipping generic ablation generation...")
    # ablation_effects = gen_ablations(mean_activation, q, autoencoder)
    
    print(f"Data successfully generated! Input a feature # (up to but not including {HIDDEN_SIZE}) to see a visualization of that feature.")
    f = FeatureExplorer(root, (board_states, feature_activations), None, q, autoencoder)
    root.mainloop()
        

if __name__ == "__main__":
    main()
    print("Program terminated successfully!")
