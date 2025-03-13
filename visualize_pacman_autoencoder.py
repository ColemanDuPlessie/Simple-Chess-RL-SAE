import numpy as np
import tkinter as tk
import matplotlib.pyplot as plt
import gym
import torch
from autoencoder import QNetAutoencoder
from train_pacman_dqn import AtariQnet
from tqdm import tqdm

device = "cuda" if torch.cuda.is_available() else "cpu"

QNET_PATH = "robust_pacman_qnet.pt"

AUTOENCODER_PATH = "trained_models/pacman/robust_autoencoder.pt"

PRETRAINED_HIDDEN_SIZE = 512
HIDDEN_SIZE = 2048
TOPK_ACT = True
K = 50

alternate_layer = False
layers_skipped = 2

preencoder_bias = -1

FEAT_ACT_GAME_PATH = "feature_activations/1000_games.pt"
FEAT_ACT_PATH = "feature_activations/1000_games_feat_acts.pt"
FEAT_ACT_HIGHLIGHTS_PATH = "feature_activations/highlights/"
FEAT_ACT_HIGHLIGHTS_NUM_SAMPLES = 1556654
LOADED_IGNORED_STEPS = 66

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

def ints_to_hex(r, g, b):
    r = min(r, 255)
    g = min(g, 255)
    b = min(b, 255)
    return "#" + hex(int(r)*16**4+int(g)*16**2+int(b))[2:].zfill(6)
            
class ControlPanel:

    def __init__(self, tk_root, qnet, autoencoder, env):
        self.root = tk_root
        self.q = qnet
        self.autoencoder = autoencoder
        self.env = env
        
        self.obs = self.env.reset()[0]
        self.move_list = []
        self.end_state = False
        self.feat_act_board_states = None
        self.feat_acts = None
        
        self.init_control_panel()
        
        self.MIN_COLOR = np.array((0, 0, 255))
        self.ZERO_COLOR = np.array((127, 127, 127))
        self.MAX_COLOR = np.array((255, 0, 0))
    
    def init_control_panel(self):
        self.autoencoder_verbose = tk.BooleanVar()
        self.premade_feat_acts = tk.BooleanVar()
        self.premade_feat_act_highlights = tk.BooleanVar()
    
        self.frame = tk.Frame(self.root)
        self.step_button = tk.Button(self.frame, text="Step", width=5, command=self.step)
        self.manual_step_label = tk.Label(self.frame, text="Step in a forced direction with arrow keys (make sure this window is focused!)")
        self.backstep_button = tk.Button(self.frame, text="Step reverse (SLOW!)", width=20, command=self.backstep)
        self.move_input = tk.Text(self.frame, width=2, height=1)
        self.step_manual = tk.Button(self.frame, text="Step in chosen direction", width=20, command=self.step_input)
        self.autoencoder_verbose_checkbox = tk.Checkbutton(self.frame, text="List active autoencoder features?", variable=self.autoencoder_verbose, onvalue=True, offvalue=False)
        
        self.canv_frame = tk.Frame(self.frame)
        self.generic_canv_frame = tk.Frame(self.canv_frame)
        self.canv_label = tk.Label(self.generic_canv_frame, text="DQN next move weights")
        self.canvas = tk.Canvas(self.generic_canv_frame, width=200, height=200, bg="#aabbcc")
        self.autoencoder_canv_frame = tk.Frame(self.canv_frame)
        self.autoencoder_canv_label = tk.Label(self.autoencoder_canv_frame, text="Autoencoder next move weights")
        self.autoencoder_canvas = tk.Canvas(self.autoencoder_canv_frame, width=200, height=200, bg="#ffeedd")
        self.diff_canv_frame = tk.Frame(self.canv_frame)
        self.diff_canv_label = tk.Label(self.diff_canv_frame, text="Autoencoder move minus DQN move")
        self.diff_canvas = tk.Canvas(self.diff_canv_frame, width=200, height=200, bg="#ffffff")
        
        self.ablation_canv_frame = tk.Frame(self.frame)
        self.ablation_label = tk.Label(self.ablation_canv_frame, text="As above, but with feature below ablated")
        self.ablation_input = tk.Text(self.ablation_canv_frame, width=5, height=1)
        self.ablation_autoencoder_canvas = tk.Canvas(self.ablation_canv_frame, width=200, height=200, bg="#ffeedd")
        self.ablation_diff_canvas = tk.Canvas(self.ablation_canv_frame, width=200, height=200, bg="#ffffff")
        
        self.gen_feat_acts_button = tk.Button(self.frame, text="Get feature activations (using 10 games, takes multiple minutes)", command=self.gen_feat_acts)
        self.load_feat_acts_checkbox = tk.Checkbutton(self.frame, text="Load, not generate, feature activations (1000, not 10)", variable=self.premade_feat_acts, onvalue=True, offvalue=False)
        self.get_max_feat_setup_button = tk.Button(self.frame, text="Get max activating game state of feature above", command=self.play_max_activating_game)
        self.get_act_fraction_button = tk.Button(self.frame, text="Get number of game states feature above activates on", command=self.get_act_fraction)
        self.get_nth_feat_setup_button = tk.Button(self.frame, text="Get Nth highest activating game state of feature above", command=self.play_nth_activating_game)
        self.load_feat_act_highlights = tk.Checkbutton(self.frame, text="Only load feature activation highlights", variable=self.premade_feat_act_highlights, onvalue=True, offvalue=False)
        
        self.step_button.pack()
        self.manual_step_label.pack()
        self.backstep_button.pack()
        self.move_input.pack()
        self.step_manual.pack()
        self.autoencoder_verbose_checkbox.pack()
        self.canv_frame.pack()
        self.generic_canv_frame.pack(side="left")
        self.canv_label.pack()
        self.canvas.pack()
        self.autoencoder_canv_frame.pack(side="left")
        self.autoencoder_canv_label.pack()
        self.autoencoder_canvas.pack()
        self.diff_canv_frame.pack(side="left")
        self.diff_canv_label.pack()
        self.diff_canvas.pack()
        self.ablation_canv_frame.pack()
        self.ablation_label.pack()
        self.ablation_input.pack()
        self.ablation_autoencoder_canvas.pack(side="left")
        self.ablation_diff_canvas.pack(side="right")
        self.gen_feat_acts_button.pack()
        self.load_feat_acts_checkbox.pack()
        self.get_max_feat_setup_button.pack()
        self.get_act_fraction_button.pack()
        self.get_nth_feat_setup_button.pack()
        self.load_feat_act_highlights.pack()
        self.frame.pack()
        
        self.root.bind("<KeyRelease-Left>", self.step_left)
        self.root.bind("<KeyRelease-Right>", self.step_right)
        self.root.bind("<KeyRelease-Up>", self.step_up)
        self.root.bind("<KeyRelease-Down>", self.step_down)
    
    def _draw_square(self, coords, color, scale, canv):
        return canv.create_rectangle(coords[0]-scale/2, coords[1]-scale/2, coords[0]+scale/2, coords[1]+scale/2, fill=color)
    
    def _draw_canvas(self, weights, canv, renorm=True, highlight=True):
        max_str = torch.max(weights).item()
        min_str = torch.min(weights).item()
        weights = (weights-min_str)/(max_str-min_str)*2-1
        normed_strs = [i.item() for i in weights]
        colors = [self.ZERO_COLOR*(1-abs(strength))+self.MIN_COLOR*abs(strength) if strength < 0 else self.ZERO_COLOR*(1-abs(strength))+self.MAX_COLOR*abs(strength) for strength in normed_strs]
        colors = [ints_to_hex(*color) for color in colors]
        
        canv.delete("all")
        chosen = torch.argmax(weights) if highlight else -1
        for i, color in enumerate(colors):
            self._draw_square((100+DIRECTIONS[i][0]*50, 100+DIRECTIONS[i][1]*50), color, 50, canv)
            if i == chosen:
                canv.create_oval(80+DIRECTIONS[i][0]*50, 80+DIRECTIONS[i][1]*50, 120+DIRECTIONS[i][0]*50, 120+DIRECTIONS[i][1]*50, fill="#00ff00")
        
    def step_input(self):
        self.step_direction(int(self.move_input.get("1.0", "end-1c").split()[0]))
    
    def _get_autoencoder_predicted_move(self, obs, verbose=False, ablated=-1):
        if alternate_layer:
            activation = self.q.get_activations_early(torch.from_numpy(np.transpose(obs, (2, 0, 1))).float(), layers_skipped)
        else:
            activation = self.q.get_activations(torch.from_numpy(np.transpose(obs, (2, 0, 1))).float())
        features = self.autoencoder.get_features(activation)
        post_act_func_feats = self.autoencoder.activation_func(features)
        if verbose:
            for i in range(len(features)):
                if post_act_func_feats[i] != 0.0:
                    print(f"Feature {i} is active with value {post_act_func_feats[i]}!")
        if ablated >= 0:
            post_act_func_feats[ablated] = 0
        activation = self.autoencoder.out_layer(post_act_func_feats)
        if alternate_layer:
            out = self.q.early_activations_to_out(activation, layers_skipped)
        else:
            out = self.q.activations_to_out(activation)
        return out
    
    def draw_predicted_next_move(self):
        predicted_move = self.q(torch.from_numpy(np.transpose(self.obs, (2, 0, 1))).float())
        self._draw_canvas(predicted_move, self.canvas)
        autoencoder_predicted_move = self._get_autoencoder_predicted_move(self.obs, verbose=self.autoencoder_verbose.get())
        self._draw_canvas(autoencoder_predicted_move, self.autoencoder_canvas)
        predicted_move_diff = autoencoder_predicted_move - predicted_move
        self._draw_canvas(predicted_move_diff, self.diff_canvas, highlight=False)
        
        ablation_input = self.ablation_input.get("1.0", "end-1c").split()
        ablated_neuron = int(ablation_input[0]) if len(ablation_input) > 0 else -1
        if ablated_neuron >= 0 and ablated_neuron < HIDDEN_SIZE:
            autoencoder_ablated_move = self._get_autoencoder_predicted_move(self.obs, ablated=ablated_neuron)
            self._draw_canvas(autoencoder_ablated_move, self.ablation_autoencoder_canvas)
            predicted_ablated_diff = autoencoder_ablated_move - predicted_move
            self._draw_canvas(predicted_ablated_diff, self.ablation_diff_canvas, highlight=False)
    
    def step(self):
        move = self.q(torch.from_numpy(np.transpose(self.obs, (2, 0, 1))).float()).argmax().item()
        self.step_direction(move)
        self.draw_predicted_next_move()
    
    def step_left(self, event): self.step_direction(3)
    def step_right(self, event): self.step_direction(2)
    def step_up(self, event): self.step_direction(1)
    def step_down(self, event): self.step_direction(4)
    
    def step_direction(self, move):
        print(f"Stepping in direction {move}")
        if self.end_state:
            self.obs = self.env.reset()[0]
            self.move_list = []
            self.end_state = False
            return
        self.move_list.append(move)
        out = self.env.step(move)
        self.obs = out[0]
        self.end_state |= out[2] or out[3]
    
    def backstep(self):
        """
        This takes advantage of the fact that at eval-time
        the environment is deterministic.
        """
        self.move_list = self.move_list[:-1]
        self.end_state = False
        self.env.reset()
        for move in self.move_list[:-1]:
            self.env.step(move)
        obs = self.env.step(self.move_list[-1])[0]
    
    def gen_feat_acts(self):
        if self.feat_acts is None:
            if self.premade_feat_act_highlights.get():
                self.feat_act_board_states, self.feat_acts = None, None
                self.activation_counts = torch.load(FEAT_ACT_HIGHLIGHTS_PATH+f"act_counts.pt", map_location=device)
                self.activation_freqs = torch.load(FEAT_ACT_HIGHLIGHTS_PATH+f"act_frequencies.pt", map_location=device)
                self.clamped_act_freqs = torch.clamp(self.activation_freqs, min=0.01/FEAT_ACT_HIGHLIGHTS_NUM_SAMPLES)
            else:
                if self.premade_feat_acts.get():
                    self.feat_act_board_states, self.feat_acts = load_feature_activations(FEAT_ACT_GAME_PATH, FEAT_ACT_PATH)
                else:
                    self.feat_act_board_states, self.feat_acts = gen_feature_activations(10, self.q, self.autoencoder, 0.5)
                self.activation_counts = torch.zeros(self.autoencoder.hidden_size)
                for game in self.feat_acts:
                    for activations in game.squeeze():
                        activated = torch.logical_not(torch.eq(activations, 0.0))
                        self.activation_counts += activated
                num_samples = sum(len(game.squeeze()) for game in self.feat_acts)
                self.activation_freqs = self.activation_counts / num_samples
                self.clamped_act_freqs = torch.clamp(self.activation_freqs, min=0.01/num_samples)
        act_log_freqs = torch.log10(self.clamped_act_freqs)
        plt.hist(act_log_freqs, bins=20)
        plt.xlabel("$log_{10}$ of activation frequency")
        plt.ylabel("# of features")
        plt.title(f"SAE Feature Activation Frequency, All Features, K=50")
        plt.show() # TODO make this pretty
    
    def get_act_fraction(self):
        feat_input = self.ablation_input.get("1.0", "end-1c").split()
        feat_target = int(feat_input[0]) if len(feat_input) > 0 else -1
        if feat_target == -1: return
        
        print(f"Feature {feat_target} activates on {self.activation_counts[feat_target]} board states, {100*self.activation_freqs[feat_target]}% of all board states tested!")
    
    def get_max_act_setups(self, feat, num_acts=10):
        if self.premade_feat_act_highlights.get() and num_acts <= 25:
            return torch.load(FEAT_ACT_HIGHLIGHTS_PATH+f"neuron_{feat}_activations.pt", map_location=device), torch.zeros(num_acts) # TODO re-add [:num_acts] slice?
        if self.feat_acts is None: self.gen_feat_acts()
        max_act_values = [-999999 for i in range(num_acts)] # -999999 is just an arbitrarily large negative number. In praactice, things should almost never go below 0. This list is not necessarily sorted until the end of this function.
        max_act_locations = [None for i in range(num_acts)] # The Nth element of this corresponds to the Nth element of max_act_values
        for episode_idx in range(self.feat_acts.size(0)):
            episode_acts = self.feat_acts[episode_idx].squeeze()
            for step in range(len(episode_acts)):
                act = episode_acts[step][feat].item()
                if act > min(max_act_values):
                    modified_idx = min(range(num_acts), key=max_act_values.__getitem__)
                    max_act_values[modified_idx] = act
                    max_act_locations[modified_idx] = (episode_idx, step)
        max_act_locations = [location for values, location in sorted(zip(max_act_values, max_act_locations), key=lambda pair: pair[0], reverse=True)]
        max_act_values = sorted(max_act_values, reverse=True)
        max_act_setups = [self.feat_act_board_states[location[0]].squeeze()[:location[1]] for location in max_act_locations]
        return max_act_setups, max_act_values
    
    def play_max_activating_game(self):
        feat_input = self.ablation_input.get("1.0", "end-1c").split()
        feat_target = int(feat_input[0]) if len(feat_input) > 0 else -1
        if feat_target == -1: return
        
        max_act_setup, max_act_value = self.get_max_act_setups(feat_target, num_acts=1)
        max_act_setup = max_act_setup[0].squeeze()
        max_act_value = max_act_value[0]
        
        self.env.reset()
        self.move_list = []
        self.end_state = False
        
        if self.premade_feat_acts.get() or self.premade_feat_act_highlights.get():
            for i in range(LOADED_IGNORED_STEPS):
                self.step_direction(0)
        
        for move in max_act_setup:
            self.step_direction(int(move.item()))
        self.draw_predicted_next_move()
    
    def play_nth_activating_game(self):
        feat_input = self.ablation_input.get("1.0", "end-1c").split()
        feat_target = int(feat_input[0]) if len(feat_input) > 0 else -1
        if feat_target == -1: return
        
        n = int(self.move_input.get("1.0", "end-1c").split()[0])
        
        max_act_setup, max_act_value = self.get_max_act_setups(feat_target, num_acts=n)
        max_act_setup = max_act_setup[n-1].squeeze()
        max_act_value = max_act_value[n-1]
        
        self.env.reset()
        self.move_list = []
        self.end_state = False
        
        if self.premade_feat_acts.get() or self.premade_feat_act_highlights.get():
            for i in range(LOADED_IGNORED_STEPS):
                self.step_direction(0)
        
        for move in max_act_setup:
            self.step_direction(int(move.item()))
        self.draw_predicted_next_move()
        
def gen_feature_activations(num_epis, q, autoencoder, epsilon=0.05):
    game_states = [] # This is a list of lists, each of which represents the moves taken in one game
    feat_acts = [] # The Nth element of the Mth element of this list is the activations that led to the Nth game state of the Mth game
    game = gym.make("ALE/MsPacman-v5", repeat_action_probability=0.0)
    obs = game.reset()[0]
    
    for i in tqdm(range(num_epis)):
        done = False
        moves = []
        feats = []
        while not done:
            a = q.sample_action(torch.from_numpy(np.transpose(obs, (2, 0, 1))).float().to(device), epsilon)
            moves.append(a)
            out = game.step(a)
            obs = out[0]
            done = out[2]
            
            if alternate_layer:
                activation = q.get_activations_early(torch.from_numpy(np.transpose(obs, (2, 0, 1))).float(), layers_skipped)
            else:
                activation = q.get_activations(torch.from_numpy(np.transpose(obs, (2, 0, 1))).float())
            features = autoencoder.activation_func(autoencoder.get_features(activation))
            feats.append(features)
            
        game_states.append(moves)
        feat_acts.append(feats)
        obs = game.reset()[0]
    return game_states, feat_acts

def load_feature_activations(game_filename, feat_acts_filename):
    """
    Similar to gen_feature_activations, but output is a ragged tensor, not a list. Takes two filenames.
    """
    game_states = torch.load(game_filename, map_location=device)
    feat_acts = torch.load(feat_acts_filename, map_location=device)
    return game_states, feat_acts

def main():
    root = tk.Tk()
    e = gym.make("ALE/MsPacman-v5", render_mode="human", repeat_action_probability=0.0)
    
    q = torch.load(QNET_PATH, map_location=device)
    q.eval()
    autoencoder = QNetAutoencoder(PRETRAINED_HIDDEN_SIZE, HIDDEN_SIZE, topk_activation=TOPK_ACT, k=K, preencoder_bias=preencoder_bias).to(device)
    autoencoder.load_pretrained(AUTOENCODER_PATH)
    autoencoder.eval()
    
    display = ControlPanel(root, q, autoencoder, e)
    
    root.mainloop()

if __name__ == "__main__":
    main()
