import numpy as np
import tkinter as tk
import gym
import torch
from autoencoder import QNetAutoencoder
from train_pacman_dqn import AtariQnet

device = "cuda" if torch.cuda.is_available() else "cpu"

QNET_PATH = "pacman_qnet.pt"

AUTOENCODER_PATH = "trained_models/pacman/first_autoencoder.pt"

PRETRAINED_HIDDEN_SIZE = 512
HIDDEN_SIZE = 2048
TOPK_ACT = True
K = 20

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
        
        self.init_control_panel()
        
        self.MIN_COLOR = np.array((255, 0, 0))
        self.ZERO_COLOR = np.array((127, 127, 127))
        self.MAX_COLOR = np.array((0, 0, 255))
    
    def init_control_panel(self):
        self.frame = tk.Frame(self.root)
        self.step_button = tk.Button(self.frame, text="Step", width=5, command=self.step)
        self.manual_step_label = tk.Label(self.frame, text="Step in a forced direction with arrow keys (make sure this window is focused!)")
        self.backstep_button = tk.Button(self.frame, text="Step reverse (SLOW!)", width=20, command=self.backstep)
        self.move_input = tk.Text(self.frame, width=2, height=1)
        self.step_manual = tk.Button(self.frame, text="Step in chosen direction", width=20, command=self.step_input)
        self.canvas = tk.Canvas(self.frame, width=200, height=200, bg="#aabbcc")
        
        self.step_button.pack()
        self.manual_step_label.pack()
        self.backstep_button.pack()
        self.move_input.pack()
        self.step_manual.pack()
        self.canvas.pack()
        self.frame.pack()
        
        self.root.bind("<KeyRelease-Left>", self.step_left)
        self.root.bind("<KeyRelease-Right>", self.step_right)
        self.root.bind("<KeyRelease-Up>", self.step_up)
        self.root.bind("<KeyRelease-Down>", self.step_down)
    
    def _draw_square(self, coords, color, scale, canv):
        return canv.create_rectangle(coords[0]-scale/2, coords[1]-scale/2, coords[0]+scale/2, coords[1]+scale/2, fill=color)
    
    def draw_canvas(self, weights, renorm=True):
        max_str = max([abs(option.item()) for option in weights])
        weights = weights/max_str
        normed_strs = [option.item()/max_str for option in weights]
        colors = [self.ZERO_COLOR*(1-abs(strength))+self.MIN_COLOR*abs(strength) if strength < 0 else self.ZERO_COLOR*(1-abs(strength))+self.MAX_COLOR*abs(strength) for strength in normed_strs]
        colors = [ints_to_hex(*color) for color in colors]
        
        self.canvas.delete("all")
        chosen = torch.argmax(weights)
        for i, color in enumerate(colors):
            self._draw_square((100+DIRECTIONS[i][0]*50, 100+DIRECTIONS[i][1]*50), color, 50, self.canvas)
            if i == chosen:
                self.canvas.create_oval(80+DIRECTIONS[i][0]*50, 80+DIRECTIONS[i][1]*50, 120+DIRECTIONS[i][0]*50, 120+DIRECTIONS[i][1]*50, fill="#00ff00")
        
    def step_input(self):
        self.step_direction(int(self.move_input.get("1.0", "end-1c").split()[0]))
    
    def step(self):
        move = self.q(torch.from_numpy(np.transpose(self.obs, (2, 0, 1))).float()).argmax().item()
        self.step_direction(move)
        predicted_move = self.q(torch.from_numpy(np.transpose(self.obs, (2, 0, 1))).float())
        self.draw_canvas(predicted_move)
    
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
        

def main():
    root = tk.Tk()
    e = gym.make("ALE/MsPacman-v5", render_mode="human", repeat_action_probability=0.0)
    
    q = torch.load(QNET_PATH, map_location=device)
    q.eval()
    autoencoder = QNetAutoencoder(PRETRAINED_HIDDEN_SIZE, HIDDEN_SIZE, topk_activation=TOPK_ACT, k=K).to(device)
    autoencoder.load_pretrained(AUTOENCODER_PATH)
    autoencoder.eval()
    
    display = ControlPanel(root, q, autoencoder, e)
    
    root.mainloop()

if __name__ == "__main__":
    main()
