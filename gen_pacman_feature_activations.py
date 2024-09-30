import numpy as np
import gym
import torch
from autoencoder import QNetAutoencoder
from train_pacman_dqn import AtariQnet
from tqdm import tqdm

device = "cuda" if torch.cuda.is_available() else "cpu"

QNET_PATH = "robust_pacman_qnet.pt"

AUTOENCODER_PATH = "trained_models/pacman/robust_autoencoder.pt"

STEPS_TO_IGNORE = 66 # The first 66 steps consist of a music sting while the player remains in place and can therefore be ignored.

PRETRAINED_HIDDEN_SIZE = 512
HIDDEN_SIZE = 2048
TOPK_ACT = True
K = 50

OUT_FOLDER_PATH = "feature_activations/highlights/"

def gen_feature_activations(num_epis, q, autoencoder, epsilon=0.05):
    game_states = None # This is a 2D Tensor, each row representing the moves taken in one game
    feat_acts = None # The Nth element of the Mth element of this list is the activations that led to the Nth game state of the Mth game
    game = gym.make("ALE/MsPacman-v5", repeat_action_probability=0.0)
    obs = game.reset()[0]
    
    for i in tqdm(range(num_epis)):
        done = False
        moves = []
        feats = None
        
        step_num = 0
        while not done:
            a = q.sample_action(torch.from_numpy(np.transpose(obs, (2, 0, 1))).float().to(device), epsilon)
            out = game.step(a)
            obs = out[0]
            done = out[2]
            step_num += 1
            if step_num <= STEPS_TO_IGNORE:
                continue
            moves.append(a)
            activation = q.get_activations(torch.from_numpy(np.transpose(obs, (2, 0, 1))).float().to(device))
            features = autoencoder.activation_func(autoencoder.get_features(activation))
            if feats is None:
                feats = features.unsqueeze(0)
            else:
                feats = torch.cat((feats, features.unsqueeze(0)))
            
        if game_states is None:
            game_states = torch.nested.nested_tensor([torch.Tensor(moves).unsqueeze(0)])
        else:
            game_states = torch.cat((game_states, torch.nested.nested_tensor([torch.Tensor(moves).unsqueeze(0)])))
        if feat_acts is None:
            feat_acts = torch.nested.nested_tensor([feats.unsqueeze(0)])
        else:
            feat_acts = torch.cat((feat_acts, torch.nested.nested_tensor([feats.unsqueeze(0)])))
        obs = game.reset()[0]
    return game_states, feat_acts

def get_act_freqs(feat_acts):
    hidden_dim = feat_acts[0].squeeze()[0].size(0)
    counts = torch.zeros(hidden_dim).to(dtype=torch.long, device=device)
    num_steps = 0
    for game in feat_acts:
        num_steps += game.squeeze().size(0)
        for step in game.squeeze():
            counts += torch.gt(step, 0.0)
    freqs = counts / num_steps
    return freqs, counts, num_steps

def get_max_act_paths(feat_acts, act_counts, num_saved=20, require_different_games=True):
    """
    An act_path is a tuple of the form: (actiation strength, game #, step #)
    act_paths will contain the num_saved highest-activating act_paths for any given neuron.
    If the neuron activates on less than num_saved different frames, all active frames will be returned.
    If require_different_games is True, only the highest activating frame of each game may be included.
    """
    num_feats = act_counts.size(0)
    act_paths = [None if act_counts[i] == 0.0 else [(0.0, -1, -1) for j in range(min((num_saved, act_counts[i])))] for i in range(num_feats)]
    min_idxs = [0 for i in range(num_feats)]
    for feat in tqdm(range(num_feats)): # TODO This should probably be a vector operation, not a loop
        if act_counts[feat] > 0:
            min_act = 0.0
            min_idx = 0
            for game_idx, game in enumerate(feat_acts):
                for step_idx, step in enumerate(game.squeeze()):
                    if step[feat] > min_act:
                        if require_different_games and any(game_idx == act[1] and act[0] >= step[feat] for act in act_paths[feat]):
                            continue # Ignore this activation if we already have a better one from this game.
                        act_paths[feat][min_idx] = (step[feat], game_idx, step_idx)
                        min_idx, min_act = min(enumerate(act_paths[feat][min_idx]), key = lambda x: x[0])
    return act_paths

def expand_path(path, games_history):
    return games_history[path[1]][:path[2]]
    
def save_neuron_max_activations(neuron_act_paths, games_history, filename):
    expanded_paths = [expand_path(path, games_history) for path in neuron_act_paths if path[1] != -1]
    torch.save(expanded_paths, filename)
    
def main():
    
    q = torch.load(QNET_PATH, map_location=device)
    q.eval()
    autoencoder = QNetAutoencoder(PRETRAINED_HIDDEN_SIZE, HIDDEN_SIZE, topk_activation=TOPK_ACT, k=K).to(device)
    autoencoder.load_pretrained(AUTOENCODER_PATH)
    autoencoder.eval()
    
    game_states, feat_acts = gen_feature_activations(2500, q, autoencoder, epsilon=0.2)
    
    freqs, counts, num_steps = get_act_freqs(feat_acts)
    print("Games finished!")
    max_act_paths = get_max_act_paths(feat_acts, counts, num_saved=25, require_different_games=True)
    print("Max activation paths found!")
    
    torch.save(counts, OUT_FOLDER_PATH+"act_counts.pt")
    torch.save(freqs, OUT_FOLDER_PATH+"act_frequencies.pt")
    
    with open(OUT_FOLDER_PATH+"num_steps_surveyed.txt", 'w') as f:
        f.write((str)(num_steps))
    
    print("Summary statistics saved!")

    for feat in range(HIDDEN_SIZE):
        if counts[feat] != 0:
            save_neuron_max_activations(max_act_paths[feat], game_states, f"{OUT_FOLDER_PATH}neuron_{feat}_activations.pt")
            print(f"Stats for neuron {feat} saved!")

if __name__ == "__main__":
    main()
