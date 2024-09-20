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

GAME_STATES_OUT_PATH = "feature_activations/1000_games.pt"
FEAT_ACTS_OUT_PATH   = "feature_activations/1000_games_feat_acts.pt"

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
            activation = q.get_activations(torch.from_numpy(np.transpose(obs, (2, 0, 1))).float())
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
    
def main():
    
    q = torch.load(QNET_PATH, map_location=device)
    q.eval()
    autoencoder = QNetAutoencoder(PRETRAINED_HIDDEN_SIZE, HIDDEN_SIZE, topk_activation=TOPK_ACT, k=K).to(device)
    autoencoder.load_pretrained(AUTOENCODER_PATH)
    autoencoder.eval()
    
    game_states, feat_acts = gen_feature_activations(1000, q, autoencoder, epsilon=0.2)
    
    torch.save(game_states, GAME_STATES_OUT_PATH)
    torch.save(feat_acts, FEAT_ACTS_OUT_PATH)

if __name__ == "__main__":
    main()
