import os
import matplotlib.pyplot as plt

import tkinter as tk # For filedialog
from tkinter import filedialog

import umap

import numpy as np

import torch as t

from autoencoder import QNetAutoencoder

from visualize_autoencoder import gen_feat_acts
from train_dqn import CheckmateQnet

device = "cuda" if t.cuda.is_available() else "cpu"

QNET_PATH = "128_neuron_trained_rook_qnet.pt"

ASK_FOR_AUTOENCODER_PATH = True
DEFAULT_AUTOENCODER_PATH = "trained_models/transpose_initialized_autoencoders/k20.pt"      

TOPK_ACT = True
K = 20            
PRETRAINED_HIDDEN_SIZE = 512
HIDDEN_SIZE = 2048

SHOW_DEAD_NEURONS = True
IGNORE_DEAD_NEURONS = False

def get_live_neurons(feat_acts, autoencoder):
    num_feats = feat_acts[0].shape[0]
    num_samples = len(feat_acts)
    act_counts = t.zeros(num_feats)
    zero = t.zeros(num_feats)
    acts = []
    for act in feat_acts:
        acted = t.gt(autoencoder.activation_func(act), zero)
        acts.append(autoencoder.activation_func(act))
        act_counts += acted
    acted = act_counts > 0
    return acted

def main():
    if ASK_FOR_AUTOENCODER_PATH:
        print(f"Please select the autoencoder file to open. It should be compatible with the Qnet stored in {QNET_PATH}")
        AUTOENCODER_PATH = filedialog.askopenfilename()
    else: AUTOENCODER_PATH = DEFAULT_AUTOENCODER_PATH
    q = t.load(QNET_PATH, map_location=device)
    q.eval()

    autoencoder = QNetAutoencoder(PRETRAINED_HIDDEN_SIZE, HIDDEN_SIZE, topk_activation=TOPK_ACT, k=K).to(device)
    autoencoder.load_pretrained(AUTOENCODER_PATH)
    
    print("Loaded model successfully...")
    
    autoencoder.eval()
    
    data = next(autoencoder.in_layer.parameters())
    
    reducer = umap.UMAP()
    
    if SHOW_DEAD_NEURONS:
        # colors = ["#1f77b4" if live else "#ff7f0e" for live in get_live_neurons(gen_feat_acts(q, autoencoder)[1], autoencoder)]
        embedding = reducer.fit_transform(data.detach().numpy())
        plt.scatter(embedding[:, 0], embedding[:, 1]) # , c=colors)
    elif IGNORE_DEAD_NEURONS:
        live_neurons = get_live_neurons(gen_feat_acts(q, autoencoder)[1], autoencoder)
        data = t.stack([data[idx] for idx in range(len(data)) if live_neurons[idx]])
        embedding = reducer.fit_transform(data.detach().numpy())
        plt.scatter(embedding[:, 0], embedding[:, 1])
    else:
        embedding = reducer.fit_transform(data.detach().numpy())
        live_neurons = get_live_neurons(gen_feat_acts(q, autoencoder)[1], autoencoder)
        embedding = np.stack([embedding[idx] for idx in range(len(embedding)) if live_neurons[idx]])
        plt.scatter(embedding[:, 0], embedding[:, 1])
    plt.gca().set_aspect('equal', 'datalim')
    plt.title("UMAP projection of feature embeddings (TODO)")
    plt.show()

if __name__ == "__main__":
    root = tk.Tk()
    root.withdraw()
    main()
    print("Program terminated successfully!")
