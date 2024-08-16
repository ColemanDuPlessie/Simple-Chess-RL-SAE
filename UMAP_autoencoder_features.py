import os
import matplotlib.pyplot as plt

import umap

import numpy as np

import torch as t

from autoencoder import QNetAutoencoder

device = "cuda" if t.cuda.is_available() else "cpu"

AUTOENCODER_PATH = "trained_models/resampled_trained_autoencoders/k20.pt"      

TOPK_ACT = True
K = 20            
PRETRAINED_HIDDEN_SIZE = 128
HIDDEN_SIZE = 1024

def main():
    autoencoder = QNetAutoencoder(PRETRAINED_HIDDEN_SIZE, HIDDEN_SIZE, topk_activation=TOPK_ACT, k=K).to(device)
    autoencoder.load_pretrained(AUTOENCODER_PATH)
    
    autoencoder.eval()
    
    data = next(autoencoder.in_layer.parameters())
    
    print("Loaded model successfully...")
    
    reducer = umap.UMAP()
    
    embedding = reducer.fit_transform(data.detach().numpy())
    
    plt.scatter(embedding[:, 0], embedding[:, 1])
    plt.gca().set_aspect('equal', 'datalim')
    plt.title("UMAP projection of feature embeddings (TODO)")
    plt.show()

if __name__ == "__main__":
    main()
    print("Program terminated successfully!")
