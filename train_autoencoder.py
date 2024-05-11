import os

import matplotlib.pyplot as plt

from autoencoder import QNetAutoencoder

from train_dqn import CheckmateQNet

QNET_PATH = "trained_rook_qnet.pt"
            
num_episodes = 100        
            
LEARNING_RATE = 0.001
REGULARIZATION_VALUE = 0.0001
PRETRAINED_HIDDEN_SIZE = 128
HIDDEN_SIZE = 1024
BATCH_SIZE = 64

def train_one_epoch(autoencoder, optimizer, data):
    optimizer.zero_grad()
    loss, out = autoencoder(data)
    loss.backward()
    optimizer.step()
    return loss, out
            

def main():
    env = gym.make('RookCheckmate-v0')
    q = CheckmateQnet().to(device)
    autoencoder = QNetAutoencoder(PRETRAINED_HIDDEN_SIZE, HIDDEN_SIZE)

    print_interval = 20
    score = 0.0  
    optimizer = optim.Adam(autoencoder.parameters(), lr=LEARNING_RATE, weight_decay=REGULARIZATION_VALUE)
    
    activations = []

    for n_epi in range(num_episodes):
        obs, _ = env.reset()
        s = np.concatenate(tuple(obs.values()))
        done = False

        while not done:
            s_tensor = torch.from_numpy(s).float().to(device)
            a = q(s_tensor)
            activations.append(q.get_activations(s_tensor))
            
            obs, r, done, truncated, info = env.step(a)
            s = np.concatenate(tuple(obs.values()))
            done_mask = 0.0 if done else 1.0
            score += r
            
    	    if len(activations) == BATCH_SIZE:
    	        loss, out = train_one_epoch(autoencoder, optimizer, t.stack(activations, dim=0))
    	        activations = []
    	        losses.append(loss.detach().item())
    	        print(f"Completed batch {i//BATCH_SIZE} with loss {loss.detach().item()}.")

            if done:
                break

        if n_epi%print_interval==0 and n_epi!=0:
            print("n_episode :{}, score : {:.1f}".format(n_epi, score/print_interval))
            score = 0.0
    env.close()
    t.save(autoencoder, "trained_autoencoder.pt")

if __name__ == "__main__":
    main()
    print("Program terminated successfully!")
