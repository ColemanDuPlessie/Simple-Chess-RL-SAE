from typing import Optional, Tuple
import torch as t
import torch.nn as nn
import torch.nn.functional as F

device = "cuda" if t.cuda.is_available() else "cpu"

class QNetAutoencoder(nn.Module):
    def __init__(
        self,
        in_size: int, hidden_size: int,
        pretrained_load_path: Optional[str] = None,
        loss_sparsity_term = 0.01,
    ):
        super().__init__()

        self.in_layer = nn.Linear(in_size, hidden_size)
        self.relu = nn.ReLU()
        self.out_layer = nn.utils.parametrizations.weight_norm(nn.Linear(hidden_size, in_size), name='weight')
        self.out_layer.parametrizations.weight.original0 = nn.Parameter(t.ones(1, in_size), requires_grad = False)
        
        self.in_size = in_size
        self.hidden_size = hidden_size
        
        self.loss_sparsity_term = loss_sparsity_term
        
        self.track_dead_neurons = False
        self.live_neurons = t.zeros(hidden_size, device=device, dtype=t.bool)

        if pretrained_load_path is not None:
            self.load_pretrained(pretrained_load_path)
    
    def prepare_for_resampling(self):
        self.track_dead_neurons = True
        self.live_neurons = t.zeros(self.hidden_size, device=device, dtype=t.bool) # Assume neurons are dead until we see them fire once

    def forward(self, x) -> Tuple[t.Tensor, t.Tensor, t.Tensor]:
        in_data = x.reshape(-1, x.shape[-1])
        with_bias = in_data + self.out_layer.bias
        acts = self.relu(self.in_layer(with_bias))
        if self.track_dead_neurons:
            acted = t.gt(t.sum(acts, dim=0), t.zeros(acts.shape[1:], device=device)) # This is only true for a neuron if it activated at least once
            self.live_neurons = t.logical_or(self.live_neurons, acted) # A neuron is live if we already knew it was live or if it just fired
        out = self.out_layer(acts)
        out = out.reshape_as(x)
        loss = F.mse_loss(out, x) + self.loss_sparsity_term * t.norm(acts, p=1)
        return loss, out
        
    def resample(self, full_dataset_in, preprocessing=lambda x: x, resample_strength_weighting=0.2, verbose=False):
        """
        The current dataset is only on the order of 10k samples.
        If the dataset gets substantially larger, it would be wiser
        to just use a large random subset (e.g. Anthropic uses 819k
        inputs when training the SAE on an LLM) of the dataset
        
        Follows the pattern laid out in Anthropic's paper:
        https://transformer-circuits.pub/2023/monosemantic-features#appendix-autoencoder-resampling
        
        DON'T FORGET TO RESET THE ADAM OPTIMIZER AFTER CALLING THIS FUNCTION
        """
        dead_neurons = t.logical_not(self.live_neurons)
        num_dead_neurons = t.sum(dead_neurons)
        if verbose: print(f"Resampling {num_dead_neurons} neurons...")
        self.track_dead_neurons = False
        if num_dead_neurons == 0:
            print("Would have resampled some neurons, but none of them needed it.")
            return
        losses = []
        with t.no_grad():
            for idx, sample in enumerate(full_dataset_in): # TODO I think this is required because we need the element-wise loss, but there's a decent chance I just forgot about an alternative, which would probably be much faster if it exists
                preprocessed = preprocessing(sample)
                with_bias = preprocessed + self.out_layer.bias
                out = self.out_layer(self.relu(self.in_layer(with_bias)))
                loss = F.mse_loss(out, preprocessed)
                losses.append(loss)
            losses = t.stack(losses)
            losses_squared = t.square(losses)
            print(losses_squared.shape)
            inputs_to_resample = t.multinomial(losses_squared, num_dead_neurons)
            
            neurons_to_resample = t.nonzero(dead_neurons)
            
            total_active_norm = 0
            for i in range(self.hidden_size):
                if i not in neurons_to_resample:
                    total_active_norm += t.norm(self.in_layer.weight[i], p=2)
            avg_active_norm = total_active_norm / (self.hidden_size - num_dead_neurons)
            print(f"Average norm of encoder weights of active neurons: {avg_active_norm}")
            
            inputs_to_resample = [preprocessing(full_dataset_in[idx.item()]) for idx in inputs_to_resample]
            inputs_to_resample = [act/t.norm(act, p=2) for act in inputs_to_resample]
            for i in range(len(inputs_to_resample)):
                self.out_layer.parametrizations.weight.original1[:, neurons_to_resample[i]] = t.reshape(inputs_to_resample[i], (-1, 1))
                self.in_layer.weight[neurons_to_resample[i], :] = t.reshape(inputs_to_resample[i], (-1,))*resample_strength_weighting*avg_active_norm
                self.in_layer.bias[neurons_to_resample[i]] = 0
        if verbose:
            print(f"Successfully resampled {num_dead_neurons} dead neurons!")
    
    def features_to_out(self, feats):
        return self.out_layer(self.relu(feats))
       
    def get_features(self, x):
        with_bias = x + self.out_layer.bias
        return self.in_layer(with_bias)
    
    def forward_with_feature_ablation(self, x, feat_to_ablate, ablation_function=lambda x: 0):
        in_data = x.reshape(-1, x.shape[-1])
        with_bias = in_data + self.out_layer.bias
        features = self.in_layer(with_bias)
        features[feat_to_ablate] = ablation_function(features[feat_to_ablate])
        out = self.out_layer(self.relu(features))
        out = out.reshape_as(x)
        return out

    def load_pretrained(self, path: str = "model/pytorch_model.bin") -> None:
        try:
            self.load_state_dict(t.load(path, map_location=device))
        except RuntimeError: # Parameter mismatch due to loading old version of model
            print("Autoencoder file does not match current implementation. Checking backwards compatibility...\nWarning: If backwards compatible, autoencoder will train poorly/have undefined behavior (you should probably try retraining from scratch). Inference will be fine, though")
            self.out_layer = nn.Linear(self.hidden_size, self.in_size)
            self.load_state_dict(t.load(path, map_location=device))

if __name__ == "__main__":
    from train_dqn import CheckmateQnet
    from train_autoencoder import gen_all_board_state_tensors
    q = CheckmateQnet(24, 75, 512)
    a = QNetAutoencoder(512, 2024)
    a.prepare_for_resampling()
    rand_in = t.rand(16, 75)
    acts = q.get_activations(rand_in)
    loss, acts = a(acts)
    out = q.activations_to_out(acts)
    all_boards = gen_all_board_state_tensors()
    a.resample(all_boards, q.get_activations, verbose=True)
    print("Done")
