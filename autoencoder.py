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
        self.dead_neurons = t.ones(hidden_size, device=device, dtype=t.bool)

        if pretrained_load_path is not None:
            self.load_pretrained(pretrained_load_path)
    
    def prepare_for_resampling(self):
        self.track_dead_neurons = True
        self.dead_neurons = t.ones(self.hidden_size, device=device, dtype=t.bool)

    def forward(self, x) -> Tuple[t.Tensor, t.Tensor, t.Tensor]:
        in_data = x.reshape(-1, x.shape[-1])
        with_bias = in_data + self.out_layer.bias
        acts = self.relu(self.in_layer(with_bias))
        if self.track_dead_neurons:
            acted = t.gt(acts, t.zeros(acts.shape, device=device))
            self.dead_neurons = t.logical_and(self.dead_neurons, acted)
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
        num_dead_neurons = t.sum(self.dead_neurons)
        self.track_dead_neurons = False
        if num_dead_neurons == 0:
            print("Would have resampled some neurons, but none of them needed it.")
            return
        losses = []
        for sample in full_dataset_in: # TODO I think this is required because we need the element-wise loss, but there's a decent chance I just forgot about an alternative, which would probably be much faster if it exists
            preprocessed = preprocessing(sample)
            with_bias = preprocessed + self.out_layer.bias
            out = self.out_layer(self.relu(self.in_layer(with_bias)))
            loss = F.mse_loss(out, preprocessed)
            losses.append(loss)
        losses = t.stack(losses)
        losses_squared = t.square(losses)
        print(losses_squared.shape)
        inputs_to_resample = t.multinomial(losses_squared, num_dead_neurons)
        
        inputs_to_resample = [preprocessing(full_dataset_in[idx.item()]) for idx in inputs_to_resample]
        neurons_to_resample = t.nonzero(self.dead_neurons)
        with t.no_grad():
            for i in range(len(inputs_to_resample)):
                self.out_layer.parametrizations.weight.original1[:, neurons_to_resample[i]] = t.reshape(inputs_to_resample[i], (-1,))
                self.in_layer.weight[neurons_to_resample[i], :] = t.reshape(inputs_to_resample[i], (-1,))*resample_strength_weighting
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
    test_model = QNetAutoencoder(10, 20)
    input_ids = t.rand(8, 1, 16, 10)
    loss, out = test_model(input_ids)
    print("Done")
