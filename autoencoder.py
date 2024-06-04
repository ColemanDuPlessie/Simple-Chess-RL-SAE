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
        loss_sparsity_term = 0.01
    ):
        super().__init__()

        self.in_layer = nn.Linear(in_size, hidden_size)
        self.relu = nn.ReLU()
        self.out_layer = nn.utils.parametrizations.weight_norm(nn.Linear(hidden_size, in_size), name='weight')
        self.out_layer.parametrizations.weight.original0 = nn.Parameter(t.ones(1, in_size), requires_grad = False)
        
        self.in_size = in_size
        self.hidden_size = hidden_size
        
        self.loss_sparsity_term = loss_sparsity_term

        if pretrained_load_path is not None:
            self.load_pretrained(pretrained_load_path)

    def forward(self, x) -> Tuple[t.Tensor, t.Tensor, t.Tensor]:
        in_data = x.reshape(-1, x.shape[-1])
        with_bias = in_data + self.out_layer.bias
        acts = self.relu(self.in_layer(with_bias))
        out = self.out_layer(acts)
        out = out.reshape_as(x)
        loss = F.mse_loss(out, x) + self.loss_sparsity_term * t.norm(acts, p=1)
        return loss, out
    
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
