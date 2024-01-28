import torch
import torch.nn as nn
import numpy as np
from torch.distributions.multivariate_normal import MultivariateNormal

class CouplingLayer(nn.Module):
    def __init__(self, input_dim, hidden_dim) -> None:
        super().__init__()
        self.s_layer = nn.Sequential(
            nn.Linear(input_dim, hidden_dim), # s_layer_1
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), # s_layer_2
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), # s_layer_3
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), # s_layer_4
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim), # s_layer_5
            nn.Tanh()
        )

        self.t_layer = nn.Sequential(
            nn.Linear(input_dim, hidden_dim), # t_layer_1
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), # t_layer_2
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), # t_layer_3
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), # t_layer_4
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim), # t_layer_5
        )

    def forward(self, x):
        s = self.s_layer(x)
        t = self.t_layer(x)
        return s, t

class RealNVP(nn.Module):
    def __init__(self, input_dim, coupling_layers, hidden_dim):
        super(RealNVP, self).__init__()
        # Initialize layers and networks for affine transformations
        self.coupling_layers = coupling_layers
        self.layers = nn.ModuleList([CouplingLayer(input_dim, hidden_dim) for _ in range(coupling_layers)])

        self.distribution = MultivariateNormal(torch.zeros(4), torch.eye(4))
        self.masks = np.array(
            [[0, 0, 1, 1], [1, 1, 0, 0]] * (coupling_layers // 2), dtype="float32"
        )
        self.masks = torch.tensor(self.masks)


    def forward(self, x, training=True):
        # Forward pass for generation
        log_det_inv = 0
        direction = -1 if training else 1
        for i in range(self.coupling_layers)[::direction]:
            x_masked = x * self.masks[i]
            reversed_mask = 1 - self.masks[i]
            s, t = self.layers[i](x_masked)
            s = s * reversed_mask
            t = t * reversed_mask

            gate = (direction - 1) / 2

            x = (
                reversed_mask
                * (x * torch.exp(direction * s) + direction * t * torch.exp(gate * s))
                + x_masked
            )
            log_det_inv += gate * torch.sum(s, axis=1)
        return x, log_det_inv
    
    def log_loss(self, x):
        y, logdet = self(x)
        log_likelihood = self.distribution.log_prob(y) + logdet
        return -torch.mean(log_likelihood)
    
    @torch.inference_mode()
    def sample(self, n_samples):
        z = self.distribution.sample((n_samples,))
        return self.forward(z, training=False)[0]