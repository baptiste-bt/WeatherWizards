import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import torch.nn.functional as F
import numpy as np
import ot

class VAE(nn.Module):
    def __init__(self, input_dim, hidden_dim, latent_dim, device='cpu'):
        super(VAE, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.latent_dim = latent_dim

        self.mu_feedforward = nn.Linear(hidden_dim, latent_dim)
        self.logvar_feedforward = nn.Linear(hidden_dim, latent_dim)

        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim)
        )
        self.device = device
        self.to(device)

    def forward(self, x):
        x = self.encoder(x)
        mu = self.mu_feedforward(x)
        logvar = self.logvar_feedforward(x)
        z = self.reparameterize(mu, logvar)
        x = self.decoder(z)
        return x, mu, logvar
    
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def fit(self, y, epochs=10, batch_size=32, lr=1e-3, beta=100., y_val=None, verbose=1, wasserstein_coeff=1.):
        # Check y is a tensor
        assert isinstance(y, torch.Tensor), 'y must be a tensor'
        if y_val is not None:
            assert isinstance(y_val, torch.Tensor), 'y_val must be a tensor'

        # Check y is normalized
        if not torch.allclose(y.mean(), torch.zeros(1), atol=1e-2):
            print(f'y must be normalized, mean is {y.mean()}')
        if not torch.allclose(y.std(), torch.ones(1), atol=1e-2):
            print(f'y must be normalized, std is {y.std()}')

        dataloader = DataLoader(TensorDataset(y), batch_size=batch_size, shuffle=True)
        optimizer = optim.AdamW(self.parameters(), lr=lr, weight_decay=0.01)

        losses = []
        was_distances = []
        was_distances_test = [] 

        for epoch in range(epochs):
            for batch_targets, in dataloader:
                optimizer.zero_grad()
                
                batch_targets = batch_targets.to(self.device)
                x, mu, logvar = self(batch_targets)
                total_loss, recon_loss, kl_div = vae_loss(x, batch_targets, mu, logvar, beta=beta)
                if wasserstein_coeff > 0.:
                    total_loss = total_loss + wasserstein_coeff * ot.sliced_wasserstein_distance(x, y, n_projections=100)
                total_loss.backward()
                optimizer.step()
                losses.append([total_loss.detach().item(), recon_loss.detach().item(), kl_div.detach().item()])

            # Compute ot distance
            generated_yield = self.sample(1000)
            was_dist = ot.sliced_wasserstein_distance(generated_yield.cpu(), y, n_projections=1000).item()
            was_dist_test = ot.sliced_wasserstein_distance(generated_yield.cpu(), y_val, n_projections=1000).item() if y_val is not None else 0.
            if verbose:
                print(f'Epoch [{epoch+1}/{epochs}], Total Loss: {total_loss.detach().item():.4f}, Recon Loss: {recon_loss.detach().item():.4f}, KL Div: {kl_div.detach().item():.4f}, WAS dist: {was_dist:.4f}, WAS dist test: {was_dist_test:.4f}')
            was_distances.append(was_dist)
            was_distances_test.append(was_dist_test)

        return {
            'losses': losses,
            'was_distances': was_distances,
            'was_distances_test': was_distances_test
        }

    
    @torch.inference_mode()
    def sample(self, n_samples):
        z = torch.randn(n_samples, self.latent_dim, device=self.device)
        return self.decoder(z)
    
def vae_loss(reconstructed_x, x, mu, logvar, beta=100.):
    """
    Calculate the loss for a VAE.

    Parameters:
    - reconstructed_x: the output of the decoder.
    - x: the original input data.
    - mu: the mean from the latent space distribution.
    - logvar: the log variance from the latent space distribution.
    - beta: the weight for the reconstruction loss term.

    Returns:
    - Total loss as a PyTorch scalar.
    """

    # Reconstruction loss (MSE)
    recon_loss = F.mse_loss(reconstructed_x, x, reduction='mean')

    # KL divergence
    kl_div = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1).mean()

    # Total loss
    total_loss = beta*recon_loss + kl_div

    return total_loss, recon_loss, kl_div