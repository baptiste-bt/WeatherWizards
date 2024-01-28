import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import torch.nn.functional as F

class VAE(nn.Module):
    def __init__(self, input_dim, hidden_dim, latent_dim):
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
    
    @torch.inference_mode()
    def sample(self, n_samples):
        z = torch.randn(n_samples, self.latent_dim)
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