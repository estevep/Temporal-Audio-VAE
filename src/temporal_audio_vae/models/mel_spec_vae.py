import torch
from torch import nn
import torch.distributions as distrib


class AE(nn.Module):
    def __init__(self, encoder, decoder, encoding_dim):
        super(AE, self).__init__()
        self.encoding_dims = encoding_dim
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded


class MelSpecVAE(AE):
    def __init__(self, encoder, decoder, encoding_dims, latent_dims):
        super(MelSpecVAE, self).__init__(encoder, decoder, encoding_dims)
        self.latent_dims = latent_dims
        self.mu = nn.Sequential(nn.Linear(self.encoding_dims, self.latent_dims))
        self.sigma = nn.Sequential(
            nn.Linear(self.encoding_dims, self.latent_dims), nn.Softplus()
        )

    def encode(self, x):
        x_encoded = self.encoder(x)

        mu = self.mu(x_encoded)
        sigma = self.sigma(x_encoded)

        return (mu, sigma)

    def decode(self, z):
        return self.decoder(z)

    def forward(self, x):
        # Encode the inputs
        z_params = self.encode(x)
        # Obtain latent samples and latent loss
        z_tilde, kl_div = self.latent(z_params)
        # Decode the samples
        x_tilde = self.decode(z_tilde)
        return x_tilde, kl_div

    def latent(self, z_params):
        normal = distrib.Normal(loc=0.0, scale=1.0)
        mu, sigma = z_params
        device = mu.device
        kl_div = torch.sum(1 + torch.log(sigma**2) - mu**2 - 2 * (sigma**2)) / 2
        z = mu + sigma * normal.sample(sigma.shape).to(device)

        return z, kl_div
    
def construct_encoder_decoder(n_latent, n_hidden):
    # Encoder network
    encoder = nn.Sequential(
        nn.Conv1d(512, 256, 8, 2, 3),
        nn.LeakyReLU(),
        nn.BatchNorm1d(256),
        nn.Conv1d(256, 128, 8, 2, 3),
        nn.LeakyReLU(),
        nn.BatchNorm1d(128),
        nn.Conv1d(128, 64, 8, 2, 3),
        nn.LeakyReLU(),
        nn.BatchNorm1d(64),
        nn.Conv1d(64, 32, 8, 2, 3),
        nn.LeakyReLU(),
        nn.BatchNorm1d(32),
        nn.Conv1d(32, 16, 8, 2, 3),
        nn.LeakyReLU(),
        nn.BatchNorm1d(16),
        nn.Flatten(),
    )

    # Decoder network
    decoder = nn.Sequential(
        nn.Linear(n_latent, n_hidden),
        nn.LeakyReLU(),
        nn.Unflatten(1, torch.Size([n_latent, int(n_hidden / n_latent)])),
        nn.ConvTranspose1d(16, 32, 8, 2, 3),
        nn.LeakyReLU(),
        nn.BatchNorm1d(32),
        nn.ConvTranspose1d(32, 64, 8, 2, 3),
        nn.LeakyReLU(),
        nn.BatchNorm1d(64),
        nn.ConvTranspose1d(64, 128, 8, 2, 3),
        nn.LeakyReLU(),
        nn.BatchNorm1d(128),
        nn.ConvTranspose1d(128, 256, 8, 2, 3),
        nn.LeakyReLU(),
        nn.BatchNorm1d(256),
        nn.ConvTranspose1d(256, 512, 8, 2, 3),
        nn.LeakyReLU(),
        nn.BatchNorm1d(512),
    )
    return encoder, decoder
