import torch
import torch.nn as nn
from torch.distributions import Multinomial, Normal
from torch.distributions.kl import kl_divergence


class LinearVAE(nn.Module):
    def __init__(self, input_dim, latent_dim=None):
        """ Initializes the model

        Parameters
        ----------
        input_dim : int
            Number of input dimensions (aka number of microbes)
        latent_dim : int
            Number of latent dimensions (aka number of principal components)
        """
        super(LinearVAE, self).__init__()
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.encoder = nn.Linear(input_dim, latent_dim)
        self.decoder = nn.Linear(latent_dim, input_dim)
        self.variational_logvars = nn.Parameter(torch.zeros(self.latent_dim))
        self.log_sigma_sq = nn.Parameter(torch.tensor(0.0))

    def forward(self, x):
        """ Computes forward pass and evaluates the ELBO

        Parameters
        ----------
        x : torch.Tensor
            Input microbe counts. This is of dimension B x D
            where B = batch size (aka number of samples in the batch).
            and D = input dimensions (aka number of microbial dimensions)


        Return
        ------
        loglike : torch.Tensor
            Log likelihood

        Notes
        -----
        Look into named tensors
        https://pytorch.org/docs/stable/named_tensor.html

        ELBO : expected log-likelihood - KL(posterior || prior)

        See : https://arxiv.org/abs/1312.6114
        and https://en.wikipedia.org/wiki/Evidence_lower_bound

        TODO:
        1. Need to convert inputs into centered log proportions (CLR)
           ```python
               p = (x + 1) / (x + 1).sum(axis=1)  # axis=1 -> total counts per sample
               cp = torch.log(p) - torch.mean(p, axis=1)
               latent = self.encoder(cp)
           ```
        """
        # Step 1: parameters for the latent space
        z_mean = self.encoder(x)   # TODO: return CLR proportions
        z_std = torch.exp(0.5 * self.variational_logvars)  # standard deviation
        qz = Normal(z_mean, z_std)  # posterior over z
        z_sample = qz.rsample()
        # Step 2: reconstructing the original counts
        decoded = self.decoder(z_sample)
        exp_loglike = Multinomial(
                logits=decoded, validate_args=False  # weird ...
        ).log_prob(x).mean()
        # Step 3: compute KL divergence
        pz = Normal(0, 1)
        kl_div = kl_divergence(qz, pz).mean(0).sum()
        # Step 4: compute the elbo
        elbo = exp_loglike - kl_div
        return - elbo
