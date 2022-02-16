import torch
import unittest
from cockatoo.models.vae import MultVAE
from cockatoo.sim import multinomial_bioms


class TestVAE(unittest.TestCase):

    def setUp(self):
        input_dim = 100
        latent_dim = 5
        num_samples = 100
        seq_depth = 1000
        self.all_params = multinomial_bioms(latent_dim, input_dim,
                                            num_samples, seq_depth)
        self.counts = torch.Tensor(self.all_params['Y'])
        self.input_dim = input_dim
        self.latent_dim = latent_dim

    def test_forward(self):
        self.model = MultVAE(self.input_dim, self.latent_dim)
        # TODO : make this run
        neg_elbo = self.model(self.counts)
        ne = neg_elbo.detach().cpu().numpy()
        self.assertGreater(ne, 0)

    def test_optimize(self):
        # TODO : fill this out
        # see pytorch-lightning documentation
        # https://github.com/PyTorchLightning/pytorch-lightning
        pass


if __name__ == '__main__':
    unittest.main()
