import torch
import torch.nn as nn
import torch.nn.functional as F


class AutoEncoder(nn.Module):

    def __init__(self, generative, approx_post):
        super().__init__()
        self.decoder = generative
        self.encoder = approx_post

    def forward(self, x):
        batch = x.size(0)
        dist, enc_samples = self.encoder(x)
        sample_size = enc_samples[0].size(1)
        # merge samples into batch for the decoder network
        reshaped_samples = [enc.view(batch * sample_size, *enc.size()[2:])
                            for enc in enc_samples]
        output = self.decoder(reshaped_samples)
        # reconstitute samples within a batch
        output = output.view(batch, sample_size, *output.size()[1:])
        return dist, output

    def loss(self, data, dist, output):
        batch_size = data.size(0)
        kl_loss = torch.mean(self.encoder.kl_loss(dist))
        # print('KL-loss:', kl_loss)
        diff = (data.unsqueeze(1) - output).view(batch_size, -1)
        # reconst_loss = torch.mean((diff * diff).sum(dim=1))  # -LL of stdnorm
        reconst_loss = torch.mean(diff * diff)
        # print('Reconstruction loss:', reconst_loss)
        return kl_loss + reconst_loss


class GenerativeModel(nn.Module):
    """The generative model (decoder) for the VAE.

    The decoder takes as input multiple layers of variables with Gaussian
    prior. It outputs some encoded state vector.

    The layer transform[i] must take a tensor of size [batch x state_dim[i]] to
    one of size [batch x state_dim[i+1]], or [batch_size x (output_shape)] for
    the last transform.
    """
    def __init__(self, state_dims, transforms):
        super().__init__()
        if len(state_dims) == 0:
            raise ValueError('Must have a nonzero number of generative parameters')
        if len(state_dims) != len(transforms):
            raise ValueError('Cannot have different numbers of generative parameters and transforms')
        for idx, t in enumerate(transforms):
            self.add_module('layer_{}'.format(idx), t)
        self.layers = transforms
        self.state_dims = state_dims

    def forward(self, x):
        """Apply the decoder to encoding x.
        Input should be a list, where entry i has shape [batch x state_dim[i]]
        """
        result = torch.zeros_like(x[0])  # additive identity for ease of coding
        for layer, inp in zip(self.layers, x):
            result = layer(result + inp)
        return result

    def sample(self, batch_size):
        """Generate a sample from this distribution's prior."""
        enc = [torch.randn(batch_size, dim) for dim in self.state_dims]
        return self.forward(enc)


def build_mlp(input_dim, mlp_sizes):
    layers = []
    current_dim = input_dim
    for size in mlp_sizes[:-1]:
        layers.append(nn.Linear(current_dim, size))
        layers.append(nn.ReLU())
        current_dim = size
    layers.append(nn.Linear(current_dim, mlp_sizes[-1]))
    return nn.Sequential(*layers)


def mlp_generative_model(state_dims, mlp_hidden_sizes, output_shape):
    """Factory function for GenerativeModels with MLP transformations"""
    transforms = []
    for i in range(len(state_dims) - 1):
        dims = mlp_hidden_sizes[i] + [state_dims[i+1]]
        transforms.append(build_mlp(state_dims[i], dims))
    dims = mlp_hidden_sizes[-1] + [output_shape]
    transforms.append(build_mlp(state_dims[-1], dims))
    return GenerativeModel(state_dims, transforms)


def kl_stdnorm_diag(mu, c_diag):
    """Compute the KL-divergence from the standard normal to a normal with
    mean mu and covariance c_diag.
    mu - tensor of size [batch x dim]
    c_diag - tensor of size [batch x dim]
    """
    return 0.5 * (-c_diag.log().sum(dim=1) - mu.size(1) + c_diag.sum(dim=1)
                  + (mu * mu).sum(dim=1))


class ApproxPosterior(nn.Module):
    """Approximate posterior distribution (encoder) for the VAE.

    Learns the parameters of one Gaussian per layer of the generative model.
    Assumes that the distribution factors across layers.

    The network is structured as an MLP, where the last layer has a number of
    different outputs corresponding to the parameters of the decoder.
    """

    def __init__(self, input_dim, mlp_sizes, state_dims, sample_size):
        super().__init__()
        self.representation = build_mlp(input_dim, mlp_sizes)
        rep_dim = mlp_sizes[-1] if len(mlp_sizes) > 0 else input_dim
        output_layers = []
        for idx, dim in enumerate(state_dims):
            mu_layer = nn.Linear(rep_dim, dim)
            cov_layer = nn.Linear(rep_dim, dim)
            self.add_module('layer_{}_mu'.format(idx), mu_layer)
            self.add_module('layer_{}_cov'.format(idx), cov_layer)
            output_layers.append((mu_layer, cov_layer))
        self.output_layers = output_layers

        self.state_dims = state_dims
        self.sample_size = sample_size

    def encode_distribution(self, x):
        rep = F.relu(self.representation(x))
        output = []
        for mu_layer, cov_layer in self.output_layers:
            # TODO: figure out if 0 variance blows up
            output.append((F.relu(mu_layer(rep)), F.softplus(cov_layer(rep))))
        return output

    def forward(self, x):
        """Sample from the approximate posterior.
        Output is a list of tensors, where the tensor at index i has shape:
        [batch x samples x state_dims[i]
        """
        # Use the fact that N(m, C) = sqrt(C) * N(0, 1) + m to allow autograd
        # to backprop through the Gaussian parameters properly (I think)
        batch = x.size(0)
        params = self.encode_distribution(x)
        samples = []
        for (mu, cov), dim in zip(params, self.state_dims):
            # draw multivariate Gaussian samples:
            # [batch x sample_size x state_dims[i]]
            # scale by cov, add params
            sample = torch.randn(batch, self.sample_size, dim)
            samples.append(sample * torch.sqrt(cov.unsqueeze(1))
                           + mu.unsqueeze(1))
        return params, samples

    def kl_loss(self, dist):
        return sum(kl_stdnorm_diag(mu, c_diag) for mu, c_diag in dist)
