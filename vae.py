import torch
import torch.nn as nn
import torch.nn.functional as F


class AutoEncoder(nn.Module):

    def __init__(self, generative, approx_post):
        super().__init__()
        self.decoder = generative
        self.encoder = approx_post
        self.loss_func = nn.BCEWithLogitsLoss(reduction='sum')

    def forward(self, x):
        batch = x.size(0)
        dist, enc_samples = self.encoder(x)
        sample_size = enc_samples.size(1)
        # merge samples into batch for the decoder network
        reshaped_samples = enc_samples.view(batch * sample_size, *enc_samples.size()[2:])
        output = self.decoder(reshaped_samples)
        # reconstitute samples within a batch
        output = output.view(batch, sample_size, *output.size()[1:])
        return dist, output

    def loss(self, data, dist, output):
        batch_size = data.size(0)
        kl_loss = torch.sum(self.encoder.kl_loss(dist))
        # print()
        # print('KL-loss:', kl_loss)
        # data has dimension batch x pixels
        # output has dimension batch x samples x pixels
        # diff = (data.unsqueeze(1) - output).view(batch_size, -1)
        # reconst_loss = torch.mean((diff * diff).sum(dim=1))  # -LL of stdnorm
        # reconst_loss = torch.mean(diff * diff)
        sample_size = output.size(1)
        reconst_loss = self.loss_func(output, data.unsqueeze(1).expand(-1, sample_size, -1))
        # print('Reconstruction loss:', reconst_loss)
        # print()
        return (kl_loss + reconst_loss) / batch_size


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
        indices = [0]
        for dim in state_dims:
            indices.append(indices[-1] + dim)
        self.indices = indices

    def forward(self, x):
        """Apply the decoder to encoding x.
        Input should be a list, where entry i has shape [batch x state_dim[i]]
        """
        dev = x.device
        result = torch.zeros(self.state_dims[0], device=dev)  # additive identity for ease of coding
        for index, layer in enumerate(self.layers):
            inp = x[:, self.indices[index]:self.indices[index+1]]
            result = layer(result + inp)
        return result

    def sample(self, batch_size):
        """Generate a sample from this distribution's prior."""
        dim = self.indices[-1]
        enc = torch.randn(batch_size, dim)
        params = self.forward(enc)
        return torch.sigmoid(params)


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
        # TODO: allow length 0 mlp_sizes in a reasonable way
        self.representation = build_mlp(input_dim, mlp_sizes)
        rep_dim = mlp_sizes[-1] if len(mlp_sizes) > 0 else input_dim
        self.state_size = sum(state_dims)
        self.output_mu = nn.Linear(rep_dim, self.state_size)
        self.output_cov = nn.Linear(rep_dim, self.state_size)
        self.sample_size = sample_size

    def encode_distribution(self, x):
        rep = F.relu(self.representation(x))
        mu = self.output_mu(rep)
        cov = F.softplus(self.output_cov(rep))
        return (mu, cov)

    def forward(self, x):
        """Sample from the approximate posterior.
        Output is a list of tensors, where the tensor at index i has shape:
        [batch x samples x state_dims[i]
        """
        # Use the fact that N(m, C) = sqrt(C) * N(0, 1) + m to allow autograd
        # to backprop through the Gaussian parameters properly (I think)
        batch = x.size(0)
        dev = x.device
        mu, cov = self.encode_distribution(x)
        sample = torch.randn(batch, self.sample_size, self.state_size,
                             device=dev)
        sample = sample * torch.sqrt(cov.unsqueeze(1)) + mu.unsqueeze(1)
        return (mu, cov), sample

    def kl_loss(self, dist):
        mu, cov = dist
        return kl_stdnorm_diag(mu, cov)
