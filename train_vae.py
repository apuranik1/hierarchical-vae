import math
import torch


def mean_logspace(x, *args, **kwargs):
    if 'dim' in kwargs:
        kwargs['keepdim'] = True
    max_val = x.max(*args, **kwargs)
    if isinstance(max_val, tuple):
        max_val, _ = max_val
    stable_mean = torch.exp(x - max_val).mean(*args, **kwargs)
    return torch.log(stable_mean) + max_val


def train(autoencoder, train_data, val_data, epochs, batch_size, lr, momentum,
          decay, cuda):
    n = train_data.size(0)
    device = torch.device('cuda') if cuda else torch.device('cpu')
    # this will probably leave out the last couple training points...
    num_batches = n // batch_size
    autoencoder = autoencoder.to(device=device)
    # vanilla SGD + momentum takes annoyingly long to converge
    sgd = torch.optim.Adam(autoencoder.parameters(), lr, weight_decay=decay)
    width = int(math.log10(num_batches) + 1)
    width_format = '{:' + str(width) + '}'
    batch_format = 'Batch ' + width_format + '/' + width_format + '.'
    for epoch in range(epochs):
        autoencoder.train()
        loss_sum = 0
        indices = torch.randperm(num_batches)
        for i in range(num_batches):
            idx = indices[i].item()
            start = idx * batch_size
            data = train_data[start:start+batch_size].to(device=device)
            loss = train_batch(autoencoder, data, sgd)
            loss_sum += loss
            print('\r' + ' ' * 40, end='')
            print('\r' + batch_format.format(i, num_batches), end='')
            print(' Average loss = {:2.4}'.format(loss_sum / (i + 1)),
                  end='', flush=True)
        print()
        val_loss = evaluate_ll(autoencoder, val_data, batch_size, cuda)
        print('Epoch {} complete. Training loss = {:2.3}. Validation LL = {:2.3}'
              .format(epoch, loss_sum / num_batches, val_loss))
    return autoencoder


def train_batch(autoencoder, data, optimizer):
    optimizer.zero_grad()
    dist, output = autoencoder(data)
    loss = autoencoder.loss(data, dist, output)
    loss.backward()
    optimizer.step()
    return loss.item()


def evaluate_elbo(autoencoder, dataset, batch_size, cuda):
    autoencoder.eval()
    n = dataset.size(0)
    device = torch.device('cuda') if cuda else torch.device('cpu')
    num_batches = n // batch_size
    autoencoder = autoencoder.to(device=device)
    loss_sum = 0
    for idx in range(num_batches):
        start = idx * batch_size
        data = dataset[start:start+batch_size].to(device=device)
        dist, output = autoencoder(data)
        loss = autoencoder.loss(data, dist, output).item()
        loss_sum += loss
    avg_loss = loss_sum / num_batches
    return avg_loss


def evaluate_ll(autoencoder, dataset, batch_size, cuda):
    """Use importance sampling to estimate the average marginal log-likelihood
    of the dataset."""
    # -LL(x) = E_p[-ln(p(x|z))] = E_q_x[-ln(p(x|z) * p(z) / q_x(z))]
    # sample values z from q_x
    # compute -ll(x|z)
    # weight by prior density / posterior density
    autoencoder.eval()
    n = dataset.size(0)
    device = torch.device('cuda') if cuda else torch.device('cpu')
    num_batches = n // batch_size
    autoencoder = autoencoder.to(device=device)
    nll_cond_func = torch.nn.BCEWithLogitsLoss(reduction='none')
    ll_sum = 0
    for idx in range(num_batches):
        start = idx * batch_size
        data = dataset[start:start+batch_size].to(device=device)
        dist, enc_samples = autoencoder.encoder(data)
        mu, cov = dist
        mu = mu.unsqueeze(1)
        cov = cov.unsqueeze(1)
        sample_size = enc_samples.size(1)
        # compute prior LL of enc_samples
        prior_dim = autoencoder.decoder.indices[-1]
        ll_prior = -0.5 * (enc_samples * enc_samples).sum(dim=2) \
            - prior_dim / 2 * math.log(2 * math.pi)
        ll_posterior = -0.5 * ((enc_samples - mu) ** 2 / cov).sum(dim=2) \
            - prior_dim / 2 * math.log(2 * math.pi) \
            - 0.5 * torch.log(cov).sum(dim=2)

        reshaped_samples = enc_samples.view(batch_size * sample_size,
                                            *enc_samples.size()[2:])
        output = autoencoder.decoder(reshaped_samples)
        output = output.reshape(batch_size, sample_size, *output.size()[1:])
        truth = data.unsqueeze(1).expand(-1, sample_size, -1)

        ll_cond = -nll_cond_func(output, truth)  # shape batch x samples x 784
        ll_cond = ll_cond.sum(dim=2)
        avg_ll = mean_logspace(ll_cond - ll_posterior + ll_prior, dim=1)
        ll_sum += avg_ll.mean().item()
    return ll_sum / num_batches
