import math
import torch


def train(autoencoder, dataset, epochs, batch_size, lr, momentum, decay, cuda):
    n = dataset.size(0)
    device = torch.device('cuda') if cuda else torch.device('cpu')
    # this will probably leave out the last couple training points...
    num_batches = n // batch_size
    autoencoder = autoencoder.to(device=device)
    sgd = torch.optim.SGD(autoencoder.parameters(), lr, momentum=momentum, weight_decay=decay)
    # sgd = torch.optim.Adam(autoencoder.parameters(), lr, weight_decay=decay)
    width = int(math.log10(num_batches) + 1)
    width_format = '{:' + str(width) + '}'
    batch_format = 'Batch ' + width_format + '/' + width_format + '.'
    for epoch in range(epochs):
        loss_sum = 0
        indices = torch.randperm(num_batches)
        for i in range(num_batches):
            idx = indices[i].item()
            start = idx * batch_size
            data = dataset[start:start+batch_size].to(device=device)
            loss = train_batch(autoencoder, data, sgd)
            loss_sum += loss
            print('\r' + ' ' * 40, end='')
            print('\r' + batch_format.format(i, num_batches), end='')
            print(' Average loss = {:2.5}'.format(loss_sum / (i + 1)),
                  end='', flush=True)
        print()
        print('Epoch {} complete. Average loss = {:2.3}'
              .format(epoch, loss_sum / num_batches))
    return autoencoder


def train_batch(autoencoder, data, optimizer):
    optimizer.zero_grad()
    dist, output = autoencoder(data)
    loss = autoencoder.loss(data, dist, output)
    loss.backward()
    optimizer.step()
    return loss.item()
