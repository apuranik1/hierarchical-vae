import math
import torch


def train(autoencoder, dataset, epochs, batch_size=64, lr=0.001, momentum=0.9,
          decay=0.1):
    n = dataset.size(0)
    # this will probably leave out the last couple training points...
    num_batches = n // batch_size
    # sgd = torch.optim.SGD(autoencoder.parameters(), lr, momentum=momentum, weight_decay=decay)
    width = int(math.log10(num_batches) + 1)
    width_format = '{:' + str(width) + '}'
    batch_format = 'Batch ' + width_format + '/' + width_format + '.'
    sgd = torch.optim.Adam(autoencoder.parameters(), 0.01, weight_decay=decay)
    for epoch in range(epochs):
        loss_sum = 0
        for i in range(num_batches):
            start = i * batch_size
            loss = train_batch(autoencoder, dataset[start:start+batch_size],
                               sgd)
            loss_sum += loss
            print('\r' + ' ' * 40, end='')
            print('\r' + batch_format.format(i, num_batches), end='')
            print(' Average loss = {:2.5}'.format(loss_sum / (i + 1)),
                  end='', flush=True)
        print()
        print('Epoch {} complete. Average loss = {:2.3}'
              .format(epoch, loss_sum / num_batches))
    return autoencoder


def train_batch(autoencoder, datapoints, optimizer):
    optimizer.zero_grad()
    dist, output = autoencoder(datapoints)
    loss = autoencoder.loss(datapoints, dist, output)
    loss.backward()
    optimizer.step()
    return loss
