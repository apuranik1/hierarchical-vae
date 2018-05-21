import numpy as np
import torch


pixels = 784


def sync_shuffle(data1, data2):
    n = data1.shape[0]
    perm = torch.randperm(n)
    return (data1[perm], data2[perm])


def load_mnist(num_points, digits, offset=0, normalize=True):
    X = []
    Y = []
    # X = np.zeros((num_points * totalcount, pixels))
    # Y = np.zeros((num_points * totalcount, 1), dtype=np.int)
    for i, num in enumerate(digits):
        filename = 'data/mnist_digit_{}.csv'.format(num)
        fullset = np.loadtxt(filename, dtype=np.float32)
        # normalize
        if normalize:
            fullset *= 2 / 255
            fullset -= 1
        X.append(torch.from_numpy(fullset[offset:offset+num_points, :]))
        Y.append(num + torch.zeros((num_points, 1), dtype=torch.int32))
        # X[num_points * i:num_points * (i+1), :] = fullset[offset:offset+num_points, :]
        # Y[num_points * i:num_points * (i+1), :] = num
    X = torch.cat(X, dim=0)
    Y = torch.cat(Y, dim=0)
    return sync_shuffle(X, Y)
