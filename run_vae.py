import argh
import numpy as np
from PIL import Image
import torch

import vae
import load_mnist
import train_vae


# TODO: add t-convs to generative model
def build_vae():
    state_sizes = [200]
    generative = vae.GenerativeModel(state_sizes,
                                     [vae.build_mlp(200, [200, 200, 784])])
    recog = vae.ApproxPosterior(load_mnist.pixels, [200], state_sizes, 3)
    model = vae.AutoEncoder(generative, recog)
    print('Model structure:')
    print(list(model.children()))
    return model


def dataprep(datafile, train=False, val=False, test=False):
    # throw out labels
    # data, _ = load_mnist.load_mnist(1024, list(range(10)))
    if train + val + test != 1:
        raise ValueError('Must specify exactly one of --train, --val, or --test')
    fname = (load_mnist.BIN_MNIST_TRAIN if train else
             load_mnist.BIN_MNIST_VAL if val else
             load_mnist.BIN_MNIST_TEST)
    data = load_mnist.load_bin_mnist(fname)
    torch.save(data, datafile)


def train(modelfile, datafile, epochs=1, batch_size=512, lr=0.000002,
          momentum=0.9, decay=0.00002, load=False, cuda=False):
    if load:
        print('Loading existing model')
        model = torch.load(modelfile)
    else:
        print('Training new model')
        model = build_vae()
    model.train()
    dataset = torch.load(datafile)
    train_vae.train(model, dataset, epochs, batch_size, lr, momentum, decay, cuda)
    torch.save(model, modelfile)


def sample(modelfile, count=5, name='samples/sample'):
    model = torch.load(modelfile).cpu()
    model.eval()
    print(model.decoder.layers)
    imgs = model.decoder.sample(count).detach()
    print(imgs)
    for idx in range(count):
        probs = imgs[idx, :].numpy().reshape(28, 28)
        sample = np.random.uniform(size=probs.shape)
        img = np.where(sample < probs, 255, 0).astype('uint8')
        # img = imgs[idx, :].numpy().reshape(28, 28) * 255
        # img = np.rint(np.clip(img, 0, 255)).astype('uint8')
        fname = name + '-{}.png'.format(idx)
        image = Image.fromarray(img, mode='L')
        image.save(fname)


def elbo(modelfile, datafile, batch_size=512, cuda=False):
    model = torch.load(modelfile)
    data = torch.load(datafile)
    print(train_vae.evaluate_elbo(model, data, batch_size, cuda))


def evaluate(modelfile, datafile, batch_size=512, samples=10, cuda=False):
    model = torch.load(modelfile)
    data = torch.load(datafile)
    model.encoder.sample_size = samples
    print(train_vae.evaluate_ll(model, data, batch_size, cuda))


if __name__ == '__main__':
    argh.dispatch_commands([dataprep, train, sample, elbo, evaluate])
