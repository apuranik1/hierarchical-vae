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
    generative = vae.GenerativeModel(state_sizes, [vae.build_mlp(200, [200, 200, 784])])
    recog = vae.ApproxPosterior(load_mnist.pixels, [200], state_sizes, 5)
    model = vae.AutoEncoder(generative, recog)
    print('Model structure:')
    print(list(model.children()))
    return model


def dataprep(datafile):
    # data, _ = load_mnist.load_mnist(1024, list(range(10)))  # throw out labels
    data = load_mnist.load_bin_mnist()
    torch.save(data, datafile)


def train(modelfile, datafile, epochs=1, batch_size=512, lr=0.002, momentum=0.9, decay=0.0001):
    model = build_vae()
    dataset = torch.load(datafile)
    train_vae.train(model, dataset, epochs, batch_size, lr, momentum, decay)
    torch.save(model, modelfile)


def sample(modelfile, count=5, name='samples/sample'):
    model = torch.load(modelfile)
    print(model.decoder.__dict__)
    imgs = model.decoder.sample(count).detach()
    print(imgs)
    for idx in range(count):
        img = imgs[idx, :].numpy().reshape(28, 28) * 255
        img = np.rint(np.clip(img, 0, 255)).astype('uint8')
        fname = name + '-{}.png'.format(idx)
        image = Image.fromarray(img, mode='L')
        image.save(fname)


if __name__ == '__main__':
    argh.dispatch_commands([dataprep, train, sample])
