import argh
import numpy as np
from PIL import Image
import torch

import vae
import load_mnist
import train_vae


def build_vae():
    state_sizes = [100, 100]
    generative = vae.GenerativeModel(state_sizes, [vae.build_mlp(100, [100]),
                                                vae.build_mlp(100, [784])])
    recog = vae.ApproxPosterior(load_mnist.pixels, [100, 100], state_sizes, 3)
    model = vae.AutoEncoder(generative, recog)
    print('Model structure:')
    print(list(model.children()))
    return model


def dataprep(datafile):
    data, _ = load_mnist.load_mnist(1024, list(range(10)))
    torch.save(data, datafile)


def train(modelfile, datafile, epochs=50, batch_size=512, lr=0.002, momentum=0.9, decay=0.0001):
    model = build_vae()
    dataset = torch.load(datafile) # throw out labels
    train_vae.train(model, dataset, epochs, batch_size, lr, momentum, decay)
    torch.save(model, modelfile)


def sample(modelfile, count=5, name='sample'):
    model = torch.load(modelfile)
    print(model.decoder.__dict__)
    imgs = model.decoder.sample(count).detach()
    print(imgs)
    for idx in range(count):
        img = (imgs[idx, :].numpy().reshape(28, 28) + 1) * 255 / 2
        img = np.rint(np.clip(img, 0, 255)).astype('uint8')
        print(img)
        fname = name + '-{}.png'.format(idx)
        image = Image.fromarray(img, mode='L')
        image.save(fname)


if __name__ == '__main__':
    argh.dispatch_commands([dataprep, train, sample])
