import argh
import torch

import vae
import load_mnist
import train_vae


def build_vae():
    generative = vae.GenerativeModel([100], [vae.build_mlp(100, [784])])
    recog = vae.ApproxPosterior(load_mnist.pixels, [100], [100], 3)
    model = vae.AutoEncoder(generative, recog)
    print('Model structure:')
    print(list(model.children()))
    return model


def dataprep(datafile):
    data = load_mnist.load_mnist(1024, list(range(10)))
    torch.save(data, datafile)


def train(modelfile, datafile, epochs=50, batch_size=64, lr=0.002, momentum=0.9, decay=0.1):
    model = build_vae()
    dataset, _ = torch.load(datafile) # throw out labels
    train_vae.train(model, dataset, epochs, batch_size, lr, momentum, decay)
    torch.save(model, modelfile)


if __name__ == '__main__':
    argh.dispatch_commands([dataprep, train])
