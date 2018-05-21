# hierarchical-vae
Implementation of a hierarchical variational autoencoder

The implementation is based on the hierarchical VAE described in the paper [1401.4082](https://arxiv.org/pdf/1401.4082.pdf).

Dependencies:
 - argh
 - torch
 - numpy

Example execution:
```
python run_vae.py dataprep mnist.pt
python run_vae.py train minivae.pt mnist.pt
```
