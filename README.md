# :yum: Generative Adversarial Networks (GAN)

This project is an example for GAN models but new models will not be added in the near future as I do not have time for it. It is still maintained and updated so do not hesitate to add new models yourself ! :smile:

**NEW : [CHANGELOG](https://github.com/yui-mhcp/yui-mhcp/blob/main/CHANGELOG.md) file ! Check it to have a global overview of the latest modifications !** :yum:

## Project structure

```bash
├── custom_architectures
│   └── base_dcgan_arch.py  : DCGAN architecture
├── custom_layers
├── custom_train_objects
│   ├── losses
│   │   └── gan_loss.py     : custom loss for GAN models
│   ├── metrics
│   │   └── gan_metric.py   : custom metric for GAN models
├── datasets
├── hparams
├── loggers
├── models
│   ├── generation
│   │   ├── base_gan.py     : abstract GAN class
│   │   └── dcgan.py        : DCGAN implementation based on the tensorflow's tutorial
├── pretrained_models
├── unitest
├── utils
└── example_dcgan.ipynb
```

Check [the main project](https://github.com/yui-mhcp/base_dl_project) for more information about the unextended modules / structure / main classes. 

## Available models

### Model architectures

Available architectures : 
- `Image GAN` :
    - [Deep Convolutional Generative Adversarial Network (DCGAN)](https://arxiv.org/abs/1511.06434v2)

### Model weights

| Classes   | Conditionned  | Dataset   | Architecture  | Trainer   | Weights   |
| :-------: | :-----------: | :-------: | :-----------: | :-------: | :-------: |
| 10        | No            | `MNIST`   | `DCGAN`       | [me](https://github.com/yui-mhcp) | [Google Drive](https://drive.google.com/file/d/1iaj1d8dP8OdYdyNFiSEYEMPCI1Acu04m/view?usp=sharing)  | 
| 10        | Yes           | `MNIST`   | `DCGAN`       | [me](https://github.com/yui-mhcp) | [Google Drive](https://drive.google.com/file/d/1Do8pPhAIMpxty4-stAcfsrrJgMZLVR7K/view?usp=sharing)  | 

Models must be unzipped in the `pretrained_models/` directory !

## Installation and usage

1. Clone this repository : `git clone https://github.com/yui-mhcp/generation.git`
2. Go to the root of this repository : `cd generation`
3. Install requirements : `pip install -r requirements.txt`
4. Open an example notebook and follow the instructions !


## TO-DO list

- [x] Make the TO-DO list.
- [x] Implement the `GANLoss` class.
- [x] Implement a `GANmetric`.
- [x] Implement the training loop procedure
- [x] Reproduce [tensorflow's DCGAN tutorial](https://www.tensorflow.org/tutorials/generative/dcgan) results
- [x] Implement `label-conditionned` encoder
- [ ] Implement `StyleGAN2` architecture
- [ ] Implement a `discriminator` wrapper to allow every generative models to have a discriminator (such as TTS, Q&A, ...). 
- [ ] Implement `DALL-E` -like models
- [ ] Implement `DeepFakes`-like models (auto-encoders)
- [ ] Implement `CycleGAN` architecture

## What is Generative Adversarial Network

**Generative Adversarial Networks (GANs)** are a special architecture composed of 2 networks : 
- The **generator** which tries to generate realistic outputs
- The **discriminator** which tries to distinguish *real* data from *fake* (generated by the generator) data

Note that I say *output* and not *image* as GAN is a general type of architecture and can be applied to whathever you want and not only to generate images (even if it is the most popular usage of GANs). 

### GAN training procedure

The idea is to train both models simultaneously such that the discriminator will learn *what is a real data* and the generator will try to *flood the discriminator* by producing data it thinks to bereal.

The idea is that the discriminator will lean caracteristics of real data and the generator will lean how to reproduce these caracteristics in order to flood the discriminator.


- Generator : the generator's objective is that the discriminator classifies its data as `true` data. The `target` is therefore that the discriminator predicts `1` and the loss will be a `binary_crossentropy` with `1-values` as target and `discriminator output` as prediction
```python
def generator_loss(fake_scores):
    return tf.reduce_mean(binary_crossentropy(tf.ones_like(fake_scores), fake_scores))
```

- Discriminator : the discriminator's objective has 2 parts : 
    - Classify real data as real (score of 1)
    - Classify fake data as fake (score of 0)
The loss will be the sum of both losses (fake_loss and real_loss) where both are `binary_crossentropy`
```python
def discriminator_loss(true_scores, fake_scores):
    true_loss tf.reduce_mean(binary_crossentropy(tf.ones_like(true_scores), true_scores))
    fake_loss tf.reduce_mean(binary_crossentropy(tf.zeros_like(fake_scores), fake_scores))
    return true_loss + fake_loss
```


### GAN challenges

#### Diversification

This problem appears when the `generator` tends to generate always the same kind of image. Once itfinds how to flood the discriminator, it can tend to reproduce this pattern as much as possible as it has seen that it works. 

This issue mostly appears when the dataset is relatively small which can limit the discriminator learning and make it focus on some caracteristics the generator will massively reproduce. 

#### Stability / cooperation

The main objective of this approach is that the generator will learn from the discriminator : they should collaborate to improve each other progressively. 

However, it is possible that the discriminator outperforms the generator and will never be confused which will make the generator unable to learn. 

Another problem is that after convergence of the generator, the discriminator loss will increase which will make the discriminator unstable. This can lead to a loss of quality in the discriminator decision and make the generator decrease the quality of its generation. 

#### Quality / resolution

Another issue is the  image's size / resolution which cannot be too large because it will be too difficult for the generator to flood the discriminator at the beginning and prevent any learning. 



## Contacts and licence

You can contact [me](https://github.com/yui-mhcp) at yui-mhcp@tutanota.com or on [discord](https://discord.com) at `yui#0732`

The objective of these projects is to facilitate the development and deployment of useful application using Deep Learning for solving real-world problems and helping people. 
For this purpose, all the code is under the [Affero GPL (AGPL) v3 licence](LICENCE)

All my projects are "free software", meaning that you can use, modify, deploy and distribute them on a free basis, in compliance with the Licence. They are not in the public domain and are copyrighted, there exist some conditions on the distribution but their objective is to make sure that everyone is able to use and share any modified version of these projects. 

Furthermore, if you want to use any project in a closed-source project, or in a commercial project, you will need to obtain another Licence. Please contact me for more information. 

For my protection, it is important to note that all projects are available on an "As Is" basis, without any warranties or conditions of any kind, either explicit or implied. However, do not hesitate to report issues on the repository's project or make a Pull Request to solve it :smile: 

If you use this project in your work, please add this citation to give it more visibility ! :yum:

```
@misc{yui-mhcp
    author  = {yui},
    title   = {A Deep Learning projects centralization},
    year    = {2021},
    publisher   = {GitHub},
    howpublished    = {\url{https://github.com/yui-mhcp}}
}
```

## Notes and references

Tutorials and projects : 
- [tensorflow tutorials](https://www.tensorflow.org/tutorials/generative/dcgan) : contains multiple well-explained tutorials on multiple architectures (`DCGAN`, `Pix2Pix`, ...). 
- [NVIDIA's repository](https://github.com/NVlabs/stylegan2) : official repository for the `StyleGAN2` architecture

Papers : 
- [1] [Generative Adversarial Nets](https://papers.nips.cc/paper/2014/file/5ca3e9b122f61f8f06494c97b1afccf3-Paper.pdf) : introduction paper to GANs
- [2] [Unsupervised Representation Learning with Deep Convolutional Generative Adversarial Networks](https://arxiv.org/abs/1511.06434v2) : the original `DCGAN` paper
- [3] [Analyzing and Improving the Image Quality of StyleGAN](https://arxiv.org/abs/1912.04958) : the official `StyleGAN2` architecture
- [4] [Progressive Growing of GANs for Improved Quality, Stability, and Variation](https://arxiv.org/abs/1710.10196) : paper for PGGAN which proposes a new approach to solve major GAN challenges
- [5] [MelGAN: Generative Adversarial Networks for Conditional Waveform Synthesis](https://arxiv.org/abs/1910.06711v1) : the original MelGAN paper (to show another example of usage for GAN)
