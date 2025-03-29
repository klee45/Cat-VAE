
#%matplotlib inline
import argparse
from msilib.schema import DuplicateFile
import os
import random
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.optim as optim
import torch.utils.data
import torchvision.datasets as datasets
import torchvision.transforms as tf
import torchvision.utils as torchutils
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

class GAN:

    def __init__(self):
        # Set random seed for reproducibility
        manualSeed = 1
        random.seed(manualSeed)
        torch.manual_seed(manualSeed)
        torch.use_deterministic_algorithms(True) # Needed for reproducible results
        
        # Number of workers for dataloader
        self.workers = 2

        # Batch size during training
        self.batch_size = 128

        # Spatial size of training images. All images will be resized to this
        #   size using a transformer.
        self.image_size = 128

        # Number of channels in the training images. For color images this is 3
        self.num_color_channels = 3

        # Size of z latent vector (i.e. size of generator input)
        self.num_latent_vector = 100

        # Size of feature maps in generator
        self.num_feature_maps_generator = 64

        # Size of feature maps in discriminator
        self.num_feature_maps_discriminator = 64

        # Number of training epochs
        self.num_epochs = 5

        # Learning rate for optimizers
        self.lr = 0.0002

        # Beta1 hyperparameter for Adam optimizers
        self.beta1 = 0.5

        # Number of GPUs available. Use 0 for CPU mode.
        self.num_gpus = 1
        
        # Random Perspective parameters
        self.persp = (0.2, 1.0)

    def get_dataset(self, url):
        dataset = datasets.ImageFolder(root=url,
                                       transform=tf.Compose([
                                           tf.RandomPerspective(distortion_scale=self.persp[0], p=self.persp[1]),
                                           tf.Resize(self.image_size),
                                           tf.CenterCrop(self.image_size),
                                           tf.ToTensor(),
                                           tf.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                                       ]))
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=self.batch_size, shuffle=True, num_workers=self.workers)
        device = torch.device("cuda:0" if (torch.cuda.is_available() and self.num_gpus > 0) else "cpu")
        
        real_batch = next(iter(dataloader))
        plt.figure(figsize=(8,8))
        plt.axis("off")
        plt.imshow(np.transpose(torchutils.make_grid(real_batch[0].to(device)[:64], padding=2, normalize=True).cpu(),(1,2,0)))
        plt.show()
        
    def weights_init(self, m):
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)
        
class Generator(nn.Module):
    pass

class Discriminator(nn.Module):
    pass

def main():
    gan = GAN()
    gan.get_dataset("Data")
    gan.weights_init()
    



if __name__ == '__main__':
    main()