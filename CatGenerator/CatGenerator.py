
#%matplotlib inline
import argparse
from ast import Num
from msilib.schema import DuplicateFile
import os
import itertools
import random
from numpy._typing import _256Bit
import torch
import torch.nn as nn
from torch.nn.modules import batchnorm, linear
import torch.nn.parallel
import torch.optim as optim
import torch.utils.data
import torchvision.datasets as datasets
import torchvision.transforms as tf
import torchvision.utils as torchutils
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from torchvision.utils import save_image

parser = argparse.ArgumentParser()
parser.add_argument("--num_epochs", type=int, default=500, help="number of epochs of training")
parser.add_argument("--batch_size", type=int, default=128, help="size of the batches")
parser.add_argument("--image_size", type=int, default=128, help="size of each image dimension")
parser.add_argument("--lr_g", type=float, default=1e-3, help="adam: learning rate for generator")
parser.add_argument("--lr_d", type=float, default=1e-5, help="adam: learning rate for discriminator")
parser.add_argument("--b1", type=float, default=0.5, help="adam: decay of first order momentum of gradient")
parser.add_argument("--b2", type=float, default=0.999, help="adam: decay of first order momentum of gradient")
parser.add_argument("--num_gpus", type=int, default=1, help="number of gpu threads to use")
parser.add_argument("--latent_size", type=int, default=128, help="size of the latent vector i.e. size of generator input")
parser.add_argument("--discriminator_layer_size", type=int, default=64, help="The size of the linear layers in the discriminator")
parser.add_argument("--num_channels", type=int, default=3, help="number of image channels")
parser.add_argument("--sample_interval", type=int, default=100, help="interval between image sampling")
parser.add_argument("--plot_interval", type=int, default=50, help="interval between datapoint plotting")
parser.add_argument("--workers", type=int, default=2, help="number of dataloader workers")
parser.add_argument("--persp1", type=float, default=0.2, help="perspective distortion")
parser.add_argument("--persp2", type=float, default=0.5, help="perspective chance")
opt = parser.parse_args()

img_shape = (opt.num_channels, opt.image_size, opt.image_size)

'''
Encoder
    4x (Conv2d -> BatchNorm2d -> ReLU)
        16 32 64 128 256
    Linear -> BatchNorm1d -> ReLU
        xx latent
    
Decoder
    Linear -> BatchNorm1d -> ReLU
        latent xx
    4x (ConvTranspose2d -> BatchNorm2d -> ReLU)
        256 128 64 32 16
    
Generator
    4x (Linear -> BatchNorm1d -> ReLU)
        128 256 512 256 latent
    
Discriminator
    4x (Conv2d -> BatchNorm2d -> ReLU)
        16 32 64 128 256
    Linear -> BatchNorm1d -> ReLU -> Linear
        xx 512 1

real images -> Encoder -> Decoder -> autoencoded images
noise -> Generator -> Decoder -> generated images

autoencoded images -> Discriminator
generated images -> Discriminator
2x real images -> Discriminator
'''

class encoder_block(nn.Module):
    def __init__(self, channel_in, channel_out, leaky=False):
        self.conv = nn.Sequential(
            nn.Conv2d(channel_in, channel_out, kernel_size=5, stride=2, padding=2, bais=False),
            nn.BatchNorm2d(channel_out, momentum=0.9),
            nn.LeakyReLU(0.2, True) if leaky else nn.ReLU(True) 
        )
    def forward(self, ten):
        return self.conv(ten)

class decoder_block(nn.Module):
    def __init__(self, channel_in, channel_out, leaky=False):
        self.conv = nn.Sequential(
            nn.ConvTranspose2d(channel_in, channel_out, kernel_size=5, stride=2, padding=2, output_padding=1, bais=False),
            nn.BatchNorm2d(channel_out, momentum=0.9),
            nn.LeakyReLU(0.2, True) if leaky else nn.ReLU(True) 
        )
    def forward(self, ten):
        return self.conv(ten)

class fully_connected_block(nn.Module):
    def __init__(self, channel_in, channel_out, leaky=False):
        self.conv = nn.Sequential(
            nn.Linear(channel_in, channel_out, bias=False),
            nn.BatchNorm1d(channel_out, momentum=0.9),
            nn.LeakyReLU(0.2, True) if leaky else nn.ReLU(True) 
    )
    def foward(self, ten):
        return self.conv(ten)

class Decoder(nn.Module):
    def __init__(self, channel_in, abstract_size, scaling):
        super(Decoder, self).__init__()
        
        abstract_features = abstract_size * abstract_size * scaling
        self.fc = nn.Sequential(
            nn.Linear(channel_in, abstract_features, bias=False),
            nn.BatchNorm1d(abstract_features),
            nn.ReLU(True)
        )
        
        layers_list = []
        layers_list.append(decoder_block)

        self.end = nn.Sequential(
            nn.Conv2d()
            nn.Tanh(),
        )
        
    def forward(self, input_data):
        return self.main(input_data)
    
class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()

        self.main = nn.Sequential(
            
            # 256 x 256
            nn.Conv2d(opt.num_channels,
                      32,
                      kernel_size=5,
                      stride=2,
                      padding=2),
            nn.BatchNorm2d(32, momentum=0.5),          
            nn.LeakyReLU(0.2),
            # 128 x 128
            nn.MaxPool2d(kernel_size=2, stride=2),
            # 64 x 64
            nn.Conv2d(32,
                      64,
                      kernel_size=5,
                      stride=2,
                      padding=2),
            nn.BatchNorm2d(64, momentum=0.5),          
            nn.LeakyReLU(0.2),
            # 32 x 32
            nn.MaxPool2d(kernel_size=2, stride=2),
            # 16 x 16
            nn.Conv2d(64,
                      128,
                      kernel_size=5,
                      stride=2,
                      padding=2),
            nn.BatchNorm2d(128, momentum=0.5),          
            nn.LeakyReLU(0.2),
            # 8 x 8
            nn.Flatten(),
            nn.Linear(4 * 4 * 128, opt.discriminator_layer_size),
            nn.BatchNorm1d(opt.discriminator_layer_size, momentum=0.9),
            nn.LeakyReLU(0.2),
            nn.Linear(opt.discriminator_layer_size, 1),
            nn.Sigmoid()
        )
            
    def forward(self, input_data):
        # Expects (batch_size, 3, image_size, image_size)
        return self.main(input_data)
        
def sample_images(imgs, epoch_num):
    save_image(imgs.data, "Results/Generated_Images/%d.png" % epoch_num, nrow=5, normalize=True)
    
def generate_noise(length):
    return torch.cuda.FloatTensor(np.random.normal(0, 1, size=(length, opt.latent_size, 8, 8)))

def main():    
    print("Setup")
    cuda = True if torch.cuda.is_available() else False
    
    '''
    # Set random seed for reproducibility
    manualSeed = 1
    random.seed(manualSeed)
    torch.manual_seed(manualSeed)
    torch.use_deterministic_algorithms(True) # Needed for reproducible results
    '''
    
    # Data loading
    dataset = datasets.ImageFolder(root="Data/Full",
                                    transform=tf.Compose([
                                        tf.RandomPerspective(distortion_scale=opt.persp1, p=opt.persp2),
                                        tf.RandomRotation(15),
                                        tf.Resize(opt.image_size),
                                        tf.CenterCrop(opt.image_size),
                                        tf.ToTensor(),
                                        #tf.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                                    ]))
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=opt.batch_size, shuffle=True, num_workers=opt.workers)
      
    '''
    real_batch = next(iter(dataloader))
    plt.figure(figsize=(8,8))
    plt.axis("off")
    plt.imshow(np.transpose(torchutils.make_grid(real_batch[0].to(self.device)[:64], padding=2, normalize=True).cpu(),(1,2,0)))
    plt.show()
    '''
        
    # Neural net
    decoder = Decoder()
    discriminator = Discriminator()
    
    # Use binary cross-entropy loss
    adversarial_loss = torch.nn.BCELoss()
    
    if cuda:
        decoder.cuda()
        discriminator.cuda()
        adversarial_loss.cuda()
        
    print(decoder)
    print(discriminator)
    
    # Training
    g_losses = []
    d_losses = []

    # Create updating figure
    plt.figure(figsize=(10,5))
    plt.title("Generator and Discriminator Loss During Training")
    plt.xlabel("iterations")
    plt.ylabel("Loss")
    plt.plot(g_losses,label="Decoder",color="blue")
    plt.plot(d_losses,label="Discriminator",color="red")
    plt.legend()

    # Optimizers
    optimizer_G = torch.optim.Adam(decoder.parameters(), lr=opt.lr_g, betas=(opt.b1, opt.b2))
    optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=opt.lr_d, betas=(opt.b1, opt.b2))    

    print("Training")
    for epoch in range(opt.num_epochs):
        for i, (imgs, _) in enumerate(dataloader):
            imgs = imgs.cuda()
            # --------------- Setup ----------------------
            # Adversarial ground truths
            valid = torch.cuda.FloatTensor(imgs.shape[0], 1).fill_(1.0).cuda()
            fake = torch.cuda.FloatTensor(imgs.shape[0], 1).fill_(0.0).cuda()

            # -------------- Training Discriminator -----------------
            discriminator.train()
            decoder.train()
            optimizer_D.zero_grad()
            optimizer_G.zero_grad()

            # Real images
            d_real_imgs = discriminator(imgs)
            
            # Fake images
            noise = generate_noise(len(imgs))
            decoded_imgs = decoder(noise)
            d_fake_imgs = discriminator(decoded_imgs.detach())
            
            # Measure discriminator's ability to classify real from generated samples
            real_loss = adversarial_loss(d_real_imgs, valid)
            fake_loss = adversarial_loss(d_fake_imgs, fake)
            d_loss = 0.5 * (fake_loss + real_loss)
            
            d_num_real = np.round(d_real_imgs.cpu().detach().numpy())
            d_num_fake = np.round(d_fake_imgs.cpu().detach().numpy())
            d_percent_real_correct = 100 * np.sum(d_num_real) / len(d_num_real)
            d_percent_fake_correct = 100 * (len(d_num_fake) - np.sum(d_num_fake)) / len(d_num_fake)

            d_loss.backward()
            optimizer_D.step()
            
            # ------------- Training Generator -------------
            score = discriminator(decoded_imgs)

            # Loss measures generator's ability to fool the discriminator
            g_loss = adversarial_loss(score, valid)
            
            g_loss.backward()
            optimizer_G.step()

            # ----------------------- Display results ------------------------
            print(
                "[Epoch %d/%d] [Batch %d/%d] [D loss: %f] [G loss: %f] [D percent correct r: %.1f | f: %.1f]"
                % (epoch, opt.num_epochs, i, len(dataloader), d_loss.item(), g_loss.item(), d_percent_real_correct, d_percent_fake_correct)
            )

            # Get the second-last batch (to get a full batch)
            if (epoch % 5 == 0 and i == len(dataloader) - 2):
                # Save noise->decoder images
                z = generate_noise(25)
                sample_images(decoder(z), epoch)
                
            # Plotting
            batch_num = epoch * len(dataloader) + i    
            if batch_num % opt.plot_interval == 0:
                g_losses.append(g_loss.item())
                d_losses.append(d_loss.item())
                plt.plot(g_losses,color="blue")
                plt.plot(d_losses,color="red")
                plt.pause(1e-10)
    

    print("Done")
    plt.savefig("Loss_Graph")
    plt.show()

if __name__ == '__main__':
    main()