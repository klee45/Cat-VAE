
#%matplotlib inline
import argparse
from ast import Num
from msilib.schema import DuplicateFile
import os
import itertools
import random
from uu import decode
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
parser.add_argument("--image_size", type=int, default=256, help="size of each image dimension")
parser.add_argument("--lr", type=float, default=5e-4, help="adam: learning rate for generator")
parser.add_argument("--b1", type=float, default=0.5, help="adam: decay of first order momentum of gradient")
parser.add_argument("--b2", type=float, default=0.999, help="adam: decay of first order momentum of gradient")
parser.add_argument("--num_gpus", type=int, default=1, help="number of gpu threads to use")
parser.add_argument("--latent_size", type=int, default=1024, help="size of the latent vector i.e. size of generator input")
parser.add_argument("--discriminator_layer_size", type=int, default=64, help="The size of the linear layers in the discriminator")
parser.add_argument("--num_channels", type=int, default=3, help="number of image channels")
parser.add_argument("--sample_interval", type=int, default=100, help="interval between image sampling")
parser.add_argument("--plot_interval", type=int, default=5, help="interval between datapoint plotting")
parser.add_argument("--workers", type=int, default=2, help="number of dataloader workers")
parser.add_argument("--persp1", type=float, default=0.2, help="perspective distortion")
parser.add_argument("--persp2", type=float, default=0.5, help="perspective chance")
parser.add_argument("--num_autoencoder_layers", type=int, default=3, help="number of conv layers in the encoder")
parser.add_argument("--num_discriminator_layers", type=int, default=3, help="number of conv layers in the discriminator")
parser.add_argument("--min_feature_size", type=int, default=16, help="Smallest feature count (default 16)")
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
    Conv2d -> Tanh()
        16 3
    
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

generated images -> Discriminator
real images -> Discriminator
'''

class Encoder_Block(nn.Module):
    def __init__(self, channel_in, channel_out, leaky=False):
       super(Encoder_Block, self).__init__()
       
       self.conv = nn.Sequential(
            nn.Conv2d(channel_in, channel_out, kernel_size=5, stride=2, padding=2, bias=False),
            nn.BatchNorm2d(channel_out, momentum=0.9),
            nn.LeakyReLU(0.2, True) if leaky else nn.ReLU(True) 
        )
    def forward(self, ten):
        return self.conv(ten)

class Decoder_Block(nn.Module):
    def __init__(self, channel_in, channel_out, leaky=False):
        super(Decoder_Block, self).__init__()
        
        self.conv = nn.Sequential(
            nn.ConvTranspose2d(channel_in, channel_out, kernel_size=5, stride=2, padding=2, output_padding=1, bias=False),
            nn.BatchNorm2d(channel_out, momentum=0.9),
            nn.LeakyReLU(0.2, True) if leaky else nn.ReLU(True) 
        )
    def forward(self, ten):
        return self.conv(ten)

class Fully_Connected_Block(nn.Module):
    def __init__(self, channel_in, channel_out, leaky=False):
        super(Fully_Connected_Block, self).__init__()

        self.conv = nn.Sequential(
            nn.Linear(channel_in, channel_out, bias=False),
            nn.BatchNorm1d(channel_out, momentum=0.9),
            nn.LeakyReLU(0.2, True) if leaky else nn.ReLU(True) 
    )
    def foward(self, ten):
        return self.conv(ten)
    
class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        
        layers_list = []
        size = opt.min_feature_size * 2
        layers_list.append(Encoder_Block(opt.num_channels, size))
        for i in range(opt.num_autoencoder_layers - 1):
            layers_list.append(Encoder_Block(size, size * 2))
            size *= 2
        assert size == opt.min_feature_size * 2**opt.num_autoencoder_layers
        
        self.conv = nn.Sequential(*layers_list)
         
        # dimensions 16 x (image_size / (2^4)) x (image_size / (2^4))
        compressed_size = int(opt.image_size / 2**opt.num_autoencoder_layers)
        self.end = nn.Sequential(
            nn.Flatten(),
            nn.Linear(size * compressed_size * compressed_size, opt.latent_size, bias=False),
            nn.BatchNorm1d(opt.latent_size, momentum=0.9),
            nn.ReLU(True)
        )
        
    def forward(self, input_data):
        data = self.conv(input_data)
        data = self.end(data)
        return data

class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()
        
        # Unflatten latent vector
        size = int(opt.min_feature_size * 2**opt.num_autoencoder_layers)
        compressed_size = int(opt.image_size / 2**opt.num_autoencoder_layers)
        compressed_features = int(size * compressed_size * compressed_size)
        self.start = nn.Sequential(
            nn.Linear(opt.latent_size, compressed_features, bias=False),
            nn.BatchNorm1d(compressed_features, momentum=0.9),
            nn.ReLU(True),
            nn.Unflatten(1, (size, compressed_size, compressed_size))
        )

        # Assemble convtranspose2d layers
        layers_list = []
        for i in range(opt.num_autoencoder_layers):
            layers_list.append(Decoder_Block(size, int(size / 2)))
            size = int(size/2)
        assert size == opt.min_feature_size
        self.conv = nn.Sequential(*layers_list)
                 
        self.end = nn.Sequential(
            nn.Conv2d(size, opt.num_channels, kernel_size=5, stride=1, padding=2),
            nn.Tanh()
        )
        
    def forward(self, input_data):
        data = self.start(input_data)
        data = self.conv(data)
        data = self.end(data)
        return data
        
class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
            
    def forward(self, input_data):
        pass
    
def sample_images(imgs, name, path):
    save_image(imgs.data[:25], "Results/" + path + "_Images/%s.png" % name, nrow=5, normalize=True)
    
def generate_noise(length):
    return torch.cuda.FloatTensor(np.random.normal(0, 1, size=(length, opt.latent_size)))

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
                                        #tf.RandomPerspective(distortion_scale=opt.persp1, p=opt.persp2),
                                        #tf.RandomRotation(15),
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
    encoder = Encoder()
    
    # Use binary cross-entropy loss
    adversarial_loss = torch.nn.BCELoss()
    pixelwise_loss = torch.nn.L1Loss()
    
    if cuda:
        encoder.cuda()
        decoder.cuda()
        adversarial_loss.cuda()
        
    print(encoder)
    print(decoder)
    
    # Training
    a_losses = []
    g_losses = []
    d_losses = []

    # Create updating figure
    plt.figure(figsize=(10,5))
    plt.title("Losses During Training")
    plt.xlabel("iterations")
    plt.ylabel("Loss")
    plt.yscale("log")
    plt.plot(a_losses,label="Autoencoder",color="blue")
    plt.plot(g_losses,label="Generator", color="green")
    plt.plot(d_losses,label="Discriminator",color="red")
    plt.legend()

    # Optimizers
    optimizer_encoder = torch.optim.Adam(params=encoder.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))
    optimizer_decoder = torch.optim.Adam(params=decoder.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))

    print("Training")
    for epoch in range(opt.num_epochs):
        for i, (imgs, _) in enumerate(dataloader):
            imgs = imgs.cuda()

            # Adversarial ground truths
            valid = torch.cuda.FloatTensor(imgs.shape[0], 1).fill_(1.0).cuda()
            fake = torch.cuda.FloatTensor(imgs.shape[0], 1).fill_(0.0).cuda()

            # --------------- Setup ----------------------

            # -------------- Training Autoencoder ------------------
            optimizer_encoder.zero_grad()
            optimizer_decoder.zero_grad()

            encoded_images = encoder(imgs)
            decoded_images = decoder(encoded_images)

            autoenc_loss = pixelwise_loss(decoded_images, imgs)
            a_losses.append(autoenc_loss.item())
            
            autoenc_loss.backward()
            optimizer_encoder.step()
            optimizer_decoder.step()

            # -------------- Training Discriminator -----------------
            '''
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
            '''
            
            # ------------- Training Generator -------------
            '''
            score = discriminator(decoded_imgs)

            # Loss measures generator's ability to fool the discriminator
            g_loss = adversarial_loss(score, valid)
            
            g_loss.backward()
            optimizer_G.step()
            '''
            
            # ----------------------- Display results ------------------------
            print(
                "[Epoch %d/%d] [Batch %d/%d]" % (epoch, opt.num_epochs, i, len(dataloader)) +
                "\n\t [Autoencoder loss: %f]" % autoenc_loss.item() + 
                "\n\t [Generator loss: %f]" % 0 +  
                "\n\t [Discriminator loss: %f]" % 0 + 
                "\n\t [percent correct r: %.2f | f: %.2f]" % (0, 0)
                #d_percent_real_correct, d_percent_fake_correct)
            )

            # Get the second-last batch (to get a full batch)
            if (epoch % 5 == 0 and i == len(dataloader) - 2):
                # Save noise->decoder images
                sample_images(decoded_images, epoch, "Autoencoder")
                sample_images(imgs, str(epoch)+"_real", "Autoencoder")
                
            # Plotting
            batch_num = epoch * len(dataloader) + i    
            if batch_num % opt.plot_interval == 0:
                plt.plot(a_losses,color="blue")
                plt.pause(1e-10)
    

    print("Done")
    plt.savefig("Loss_Graph")
    plt.show()

if __name__ == '__main__':
    main()