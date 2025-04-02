
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
parser.add_argument("--num_epochs", type=int, default=50, help="number of epochs of training")
parser.add_argument("--batch_size", type=int, default=128, help="size of the batches")
parser.add_argument("--image_size", type=int, default=128, help="size of each image dimension")
parser.add_argument("--lr", type=float, default=5e-4, help="adam: learning rate for generator")
parser.add_argument("--b1", type=float, default=0.5, help="adam: decay of first order momentum of gradient")
parser.add_argument("--b2", type=float, default=0.999, help="adam: decay of first order momentum of gradient")
parser.add_argument("--num_gpus", type=int, default=1, help="number of gpu threads to use")
parser.add_argument("--latent_size", type=int, default=1024, help="size of the latent vector i.e. size of decoder input")
parser.add_argument("--generator_size", type=int, default=128, help="size of the noise vector for the generator")
parser.add_argument("--num_channels", type=int, default=3, help="number of image channels")
parser.add_argument("--sample_interval", type=int, default=100, help="interval between image sampling")
parser.add_argument("--plot_interval", type=int, default=25, help="interval between datapoint plotting")
parser.add_argument("--workers", type=int, default=2, help="number of dataloader workers")
parser.add_argument("--persp1", type=float, default=0.2, help="perspective distortion")
parser.add_argument("--persp2", type=float, default=0.5, help="perspective chance")
parser.add_argument("--num_autoencoder_layers", type=int, default=3, help="number of conv layers in the autoencoder")
parser.add_argument("--num_generator_layers", type=int, default=3, help="number of dense layers in the generator")
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
    Conv2d -> Tanh
        16 3
    
Generator
    4x (Linear -> BatchNorm1d -> ReLU)
        64 128 256 512 latent
    
Discriminator
    4x (Conv2d -> BatchNorm2d -> ReLU)
        16 32 64 128 256
    Linear -> BatchNorm1d -> ReLU -> Linear -> Sigmoid
        xx 512 1

real images -> Encoder -> Decoder -> autoencoded images
noise -> Generator -> Decoder -> generated images

generated images -> Discriminator
real images -> Discriminator
'''

# Conv -> Batch norm -> ReLU
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

# Conv tranpose -> Batch norm -> ReLU
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

# Linear -> Batch norm -> ReLU
class Fully_Connected_Block(nn.Module):
    def __init__(self, channel_in, channel_out, leaky=False):
        super(Fully_Connected_Block, self).__init__()

        self.conv = nn.Sequential(
            nn.Linear(channel_in, channel_out, bias=False),
            nn.BatchNorm1d(channel_out, momentum=0.9),
            nn.LeakyReLU(0.2, True) if leaky else nn.ReLU(True) 
    )
    def forward(self, ten):
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
    
class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()

        layers_list = []
        size = opt.generator_size
        for i in range(opt.num_generator_layers - 1):
            layers_list.append(Fully_Connected_Block(size, size * 2))
            size *= 2
        layers_list.append(Fully_Connected_Block(size, opt.latent_size))

        self.fc = nn.Sequential(*layers_list)

    def forward(self, input_data):
        return self.fc(input_data)
        
class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
            
        layers_list = []
        size = opt.min_feature_size * 2
        layers_list.append(Encoder_Block(opt.num_channels, size))
        for i in range(opt.num_discriminator_layers - 1):
            layers_list.append(Encoder_Block(size, size * 2))
            size *= 2
        assert size == opt.min_feature_size * 2**opt.num_discriminator_layers
        
        self.conv = nn.Sequential(*layers_list)
         
        # dimensions 16 x (image_size / (2^4)) x (image_size / (2^4))
        compressed_size = int(opt.image_size / 2**opt.num_discriminator_layers)
        self.end = nn.Sequential(
            nn.Flatten(),
            nn.Linear(size * compressed_size * compressed_size, 512, bias=False),
            nn.BatchNorm1d(512, momentum=0.9),
            nn.ReLU(True),
            nn.Linear(512, 1, bias=False),
            nn.Sigmoid()
        )
        
    def forward(self, input_data):
        data = self.conv(input_data)
        data = self.end(data)
        return data
    
def sample_images(imgs, name, path):
    save_image(imgs.data[:25], "Results/" + path + "_Images/%s.png" % name, nrow=5, normalize=True)
    
def generate_noise(length, channels):
    return torch.cuda.FloatTensor(np.random.normal(0, 1, size=(length, channels)))

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
    generator = Generator()
    discriminator = Discriminator()
    
    # Use binary cross-entropy loss
    adversarial_loss = torch.nn.BCELoss()
    pixelwise_loss = torch.nn.L1Loss()
    
    if cuda:
        encoder.cuda()
        decoder.cuda()
        generator.cuda()
        discriminator.cuda()
        adversarial_loss.cuda()
        pixelwise_loss.cuda()

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
    plt.plot(d_losses,label="Discriminator",color="red")
    plt.legend()

    # Optimizers
    optimizer_encoder = torch.optim.Adam(params=encoder.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))
    optimizer_decoder = torch.optim.Adam(params=decoder.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))
    optimizer_generator = torch.optim.Adam(params=generator.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))
    optimizer_discriminator = torch.optim.Adam(params=discriminator.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))

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

            autoencoder_loss = pixelwise_loss(decoded_images, imgs)
            a_losses.append(autoencoder_loss.item())
            
            autoencoder_loss.backward()
            optimizer_encoder.step()
            optimizer_decoder.step()

            # -------------- Training Discriminator and Generator -----------------
            optimizer_generator.zero_grad()
            optimizer_discriminator.zero_grad()
            
            noise = generate_noise(len(imgs), opt.generator_size)
            generated_images = decoder(generator(noise))
            
            # Score both the real and fake images in the discriminator
            d_real_imgs = discriminator(imgs)
            d_fake_imgs = discriminator(generated_images)

            # Calculate loss
            real_loss = adversarial_loss(d_real_imgs, valid)
            fake_loss = adversarial_loss(d_fake_imgs, fake)
            discriminator_loss = 0.5 * (real_loss + fake_loss)
            d_losses.append(discriminator_loss.item())

            c = d_real_imgs.detach().cpu().numpy()
            d = d_fake_imgs.detach().cpu().numpy()

            a = real_loss.item()
            b = fake_loss.item()
            
            # Information for debugging
            d_num_real = np.round(d_real_imgs.detach().cpu().numpy())
            d_num_fake = np.round(d_fake_imgs.detach().cpu().numpy())
            d_percent_real_correct = 100 * np.sum(d_num_real) / len(d_num_real)
            d_percent_fake_correct = 100 * (len(d_num_fake) - np.sum(d_num_fake)) / len(d_num_fake)

            discriminator_loss.backward()
            optimizer_generator.step()
            optimizer_discriminator.step()

            # ----------------------- Display results ------------------------
            print(
                "[Epoch %d/%d] [Batch %d/%d]" % (epoch, opt.num_epochs, i, len(dataloader)) +
                "\n\t [Autoencoder loss: %f]" % autoencoder_loss.item() + 
                "\n\t [Discriminator loss: %f]" % discriminator_loss.item() + 
                "\n\t [percent correct r: %.2f | f: %.2f]" % (d_percent_real_correct, d_percent_fake_correct)
                #d_percent_real_correct, d_percent_fake_correct)
            )

            # Get the second-last batch (to get a full batch)
            if (epoch % 5 == 0 and i == len(dataloader) - 2):
                # Save noise->decoder images
                sample_images(decoded_images, epoch, "Autoencoder")
                sample_images(imgs, str(epoch)+"_real", "Autoencoder")
                sample_images(generated_images, epoch, "Generator")
                
            # Plotting
            batch_num = epoch * len(dataloader) + i    
            if (batch_num + 1) % opt.plot_interval == 0:
                plt.plot(a_losses,color="blue")
                plt.plot(d_losses,color="red")
                plt.pause(1e-10)
    

    print("Done")
    plt.savefig("Loss_Graph")
    plt.show()

if __name__ == '__main__':
    main()