
#%matplotlib inline
import argparse
from ast import Num
from msilib.schema import DuplicateFile
import os
import itertools
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
from torchvision.utils import save_image

parser = argparse.ArgumentParser()
parser.add_argument("--num_epochs", type=int, default=100, help="number of epochs of training")
parser.add_argument("--batch_size", type=int, default=128, help="size of the batches")
parser.add_argument("--image_size", type=int, default=128, help="size of each image dimension")
parser.add_argument("--lr", type=float, default=0.0002, help="adam: learning rate")
parser.add_argument("--b1", type=float, default=0.5, help="adam: decay of first order momentum of gradient")
parser.add_argument("--b2", type=float, default=0.999, help="adam: decay of first order momentum of gradient")
parser.add_argument("--num_gpus", type=int, default=1, help="number of gpu threads to use")
parser.add_argument("--latent_size", type=int, default=100, help="size of the latent vector i.e. size of generator input")
parser.add_argument("--num_channels", type=int, default=3, help="number of image channels")
parser.add_argument("--sample_interval", type=int, default=200, help="interval between image sampling")
parser.add_argument("--workers", type=int, default=2, help="number of dataloader workers")
parser.add_argument("--feature_maps_generator", type=int, default=64, help="number of feature maps in the generator")
parser.add_argument("--feature_maps_discriminator", type=int, default=64, help="number of feature maps in the discriminator")
parser.add_argument("--persp1", type=float, default=0.2, help="perspective distortion")
parser.add_argument("--persp2", type=float, default=0.5, help="perspective chance")
opt = parser.parse_args()

img_shape = (opt.num_channels, opt.image_size, opt.image_size)

class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        
        self.main = nn.Sequential(
            nn.Conv2d(opt.num_channels,
                      256,
                      kernel_size=5,
                      stride=2,
                      padding=2),
            nn.ReLU(),
            nn.Conv2d(256,
                      128,
                      kernel_size=5,
                      stride=2,
                      padding=2),
            nn.ReLU(),
            nn.Conv2d(128,
                      64,
                      kernel_size=5,
                      stride=2,
                      padding=2),
            nn.ReLU(),
            nn.Conv2d(64,
                      opt.latent_size,
                      kernel_size=5,
                      stride=2,
                      padding=2),
            nn.ReLU()
        )

    def forward(self, input_data):
        # Expects (batch_size, 3, image_size, image_size)
        return self.main(input_data)
        
class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()
        
        self.main = nn.Sequential(
            nn.ConvTranspose2d(opt.latent_size,
                               64,
                               kernel_size=5,
                               stride=2,
                               padding=2,
                               output_padding=1),
            nn.ReLU(),            
            nn.ConvTranspose2d(64,
                               128,
                               kernel_size=5,
                               stride=2,
                               padding=2,
                               output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(128,
                               256,
                               kernel_size=5,
                               stride=2,
                               padding=2,
                               output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(256,
                               3,
                               kernel_size=5,
                               stride=2,
                               padding=2,
                               output_padding=1),
            nn.Sigmoid()
        )
        
    def forward(self, input_data):
        # Expects (batch_size, latent_size, 8, 8)
        return self.main(input_data)
    
class Discriminator(nn.Module):
    def __init__(self, item_count):
        super(Discriminator, self).__init__()

        self.main = nn.Sequential(
            nn.Conv2d(opt.num_channels,
                      1,
                      kernel_size=3,
                      stride=2,
                      padding=1),
            nn.Sigmoid()
        )
            
    def forward(self, input_data):
        # Expects (batch_size, 3, image_size, image_size)
        return self.main(input_data)
    
def get_average(tensor):
    return torch.mean(torch.mean(tensor, 3), 2)
    
def sample_images(decoder, batches_done):
    z = torch.cuda.FloatTensor(np.random.normal(0, 1, (10, opt.latent_size, 8, 8)))
    gen_imgs = decoder(z)
    save_image(gen_imgs.data, "Generated_Images/%d.png" % batches_done, nrow=10, normalize=True)
    
def sample_autoencoder_images(imgs, pos):
    plt.figure(figsize=(8,8))
    plt.axis("off")
    plt.imshow(np.transpose(torchutils.make_grid(imgs.cuda()[:64], padding=2, normalize=True).cpu(),(1,2,0)))
    plt.savefig("Autoencoder_Images/"+str(pos)+".png")


def main():    
    print("Setup")
    cuda = True if torch.cuda.is_available() else False
    
    # Set random seed for reproducibility
    manualSeed = 1
    random.seed(manualSeed)
    torch.manual_seed(manualSeed)
    torch.use_deterministic_algorithms(True) # Needed for reproducible results

    # Data loading
    dataset = datasets.ImageFolder(root="Data/Full",
                                    transform=tf.Compose([
                                        #tf.RandomPerspective(distortion_scale=opt.persp1, p=opt.persp2),
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
    encoder = Encoder()
    decoder = Decoder()
    discriminator = Discriminator(len(dataloader))
    
    # Use binary cross-entropy loss
    adversarial_loss = torch.nn.BCELoss()
    pixelwise_loss = torch.nn.L1Loss()
    
    if cuda:
        encoder.cuda()
        decoder.cuda()
        discriminator.cuda()
        adversarial_loss.cuda()
        pixelwise_loss.cuda()
    
    # Optimizers
    optimizer_G = torch.optim.Adam(
        itertools.chain(encoder.parameters(), decoder.parameters()), lr=opt.lr, betas=(opt.b1, opt.b2)
    )
    optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))    

    # Training
    g_losses = []
    d_losses = []

    print("Training")
    for epoch in range(opt.num_epochs):
        for i, (imgs, _) in enumerate(dataloader):
            # --------------- Setup ----------------------
            # Adversarial ground truths
            valid = torch.cuda.FloatTensor(imgs.shape[0], 1).fill_(1.0).cuda()
            fake = torch.cuda.FloatTensor(imgs.shape[0], 1).fill_(0.0).cuda()

            # Configure input
            real_imgs = imgs.type(torch.cuda.FloatTensor).cuda()
            

            # ------------- Training Autoencoder -------------
            optimizer_G.zero_grad()

            encoded_imgs = encoder(real_imgs)
            decoded_imgs = decoder(encoded_imgs)
            
            score = get_average(discriminator(decoded_imgs))

            pos = (epoch * len(dataloader) + i)

            # Loss measures generator's ability to fool the discriminator
            g_loss = 0.001 * adversarial_loss(score, valid) + 0.999 * pixelwise_loss(
                decoded_imgs, real_imgs
            )

            g_loss.backward()
            optimizer_G.step()
            g_losses.append(g_loss.item())


            # -------------- Training Discriminator -----------------
            optimizer_D.zero_grad()

            # Sample noise as discriminator ground truth
            z = torch.cuda.FloatTensor(np.random.normal(0, 1, (imgs.shape[0], opt.num_channels, opt.image_size, opt.image_size)))

            # Measure discriminator's ability to classify real from generated samples
            real_loss = adversarial_loss(get_average(discriminator(z)), valid)
            fake_loss = adversarial_loss(get_average(discriminator(decoded_imgs.detach())), fake)
            d_loss = 0.5 * (real_loss + fake_loss)

            d_loss.backward()
            optimizer_D.step()
            d_losses.append(d_loss.item())


            # ----------------------- Display results ------------------------
            print(
                "[Epoch %d/%d] [Batch %d/%d] [D loss: %f] [G loss: %f]"
                % (epoch, opt.num_epochs, i, len(dataloader), d_loss.item(), g_loss.item())
            )

            batches_done = epoch * len(dataloader) + i
            if batches_done % opt.sample_interval == 0:
                # Save noise->decoder images
                sample_images(decoder, batches_done)
                # Sample true->autoencoder images
                sample_autoencoder_images(decoded_imgs, batches_done)

    print("Done")
    plt.figure(figsize=(10,5))
    plt.title("Generator and Discriminator Loss During Training")
    plt.plot(g_losses,label="G")
    plt.plot(d_losses,label="D")
    plt.xlabel("iterations")
    plt.ylabel("Loss")
    plt.legend()
    plt.show()

if __name__ == '__main__':
    main()