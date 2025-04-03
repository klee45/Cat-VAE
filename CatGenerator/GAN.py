import Utility
import Autoencoder
import argparse
import torch
import torch.nn as nn
import torchvision.datasets as datasets
import torchvision.transforms as tf
import torchvision.utils as torchutils
import numpy as np
import matplotlib.pyplot as plt

opt = Utility.get_opt()

'''
Generator
    4x (Linear -> BatchNorm1d -> ReLU)
        64 128 256 512 latent
    
Discriminator
    4x (Conv2d -> BatchNorm2d -> ReLU)
        16 32 64 128 256
    Linear -> BatchNorm1d -> ReLU -> Linear -> Sigmoid
        xx 512 1
'''
'''

generated images -> Discriminator
real images -> Discriminator

'''

# Conv -> Batch norm -> ReLU
class Discriminator_Block(nn.Module):
    def __init__(self, channel_in, channel_out, leaky=False):
       super(Discriminator_Block, self).__init__()
       
       self.conv = nn.Sequential(
            nn.Conv2d(channel_in, channel_out, kernel_size=5, stride=2, padding=2, bias=False),
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
        layers_list.append(Discriminator_Block(opt.num_channels, size))
        for i in range(opt.num_discriminator_layers - 1):
            layers_list.append(Discriminator_Block(size, size * 2))
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
    
def generate_noise(length, channels):
    return torch.cuda.FloatTensor(np.random.normal(0, 1, size=(length, channels)))
    
def main():    
    print("Setup")
    
    # Neural net
    generator = Generator()
    discriminator = Discriminator()
        
    # Use binary cross-entropy loss
    adversarial_loss = torch.nn.BCELoss()

    if torch.cuda.is_available:
        generator.cuda()
        discriminator.cuda()
        adversarial_loss.cuda()

    print(generator)
    print(discriminator)

    # Optimizers
    optimizer_generator = torch.optim.Adam(params=generator.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))
    optimizer_discriminator = torch.optim.Adam(params=discriminator.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))
        
    # Training
    d_losses = []

    # Create updating figure
    plt.figure(figsize=(10,5))
    plt.title("Losses During Training")
    plt.xlabel("iterations")
    plt.ylabel("Loss")
    plt.yscale("log")
    plt.plot(d_losses,label="Discriminator",color="red")
    plt.legend()

    # Data loading
    dataset = datasets.ImageFolder(root="Data/Full",
                                    transform=tf.Compose([
                                        #tf.RandomPerspective(distortion_scale=opt.persp1, p=opt.persp2),
                                        #tf.RandomRotation(15),
                                        tf.Resize(opt.image_size),
                                        tf.CenterCrop(opt.image_size),
                                        tf.ToTensor(),
                                    ]))
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=opt.batch_size, shuffle=True, num_workers=opt.workers)
    
    # Define NN
    generator = Generator()
    discriminator = Discriminator()
    
    # Load and freeze decoder
    decoder = Autoencoder.Decoder()
    decoder.load_state_dict(torch.load("Models/Decoder"))
    for param in decoder.parameters():
        param.requires_grad = False

    print("Training")
    for epoch in range(opt.num_epochs):
        for i, (imgs, _) in enumerate(dataloader):
            imgs = imgs.cuda()
            
            # Adversarial ground truths
            valid = torch.cuda.FloatTensor(imgs.shape[0], 1).fill_(1.0).cuda()
            fake = torch.cuda.FloatTensor(imgs.shape[0], 1).fill_(0.0).cuda()

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
                "[Epoch %d/%d] [Batch %d/%d]" % (epoch + 1, opt.num_epochs, i, len(dataloader)) +
                "\n\t [Discriminator loss: %f]" % discriminator_loss.item() + 
                "\n\t [percent correct r: %.2f | f: %.2f]" % (d_percent_real_correct, d_percent_fake_correct)
                #d_percent_real_correct, d_percent_fake_correct)
            )
            
            # Get the second-last batch (to get a full batch)
            if ((epoch + 1) % 5 == 0 and i == len(dataloader) - 2):
                # Save noise->decoder images
                Utility.sample_images(generated_images, epoch + 1, "Generator")
                
            # Plotting
            batch_num = epoch * len(dataloader) + i    
            if (batch_num + 1) % opt.plot_interval == 0:
                plt.plot(d_losses,color="red")
                plt.pause(1e-10)
    
    print("Done")
    torch.save(generator.state_dict(), "Models/Generator")
    torch.save(discriminator.state_dict(), "Models/Discriminator")

    plt.savefig("GAN_Loss_Graph")
    plt.show()

if __name__ == '__main__':
    main()