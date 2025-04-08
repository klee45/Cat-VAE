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
import os

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
            nn.LeakyReLU(0.2, True),
            nn.Linear(512, 1, bias=False),
            nn.Sigmoid()
        )
        
    def forward(self, input_data):
        data = self.conv(input_data)
        data = self.end(data)
        return data

def weights_init(m):
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
        torch.nn.init.normal_(m.weight, 0.0, 0.02)
    if isinstance(m, nn.BatchNorm2d):
        torch.nn.init.normal_(m.weight, 0.0, 0.02)
        torch.nn.init.constant_(m.bias, 0)
    if isinstance(m, nn.Linear):
        torch.nn.init.normal_(m.weight, 0.0, 0.02)
    
def generate_noise(length, channels):
    return torch.cuda.FloatTensor(np.random.normal(0, 1, size=(length, channels)))
        
def get_gradient(discriminator, real, fake, epsilon):
    mixed_images = real * epsilon + fake * (1 - epsilon)

    mixed_scores = discriminator(mixed_images)
    
    gradient = torch.autograd.grad(
        inputs=mixed_images,
        outputs=mixed_scores,
        grad_outputs=torch.ones_like(mixed_scores), 
        create_graph=True,
        retain_graph=True,
        
    )[0]
    return gradient

def gradient_penalty(gradient):
    gradient = gradient.view(len(gradient), -1)
    gradient_norm = gradient.norm(2, dim=1)
    penalty = torch.mean((gradient_norm -1)**2)
    return penalty

def get_gen_loss(fake_pred):
    return -torch.mean(fake_pred)

def get_disc_loss(fake_pred, real_pred, grad_pen, c_lambda=10):
    return torch.mean(fake_pred) - torch.mean(real_pred) + c_lambda * grad_pen

    
def main():    
    print("Setup")
    
    # Neural net
    generator = Generator()
    discriminator = Discriminator()
    
    generator.apply(weights_init)
    discriminator.apply(weights_init)
            
    # Load and freeze decoder
    decoder = Autoencoder.Decoder()
    decoder.load_state_dict(torch.load("Models/Decoder"))
    for param in decoder.parameters():
        param.requires_grad = False
        
    if torch.cuda.is_available:
        generator.cuda()
        discriminator.cuda()
        decoder.cuda()

    print(generator)
    print(discriminator)

    # Optimizers
    optimizer_generator = torch.optim.Adam(params=generator.parameters(), lr=opt.lr_g, betas=(opt.b1, opt.b2))
    optimizer_discriminator = torch.optim.Adam(params=discriminator.parameters(), lr=opt.lr_d, betas=(opt.b1, opt.b2))
        
    # Training
    g_losses = []
    d_losses = []

    # Create updating figure
    plt.figure(figsize=(10,5))
    plt.title("Losses During Training")
    plt.xlabel("iterations")
    plt.ylabel("Loss")
    #plt.yscale("log")
    plt.plot(g_losses, label="Generator", color="green")
    plt.plot(d_losses, label="Discriminator",color="red")
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
    
    start_epoch = 0
    if os.path.isfile("Progress/epoch.txt"):
        start_epoch = int(np.loadtxt("Progress/epoch.txt")) + 1
        g_losses, d_losses = [x.tolist() for x in np.loadtxt("Progress/loss.csv", delimiter=",")]
        generator.load_state_dict(torch.load("Models/Generator"))
        discriminator.load_state_dict(torch.load("Models/Discriminator"))
        

    print("Training")
    for epoch in range(start_epoch, opt.num_epochs_gan):
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
            d_fake_imgs = discriminator(generated_images.detach())

            # Calculate loss
            epsilon = torch.rand(len(imgs), 1, 1, 1, device="cuda", requires_grad=True)
            gradient = get_gradient(discriminator, imgs, generated_images.detach(), epsilon)
            penalty = gradient_penalty(gradient)
            
            # --------------- Discriminator ----------------------------
            d_loss = get_disc_loss(d_fake_imgs.detach(), d_real_imgs, penalty)

            # Information for debugging
            d_num_real = np.round(d_real_imgs.detach().cpu().numpy())
            d_num_fake = np.round(d_fake_imgs.detach().cpu().numpy())
            d_percent_real_correct = 100 * np.sum(d_num_real) / len(d_num_real)
            d_percent_fake_correct = 100 * (len(d_num_fake) - np.sum(d_num_fake)) / len(d_num_fake)

            d_loss.backward()
            d_losses.append(d_loss.item())
            optimizer_discriminator.step()
            
            # ------------------- Generator --------------------------
            noise_2 = generate_noise(len(imgs), opt.generator_size)
            generated_images_2 = decoder(generator(noise_2))
            d_fake_imgs_2 = discriminator(generated_images)

            g_loss = get_gen_loss(d_fake_imgs_2)
            g_loss.backward()
            g_losses.append(g_loss.item())
            optimizer_generator.step()

            # ----------------------- Display results ------------------------
            print(
                "[Epoch %d/%d] [Batch %d/%d]" % (epoch + 1, opt.num_epochs_gan, i + 1, len(dataloader)) +
                "\n\t [Generator loss: %f]" % g_loss.item() + 
                "\n\t [Discriminator loss: %f]" % d_loss.item() + 
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
                plt.plot(g_losses, color="green")
                plt.plot(d_losses, color="red")
                plt.pause(1e-10)

        # ----------------------- Save the model ---------------------------
        torch.save(generator.state_dict(), "Models/Generator")
        torch.save(discriminator.state_dict(), "Models/Discriminator")
                
        np.savetxt("Progress/loss.csv", [g_losses, d_losses], delimiter=",", fmt="%f")
        np.savetxt("Progress/epoch.txt", [epoch], fmt="%d")
    
    print("Done")
    os.remove("Progress/epoch.txt")
    plt.savefig("GAN_Loss_Graph")
    plt.show()
        

if __name__ == '__main__':
    main()