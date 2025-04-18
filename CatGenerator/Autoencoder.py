
#%matplotlib inline
from uu import decode
import os
import Utility
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.optim as optim
import torch.utils.data
import torchvision.datasets as datasets
import torchvision.transforms.v2 as tf
import torchvision.utils as torchutils
import numpy as np
import matplotlib.pyplot as plt
import torchvision

opt = Utility.get_opt()

'''
Encoder
    4x (Conv2d -> BatchNorm2d -> ReLU)
        3 16 32 64 128
    Linear -> BatchNorm1d -> ReLU
        128 latent
    
Decoder
    Linear -> BatchNorm1d -> ReLU
        latent 256
    4x (ConvTranspose2d -> BatchNorm2d -> ReLU)
        256 128 64 32 16
    Conv2d -> Tanh
        16 3

real images -> Encoder -> Decoder -> autoencoded images
noise -> Decoder -> generated images
'''

# Conv -> ReLU -> Batch norm
class Encoder_Block(nn.Module):
    def __init__(self, channel_in, channel_out, leaky=False):
       super(Encoder_Block, self).__init__()
       
       self.conv = nn.Sequential(
            nn.Conv2d(channel_in, channel_out, kernel_size=5, stride=2, padding=2, bias=False), 
            nn.LeakyReLU(0.2, True) if leaky else nn.ReLU(True),
            nn.BatchNorm2d(channel_out, momentum=0.9)
        )
    def forward(self, ten):
        return self.conv(ten)

# Conv tranpose -> ReLU-> Batch norm 
class Decoder_Block(nn.Module):
    def __init__(self, channel_in, channel_out, leaky=False):
        super(Decoder_Block, self).__init__()
        
        self.conv = nn.Sequential(
            nn.ConvTranspose2d(channel_in, channel_out, kernel_size=5, stride=2, padding=2, output_padding=1, bias=False),
            nn.LeakyReLU(0.2, True) if leaky else nn.ReLU(True),
            nn.BatchNorm2d(channel_out, momentum=0.9)
        )
    def forward(self, ten):
        return self.conv(ten)

class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        
        layers_list = []
        size = opt.min_feature_size
        layers_list.append(Encoder_Block(opt.num_channels, size))
        for i in range(opt.num_autoencoder_layers - 1):
            layers_list.append(Encoder_Block(size, size * 2))
            size *= 2
        assert size == opt.min_feature_size * 2**(opt.num_autoencoder_layers - 1)

        layers_list.append(nn.Flatten())
        self.conv = nn.Sequential(*layers_list)
         
        # dimensions 16 x (image_size / (2^4)) x (image_size / (2^4))
        compressed_size = int(opt.image_size / 2**opt.num_autoencoder_layers)
        self.fc_mean = nn.Sequential(
            nn.Linear(size * compressed_size * compressed_size, opt.latent_size, bias=False)
        )
        
        self.fc_log_var  = nn.Sequential(
            nn.Linear(size * compressed_size * compressed_size, opt.latent_size, bias=False)
        )
        
    def forward(self, input_data):
        data = self.conv(input_data)
        z_mean = self.fc_mean(data)
        z_log_var = self.fc_log_var(data)
        return z_mean, z_log_var

class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()
        
        # Unflatten latent vector
        size = int(opt.min_feature_size * 2**(opt.num_autoencoder_layers))
        compressed_size = int(opt.image_size / 2**opt.num_autoencoder_layers)
        compressed_features = int(size * compressed_size * compressed_size)
        self.start = nn.Sequential(
            nn.Linear(opt.latent_size, compressed_features, bias=False),
            nn.ReLU(True),
            nn.BatchNorm1d(compressed_features, momentum=0.9),
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
            nn.Conv2d(size, opt.num_channels, kernel_size=3, stride=1, padding=1),
            nn.Sigmoid()
        )
        
    def forward(self, input_data):
        data = self.start(input_data)
        data = self.conv(data)
        data = self.end(data)
        return data
    
def weights_init(m):
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
        nn.init.normal_(m.weight, 0.0, 0.02)
    if isinstance(m, nn.BatchNorm2d):
        nn.init.normal_(m.weight, 0.0, 0.02)
        nn.init.constant_(m.bias, 0)
        
def vae_gaussian_kl_loss(mu, logvar):
    kld = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1)
    return kld.sum()
        
def reconstruction_loss(sampled, real):
    bce_loss = nn.MSELoss(reduction="sum")
    return bce_loss(sampled, real)

def vae_loss(mean, logvar, sampled, real):
    recon_loss = reconstruction_loss(sampled, real)
    kld_loss = vae_gaussian_kl_loss(mean, logvar)
    return recon_loss, kld_loss

def generate_noise(length):
    return torch.normal(0, 1, size=(length, opt.latent_size)).cuda()

def generate_latent(mean, var):
    noise = generate_noise(len(mean))
    return mean + var * noise

def make_folder(path):
    if not os.path.exists(path):
        os.mkdir(path)

def main():    
    print("Setup")

    make_folder("Models")
    make_folder("Progress")
    make_folder("Results/Autoencoder_Images")
    make_folder("Results/Sampled_Images")

    # Data loading
    if (opt.test_generator):
        dataloader= torch.utils.data.DataLoader(
            torchvision.datasets.MNIST('/files/', train=True, download=True,
                                       transform=tf.Compose([
                                            tf.Resize(opt.image_size),
                                            tf.ToTensor(),
                                            tf.Normalize((0.1307,), (0.3081,))
                                 ])),
        batch_size=opt.batch_size, shuffle=True)
         
    else:
       dataset = datasets.ImageFolder(root="Data/Full",
                                        transform=tf.Compose([
                                            tf.RandomResize(int(opt.image_size * 1.5), 3 * opt.image_size),
                                            tf.RandomCrop(int(opt.image_size * 1.5)),
                                            tf.RandomPerspective(distortion_scale=opt.persp1, p=opt.persp2),
                                            tf.RandomRotation(15),
                                            tf.CenterCrop(opt.image_size),
                                            tf.ToTensor(),
                                        ]))
       dataloader = torch.utils.data.DataLoader(dataset, batch_size=opt.batch_size, shuffle=True, num_workers=opt.workers, drop_last=True)
       
    # Neural net
    decoder = Decoder()
    encoder = Encoder()
    
    encoder.apply(weights_init)
    decoder.apply(weights_init)
    
    if torch.cuda.is_available:
        encoder.cuda()
        decoder.cuda()

    print(encoder)
    print(decoder)
    
    # Training
    r_losses = []
    k_losses = []

    # Create updating figure
    plt.figure(figsize=(10,5))
    plt.title("Losses During Training")
    plt.xlabel("iterations")
    plt.ylabel("Loss")
    plt.plot(r_losses,label="Reconstruction", color="blue")
    plt.plot(k_losses, label="KLD", color="red")
    plt.legend()

    # Optimizers
    optimizer_encoder = torch.optim.Adam(params=encoder.parameters(), lr=opt.lr_a, betas=(opt.b1, opt.b2))
    optimizer_decoder = torch.optim.Adam(params=decoder.parameters(), lr=opt.lr_a, betas=(opt.b1, opt.b2))    

    # Load progress if it exists
    start_epoch = 0
    if os.path.isfile("Progress/epoch.txt"):
        start_epoch = int(np.loadtxt("Progress/epoch.txt")) + 1
        r_losses, k_losses = [x.tolist() for x in np.loadtxt("Progress/loss.csv", delimiter=",")]
        encoder.load_state_dict(torch.load("Models/Encoder"))
        decoder.load_state_dict(torch.load("Models/Decoder"))

    print("Training")
    for epoch in range(start_epoch, opt.num_epochs_autoencoder):
        for i, (imgs, _) in enumerate(dataloader):
            imgs = imgs.cuda()

            # --------------- Setup ----------------------

            # -------------- Training autoencoder ------------------
            optimizer_encoder.zero_grad()
            optimizer_decoder.zero_grad()

            mean, logvar = encoder(imgs)
            var =  torch.exp(logvar * 0.5)
            latent = generate_latent(mean, var)

            decoded_images = decoder(latent)

            r_loss, k_loss = vae_loss(mean, logvar, decoded_images, imgs)
            r_losses.append(r_loss.item())
            k_losses.append(k_loss.item())

            v_loss = r_loss + k_loss
            v_loss.backward()
            optimizer_encoder.step()
            optimizer_decoder.step()


            # ----------------------- Display results ------------------------
            print(
                "[Epoch %d/%d] [Batch %d/%d]" % (epoch + 1, opt.num_epochs_autoencoder, i + 1, len(dataloader)) +
                "\n\t [Reconstruction loss: %f]" % r_loss.item() +
                "\n\t [KLD            loss: %f]" % k_loss.item() + 
                "\n\t [Latent var, m: %f | %f]" % torch.var_mean(latent)
            )

            # Get the last batch
            if ((epoch + 1) % 5 == 0 and i == len(dataloader) - 1):
                # Generate images
                noise = generate_noise(len(imgs))
                sampled_images = decoder(noise)

                # Save noise->decoder images
                Utility.sample_images(decoded_images, epoch + 1, "Autoencoder")
                Utility.sample_images(imgs, str(epoch + 1) + "_real", "Autoencoder")
                Utility.sample_images(sampled_images, epoch + 1, "Sampled")

            # Plotting
            batch_num = epoch * len(dataloader) + i    
            if (batch_num + 1) % opt.plot_interval == 0:
                plt.plot(r_losses, color="blue")
                plt.plot(k_losses, color="red")
                plt.pause(1e-10)
    
        # ----------------------- Save the model ---------------------------
        torch.save(encoder.state_dict(), "Models/Encoder")
        torch.save(decoder.state_dict(), "Models/Decoder")
                
        np.savetxt("Progress/loss.csv", [r_losses, k_losses], delimiter=",", fmt="%f")
        np.savetxt("Progress/epoch.txt", [epoch], fmt="%d")
    

    print("Done")
    os.remove("Progress/epoch.txt")

    plt.savefig("Results/Autoencoder_Loss_Graph")
    plt.show()

if __name__ == '__main__':
    main()