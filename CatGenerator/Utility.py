import argparse
from torchvision.utils import save_image

def get_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_epochs_autoencoder", type=int, default=50, help="number of epochs of training for the autoencoder")
    parser.add_argument("--num_epochs_gan", type=int, default=200, help="number of epochs of training for the gan")
    parser.add_argument("--batch_size", type=int, default=128, help="size of the batches")
    parser.add_argument("--image_size", type=int, default=256, help="size of each image dimension")
    parser.add_argument("--lr_a", type=float, default=1e-4, help="adam: learning rate for autoencoder")
    parser.add_argument("--lr_g", type=float, default=1e-4, help="adam: learning rate for generator")
    parser.add_argument("--lr_d", type=float, default=1e-6, help="adam: learning rate for discriminator")
    parser.add_argument("--b1", type=float, default=0.5, help="adam: decay of first order momentum of gradient")
    parser.add_argument("--b2", type=float, default=0.999, help="adam: decay of first order momentum of gradient")
    parser.add_argument("--num_gpus", type=int, default=1, help="number of gpu threads to use")
    parser.add_argument("--latent_size", type=int, default=64, help="size of the latent vector i.e. size of decoder input")
    parser.add_argument("--generator_size", type=int, default=128, help="size of the noise vector for the generator")
    parser.add_argument("--num_channels", type=int, default=3, help="number of image channels")
    parser.add_argument("--sample_interval", type=int, default=100, help="interval between image sampling")
    parser.add_argument("--plot_interval", type=int, default=100, help="interval between datapoint plotting")
    parser.add_argument("--workers", type=int, default=2, help="number of dataloader workers")
    parser.add_argument("--persp1", type=float, default=0.2, help="perspective distortion")
    parser.add_argument("--persp2", type=float, default=0.5, help="perspective chance")
    parser.add_argument("--num_autoencoder_layers", type=int, default=4, help="number of conv layers in the autoencoder")
    parser.add_argument("--num_generator_layers", type=int, default=4, help="number of dense layers in the generator")
    parser.add_argument("--num_discriminator_layers", type=int, default=4, help="number of conv layers in the discriminator")
    parser.add_argument("--min_feature_size", type=int, default=16, help="Smallest feature count for the conv layers (default 4)")
    parser.add_argument("--test_generator", type=bool, default=True, help="Whether to run the handwriting dataset for testing")
    opt = parser.parse_args()
    return opt

def sample_images(imgs, name, path):
    save_image(imgs.data[:25], "Results/" + path + "_Images/%s.png" % name, nrow=5, normalize=True)
    
