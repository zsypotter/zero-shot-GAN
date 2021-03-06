from __future__ import print_function
import argparse
import os
import random
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from IPython.display import HTML
from network import Generator, Discriminator
from tensorboardX import SummaryWriter
from utils import *

parser = argparse.ArgumentParser()
parser.add_argument("--dataroot", type=str, default="/data2/zhousiyu/dataset/Animals_with_Attributes2/JPEGImages")
parser.add_argument("--workers", type=int, default=2)
parser.add_argument("--ngpu", type=int, default=1)
parser.add_argument("--batch_size", type=int, default=128)
parser.add_argument("--image_size", type=int, default=64)
parser.add_argument("--nc", type=int, default=3)
parser.add_argument("--nz", type=int, default=100)
parser.add_argument("--ndf", type=int, default=64)
parser.add_argument("--ngf", type=int, default=64)
parser.add_argument("--num_epochs", type=int, default=100)
parser.add_argument("--lr", type=float, default=0.0002)
parser.add_argument("--beta1", type=float, default=0.5)
parser.add_argument("--gan_type", type=str, default="LogGAN")
parser.add_argument("--gp_weight", type=float, default=10.)
parser.add_argument("--tc_th", type=float, default=2.)
parser.add_argument("--manualSeed", type=int, default=999)
parser.add_argument("--truncnorm", type=bool, default=False)
args = parser.parse_args()

#manualSeed = random.randint(1, 10000) # use if you want new results
print("Random Seed: ", args.manualSeed)
random.seed(args.manualSeed)
torch.manual_seed(args.manualSeed)

dataset = dset.ImageFolder(root=args.dataroot,
                           transform=transforms.Compose([
                               transforms.Resize(args.image_size),
                               transforms.CenterCrop(args.image_size),
                               transforms.ToTensor(),
                               transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                           ]))
# Create the dataloader
dataloader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size,
                                         shuffle=True, num_workers=args.workers)

# Decide which device we want to run on
device = torch.device("cuda:0" if (torch.cuda.is_available() and args.ngpu > 0) else "cpu")

# Create the generator
netG = Generator(args.ngpu, args.nc, args.nz, args.ndf, args.ngf).to(device)
if (device.type == 'cuda') and (args.ngpu > 1):
    netG = nn.DataParallel(netG, list(range(args.ngpu)))
netG.apply(weights_init)
print(netG)


# Create the Discriminator
netD = Discriminator(args.ngpu, args.nc, args.nz, args.ndf, args.ngf).to(device)
if (device.type == 'cuda') and (args.ngpu > 1):
    netD = nn.DataParallel(netD, list(range(args.ngpu)))
netD.apply(weights_init)
print(netD)


# Initialize BCELoss function
criterion = nn.BCELoss()

# Create batch of latent vectors that we will use to visualize
#  the progression of the generator
if args.truncnorm:
    fixed_noise = torch.from_numpy(truncated_z_sample(64, args.nz, args.tc_th, args.manualSeed)).float().to(device)
else:
    fixed_noise = torch.randn(64, args.nz, 1, 1, device=device)

# Establish convention for real and fake labels during training
real_label = 1
fake_label = 0

# Setup Adam optimizers for both G and D
optimizerD = optim.Adam(netD.parameters(), lr=args.lr, betas=(args.beta1, 0.999))
optimizerG = optim.Adam(netG.parameters(), lr=args.lr, betas=(args.beta1, 0.999))


# Training Loop

# Lists to keep track of progress
img_list = []
G_losses = []
D_losses = []
iters = 0

writer = SummaryWriter()
print("Starting Training Loop...")
# For each epoch
for epoch in range(args.num_epochs):
    # For each batch in the dataloader
    for i, data in enumerate(dataloader, 0):
        
        if args.gan_type == "WGAN":
            loss_d, loss_g = wgan_with_gp(data, netD, netG, optimizerD, optimizerG, device, args)
        elif args.gan_type == "LogGAN" or args.gan_type == "MseGAN":
            loss_d, loss_g = gan(data, netD, netG, optimizerD, optimizerG, device, args)
        
        # Output training stats
        if i % 1 == 0:
            print('[%d/%d][%d/%d]\tLoss_D: %.4f\tLoss_G: %.4f\t'
                  % (epoch, args.num_epochs, i, len(dataloader),
                     loss_d.item(), loss_g.item()))
        
        
        # Check how the generator is doing by saving G's output on fixed_noise
        if (iters % 1 == 0) or ((epoch == num_epochs-1) and (i == len(dataloader)-1)):
            fake = netG(fixed_noise)
            real = data[0].to(device)
            vis_fake = (fake + 1) / 2
            vis_real = (real + 1) / 2
            writer.add_image("fake", vis_fake, iters)
            writer.add_image("real", vis_real[0:64], iters)
            
        iters += 1

writer.close()

