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

# Set random seem for reproducibility
manualSeed = 999
#manualSeed = random.randint(1, 10000) # use if you want new results
print("Random Seed: ", manualSeed)
random.seed(manualSeed)
torch.manual_seed(manualSeed)


dataroot = "/data2/zhousiyu/dataset/Animals_with_Attributes2/JPEGImages"
workers = 2
batch_size = 128
image_size = 64
nc = 3
nz = 100
ngf = 64
ndf = 64
num_epochs = 100
lr = 0.0002
beta1 = 0.5
ngpu = 1
gp_weight = 10

dataset = dset.ImageFolder(root=dataroot,
                           transform=transforms.Compose([
                               transforms.Resize(image_size),
                               transforms.CenterCrop(image_size),
                               transforms.ToTensor(),
                               transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                           ]))
# Create the dataloader
dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size,
                                         shuffle=True, num_workers=workers)

# Decide which device we want to run on
device = torch.device("cuda:0" if (torch.cuda.is_available() and ngpu > 0) else "cpu")


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)

# Create the generator
netG = Generator(ngpu, nc, nz, ndf, ngf).to(device)
if (device.type == 'cuda') and (ngpu > 1):
    netG = nn.DataParallel(netG, list(range(ngpu)))
netG.apply(weights_init)
print(netG)


# Create the Discriminator
netD = Discriminator(ngpu, nc, nz, ndf, ngf).to(device)
if (device.type == 'cuda') and (ngpu > 1):
    netD = nn.DataParallel(netD, list(range(ngpu)))
netD.apply(weights_init)
print(netD)


# Initialize BCELoss function
criterion = nn.BCELoss()

# Create batch of latent vectors that we will use to visualize
#  the progression of the generator
fixed_noise = torch.randn(64, nz, 1, 1, device=device)

# Establish convention for real and fake labels during training
real_label = 1
fake_label = 0

# Setup Adam optimizers for both G and D
optimizerD = optim.Adam(netD.parameters(), lr=lr, betas=(beta1, 0.999))
optimizerG = optim.Adam(netG.parameters(), lr=lr, betas=(beta1, 0.999))


# Training Loop

# Lists to keep track of progress
img_list = []
G_losses = []
D_losses = []
iters = 0

writer = SummaryWriter()
print("Starting Training Loop...")
# For each epoch
for epoch in range(num_epochs):
    # For each batch in the dataloader
    for i, data in enumerate(dataloader, 0):
        
        ############################
        # (1) Update D network: maximize log(D(x)) + log(1 - D(G(z)))
        ###########################
        netD.zero_grad()

        real = data[0].to(device)
        b_size = real.size(0)
        read_d = netD(real)

        noise = torch.randn(b_size, nz, 1, 1, device=device)
        fake = netG(noise).detach()
        fake_d = netD(fake)

        epsilon = torch.rand(b_size, 1, 1, 1).to(device)
        interpolates = torch.tensor((epsilon * real + (1 - epsilon) * fake).data, requires_grad=True)
        gradients = torch.autograd.grad(
            netD(interpolates).view(b_size),
            interpolates,
            grad_outputs=torch.ones(b_size).to(device),
            create_graph=True)[0]
        gp = ((gradients.view(b_size, -1).norm(2, dim=1) - 1).pow(2)).mean()

        loss_d_without_gp = (-read_d + fake_d).mean()
        loss_d = loss_d_without_gp + gp_weight * gp
        loss_d.backward()
        optimizerD.step()


        ############################
        # (2) Update G network: maximize log(D(G(z)))
        ###########################
        netG.zero_grad()

        noise = torch.randn(b_size, nz, 1, 1, device=device)
        fake = netG(noise)
        fake_d = netD(fake).mean()

        loss_g = -fake_d
        loss_g.backward()
        optimizerG.step()

        
        # Output training stats
        if i % 1 == 0:
            print('[%d/%d][%d/%d]\tLoss_D: %.4f\tLoss_G: %.4f\tLoss_D_Without_GP: %.4f'
                  % (epoch, num_epochs, i, len(dataloader),
                     loss_d.item(), loss_g.item(), loss_d_without_gp.item()))
        
        
        # Check how the generator is doing by saving G's output on fixed_noise
        if (iters % 1 == 0) or ((epoch == num_epochs-1) and (i == len(dataloader)-1)):
            fake = netG(fixed_noise)
            vis_fake = (fake + 1) / 2
            vis_real = (real + 1) / 2
            writer.add_image("fake", vis_fake, iters)
            writer.add_image("real", vis_real[0:64], iters)
            
        iters += 1

writer.close()

