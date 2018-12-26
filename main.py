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
import time
from IPython.display import HTML
from network import Generator, Discriminator
from tensorboardX import SummaryWriter
from utils import *

parser = argparse.ArgumentParser()
parser.add_argument("--dataroot", type=str, default="/data2/zhousiyu/dataset/CUB_200_2011/images")
parser.add_argument("--num_class", type=int, default=200)
parser.add_argument("--log_dir", type=str, default="runs")
parser.add_argument("--workers", type=int, default=2)
parser.add_argument("--ngpu", type=int, default=1)
parser.add_argument("--batch_size", type=int, default=64)
parser.add_argument("--image_size", type=int, default=256)
parser.add_argument("--nc", type=int, default=3)
parser.add_argument("--nz", type=int, default=100)
parser.add_argument("--ndf", type=int, default=16)
parser.add_argument("--ngf", type=int, default=16)
parser.add_argument("--num_epochs", type=int, default=100)
parser.add_argument("--lr", type=float, default=0.0002)
parser.add_argument("--decay_begin_step", type=int, default=50)
parser.add_argument("--decay_step", type=int, default=5)
parser.add_argument("--decay_gama", type=float, default=0.9)
parser.add_argument("--beta1", type=float, default=0.5)
parser.add_argument("--gan_type", type=str, default="LogGAN")
parser.add_argument("--gp_weight", type=float, default=10.)
parser.add_argument("--tc_th", type=float, default=2.)
parser.add_argument("--manualSeed", type=int, default=999)
parser.add_argument("--truncnorm", type=bool, default=False)
parser.add_argument("--display_step", type=int, default=100)
parser.add_argument("--display_num", type=int, default=64)
args = parser.parse_args()

model_name = os.path.join(args.log_dir, args.gan_type + "_" + time.asctime(time.localtime(time.time())))

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
netG = Generator(args).to(device)
if (device.type == 'cuda') and (args.ngpu > 1):
    netG = nn.DataParallel(netG, list(range(args.ngpu)))
netG.apply(weights_init)
print(netG)


# Create the Discriminator
netD = Discriminator(args).to(device)
if (device.type == 'cuda') and (args.ngpu > 1):
    netD = nn.DataParallel(netD, list(range(args.ngpu)))
netD.apply(weights_init)
print(netD)


# Initialize BCELoss function
criterion = nn.BCELoss()

# Create batch of latent vectors that we will use to visualize
#  the progression of the generator
if args.truncnorm:
    fixed_noise = truncated_z_sample(args.display_num, args.nz + args.num_class, args.tc_th, args.manualSeed)
    print("Use Truncnorm", args.tc_th)
else:
    fixed_noise = np.random.randn(args.display_num, args.nz + args.num_class)
label = np.arange(args.display_num)
label_onehot = np.zeros((args.display_num, args.num_class))
label_onehot[np.arange(args.display_num), label[np.arange(args.display_num)]] = 1
fixed_noise[np.arange(args.display_num), :args.num_class] = label_onehot[np.arange(args.display_num)]
fixed_noise = torch.from_numpy(fixed_noise).float().to(device)



# Establish convention for real and fake labels during training
real_label = 1
fake_label = 0

# Setup Adam optimizers for both G and D
if args.gan_type == "WGAN":
    optimizerD = optim.SGD(netD.parameters(), lr = args.lr, momentum=0.9)
    optimizerG = optim.SGD(netG.parameters(), lr = args.lr, momentum=0.9)
else:
    optimizerD = optim.Adam(netD.parameters(), lr=args.lr, betas=(args.beta1, 0.999))
    optimizerG = optim.Adam(netG.parameters(), lr=args.lr, betas=(args.beta1, 0.999))

mt = [i for i in range(args.decay_begin_step, args.num_epochs, args.decay_step)]

schedulerD = optim.lr_scheduler.MultiStepLR(optimizerD, milestones=mt, gamma=args.decay_gama)
schedulerG = optim.lr_scheduler.MultiStepLR(optimizerG, milestones=mt, gamma=args.decay_gama)


# Training Loop

# Lists to keep track of progress
img_list = []
G_losses = []
D_losses = []
iters = 0

writer = SummaryWriter(model_name)
print("Starting Training Loop...")
# For each epoch
for epoch in range(args.num_epochs):
    # For each batch in the dataloader
    schedulerD.step()
    schedulerG.step()
    for i, data in enumerate(dataloader, 0):
        
        if args.gan_type == "WGAN":
            loss_d, loss_g, real_dis, fake_dis, real_aux_loss, fake_aux_loss = wgan_with_gp(data, netD, netG, optimizerD, optimizerG, device, args)
        elif args.gan_type == "LogGAN" or args.gan_type == "MseGAN":
            loss_d, loss_g, real_dis, fake_dis, real_aux_loss, fake_aux_loss = gan(data, netD, netG, optimizerD, optimizerG, device, args)
        
        # Output training stats
        if i % 1 == 0:
            print('[%d/%d][%d/%d]\tLoss_D: %.4f\tLoss_G: %.4f\tdis: %.4f(%.4f)\taux_loss: %.4f(%.4f)'
                  % (epoch, args.num_epochs, i, len(dataloader),
                     loss_d.item(), loss_g.item(), real_dis.mean().item(), fake_dis.mean().item(), real_aux_loss.item(), fake_aux_loss.item()))
        
        
        # Check how the generator is doing by saving G's output on fixed_noise
        if (iters % args.display_step == 0) or ((epoch == args.num_epochs-1) and (i == len(dataloader)-1)):
            fake = netG(fixed_noise)
            vis_fake = (fake + 1) / 2
            writer.add_image("fake", vis_fake, iters)
            
        iters += 1

writer.close()

