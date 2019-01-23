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
import time
from network import Generator, Discriminator
from tensorboardX import SummaryWriter
from data_loader import customData
from utils import *

parser = argparse.ArgumentParser()
parser.add_argument("--dataset", type=str, default="/home/disk1/zhousiyu/dataset/CUB_200_2011/zeroshot")
parser.add_argument("--image_dir", type=str, default="/home/disk1/zhousiyu/dataset/CUB_200_2011/images")
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
parser.add_argument("--num_epochs", type=int, default=10000)
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
parser.add_argument("--process_att", type=bool, default=False)
args = parser.parse_args()

model_name = os.path.join(args.log_dir, args.gan_type + "_" + time.asctime(time.localtime(time.time())))

#manualSeed = random.randint(1, 10000) # use if you want new results
print("Random Seed: ", args.manualSeed)
random.seed(args.manualSeed)
torch.manual_seed(args.manualSeed)

# set GPU option
device = torch.device("cuda:0" if (torch.cuda.is_available() and args.ngpu > 0) else "cpu")

# init file_path
att_path = os.path.join(args.dataset, 'class_attribute_labels_continuous.txt')
train_img_path = os.path.join(args.dataset, 'train.txt')
train_cls_path = os.path.join(args.dataset, 'train_classes.txt')
testR_img_path = os.path.join(args.dataset, 'testRecg.txt')
testZ_img_path = os.path.join(args.dataset, 'testZS.txt')
test_cls_path = os.path.join(args.dataset, 'test_classes.txt')

# init attributes
train_cls_file = open(train_cls_path)
test_cls_file = open(test_cls_path)
lines = train_cls_file.readlines()
train_cls_dict = [(int(line.split(' ')[0]) - 1) for line in lines]
lines = test_cls_file.readlines()
test_cls_dict = [(int(line.split(' ')[0]) - 1) for line in lines]
att_dict = np.loadtxt(att_path)
train_class_num = len(train_cls_dict)
test_class_num = len(test_cls_dict)
num_class, att_size = att_dict.shape

# modify att value
if args.process_att:
    if att_dict.max() > 1.:
        att_dict /= 100.
    att_mean = att_dict[train_cls_dict, :].mean(axis=0)
    for i in range(att_size):
        att_dict[att_dict[:, i] < 0, i] = att_mean[i]
    for i in range(att_size):
        att_dict[:, i] = att_dict[:, i] - att_mean[i] + 0.5
train_att_dict = att_dict[train_cls_dict, :]
test_att_dict = att_dict[test_cls_dict, :]

# set dataLoader
# set transform
data_transforms = transforms.Compose([
                    transforms.Resize(args.image_size),
                    transforms.CenterCrop(args.image_size),
                    transforms.ToTensor(),
                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                ])  

# set data_loader
trainset = customData(args.image_dir, train_img_path, train_cls_path, train_class_num, data_transforms)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size, shuffle=True, num_workers=2)
testRset = customData(args.image_dir, testR_img_path, train_cls_path, train_class_num, data_transforms)
testRloader = torch.utils.data.DataLoader(testRset, batch_size=1, shuffle=True, num_workers=2)
testZset = customData(args.image_dir, testZ_img_path, test_cls_path, test_class_num, data_transforms)
testZloader = torch.utils.data.DataLoader(testZset, batch_size=1, shuffle=True, num_workers=2)

# Create the generator
netG = Generator(args, att_size).to(device)
if (device.type == 'cuda') and (args.ngpu > 1):
    netG = nn.DataParallel(netG, list(range(args.ngpu)))
netG.apply(weights_init)
print(netG)


# Create the Discriminator
netD = Discriminator(args, att_size).to(device)
if (device.type == 'cuda') and (args.ngpu > 1):
    netD = nn.DataParallel(netD, list(range(args.ngpu)))
netD.apply(weights_init)
print(netD)

# Create batch of latent vectors that we will use to visualize
#  the progression of the generator
if args.truncnorm:
    fixed_noise = truncated_z_sample(args.display_num, args.nz + att_size, args.tc_th, args.manualSeed)
    print("Use Truncnorm", args.tc_th)
else:
    fixed_noise = np.random.randn(args.display_num, args.nz + att_size)
fixed_noise[np.arange(args.display_num), :att_size] = att_dict[np.arange(args.display_num), :att_size]
fixed_noise = torch.from_numpy(fixed_noise).float().to(device)


# Setup Adam optimizers for both G and D
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
    for i, data in enumerate(trainloader, 0):
        
        if args.gan_type == "WGAN":
            loss_d, loss_g, real_dis, fake_dis, real_aux_loss, fake_aux_loss = wgan_with_gp(data, netD, netG, optimizerD, optimizerG, device, args, att_size)
        elif args.gan_type == "LogGAN" or args.gan_type == "MseGAN":
            loss_d, loss_g, real_dis, fake_dis, real_aux_loss, fake_aux_loss = gan(data, netD, netG, optimizerD, optimizerG, device, args, att_size, train_att_dict, test_att_dict, att_dict)
        
        # Output training stats
        if i % 1 == 0:
            print('[%d/%d][%d/%d]\tLoss_D: %.4f\tLoss_G: %.4f\tdis: %.4f(%.4f)\taux_loss: %.4f(%.4f)'
                  % (epoch, args.num_epochs, i, len(trainloader),
                     loss_d.item(), loss_g.item(), real_dis.mean().item(), fake_dis.mean().item(), real_aux_loss.item(), fake_aux_loss.item()))
        
        
        # Check how the generator is doing by saving G's output on fixed_noise
        if (iters % args.display_step == 0) or ((epoch == args.num_epochs-1) and (i == len(trainloader)-1)):
            fake = netG(fixed_noise)
            vis_fake = (fake + 1) / 2
            writer.add_image("fake", vis_fake, iters)
            
        iters += 1

writer.close()

