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
from utils import *
from data_loader import customData

############################
    # set parameter
############################
parser = argparse.ArgumentParser()
parser.add_argument("--dataset", type=str, default="/data2/zhousiyu/dataset/CUB_200_2011/zeroshot")
parser.add_argument("--image_dir", type=str, default="/data2/zhousiyu/dataset/CUB_200_2011/images")
parser.add_argument("--log_dir", type=str, default="runs")
parser.add_argument("--show_num", type=int, default=64)
parser.add_argument("--workers", type=int, default=2)
parser.add_argument("--ngpu", type=int, default=1)
parser.add_argument("--batch_size", type=int, default=64)
parser.add_argument("--image_size", type=int, default=224)
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
parser.add_argument("--gan_weight", type=float, default=1)
parser.add_argument("--L2_weight", type=float, default=100)
args = parser.parse_args()

############################
    # init model_name
############################
model_name = os.path.join(args.log_dir, args.gan_type + '_' + time.asctime( time.localtime(time.time()) ))

############################
    # set random_seed
############################
print("Random Seed: ", args.manualSeed)
random.seed(args.manualSeed)
torch.manual_seed(args.manualSeed)

############################
    # set GPU option
############################
device = torch.device("cuda:0" if (torch.cuda.is_available() and args.ngpu > 0) else "cpu")

############################
    # init file_path
############################
att_path = os.path.join(args.dataset, 'class_attribute_labels_continuous.txt')
train_img_path = os.path.join(args.dataset, 'train.txt')
train_cls_path = os.path.join(args.dataset, 'train_classes.txt')
testR_img_path = os.path.join(args.dataset, 'testRecg.txt')
testZ_img_path = os.path.join(args.dataset, 'testZS.txt')
test_cls_path = os.path.join(args.dataset, 'test_classes.txt')

############################
    # init attributes
############################
# load data
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
if att_dict.max() > 1.:
    att_dict /= 100.
att_mean = att_dict[train_cls_dict, :].mean(axis=0)
for i in range(att_size):
    att_dict[att_dict[:, i] < 0, i] = att_mean[i]
for i in range(att_size):
    att_dict[:, i] = att_dict[:, i] - att_mean[i] + 0.5
train_att_dict = att_dict[train_cls_dict, :]
test_att_dict = att_dict[test_cls_dict, :]

############################
    # set dataLoader
############################
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

############################
    # init network
############################
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
print(netD)

############################
    # set loss
############################
if args.gan_type == "LogGAN":
    dis_criterion = nn.BCELoss()
else:
    dis_criterion = nn.MSELoss()
att_criterion = nn.CrossEntropyLoss()
mse_criterion = nn.MSELoss()
dis_criterion = dis_criterion.to(device)
att_criterion = att_criterion.to(device)
mse_criterion = mse_criterion.to(device)

############################
    # set fixed noise for test
############################
if args.truncnorm:
    fixed_noise = truncated_z_sample(num_class, args.nz + att_size, args.tc_th, args.manualSeed)
else:
    fixed_noise = np.random.normal(0, 1, (num_class, args.nz + att_size))
fixed_noise[np.arange(num_class), :att_size] = att_dict[np.arange(num_class)]
fixed_noise = torch.from_numpy(fixed_noise).float().to(device)


############################
    # set optimzer
############################
optimizerD = optim.Adam(netD.parameters(), lr=args.lr, betas=(args.beta1, 0.999))
optimizerG = optim.Adam(netG.parameters(), lr=args.lr, betas=(args.beta1, 0.999))
mt = [i for i in range(args.decay_begin_step, args.num_epochs, args.decay_step)]
schedulerD = optim.lr_scheduler.MultiStepLR(optimizerD, milestones=mt, gamma=args.decay_gama)
schedulerG = optim.lr_scheduler.MultiStepLR(optimizerG, milestones=mt, gamma=args.decay_gama)

############################
    # train_loop
############################
iters = 0
writer = SummaryWriter(model_name)
print("Starting Training Loop...")
# For each epoch
for epoch in range(args.num_epochs):
    # For each batch in the dataloader
    schedulerD.step()
    schedulerG.step()
    train_ac = 0
    for i, data in enumerate(trainloader, 0):
        ############################
        # (1) Update D network: maximize log(D(x)) + log(1 - D(G(z)))
        ###########################
        ## Train with all-real batch
        netD.zero_grad()
        real, train_att_label, att_label = data
        real =real.to(device)
        att_label = att_label.to(device)
        train_att_label = train_att_label.to(device)
        b_size = real.size(0)
        dis_label = torch.full((b_size,), 1, device=device)

        att, dis = netD(real)
        dis = dis.view(-1)
        if args.gan_type == "LogGAN":
            dis = F.sigmoid(dis)
        similarity = torch.mm(att, torch.from_numpy(train_att_dict).float().to(device).t())
        errD_real = args.gan_weight * dis_criterion(dis, dis_label) + att_criterion(similarity, train_att_label)
        dis_real = dis
        att_real = att_criterion(similarity, train_att_label)
        errD_real.backward()

        predict = torch.argmax(similarity, 1)
        train_ac += (predict == train_att_label).sum().item()

        ## Train with all-fake batch
        # Generate batch of latent vectors
        # prepare latent vectors
        if args.truncnorm:
            noise = truncated_z_sample(b_size, args.nz + att_size, args.tc_th, args.manualSeed)
        else:
            noise = np.random.normal(0, 1, (b_size, args.nz + att_size)) 
        att_d = att.detach()   
        noise[np.arange(b_size), :att_size] = att_d[np.arange(b_size)]
        noise = torch.from_numpy(noise).float().to(device)
        # feed in network
        fake = netG(noise)
        dis_label.fill_(0)
        att, dis = netD(fake.detach())
        dis = dis.view(-1)
        if args.gan_type == "LogGAN":
            dis = F.sigmoid(dis)
        similarity = torch.mm(att, torch.from_numpy(train_att_dict).float().to(device).t())
        errD_fake = args.gan_weight * dis_criterion(dis, dis_label) + att_criterion(similarity, train_att_label)
        errD_fake.backward()

        loss_d = (errD_real + errD_fake) / 2
        optimizerD.step()

        ############################
        # (2) Update G network: maximize log(D(G(z)))
        ###########################
        netG.zero_grad()
        dis_label.fill_(1)
        att, dis = netD(fake) 
        dis = dis.view(-1)
        if args.gan_type == "LogGAN":
            dis = F.sigmoid(dis)
        similarity = torch.mm(att, torch.from_numpy(train_att_dict).float().to(device).t())
        errG = args.gan_weight * dis_criterion(dis, dis_label) + att_criterion(similarity, train_att_label) + args.L2_weight * mse_criterion(fake, real)
        dis_fake = dis
        att_fake = att_criterion(similarity, train_att_label)
        errG.backward()
        loss_g = errG
        optimizerG.step()
        
        ############################
            # print loss
        ############################ 
        if iters % 1 == 0:
            print('[%d/%d][%d/%d]\tLoss_D: %.4f\tLoss_G: %.4f\tDIS: %.4f(%.4f)\tATT: %.4f(%.4f)'
                  % (epoch, args.num_epochs, i, len(trainloader),
                     loss_d.item(), loss_g.item(), dis_real.mean().item(), dis_fake.mean().item(), att_real.mean().item(), att_fake.mean().item()))

        ############################
            # tensorboard summary
        ############################    
        if iters % 100 == 0:  
            vis_fake = (fake + 1) / 2
            vis_real = (real + 1) / 2
            writer.add_image("fake", vis_fake, iters)
            writer.add_image("real", vis_real, iters)
            writer.add_scalar("loss_d", loss_d, iters)
            writer.add_scalar("loss_g", loss_g, iters)
            writer.add_scalar("dis_real", dis_real.mean(), iters)
            writer.add_scalar("dis_fake", dis_fake.mean(), iters)
            writer.add_scalar("att_real", att_real.mean(), iters)
            writer.add_scalar("att_fake", att_fake.mean(), iters)
        iters += 1
    train_ac = train_ac / len(trainloader.dataset)

    ############################
        # testR_loop
    ############################
    testR_ac = 0
    for i, data in enumerate(testRloader, 0):
        netD.zero_grad()
        real, testR_att_label, att_label = data
        real =real.to(device)
        testR_att_label = testR_att_label.to(device)
        b_size = real.size(0)

        att, dis = netD(real)
        similarity = torch.mm(att, torch.from_numpy(train_att_dict).float().to(device).t())

        predict = torch.argmax(similarity, 1)
        testR_ac += (predict == testR_att_label).sum().item()
    testR_ac = testR_ac / len(testRloader.dataset)

    ############################
        # testZ_loop
    ############################
    testZ_ac = 0
    for i, data in enumerate(testZloader, 0):
        netD.zero_grad()
        real, testZ_att_label, att_label = data
        real =real.to(device)
        testZ_att_label = testZ_att_label.to(device)
        b_size = real.size(0)

        att, dis = netD(real)
        similarity = torch.mm(att, torch.from_numpy(test_att_dict).float().to(device).t())

        predict = torch.argmax(similarity, 1)
        testZ_ac += (predict == testZ_att_label).sum().item()
    testZ_ac = testZ_ac / len(testZloader.dataset)

    print('[%d/%d]\ttrain_ac: %.4f\ttestR_ac: %.4f\ttestZ_ac: %.4f'
                  % (epoch, args.num_epochs,
                     train_ac, testR_ac, testZ_ac))
    writer.add_scalar("train_ac", train_ac, epoch)
    writer.add_scalar("testR_ac", testR_ac, epoch)
    writer.add_scalar("testZ_ac", testZ_ac, epoch)
    
writer.close()

