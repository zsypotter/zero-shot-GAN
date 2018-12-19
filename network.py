import torch
import torch.nn as nn
from torchvision import models

#########################################################################
# Discriminator Code

class Discriminator(nn.Module):
    def __init__(self, args, att_size):
        super(Discriminator, self).__init__()
        self.VGG = models.vgg19(pretrained=True)
        self.dis_linear = nn.Linear(4096, 1)
        self.att_linear = nn.Linear(4096, att_size, bias=False)

    def forward(self, x):
        x = self.VGG.features(x)
        x = x.view(x.size(0), -1)
        x = self.VGG.classifier[0](x)
        x = self.VGG.classifier[1](x)
        x = self.VGG.classifier[2](x)
        x = self.VGG.classifier[3](x)
        x = self.VGG.classifier[4](x) 
        dis = self.dis_linear(x)
        att = self.att_linear(x)

        return att, dis

        

#########################################################################
# Generator Code
        
class Generator(nn.Module):
    def __init__(self, args, att_size):
        super(Generator, self).__init__()
        self.ngpu = args.ngpu
        self.nc = args.nc
        self.nz = args.nz + att_size
        self.ndf = args.ndf
        self.ngf = args.ngf
        self.init_size = args.image_size // 8

        self.linear = nn.Linear(self.nz, self.init_size * self.init_size * 64)
        self.residual_block = nn.Sequential(
            nn.Conv2d(64, 64, 3, 1, 1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            nn.Conv2d(64, 64, 3, 1, 1, bias=False),
            nn.BatchNorm2d(64)
        ) 
        self.batch_norm = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(True)
        self.conv_sq = nn.Sequential(
            nn.Conv2d(64, 256, 3, 1, 1),
            nn.PixelShuffle(2),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            nn.Conv2d(64, 256, 3, 1, 1),
            nn.PixelShuffle(2),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            nn.Conv2d(64, 256, 3, 1, 1),
            nn.PixelShuffle(2),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            nn.Conv2d(64, 3, 9, 1, 4),
            nn.Tanh()
        )


    def forward(self, input):
        x = input.view(-1, self.nz)
        x = self.linear(x)
        x = x.view(-1, 64, self.init_size, self.init_size)
        
        # resblock part
        c = x
        for i in range(1, 17, 1):
            o = self.residual_block(c)
            c = o + c
        c = self.batch_norm(c)
        c = self.relu(c)
        x = c + x

        # upsample part
        x = self.conv_sq(x)

        return x