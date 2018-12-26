import torch
import torch.nn as nn

#########################################################################
# Discriminator Code

class Discriminator(nn.Module):
    def __init__(self, args):
        super(Discriminator, self).__init__()
        self.nc = args.nc
        self.nz = args.nz
        self.ndf = args.ndf
        self.main = nn.Sequential(
            # input is (nc) x 256 x 256
            nn.Conv2d(self.nc, self.ndf, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf) x 128 x 128
            nn.Conv2d(self.ndf, self.ndf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(self.ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*2) x 64 x 64
            nn.Conv2d(self.ndf * 2, self.ndf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(self.ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*4) x 32 x 32
            nn.Conv2d(self.ndf * 4, self.ndf * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(self.ndf * 8),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*8) x 16 x 16
            nn.Conv2d(self.ndf * 8, self.ndf * 16, 4, 2, 1, bias=False),
            nn.BatchNorm2d(self.ndf * 16),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*16) x 8 x 8
            nn.Conv2d(self.ndf * 16, self.ndf * 32, 4, 2, 1, bias=False),
            nn.BatchNorm2d(self.ndf * 32),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*8) x 4 x 4
            nn.Conv2d(self.ndf * 32, 1, 4, 1, 0, bias=False),
            #nn.Sigmoid()
        )

    def forward(self, input):
        output = self.main(input)
        return output

#########################################################################
# Generator Code
        
class Generator(nn.Module):
    def __init__(self, args):
        super(Generator, self).__init__()
        self.nc = args.nc
        self.nz = args.nz
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
        c = x

        for i in range(1, 17, 1):
            o = self.residual_block(c)
            c = o + c
        
        c = self.batch_norm(c)
        c = self.relu(c)
        x = c + x
        x = self.conv_sq(x)

        return x