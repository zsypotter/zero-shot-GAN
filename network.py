import torch
import torch.nn as nn

#########################################################################
# Discriminator Code

class Discriminator(nn.Module):
    def __init__(self, ngpu, nc, nz, ndf, ngf):
        super(Discriminator, self).__init__()
        self.ngpu = ngpu
        self.nc = nc
        self.nz = nz
        self.ndf = ndf
        self.ngf = ngf
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
        return self.main(input)

#########################################################################
# Generator Code
        
class Generator(nn.Module):
    def __init__(self, ngpu, nc, nz, ndf, ngf):
        super(Generator, self).__init__()
        self.ngpu = ngpu
        self.nc = nc
        self.nz = nz
        self.ndf = ndf
        self.ngf = ngf
        self.main = nn.Sequential(
            # input is Z, going into a convolution
            nn.ConvTranspose2d( self.nz, self.ngf * 32, 4, 1, 0, bias=False),
            nn.BatchNorm2d(self.ngf * 32),
            nn.ReLU(True),
            # state size. (ngf*32) x 4 x 4
            nn.ConvTranspose2d(self.ngf * 32, self.ngf * 16, 4, 2, 1, bias=False),
            nn.BatchNorm2d(self.ngf * 16),
            nn.ReLU(True),
            # state size. (ngf*16) x 8 x 8
            nn.ConvTranspose2d( self.ngf * 16, self.ngf * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 8),
            nn.ReLU(True),
            # state size. (ngf*8) x 16 x 16
            nn.ConvTranspose2d( self.ngf * 8, self.ngf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 4),
            nn.ReLU(True),
            # state size. (ngf*4) x 32 x 32
            nn.ConvTranspose2d( self.ngf * 4, self.ngf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(True),
            # state size. (ngf*2) x 64 x 64
            nn.ConvTranspose2d( self.ngf * 2, self.ngf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf),
            nn.ReLU(True),
            # state size. (ngf) x 128 x 128
            nn.ConvTranspose2d( self.ngf, self.nc, 4, 2, 1, bias=False),
            nn.Tanh()
            # state size. (nc) x 256 x 256
        )

    def forward(self, input):
        return self.main(input)