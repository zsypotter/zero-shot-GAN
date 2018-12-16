import torch
import torch.nn as nn

#########################################################################
# Discriminator Code

class Discriminator(nn.Module):
    def __init__(self, args, att_size):
        super(Discriminator, self).__init__()
        self.ngpu = args.ngpu
        self.nc = args.nc
        self.nz = args.nz
        self.ndf = args.ndf
        self.ngf = args.ngf
        self.att_size = att_size
        self.last_size = args.image_size // 32

        '''self.lrelu = nn.LeakyReLU(0.2, inplace=True)
        self.conv3_32 = nn.Conv2d(3, 32, 3, 2, 1, bias=False)
        self.block32 = nn.Sequential(
            nn.Conv2d(32, 32, 3, 1, 1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(32, 32, 3, 1, 1),
        )
        self.conv32_64 = nn.Conv2d(32, 64, 3, 2, 1, bias=False)
        self.block64 = nn.Sequential(
            nn.Conv2d(64, 64, 3, 1, 1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, 64, 3, 1, 1),
        )
        self.conv64_128 = nn.Conv2d(64, 128, 3, 2, 1, bias=False)
        self.block128 = nn.Sequential(
            nn.Conv2d(128, 128, 3, 1, 1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(128, 128, 3, 1, 1),
        )
        self.conv128_256 = nn.Conv2d(128, 256, 3, 2, 1, bias=False)
        self.block256 = nn.Sequential(
            nn.Conv2d(256, 256, 3, 1, 1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(256, 256, 3, 1, 1),
        )
        self.conv256_512 = nn.Conv2d(256, 512, 3, 2, 1, bias=False)
        self.block512 = nn.Sequential(
            nn.Conv2d(512, 512, 3, 1, 1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(512, 512, 3, 1, 1),
        )

        self.att_linear = nn.Linear(512 * self.last_size * self.last_size, self.att_size, bias=False)
        self.dis_linear = nn.Linear(512 * self.last_size * self.last_size, 1)'''
        self.feature = nn.Sequential(
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
            #nn.Conv2d(self.ndf * 32, 1, 4, 1, 0, bias=False),
            #nn.Sigmoid()
        )

        self.att_linear = nn.Linear(32 * self.ndf * 4 * 4, self.att_size, bias=False)
        self.dis_linear = nn.Linear(32 * self.ndf * 4 * 4, 1)

    def forward(self, input):
        '''# resblock part
        # 3 X 256 X 256
        c = self.conv3_32(input)
        c = self.lrelu(c)
        for i in range(1, 2, 1):
            o = self.block32(c)
            c = o + c
            c = self.lrelu(c)
        # 32 X 128 X 128
        c = self.conv32_64(c)
        c = self.lrelu(c)
        for i in range(1, 4, 1):
            o = self.block64(c)
            c = o + c
            c = self.lrelu(c)
        # 64 X 64 X 64
        c = self.conv64_128(c)
        c = self.lrelu(c)
        for i in range(1, 4, 1):
            o = self.block128(c)
            c = o + c
            c = self.lrelu(c)
        # 128 X 32 X32
        c = self.conv128_256(c)
        c = self.lrelu(c)
        for i in range(1, 4, 1):
            o = self.block256(c)
            c = o + c
            c = self.lrelu(c)
        # 256 X 16 X 16
        c = self.conv256_512(c)
        c = self.lrelu(c)
        for i in range(1, 4, 1):
            o = self.block512(c)
            c = o + c
            c = self.lrelu(c)
        # 512 X 8 X 8

        # mapping
        feature = c.view(-1, 512 * self.last_size * self.last_size)
        att = self.att_linear(feature)
        dis = self.dis_linear(feature)'''
        feature = self.feature(input).view(-1, 32 * self.ndf * 4 * 4)
        att = self.att_linear(feature)
        dis = self.dis_linear(feature)
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