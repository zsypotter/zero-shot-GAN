import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from scipy.stats import truncnorm

def truncated_z_sample(batch_size, dim_z, threshold=2, seed=None, truncation=1.):
    state = None if seed is None else np.random.RandomState(seed)
    values = truncnorm.rvs(-threshold, threshold, size=(batch_size, dim_z, 1, 1), random_state=state)
    return truncation * values

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)

def wgan_with_gp(data, netD, netG, optimizerD, optimizerG, device, args):
    ############################
    # (1) Update D network
    ###########################
    netD.zero_grad()

    real = data[0].to(device)
    b_size = real.size(0)
    real_d = netD(real)
    if args.truncnorm:
        noise = torch.from_numpy(truncated_z_sample(b_size, args.nz, args.tc_th, args.manualSeed)).float().to(device)
    else:
        noise = torch.randn(b_size, args.nz, 1, 1, device=device)
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

    loss_d_without_gp = (-real_d + fake_d).mean()
    loss_d = loss_d_without_gp + args.gp_weight * gp
    loss_d.backward()
    optimizerD.step()


    ############################
    # (2) Update G network
    ###########################
    netG.zero_grad()

    if args.truncnorm:
        noise = torch.from_numpy(truncated_z_sample(b_size, args.nz, args.tc_th, args.manualSeed)).float().to(device)
    else:
        noise = torch.randn(b_size, args.nz, 1, 1, device=device)
    fake = netG(noise)
    fake_d = netD(fake).mean()

    loss_g = -fake_d
    loss_g.backward()
    optimizerG.step()

    return loss_d, loss_g

def gan(data, netD, netG, optimizerD, optimizerG, device, args):
    if args.gan_type == "LogGAN":
        criterion = nn.BCELoss()
    else:
        criterion = nn.MSELoss()
    ############################
    # (1) Update D network: maximize log(D(x)) + log(1 - D(G(z)))
    ###########################
    ## Train with all-real batch
    netD.zero_grad()
    real = data[0].to(device)
    b_size = real.size(0)
    label = torch.full((b_size,), 1, device=device)
    output = netD(real).view(-1)
    if args.gan_type == "LogGAN":
        output = F.sigmoid(output)
    errD_real = criterion(output, label)
    errD_real.backward()

    ## Train with all-fake batch
    # Generate batch of latent vectors
    if args.truncnorm:
        noise = torch.from_numpy(truncated_z_sample(b_size, args.nz, args.tc_th, args.manualSeed)).float().to(device)
    else:
        noise = torch.randn(b_size, args.nz, 1, 1, device=device)
    fake = netG(noise)
    label.fill_(0)
    output = netD(fake.detach()).view(-1)
    if args.gan_type == "LogGAN":
        output = F.sigmoid(output)
    errD_fake = criterion(output, label)
    errD_fake.backward()
    loss_d = errD_real + errD_fake
    optimizerD.step()

    ############################
    # (2) Update G network: maximize log(D(G(z)))
    ###########################
    netG.zero_grad()
    label.fill_(1)
    output = netD(fake).view(-1)
    if args.gan_type == "LogGAN":
        output = F.sigmoid(output)
    errG = criterion(output, label)
    errG.backward()
    loss_g = errG
    optimizerG.step()
    
    return loss_d, loss_g