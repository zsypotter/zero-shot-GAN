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
        dis_criterion = nn.BCELoss()
    else:
        dis_criterion = nn.MSELoss()
    aux_criterion = nn.NLLLoss()
    dis_criterion = dis_criterion.to(device)
    aux_criterion = aux_criterion.to(device)
    ############################
    # (1) Update D network: maximize log(D(x)) + log(1 - D(G(z)))
    ###########################
    ## Train with all-real batch
    netD.zero_grad()
    real, aux_label = data
    real =real.to(device)
    b_size = real.size(0)
    dis_label = torch.full((b_size,), 1, device=device)
    aux_onehot = np.zeros((b_size, args.num_class))
    aux_onehot[np.arange(b_size), aux_label] = 1

    aux, dis = netD(real)
    dis = dis.view(-1)
    if args.gan_type == "LogGAN":
        dis = F.sigmoid(dis)
    errD_real = dis_criterion(dis, dis_label) + aux_criterion(aux, aux_label.to(device))
    dis_real = dis
    aux_real = aux_criterion(aux, aux_label.to(device))
    errD_real.backward()

    ## Train with all-fake batch
    # Generate batch of latent vectors
    if args.truncnorm:
        noise = truncated_z_sample(b_size, args.nz + args.num_class, args.tc_th, args.manualSeed)
    else:
        noise = np.random.normal(0, 1, (b_size, args.nz + args.num_class))
    noise[np.arange(b_size), :args.num_class] = aux_onehot[np.arange(b_size)]
    noise = torch.from_numpy(noise).float().to(device)

    fake = netG(noise)
    dis_label.fill_(0)
    aux, dis = netD(fake.detach())
    dis = dis.view(-1)
    if args.gan_type == "LogGAN":
        dis = F.sigmoid(dis)
    errD_fake = dis_criterion(dis, dis_label) + aux_criterion(aux, aux_label.to(device))
    dis_fake = dis
    aux_fake = aux_criterion(aux, aux_label.to(device))
    errD_fake.backward()
    loss_d = errD_real + errD_fake
    optimizerD.step()

    ############################
    # (2) Update G network: maximize log(D(G(z)))
    ###########################
    netG.zero_grad()
    dis_label.fill_(1)
    aux, dis = netD(fake) 
    dis = dis.view(-1)
    if args.gan_type == "LogGAN":
        dis = F.sigmoid(dis)
    errG = dis_criterion(dis, dis_label) + aux_criterion(aux, aux_label.to(device))
    errG.backward()
    loss_g = errG
    optimizerG.step()
    
    return loss_d, loss_g, dis_real, dis_fake, aux_real, aux_fake