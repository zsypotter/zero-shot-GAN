import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from scipy.stats import truncnorm

def truncated_z_sample(batch_size, dim_z, threshold=2, seed=None, truncation=1.):
    state = None if seed is None else np.random.RandomState(seed)
    values = truncnorm.rvs(-threshold, threshold, size=(batch_size, dim_z), random_state=state)
    return truncation * values

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)

def wgan_with_gp(data, netD, netG, optimizerD, optimizerG, device, args, att_size):
    aux_criterion = nn.CrossEntropyLoss()
    aux_criterion = aux_criterion.to(device)

    ############################
    # (1) Update D network
    ###########################
    netD.zero_grad()

    real, real_aux_label = data
    real = real.to(device)
    real_aux_label = real_aux_label.to(device)
    b_size = real.size(0)
    real_aux, real_dis = netD(real)

    if args.truncnorm:
        noise = truncated_z_sample(b_size, args.nz + args.num_class, args.tc_th, args.manualSeed)
    else:
        noise = np.random.randn(b_size, args.nz + args.num_class)
    aux_label = np.random.randint(0, args.num_class, b_size)
    aux_label_onehot = np.zeros((b_size, args.num_class))
    aux_label_onehot[np.arange(b_size), aux_label[np.arange(b_size)]] = 1
    noise[np.arange(b_size), :args.num_class] = aux_label_onehot[np.arange(b_size)]
    noise = torch.from_numpy(noise).float().to(device)

    fake = netG(noise)
    fake_aux_label = torch.from_numpy(aux_label).to(device)
    fake_aux, fake_dis = netD(fake.detach())

    epsilon = torch.rand(b_size, 1, 1, 1).to(device)
    interpolates = torch.tensor((epsilon * real + (1 - epsilon) * fake).data, requires_grad=True)
    gradients = torch.autograd.grad(
        netD(interpolates)[1].view(b_size),
        interpolates,
        grad_outputs=torch.ones(b_size).to(device),
        create_graph=True)[0]
    gp = ((gradients.view(b_size, -1).norm(2, dim=1) - 1).pow(2)).mean()

    real_aux_loss = aux_criterion(real_aux, real_aux_label)
    fake_aux_loss = aux_criterion(fake_aux, fake_aux_label)
    errD_real = -real_dis.mean() + args.gp_weight * gp + real_aux_loss
    errD_fake = fake_dis.mean() + args.gp_weight * gp + fake_aux_loss
    errD_fake.backward(retain_graph=True)
    errD_real.backward()
    loss_d = errD_fake + errD_real
    optimizerD.step()


    ############################
    # (2) Update G network
    ###########################
    netG.zero_grad()
    fake_aux, fake_dis = netD(fake)
    fake_aux_loss = aux_criterion(fake_aux, fake_aux_label)
    loss_g = -fake_dis.mean() + fake_aux_loss
    loss_g.backward()
    optimizerG.step()

    return loss_d, loss_g, real_dis, fake_dis, real_aux_loss, fake_aux_loss

def gan(data, netD, netG, optimizerD, optimizerG, device, args, att_size, train_att_dict, test_att_dict, att_dict):
    if args.gan_type == "LogGAN":
        dis_criterion = nn.BCELoss()
    else:
        dis_criterion = nn.MSELoss()
    aux_criterion = nn.CrossEntropyLoss()
    dis_criterion = dis_criterion.to(device)
    aux_criterion = aux_criterion.to(device)

    train_att_dict_cuda = torch.from_numpy(train_att_dict).float().to(device)
    test_att_dict_cuda = torch.from_numpy(test_att_dict).float().to(device)
    att_dict_cuda = torch.from_numpy(att_dict).float().to(device)

    ############################
    # (1) Update D network: maximize log(D(x)) + log(1 - D(G(z)))
    ###########################
    ## Train with all-real batch
    netD.zero_grad()
    real, train_aux_label, aux_label = data
    real = real.to(device)
    b_size = real.size(0)
    dis_label = torch.full((b_size,), 1, device=device)
    real_aux, real_dis = netD(real)
    real_dis = real_dis.view(-1)
    if args.gan_type == "LogGAN":
        real_dis = F.sigmoid(real_dis)
    
    similarity = torch.mm(real_aux, train_att_dict_cuda.t())
    real_aux_loss = aux_criterion(similarity, train_aux_label.to(device))
    errD_real = dis_criterion(real_dis, dis_label) + real_aux_loss
    errD_real.backward()

    ## Train with all-fake batch
    # Generate batch of latent vectors
    if args.truncnorm:
        noise = truncated_z_sample(b_size, args.nz + att_size, args.tc_th, args.manualSeed)
    else:
        noise = np.random.randn(b_size, args.nz + att_size)
    noise[np.arange(b_size), :att_size] = train_att_dict[train_aux_label]
    noise = torch.from_numpy(noise).float().to(device)

    fake = netG(noise)
    dis_label.fill_(0)
    fake_aux, fake_dis = netD(fake.detach())
    fake_dis = fake_dis.view(-1)
    if args.gan_type == "LogGAN":
        fake_dis = F.sigmoid(fake_dis)
    similarity = torch.mm(fake_aux, train_att_dict_cuda.t())
    fake_aux_loss = aux_criterion(similarity, train_aux_label.to(device))
    errD_fake = dis_criterion(fake_dis, dis_label) + fake_aux_loss
    errD_fake.backward()
    loss_d = (errD_real + errD_fake) / 2
    optimizerD.step()

    ############################
    # (2) Update G network: maximize log(D(G(z)))
    ###########################
    netG.zero_grad()
    dis_label.fill_(1)
    fake_aux, fake_dis = netD(fake)
    fake_dis = fake_dis.view(-1)
    if args.gan_type == "LogGAN":
        fake_dis = F.sigmoid(fake_dis)
    similarity = torch.mm(fake_aux, train_att_dict_cuda.t())
    fake_aux_loss = aux_criterion(similarity, train_aux_label.to(device))
    errG = dis_criterion(fake_dis, dis_label) + fake_aux_loss
    errG.backward()
    loss_g = errG
    optimizerG.step()
    
    return loss_d, loss_g, real_dis, fake_dis, real_aux_loss, fake_aux_loss