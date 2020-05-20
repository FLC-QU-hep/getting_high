
import torch
import sys
import numpy as np
import torch.nn as nn
import torch.nn.parallel
import torch.optim as optim
import torch.utils.data
from torch.autograd.variable import Variable
import models.HDF5Dataset as H
import models.GAN as GAN

# Constants
manualSeed = 2517
print("Random Seed: ", manualSeed)
np.random.seed(manualSeed)
torch.manual_seed(manualSeed)
workers = 20
batch_size = 32
nz = 100
num_epochs = 5
lr = 0.00001
beta1 = 0.5
ngpu = 1
cuda = True if torch.cuda.is_available() else False
FloatTensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor
device = torch.device("cuda:0" if (torch.cuda.is_available() and ngpu > 0) else "cpu")


PATH_save = '/*path*-to-saved-model/conv3d_corr_new2.pth'
PATH_chechpoint = '/*path*-to-saved-model/new2_1.pth'

def save(netG, netD, omtim_G, optim_D, epoch, loss, scores, path_to_save):
    torch.save({
                'Generator': netG.module.state_dict(),
                'Discriminator': netD.module.state_dict(),
                'G_optimizer': omtim_G.state_dict(),
                'D_optimizer': optim_D.state_dict(),
                'epoch': epoch,
                'loss': loss,
                'D_scores': scores
                },
                path_to_save)

netD = GAN.Discriminator(ngpu).to(device)
netG = GAN.Generator(ngpu).to(device)
print(netG, netD)

# Apply the weights_init function to randomly initialize all weights
netD.apply(GAN.weights_init)
netG.apply(GAN.weights_init)

# Initialize BCELoss function
criterion = nn.BCELoss()

# Optimizers
optimizer_G = torch.optim.Adam(netG.parameters(), lr=lr, betas=(beta1, 0.999))
optimizer_D = torch.optim.Adam(netD.parameters(), lr=lr, betas=(beta1, 0.999))


if len(sys.argv) > 1:
    if sys.argv[1] == 'from_chp':
        print('Loading from checkpoint...')
        checkpoint = torch.load(PATH_chechpoint)
        netG.load_state_dict(checkpoint['Generator'])
        netD.load_state_dict(checkpoint['Discriminator'])
        optimizer_G.load_state_dict(checkpoint['G_optimizer'])
        optimizer_D.load_state_dict(checkpoint['D_optimizer'])
        eph = checkpoint['epoch']
        chechoint_loss = checkpoint['loss']
        G_losses = chechoint_loss[0]
        D_losses = chechoint_loss[1]
        chechoint_scores = checkpoint['D_scores']
        D_scores_x = chechoint_scores[0]
        D_scores_z1 = chechoint_scores[1]
        D_scores_z2 = chechoint_scores[2]
        print('Done!')
    else:
        print('Unexpected argument: ', sys.argv[1])
        exit()
else:
    print('\nStart training from scratch...\n')
    eph = 0
    G_losses = np.array([])
    D_losses = np.array([])
    D_scores_x = np.array([])
    D_scores_z1 = np.array([])
    D_scores_z2 = np.array([])


# Handle multi-gpu if desired
if (device.type == 'cuda') and (ngpu > 1):
    netD = nn.DataParallel(netD, list(range(ngpu)))
    netG = nn.DataParallel(netG, list(range(ngpu)))
else:
    netD = nn.DataParallel(netD)
    netG = nn.DataParallel(netG)
netG.train()
netD.train()


print('Loading data...')
path = 'training_data/50k.hdf5'
data = H.HDF5Dataset(path, '30x30')
dataloader = torch.utils.data.DataLoader(data, batch_size=batch_size,
                                         shuffle=True, num_workers=workers)
print('Done!\n')

print('Start training loop...')

for epoch in range(num_epochs):
    epoch += eph + 1

    for i, batch in enumerate(dataloader):
        btch_sz = len(batch['shower'])
        real_showers = Variable(batch['shower']).float().to(device).view(btch_sz, 1, 30, 30, 30)
        real_energys = Variable(batch['energy']).float().to(device).view(btch_sz, 1, 1, 1, 1)

        # Adversarial ground truths
        valid_label = Variable(FloatTensor(btch_sz).fill_(1.0), requires_grad=False)
        fake_label = Variable(FloatTensor(btch_sz).fill_(0.0), requires_grad=False)

        ######################################################
        # Train Discriminator
        ######################################################
        netD.zero_grad()

        output = netD(real_showers, real_energys).view(-1)
        errD_real = criterion(output, valid_label)
        errD_real.backward()
        D_x = output.mean().item()
        noise = torch.FloatTensor(btch_sz, 100, 1, 1, 1).uniform_(-1, 1)

        gen_labels = np.random.uniform(10, 100, btch_sz)
        gen_labels = Variable(FloatTensor(gen_labels))
        gen_labels = gen_labels.view(btch_sz, 1, 1, 1, 1)
        fake_shower = netG(noise.to(device), gen_labels)

        output = netD(fake_shower, gen_labels).view(-1)
        errD_fake = criterion(output, fake_label)
        errD_fake.backward(retain_graph=True)
        D_G_z1 = output.mean().item()
        errD = errD_real + errD_fake

        optimizer_D.step()

        ######################################################
        # Train Generator
        ######################################################
        netG.zero_grad()

        output = netD(fake_shower, gen_labels).view(-1)
        errG = criterion(output, valid_label)
        errG.backward(retain_graph=True)
        D_G_z2 = output.mean().item()
        optimizer_G.step()

        # Output training stats
        G_losses = np.append(G_losses, errG.item())
        D_losses = np.append(D_losses, errD.item())
        D_scores_x = np.append(D_scores_x, D_x)
        D_scores_z1 = np.append(D_scores_z1, D_G_z1)
        D_scores_z2 = np.append(D_scores_z2, D_G_z2)

        if i % 10 == 0:
            print('[%d/%d] [%d/%d], (Loss_D: %.4f)  (Loss_G: %.4f),  (D(x): %.4f)  (D(G(z)): %.4f / %.4f)'
                  % (epoch, num_epochs, i, len(dataloader),
                     errD.item(), errG.item(), D_x, D_G_z1, D_G_z2))

    loss =  np.array([G_losses, D_losses])
    D_scores = np.array([D_scores_x, D_scores_z1, D_scores_z2])
    save(netG=netG, netD=netD, omtim_G=optimizer_G, optim_D=optimizer_D, epoch=epoch, loss=loss, scores=D_scores,
    path_to_save=PATH_save)
