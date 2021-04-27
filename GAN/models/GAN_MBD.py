import torch
import torch.nn as nn

karnel_G = 4
karnel_D = 3
ngf = 32
ndf = 32
nz = 100

class Generator(nn.Module):
    def __init__(self, ngpu):
        super(Generator, self).__init__()
        self.ngpu = ngpu

        self.conv1 = nn.ConvTranspose3d(nz, ngf*8, karnel_G, 1, 0, bias=False)

        self.main_conv = nn.Sequential(

            nn.ConvTranspose3d(ngf*8, ngf*4, karnel_G, 2, 1, bias=False),
            nn.BatchNorm3d(ngf*4),
            nn.ReLU(True),

            nn.ConvTranspose3d(ngf*4, ngf*2, karnel_G, 2, 1, bias=False),
            nn.BatchNorm3d(ngf*2),
            nn.ReLU(True),

            nn.ConvTranspose3d(ngf*2, ngf, karnel_G, 2, 1, bias=False),
            nn.BatchNorm3d(ngf),
            nn.ReLU(True),

            nn.ConvTranspose3d(ngf, 1, 3, 1, 2, bias=False),
            nn.ReLU()
        )

    def forward(self, noise, energy):
        input = self.conv1(noise*energy)
        return self.main_conv(input)


class Discriminator(nn.Module):
    def __init__(self, ngpu):
        super(Discriminator, self).__init__()
        self.ngpu = ngpu

        self.main_conv = nn.Sequential(

            nn.Conv3d(1, ndf, karnel_D, 2, 3, bias=False),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv3d(ndf, ndf*2, karnel_D, 2, 1, bias=False),
            nn.BatchNorm3d(ndf*2),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv3d(ndf*2, ndf*4, karnel_D, 2, 1, bias=False),
            nn.BatchNorm3d(ndf*4),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv3d(ndf*4, ndf*8, karnel_D, 2, 1, bias=False),
            nn.BatchNorm3d(ndf*8),
            nn.LeakyReLU(0.2, inplace=True)
        )
        
        self.last_conv = nn.Conv3d(ndf*8, ndf*4, karnel_D, 2, 0, bias=False)

        self.fc = nn.Sequential(

            nn.Linear(ndf*4 + 2, ndf*2),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Linear(ndf*2, 1),
        )
        
        
    def miniBatchStdDev(self, x, subGroupSize=128):
        r"""
        Add a minibatch standard deviation channel to the current layer.
        In other words:
            1) Compute the standard deviation of the feature map over the minibatch
            2) Get the mean, over all pixels and all channels of thsi ValueError
            3) expand the layer and cocatenate it with the input
        Args:
            - x (tensor): previous layer
            - subGroupSize (int): size of the mini-batches on which the standard deviation
            should be computed
        """
        size = x.size()
        subGroupSize = min(size[0], subGroupSize)
        if size[0] % subGroupSize != 0:
            subGroupSize = size[0]
        G = int(size[0] / subGroupSize)
        if subGroupSize > 1:
            y = x.view(-1, subGroupSize, size[1], size[2], size[3], size[4])
            y = torch.var(y, 1)
            y = torch.sqrt(y + 1e-8)
            y = y.view(G, -1)
            y = torch.mean(y, 1).view(G, 1)
            y = y.expand(G, size[2]*size[3]*size[4]).view((G, 1, 1, size[2], size[3], size[4]))
            y = y.expand(G, subGroupSize, -1, -1, -1, -1)
            y = y.contiguous().view((-1, 1, size[2], size[3], size[4]))
        else:
            y = torch.zeros(x.size(0), 1, x.size(2), x.size(3), x.size(4), device=x.device)

        return torch.cat([x, y], dim=1)

    def forward(self, shower, energy):
        x = self.main_conv(shower)
        x = self.last_conv(x)
        x = self.miniBatchStdDev(x)
        fc_input = torch.cat((x, energy), 1).view(-1, ndf*4 + 2)
        return self.fc(fc_input)


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)
    elif classname.find('Linear') != -1:
        m.weight.data.normal_(0.0, 0.02)
        m.bias.data.fill_(0)
