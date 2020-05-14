import torch
import torch.nn as nn
import torch.nn.parallel
import torch.nn.functional as F
from torch.autograd import Variable
from torch.utils import data

      
class DCGAN_D(nn.Module):
    """ 
    discriminator component of WGAN
    """

    def __init__(self, ndf):
        super(DCGAN_D, self).__init__()    
        self.ndf = ndf
    
        
        ### convolution
        self.conv1 = torch.nn.Conv3d(1, ndf, kernel_size=(10,2,2), stride=(1,2,2), padding=(0,1,1), bias=False)
        ## layer-normalization
        self.bn1 = torch.nn.LayerNorm([21,16,16])
        ## convolution
        self.conv2 = torch.nn.Conv3d(ndf, ndf*2, kernel_size=(6,2,2), stride=(1,2,2), padding=0, bias=False)
        ## layer-normalization
        self.bn2 = torch.nn.LayerNorm([16,8,8])
        #convolution
        self.conv3 = torch.nn.Conv3d(ndf*2, ndf*4, kernel_size=(4,2,2), stride=(1,2,2), padding=0, bias=False)
        ## layer-normalization
        self.bn3 = torch.nn.LayerNorm([13,4,4])
        #convolution
        self.conv4 = torch.nn.Conv3d(ndf*4, 1, kernel_size=(4,2,2), stride=1, padding=0, bias=False)
        
        
        # Read-out layer : 1 * isize * isize input features, ndf output features 
        self.fc1 = torch.nn.Linear((10 * 3 * 3)+1, 100)
        self.fc2 = torch.nn.Linear(100, 200)
        self.fc3 = torch.nn.Linear(200, 100)
        self.fc4 = torch.nn.Linear(100, 75)
        self.fc5 = torch.nn.Linear(75, 1)
        

    def forward(self, x, energy):
        
        
        # N (Nlayers) x 30 x 30
        
        x = F.leaky_relu(self.bn1(self.conv1(x)), 0.2)
        x = F.leaky_relu(self.bn2(self.conv2(x)), 0.2)
        x = F.leaky_relu(self.bn3(self.conv3(x)), 0.2)
        x = self.conv4(x)
        
        #Grand total --> size changes from (30, 30, 30) to (10, 3, 3)

        
        x = x.view(-1, 10 * 3 * 3)
        # Size changes from (10, 3, 3) to (1, 10 * 3 * 3) 
        #Recall that the -1 infers this dimension from the other given dimension

        x = torch.cat((x, energy), 1)

        x = F.leaky_relu(self.fc1(x), 0.2)
        x = F.leaky_relu(self.fc2(x), 0.2)
        x = F.leaky_relu(self.fc3(x), 0.2)
        x = F.leaky_relu(self.fc4(x), 0.2)
        
        # Read-out layer 
        output_wgan = self.fc5(x)
        
        #output_wgan = output_wgan.view(-1) ### flattens

        return output_wgan

class DCGAN_G(nn.Module):
    """ 
        generator component of WGAN
    """
    def __init__(self, ngf, nz):
        super(DCGAN_G, self).__init__()
        
        self.ngf = ngf
        self.nz = nz

        kernel = 4
        
        # input energy shape [batch x 1 x 1 x 1 ] going into convolutional
        self.conv1_1 = nn.ConvTranspose3d(1, ngf*4, kernel, 1, 0, bias=False)
        # state size [ ngf*4 x 4 x 4 x 4]
        
        # input noise shape [batch x nz x 1 x 1] going into convolutional
        self.conv1_100 = nn.ConvTranspose3d(nz, ngf*4, kernel, 1, 0, bias=False)
        # state size [ ngf*4 x 4 x 4 x 4]
        
        
        # outs from first convolutions concatenate state size [ ngf*8 x 4 x 4]
        # and going into main convolutional part of Generator
        self.main_conv = nn.Sequential(
            
            nn.ConvTranspose3d(ngf*8, ngf*4, kernel, 2, 1, bias=False),
            nn.LayerNorm([8, 8, 8]),
            nn.ReLU(True),
            # state shape [ (ndf*4) x 8 x 8 ]

            nn.ConvTranspose3d(ngf*4, ngf*2, kernel, 2, 1, bias=False),
            nn.LayerNorm([16, 16, 16]),
            nn.ReLU(True),
            # state shape [ (ndf*2) x 16 x 16 ]

            nn.ConvTranspose3d(ngf*2, ngf, kernel, 2, 1, bias=False),
            nn.LayerNorm([32, 32, 32]),
            nn.ReLU(True),
            # state shape [ (ndf) x 32 x 32 ]

            nn.ConvTranspose3d(ngf, 1, 3, 1, 2, bias=False),
            nn.ReLU()
            # state shape [ 1 x 30 x 30 x 30 ]
        )

    def forward(self, noise, energy):
        energy_trans = self.conv1_1(energy)
        noise_trans = self.conv1_100(noise)
        input = torch.cat((energy_trans, noise_trans), 1)
        x = self.main_conv(input)
        x = x.view(-1, 30, 30,30)
        return x
