import numpy as np
import argparse
import torch
import torch.utils.data
from torch import nn, optim
from torch.nn import functional as F
from torchvision import datasets, transforms
from torchvision.utils import save_image
from torch import autograd

    
    
    
    
class Discriminator_F_Conv_DIRENR_Diff_v3(nn.Module):
    def __init__(self, isize=30, nc=2, ndf=128):
        super(Discriminator_F_Conv_DIRENR_Diff_v3, self).__init__()    
        self.ndf = ndf
        self.isize = isize
        self.nc = nc

        
        self.conv1b = torch.nn.Conv3d(1, ndf, kernel_size=3, stride=2, padding=0, bias=False)
        self.bn1b = torch.nn.LayerNorm([14,14,14])
        self.conv2b = torch.nn.Conv3d(ndf, ndf, kernel_size=3, stride=2, padding=0, bias=False)
        self.bn2b = torch.nn.LayerNorm([6,6,6])
        self.conv3b = torch.nn.Conv3d(ndf, ndf, kernel_size=3, stride=1, padding=0, bias=False)


        self.conv1c = torch.nn.Conv3d(1, ndf, kernel_size=3, stride=2, padding=0, bias=False)
        self.bn1c = torch.nn.LayerNorm([14,14,14])
        self.conv2c = torch.nn.Conv3d(ndf, ndf, kernel_size=3, stride=2, padding=0, bias=False)
        self.bn2c = torch.nn.LayerNorm([6,6,6])
        self.conv3c = torch.nn.Conv3d(ndf, ndf, kernel_size=3, stride=1, padding=0, bias=False)

 
        #self.fc = torch.nn.Linear(5 * 4 * 4, 1)
        self.fc1a = torch.nn.Linear(30*30*30, int(ndf/2)) 
        self.fc1b = torch.nn.Linear(ndf * 4 * 4 * 4, int(ndf/2))
        self.fc1c = torch.nn.Linear(ndf * 4 * 4 * 4, int(ndf/2))
        self.fc1e = torch.nn.Linear(1, int(ndf/2))
        self.fc2 = torch.nn.Linear(int(ndf/2)*4, ndf*2)
        self.fc3 = torch.nn.Linear(ndf*2, ndf*2)
        self.fc4 = torch.nn.Linear(ndf*2, ndf*2)
        self.fc5 = torch.nn.Linear(ndf*2, 1)


    def forward(self, img, E_true):
        imga = img[:,0:1,:,:,:] #data.view(-1,1,30,30)
        imgb = img[:,1:2,:,:,:] #recon_batch.view(-1,1,30,30)

        img = imgb - imga
        
        xb = F.leaky_relu(self.bn1b(self.conv1b(imgb)), 0.2)
        xb = F.leaky_relu(self.bn2b(self.conv2b(xb)), 0.2)
        xb = F.leaky_relu(self.conv3b(xb), 0.2)
        xb = xb.view(-1, self.ndf * 4 * 4 * 4)
        xb = F.leaky_relu(self.fc1b(xb), 0.2)        
        
        xc = F.leaky_relu(self.bn1c(self.conv1c(torch.log(imgb+1.0))), 0.2)
        xc = F.leaky_relu(self.bn2c(self.conv2c(xc)), 0.2)
        xc = F.leaky_relu(self.conv3c(xc), 0.2)
        xc = xc.view(-1, self.ndf * 4 * 4 * 4)
        xc = F.leaky_relu(self.fc1c(xc), 0.2)
        
        xb = torch.cat((xb, xc, F.leaky_relu(self.fc1a(img.view(-1, 30*30*30))) , F.leaky_relu(self.fc1e(E_true), 0.2)), 1)

        xb = F.leaky_relu(self.fc2(xb), 0.2)
        xb = F.leaky_relu(self.fc3(xb), 0.2)
        xb = F.leaky_relu(self.fc4(xb), 0.2)
        xb = self.fc5(xb)

        return xb.view(-1) ### flattens
    
class Latent_Critic(nn.Module):
    def __init__(self, ):
        super(Latent_Critic, self).__init__()
        self.linear1 = nn.Linear(1, 50)
        self.linear2 = nn.Linear(50, 100)        
        self.linear3 = nn.Linear(100, 50)
        self.linear4 = nn.Linear(50, 1)

    def forward(self, x):      
        x = F.leaky_relu(self.linear1(x.view(-1,1)), inplace=True)
        x = F.leaky_relu(self.linear2(x), inplace=True)
        x = F.leaky_relu(self.linear3(x), inplace=True)
        return self.linear4(x)
    
    
    
class BiBAE_F_3D_LayerNorm_SmallLatent(nn.Module):
    """
    generator component of WGAN, adapted as VAE, with direct energy conditioning (giving true energy to both en- and de-coder)
    designed for 30x30x30 images
    faster version
    """
    def __init__(self, args, device, nc=1, ngf=8, z_rand=500, z_enc=12):
        super(BiBAE_F_3D_LayerNorm_SmallLatent, self).__init__()    
        self.ngf = ngf
        self.nc = nc
        #self.z = z
        self.z_rand = z_rand
        self.z_enc = z_enc
        self.z_full = z_enc + z_rand
        self.args = args
        self.device = device

        
        self.enconv1 = nn.Conv3d(in_channels=1, out_channels=ngf, kernel_size=(4,4,4), stride=(2,2,2),
                               padding=(2,2,2), bias=False, padding_mode='zeros')
        self.bnen1   = torch.nn.LayerNorm([16,16,16])
        self.enconv2 = nn.Conv3d(in_channels=ngf, out_channels=ngf*2, kernel_size=(4,4,4), stride=(2,2,2),
                               padding=(2,2,2), bias=False, padding_mode='zeros')
        self.bnen2   = torch.nn.LayerNorm([9,9,9])
        self.enconv3 = nn.Conv3d(in_channels=ngf*2, out_channels=ngf*4, kernel_size=(4,4,4), stride=(2,2,2),
                               padding=(2,2,2), bias=False, padding_mode='zeros')
        self.bnen3   = torch.nn.LayerNorm([5,5,5])
        self.enconv4 = nn.Conv3d(in_channels=ngf*4, out_channels=ngf*8, kernel_size=(3,3,3), stride=(1,1,1),
                               padding=(1,1,1), bias=False, padding_mode='zeros')
        self.bnen4   = torch.nn.LayerNorm([5,5,5])

     
        self.fc1 = nn.Linear(5*5*5*ngf*8+1, ngf*500, bias=True)
        self.fc2 = nn.Linear(ngf*500, int(self.z_full*1.5), bias=True)
        
        self.fc31 = nn.Linear(int(self.z_full*1.5), self.z_enc, bias=True)
        self.fc32 = nn.Linear(int(self.z_full*1.5), self.z_enc, bias=True)

        
        self.cond1 = torch.nn.Linear(self.z_full+1, int(self.z_full*1.5), bias=True)
        self.cond2 = torch.nn.Linear(int(self.z_full*1.5), ngf*500, bias=True)
        self.cond3 = torch.nn.Linear(ngf*500, 10*10*10*ngf, bias=True)
        
        self.deconv1 = torch.nn.ConvTranspose3d(ngf, ngf, kernel_size=(3,3,3), stride=(3,3,3), padding=(1,1,1), bias=False)
        self.bnde1   = torch.nn.LayerNorm([30,30,30])
        self.deconv2 = torch.nn.ConvTranspose3d(ngf, ngf*2, kernel_size=(3,3,3), stride=(2,2,2), padding=(1,1,1), bias=False)
        self.bnde2   = torch.nn.LayerNorm([60,60,60])

        #self.deconv3 = torch.nn.ConvTranspose3d(ngf*4, ngf*8, kernel_size=(3,3,3), stride=(2,2,2), padding=(0,1,1), bias=False)
        
        self.conv0 = torch.nn.Conv3d(ngf*2, ngf, kernel_size=(2,2,2), stride=(2,2,2), padding=(0,0,0), bias=False)
        self.bnco0 = torch.nn.LayerNorm([30,30,30])
        self.conv1 = torch.nn.Conv3d(ngf, ngf*2, kernel_size=(3,3,3), stride=(1,1,1), padding=(1,1,1), bias=False)
        self.bnco1 = torch.nn.LayerNorm([30,30,30])
        self.conv2 = torch.nn.Conv3d(ngf*2, ngf*4, kernel_size=(3,3,3), stride=(1,1,1), padding=(1,1,1), bias=False)
        self.bnco2 = torch.nn.LayerNorm([30,30,30])
        self.conv3 = torch.nn.Conv3d(ngf*4, ngf*2, kernel_size=(3,3,3), stride=(1,1,1), padding=(1,1,1), bias=False)      
        self.bnco3 = torch.nn.LayerNorm([30,30,30])
        self.conv4 = torch.nn.Conv3d(ngf*2, 1, kernel_size=(3,3,3), stride=(1,1,1), padding=(1,1,1), bias=False)
    
        #self.dr03 = nn.Dropout(p=0.3, inplace=False)
        #self.dr05 = nn.Dropout(p=0.5, inplace=False)

    
    def encode(self, x, E_true):
        x = F.leaky_relu(self.bnen1(self.enconv1(x.view(-1,1,30,30,30))), 0.2, inplace=True)
        x = F.leaky_relu(self.bnen2(self.enconv2(x)), 0.2, inplace=True)
        x = F.leaky_relu(self.bnen3(self.enconv3(x)), 0.2, inplace=True)
        x = F.leaky_relu(self.bnen4(self.enconv4(x)), 0.2, inplace=True)

        x = torch.cat( (x.view(-1, x.size(1)*x.size(2)*x.size(3)*x.size(4)), E_true), 1)
                       
        x = F.leaky_relu((self.fc1(x)), 0.2, inplace=True)
        x = F.leaky_relu((self.fc2(x)), 0.2, inplace=True)
        #x = F.leaky_relu((self.fc1(x)), 0.2, inplace=True)
        #x = F.leaky_relu((self.fc2(x)), 0.2, inplace=True)
        return torch.cat((self.fc31(x),torch.zeros(x.size(0), self.z_rand, device = self.device)), 1), torch.cat((self.fc32(x),torch.zeros(x.size(0), self.z_rand, device = self.device)), 1)


    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        #print(std)
        #print(mu)
        return mu + eps*std

    def decode(self, z):
        ### need to do generated 30 layers, hence the loop!
        x = F.leaky_relu((self.cond1(z)), 0.2, inplace=True)
        x = F.leaky_relu((self.cond2(x)), 0.2, inplace=True)
        x = F.leaky_relu((self.cond3(x)), 0.2, inplace=True)
        #x = F.leaky_relu(self.cond1(z), 0.2, inplace=True)
        #x = F.leaky_relu(self.cond2(x), 0.2, inplace=True)
        #x = F.leaky_relu(self.cond3(x), 0.2, inplace=True)

        ## change size for deconv2d network. Image is 10x10
        x = x.view(-1,self.ngf,10,10,10)        

        ## apply series of deconv2d and batch-norm
        x = F.leaky_relu(self.bnde1(self.deconv1(x, output_size=[x.size(0), 1, 30, 30, 30])), 0.2, inplace=True) #
        x = F.leaky_relu(self.bnde2(self.deconv2(x, output_size=[x.size(0), 1, 60, 60, 60])), 0.2, inplace=True) #
        #x = F.leaky_relu(self.deconv3(x, output_size=[x.size(0), 1, 15, 120, 120]), 0.2, inplace=True) #

        ##Image is 120x120
        x = F.leaky_relu(self.bnco0(self.conv0(x)), 0.2, inplace=True)
        x = F.leaky_relu(self.bnco1(self.conv1(x)), 0.2, inplace=True)
        x = F.leaky_relu(self.bnco2(self.conv2(x)), 0.2, inplace=True)
        x = F.leaky_relu(self.bnco3(self.conv3(x)), 0.2, inplace=True)
        x = F.relu(self.conv4(x), inplace=True)
        return x 
        
        
    def forward(self, x, E_true, z=None, mode='full'):
        #print(x.size())
        if mode == 'encode':
            mu, logvar = self.encode(x, E_true)
            z = self.reparameterize(mu, logvar)
            return mu, logvar, z 
        elif mode == 'decode':
            return self.decode(torch.cat((z,E_true), 1)) #, E_true 
        elif mode == 'full':
            mu, logvar = self.encode(x,E_true)
            z = self.reparameterize(mu, logvar)
            return self.decode(torch.cat((z,E_true), 1)), mu, logvar, z
        

    
class PostProcess_Size1Conv_EcondV2(nn.Module):
    def __init__(self, isize=30, nc=2, ndf=128, bias=False, out_funct='relu'):
        super(PostProcess_Size1Conv_EcondV2, self).__init__()    
        self.ndf = ndf
        self.isize = isize
        self.nc = nc
        self.bais = bias
        self.out_funct = out_funct
        
        self.fcec1 = torch.nn.Linear(2, int(ndf/2), bias=True)
        self.fcec2 = torch.nn.Linear(int(ndf/2), int(ndf/2), bias=True)
        self.fcec3 = torch.nn.Linear(int(ndf/2), int(ndf/2), bias=True)
        
        self.conv1 = torch.nn.Conv3d(1, ndf, kernel_size=1, stride=1, padding=0, bias=bias)
        self.bnco1 = torch.nn.LayerNorm([self.isize, self.isize, self.isize])
        
        self.conv2 = torch.nn.Conv3d(ndf+int(ndf/2), ndf, kernel_size=1, stride=1, padding=0, bias=bias)
        self.bnco2 = torch.nn.LayerNorm([self.isize, self.isize, self.isize])

        self.conv3 = torch.nn.Conv3d(ndf, ndf, kernel_size=1, stride=1, padding=0, bias=bias)
        self.bnco3 = torch.nn.LayerNorm([self.isize, self.isize, self.isize])

        self.conv4 = torch.nn.Conv3d(ndf, ndf, kernel_size=1, stride=1, padding=0, bias=bias)
        self.bnco4 = torch.nn.LayerNorm([self.isize, self.isize, self.isize])

        self.conv5 = torch.nn.Conv3d(ndf, ndf, kernel_size=1, stride=1, padding=0, bias=bias)

        self.conv6 = torch.nn.Conv3d(ndf, 1, kernel_size=1, stride=1, padding=0, bias=False)
 

    def forward(self, img, E_True=0):
        img = img.view(-1, 1, self.isize, self.isize, self.isize)
                
        econd = torch.cat((torch.sum(img.view(-1, self.isize*self.isize*self.isize), 1).view(-1, 1), E_True), 1)

        econd = F.leaky_relu(self.fcec1(econd), 0.01)
        econd = F.leaky_relu(self.fcec2(econd), 0.01)
        econd = F.leaky_relu(self.fcec3(econd), 0.01)
        
        econd = econd.view(-1, int(self.ndf/2), 1, 1, 1)
        econd = econd.expand(-1, -1, self.isize, self.isize, self.isize)        
        
        img = F.leaky_relu(self.bnco1(self.conv1(img)), 0.01)
        img = torch.cat((img, econd), 1)
        img = F.leaky_relu(self.bnco2(self.conv2(img)), 0.01)
        img = F.leaky_relu(self.bnco3(self.conv3(img)), 0.01)
        img = F.leaky_relu(self.bnco4(self.conv4(img)), 0.01)
        img = F.leaky_relu(self.conv5(img), 0.01) 
        img = self.conv6(img)

        if self.out_funct == 'relu':
            img = F.relu(img)
        elif self.out_funct == 'leaky_relu':
            img = F.leaky_relu(img, 0.01) 
              
        return img.view(-1, 1, self.isize, self.isize, self.isize)


    

