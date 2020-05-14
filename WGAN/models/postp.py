import torch
import torch.nn as nn
import torch.nn.parallel
import torch.nn.functional as F
from torch.autograd import Variable



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