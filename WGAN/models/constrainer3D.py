import torch
import torch.nn as nn
import torch.nn.parallel
import torch.nn.functional as F
from torch.autograd import Variable
import models.HDF5Dataset as H
from torch.utils import data


class energyRegressor(nn.Module):
    """ 
    Energy regressor of WGAN. 
    """

    def __init__(self):
        super(energyRegressor, self).__init__()
        
        ## 3d conv layers
        self.conv1 = torch.nn.Conv3d(1, 16, kernel_size=3, stride=2, padding=0, bias=False)
        self.bn1 = torch.nn.LayerNorm([14,14,14])
        self.conv2 = torch.nn.Conv3d(16, 32, kernel_size=3, stride=2, padding=0, bias=False)
        self.bn2 = torch.nn.LayerNorm([6,6,6])
        self.conv3 = torch.nn.Conv3d(32, 16, kernel_size=2, stride=1, padding=0, bias=False)
 
       
        ## FC layers
        self.fc1 = torch.nn.Linear(16 * 5 * 5 * 5, 100)
        self.fc2 = torch.nn.Linear(100, 1)
        
    def forward(self, x):
        #input shape :  [30, 30, 30]
        ## reshape the input: expand one dim
        #x = x.unsqueeze(1)
        
        ## image [30, 30, 30]
        ### convolution adn batch normalisation
        x = F.leaky_relu(self.bn1(self.conv1(x)), 0.2)
        x = F.leaky_relu(self.bn2(self.conv2(x)), 0.2)
        x = self.conv3(x)
      
        ## shape [5, 5, 5]
        
        ## flatten for FC
        x = x.view(-1, x.size(1) * x.size(2) * x.size(3) * x.size(4))
        
        ## pass to FC layers
        x = F.leaky_relu(self.fc1(x), 0.2)
        x = F.relu(self.fc2(x))
        return x