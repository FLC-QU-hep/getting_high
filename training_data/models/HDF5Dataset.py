from __future__ import print_function
import argparse
import random
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
from torch.utils import data
import torch.optim as optim
import torch.utils.data
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils
from torch.autograd import Variable
import os
import json
import h5py
from pathlib import Path


class HDF5Dataset(torch.utils.data.Dataset):
    def __init__(self, path, name):
        self.file_path = path
        self.dataset = None
        self.name = name
        with h5py.File(self.file_path, 'r') as file:
            self.dataset_len = len(file["{}".format(name)])

    def __getitem__(self, index):
        if self.dataset is None:
            self.dataset = h5py.File(self.file_path, 'r')["{}".format(self.name)]
        return self.dataset[index]
        
    def __len__(self):
        return self.dataset_len

