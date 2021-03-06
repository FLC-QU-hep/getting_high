import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import h5py
import logging
import os, sys
import argparse
import multiprocessing as mp
import models.HDF5Dataset as H
import torch

def correct2D(dpath, BATCH_SIZE, minibatch):

    ## Load and make them iterable
    data = H.HDF5Dataset(dpath, '30x30')
    energies = data['energy'][:].reshape(len(data['energy'][:]))
    layers = data['layers'][:]

    training_dataset = tuple(zip(layers, energies))


    dataloader = torch.utils.data.DataLoader(training_dataset, batch_size=BATCH_SIZE,
                                            shuffle=True, num_workers=4)
    dataiter = iter(dataloader)

    corr_data = []
    enr = []

    for i in range(0,minibatch):
        batch = next(dataiter, None)

        if batch is None:
            dataiter = iter(dataloader)
            batch = dataiter.next()

        real_data = batch[0] # 32x30 calo layers
        real_data = real_data.numpy()

        real_energy = batch[1]
        real_energy = real_energy.unsqueeze(-1)
        real_energy = real_energy.numpy()

        real_d = real_data.mean(axis=0)
        for layer_pos in range(len(real_d)):
            for row_pos in range(len(real_d[0])):
                if real_d[layer_pos, row_pos].sum().item() < 0.001:
                    for i in range(row_pos, 0, -1):
                        real_data[:, layer_pos, i] = real_data[:, layer_pos, i-1]
        real_data = real_data[:, :, 1:-1]
        corr_data.append(real_data)
        enr.append(real_energy)

    layers = np.vstack(np.asarray(corr_data))
    energy = np.vstack(np.asarray(enr))
    return layers, energy

parser = argparse.ArgumentParser()
parser.add_argument('--input', type=str, required=True, help='input file name')
parser.add_argument('--output', type=str, required=True, help='output name of preprocessed file')
parser.add_argument('--batchsize', type=int, required=True, help='batchsize')
parser.add_argument('--minibatch', type=int, required=True, help='mini-batchsize')

opt = parser.parse_args()
inputfile = str(opt.input)
outputfile = str(opt.output)
BS = int(opt.batchsize)
mbatch = int(opt.minibatch)

pixs, energy = correct2D(inputfile, BS, mbatch)

#Open HDF5 file for writing
hf = h5py.File(outputfile, 'w')
grp = hf.create_group("30x30")


## write to hdf5 files
grp.create_dataset('energy', data=energy)
grp.create_dataset('layers', data=pixs)


#print (pixs.shape)
