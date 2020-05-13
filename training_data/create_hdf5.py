import matplotlib
matplotlib.use('pdf')
import matplotlib.pyplot as plt
import numpy as np
import h5py
import uproot
import logging
import os, sys
import argparse
import multiprocessing as mp


## function for pixels
def pixs (strt,end):

    #binX = np.arange(-75, 76, 5.088333)
    #binZ = np.arange(-75, 76, 5.088333)
    #binX = np.arange(-90, 91, 5.600)
    #binZ = np.arange(-83, 84, 5.080)

    binX = np.arange(-81, 82, 5.088333)
    binZ = np.arange(-77, 78, 5.088333)

    ## Temporary storage for numpy arrays (layer information)
    l = []

    for i in range(strt, end):
        if len(mcPDG[i]) > 7: continue
        fig, axs = plt.subplots(30, 1, figsize=(30, 20))

        layers = []
        for j in range(0,30):
            idx = np.where((y[i] <= (hmap[j] + 0.9999)) & (y[i] > (hmap[j] + 0.0001)))
            xlayer = x[i].take(idx)[0]
            zlayer = z[i].take(idx)[0]
            elayer = e[i].take(idx)[0]

            ### GeV -- > MeV conversion for cell energies
            elayer = elayer * 1000.00

            ### 2d hist is need for energy weighted distributions
            h0 = axs[j].hist2d(xlayer, zlayer, bins=[binX, binZ], weights=elayer)

            layers.append(h0[0])

        ## accumulate for each event
        l.append(layers)
        plt.close(fig)

    layers = np.asarray(l)


    return layers

def E0(strt,end):
    ## get in incident energy
    e0 = []
    for i in range(strt, end):
        if len(mcPDG[i]) > 7: continue
        tmp = np.reshape(mcEne[i].take([0]), (1,1))
        e0.append(tmp)

    e0 = np.reshape(np.asarray(e0),(-1,1))
    return e0

if __name__=="__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--ncpu', type=int, help='number of cpus', default=1)
    parser.add_argument('--rootfile', type=str, required=True, help='input root file for streaming')
    parser.add_argument('--branch', type=str, required=True, help='branch name of the root file')
    parser.add_argument('--batchsize', type=int, help='batch size for streaming', default=100)
    opt = parser.parse_args()

    ncpu = int(opt.ncpu)
    root_path = str(opt.rootfile)
    root_branch = str(opt.branch)
    batch = int(opt.batchsize)
    ## create hit map of ECAL (in y coordinate[cm])
    hmap = np.array([1811, 1814, 1824, 1827, 1836, 1839, 1849,
                1852, 1861, 1864, 1873, 1877, 1886, 1889, 1898, 1902,
                1911, 1914, 1923, 1926, 1938, 1943, 1955, 1960,
                1971, 1976, 1988, 1993, 2005, 2010])

    #stream from root file
    ntuple = uproot.open(root_path)[root_branch]
    x = ntuple.array("scpox")
    y = ntuple.array("scpoy")
    z = ntuple.array("scpoz")
    e = ntuple.array("scene")
    mcPDG = ntuple.array("mcpdg")
    mcEne = ntuple.array("mcene")

    import time
    start_time = time.time()

    print("Creating {}-process pool".format(ncpu) )
    pool = mp.Pool(ncpu)


    evts = np.arange(0, x.shape[0], batch)
    #evts = np.arange(0, 10000, batch)
    tmp = [[evts[k-1],evts[k]] for k in range(1,len(evts))]
    events = np.vstack(tmp)

    ## execute cpu jobs
    pixels = pool.starmap(pixs, events)
    e0 = pool.starmap(E0, events)


    #Open HDF5 file for writing
    hf = h5py.File('test_30x32.hdf5', 'w')
    grp = hf.create_group("30x30")


    ## write to hdf5 files
    grp.create_dataset('energy', data=np.vstack(e0))
    grp.create_dataset('layers', data=np.vstack(pixels))

    #hf.close()

    pool.close()
    pool.join()

    print("--- %s seconds ---" % (time.time() - start_time))
