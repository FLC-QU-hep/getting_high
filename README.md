# Generative Models for High-granularity Calorimeter of ILD
We are modelling electromagnetic showers in the central region of the Silicon-Tungsten calorimeter of the proposed ILD. We investigate the use of a new architecture: Bounded-Information Bottleneck Autoencoder. In addition, we are utilising WGAN-GP and vanilla GAN approaches. In total, we train 3 generative models. 

This repository contains ingredients for repoducing *Getting High: High Fidelity Simulation of High Granularity Calorimeters with High Speed* [[`arXiv:2005.05334`](https://arxiv.org/abs/2005.05335)]

## Data Generation and Preparation 

#### Step 1: ddsim + Geant4
We use `iLCsoft` which includes `ddsim` and `Geant4`. It is better to use generation code that outputs big files to a scratch space. For DESY and NAF users: you may want to use DUST strogage in **CentOS7** NAF-WGs.

First we need to pull `ILDConfig` repository and go to its specific folder.

```
git clone https://github.com/iLCSoft/ILDConfig.git
cd ILDConfig/StandardConfig/production
```
copy all `.py`, `.sh` and `create_root_tree.xml` files to this folder from `training_data` folder. 


We use `singularity` (with `docker` image) to generate Geant4 showers. Please export necessary singularity **tmp** and **cache** for convenience. 

```
export SINGULARITY_TMPDIR=/nfs/dust/ilc/user/eren/container/tmp/
export SINGULARITY_CACHEDIR=/nfs/dust/ilc/user/eren/container/cache/

```
Please change it for **your** scratch space. Now we can start the instance and run it

```
singularity instance start -H $PWD --bind $(pwd):/home/ilc/data docker://ilcsoft/ilcsoft-centos7-gcc8.2:v02-01-pre instance1
singularity run instance://instance1 ./generateG4-gun.sh instance1

```

This generates **1000 showers**. Play with `gammaGun.mac` if you want to change it.

#### Step 2: Marlin framework to create root files
Now we would like to use `Marlin` framework in `iLCsoft`. `Marlin` takes `lcio` file, which was created in the previous step and creates `root` file

```
## copy create_root_tree.xml and marlin.sh file 
singularity run instance://instance1 ./marlin.sh "photon-shower-instance1.slcio"

```

#### Step 3: Conversion to hdf5 files
It is handy to use `uproot` framework to stream showers from `root` file in order to create `hdf5` file, which is really important for our neutral network achiterctures. 

```
singularity run -H $PWD docker://engineren/pytorch:latest python create_hdf5.py --ncpu 8 --rootfile testNonUniform.root --branch photonSIM --batchsize 100

```

#### Step 4: Remove staggering effects 
Our simulation of ILD calorimeter is a realistic one. That's why we have irregularities in geometry. This causes staggering in `x` direction; we see artifacts (i.e empty lines due to binning). In order to mitigate this effect, we apply a correction and thus remove artifacts.

```
singularity run -H $PWD docker://engineren/pytorch:latest python corrections.py --input test_30x32.hdf5 --output showers-1k.hdf5 --batchsize 100 --minibatch 10
```

choose batchsize and mini-batch size such a way that `total showers = batchsize  * minibatch` (i.e 1000 = 100 * 10 )

#### Structure of HDF5 file

Created file `showers-1k.hdf5` has the following structure: 
* Group named `30x30`
     * `energy`               : Dataset {1000, 1}
     * `layers`               : Dataset {1000, 30,30,30}

As stated in our paper, we have trained our model on 950k showers, which is approximately 200 Gb. That is why we are able to put only a small fraction of our training data to [[`Zenado`](https://zenodo.org/record/3826103#.Xrz1RBZS-EI)]
 
## Architerctures



## Training

Please make sure you downloaded `hdf5` file from Zenado and put it into `training_data` folder.
### Bib-AE

1. Choose BIBAE_Run for BIBAE training or BIBAE_PP_Run for BIBAE with post processing training
2. Modify parameters accoring to your need
3. launch python BIBAE_Run.py or python BIBAE_PP_Run.py 

```
singularity run --nv docker://engineren/pytorch:latest python BIBAE/BIBAE_Run.py

```
