import sys
import numpy as np
from data_utils.data_loader import HDF5Dataset
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.ticker import (MultipleLocator, FormatStrFormatter, AutoMinorLocator)
import torch
from scipy.stats import norm
from scipy.stats import crystalball
import matplotlib.mlab as mlab
import time
import skimage.measure
from matplotlib.colors import ListedColormap
from mpl_toolkits.mplot3d import Axes3D, art3d  # NOQA
from collections import defaultdict
import matplotlib.colors as clr
import types
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.patches import FancyArrowPatch
from mpl_toolkits.mplot3d import proj3d
from mpl_toolkits.axes_grid1 import make_axes_locatable


class Arrow3D(FancyArrowPatch):
    def __init__(self, xs, ys, zs, *args, **kwargs):
        FancyArrowPatch.__init__(self, (0,0), (0,0), *args, **kwargs)
        self._verts3d = xs, ys, zs

    def draw(self, renderer):
        xs3d, ys3d, zs3d = self._verts3d
        xs, ys, zs = proj3d.proj_transform(xs3d, ys3d, zs3d, renderer.M)
        self.set_positions((xs[0],ys[0]),(xs[1],ys[1]))
        FancyArrowPatch.draw(self, renderer)

font = {'family' : 'serif',
        'size'   : 19}
mpl.rc('font', **font)
plt.style.use('classic')
mpl.rc('font', **font)

tf_logscale = lambda x:(np.log((np.sum(x, axis=0)+1.0)))
tf_logscale_rev = lambda x:(torch.exp(x)-1.0)
tf_logscale_rev_np = lambda x:(np.exp(x)-1.0)

latent_mapped = False
latent_params_mu = None
latent_params_sigma = None

global_thresh = 0.1 
global_thresh2 = 0.01

def tf_lin_cut_F(x):
    x[x < global_thresh] = 0.0
    return x

def tf_lin_cut_F2(x):
    x[x < global_thresh2] = 0.0

    return x

global geant_label
global color_list
global fillcolor_list
global linewidth_list
global linestyle_list
global marker_list

geant_label = "Geant4"


def gen_from_model(model, model_type, number, latent_dim, device, E_max, E_min, model_P, model_P2, dataset):
        gen_start = time.time()
            
        if model_type == 'VAE_ENR' or model_type == 'BAE_ENR':
            fake_data, latent_data, data_uncut = getFakeImagesVAE_ENR(model=model, number=number,
                                                          E_max=E_max, E_min=E_min, 
                                                          latent_dim=latent_dim, device=device)
         
            
            
        gen_time = time.time() - gen_start
        print (model_type + ': {:9.4f} Seconds per Shower'.format(gen_time/number))
            
        return fake_data, latent_data, data_uncut
                
                
        
def runPlot_paper(save_locations, models, latent_dims, model_types, model_titles, save_title_comp,
                  number, numberERange, energyRange_max, energyRange_min, energyRange_step, 
                  geant_label_, color_list_, fillcolor_list_, linewidth_list_, linestyle_list_, marker_list_,
                  save_titles, device='cpu', E_max=1.0, E_min=0.1, thresh=0, SEpath='', models_P = None, models_P2 = None):

    
    NumberPerE = numberERange
    NumberERange = 110000
    E_window = 0.02
    
    global geant_label
    global color_list
    global fillcolor_list
    global linewidth_list
    global linestyle_list
    global marker_list
    
    geant_label = geant_label_
    color_list = color_list_
    fillcolor_list = fillcolor_list_
    linewidth_list = linewidth_list_
    linestyle_list = linestyle_list_
    marker_list = marker_list_

    start = time.time()
    [real_data, real_ener, real_data_uncut] = getRealImages(save_locations['Full'], 10000+int(number*(0.9/(E_max-E_min))))
    time_ = time.time() -start
    
    mask_E = ((real_ener < ((E_max)*100.0)) & (real_ener > ((E_min)*100.0)))    
    real_data = real_data[mask_E[:,0]]
    real_ener = real_ener[mask_E[:,0]]
    real_data_uncut = real_data_uncut[mask_E[:,0]]
    
    real_data = real_data[:number]
    real_ener = real_ener[:number]
    real_data_uncut = real_data_uncut[:number]


    r_real, phi_real, e_real = getRadialDistribution(real_data, xbins=30, ybins=30)
    
    hit_real = getHitE(real_data, xbins=30, ybins=30)
    hit_real_uncut = getHitE(real_data_uncut, xbins=30, ybins=30)
    spinal_real = getSpinalProfile(real_data, xbins=30, ybins=30)
    cogz_real = get0Moment(np.sum(real_data, axis=(2,3)))
    
    
        
    arr_dim0 = len(np.arange(start=energyRange_min, stop=energyRange_max, step=energyRange_step))
    arr_dim1 = len(models)

    mu_real        = np.zeros((arr_dim0, 1))
    mu_real_err    = np.zeros((arr_dim0, 1))
    sigma_real     = np.zeros((arr_dim0, 1))
    sigma_real_err = np.zeros((arr_dim0, 1))

    mu_fake        = np.zeros((arr_dim0, arr_dim1))
    mu_fake_err    = np.zeros((arr_dim0, arr_dim1))
    sigma_fake     = np.zeros((arr_dim0, arr_dim1))
    sigma_fake_err = np.zeros((arr_dim0, arr_dim1))

    e_list         = np.zeros((arr_dim0, 1))
    fit_array_counter = 0

    fake_datas = []
    fake_datas_uncut = []
    latent_datas = []
    for j in range(0, len(models)):
        temp, lat, temp_uncut = gen_from_model(model=models[j], model_type=model_types[j], 
                                               number=number, latent_dim=latent_dims[j],
                                               device=device, E_max=E_max, E_min=E_min, 
                                               model_P = models_P[j], model_P2 = models_P2[j],
                                               dataset=(real_data_uncut, real_ener))
        fake_datas.append(temp)
        fake_datas_uncut.append(temp_uncut)
        latent_datas.append(lat)


    r_fakes, phi_fakes, e_fakes = [], [], []
    hit_fakes, hit_fakes_uncut = [], []
    spinal_fakes, cogz_fakes = [], []


    for j in range(0, len(models)):
        r, phi, e = getRadialDistribution(fake_datas[j], xbins=30, ybins=30)      
        r_fakes.append(r)
        phi_fakes.append(phi)
        e_fakes.append(e)

        hit = getHitE(fake_datas[j], xbins=30, ybins=30)
        hit_fakes.append(hit)

        hit_uncut = getHitE(fake_datas_uncut[j], xbins=30, ybins=30)
        hit_fakes_uncut.append(hit_uncut)

        spinal = getSpinalProfile(fake_datas[j], xbins=30, ybins=30)
        spinal_fakes.append(spinal)

        cogz = get0Moment(np.sum(fake_datas[j], axis=(2,3)))
        cogz_fakes.append(cogz)




    cor_real,name_list = runCorrelations(real_data, real_ener, Model_title=geant_label, save_title='_GEANT4')
    for j in range(0, len(models)):
        cor_fake, _ = runCorrelations(fake_datas[j], latent_datas[j][:,-1:], 
                                      Model_title=model_titles[j], save_title=save_title_comp+save_titles[j])
        differenceCorrelations(cor_real, cor_fake, name_list,
                               Model_title=model_titles[j], save_title=save_title_comp+save_titles[j])



    try:
        plt_RadialE(data_real=r_real, data_fake=r_fakes, energy_real=e_real, energy_fake=e_fakes,
                    model_title=model_titles, save_title=save_title_comp, 
                    number=real_data.shape[0], numberf=fake_datas[0].shape[0], showLegend=True)
    except:
        plt_RadialE(data_real=r_real, data_fake=r_fakes, energy_real=e_real, energy_fake=e_fakes,
                    model_title=model_titles, save_title=save_title_comp, 
                    number=real_data.shape[0], numberf=real_data[0].shape[0], showLegend=True)


    plt_SpinalE(spinal_real, data_fake=spinal_fakes, 
                model_title=model_titles, save_title=save_title_comp, showLegend=True)

    plt_HitE_uncut(data_real=hit_real_uncut, data_fake=hit_fakes_uncut, 
                   data_real_cut=hit_real, data_fake_cut=hit_fakes, 
             model_title=model_titles, save_title=save_title_comp+'_Uncut', 
             n_bins = 70, E_max = 30, E_min = 0.01, y_max = 0.1, y_min = 0.001)

    plt_CenterOfGravity(data_real=cogz_real, data_fake=cogz_fakes, data_fake2=None, 
             model_title=model_titles, save_title=save_title_comp, showLegend=True)

    time_ = time.time() -start

    energy1 = 20.0
    energy2 = 50.0
    energy3 = 80.0

    e_sum1_real, e_sum1_fakes = None, None
    e_sum2_real, e_sum2_fakes = None, None
    e_sum3_real, e_sum3_fakes = None, None

    occ1_real, occ1_fakes = None, None
    occ2_real, occ2_fakes = None, None
    occ3_real, occ3_fakes = None, None



    [glob_real_data_ME, glob_real_Energy_ME, glob_real_data_ME_uncut] = getRealImages(save_locations['Single'], NumberERange)
    glob_sort = glob_real_Energy_ME[:,0].argsort()


    print(energyRange_min, energyRange_max, energyRange_step)
    print(np.arange(start=energyRange_min, stop=energyRange_max, step=energyRange_step))
    for e in np.arange(start=energyRange_min, stop=energyRange_max, step=energyRange_step):
        #[real_data_ME, real_Energy_ME, real_data_ME_uncut] = getRealImages(save_locations['{:d}'.format(int(e*100))], NumberERange)
        [real_data_ME, real_Energy_ME, real_data_ME_uncut] =  [glob_real_data_ME, glob_real_Energy_ME, glob_real_data_ME_uncut]
        #sort = real_Energy_ME[:,0].argsort()
        sort = glob_sort
        real_data_ME_Sort = real_data_ME[sort]
        real_data_ME_uncut_Sort = real_data_ME_uncut[sort]
        real_Energy_ME_Sort = real_Energy_ME[sort]
        #print(np.unique(real_Energy_ME))

        mask_ME = ((real_Energy_ME_Sort < ((e+E_window)*100)) & 
                    (real_Energy_ME_Sort > ((e-E_window)*100)))    

        #print(mask_ME)
        #print((e+E_window)*100, (e-E_window)*100)

        real_dataME = real_data_ME_Sort[mask_ME[:,0]]
        real_energyME = real_Energy_ME_Sort[mask_ME[:,0]]
        real_dataME_uncut = real_data_ME_uncut_Sort[mask_ME[:,0]]
        #print(real_Energy_ME_Sort[mask_ME[:,0]])
        #print(real_Energy_ME_Sort)
        #print(real_dataME.shape)
        real_dataME = real_dataME[0:NumberPerE]
        real_dataME_uncut = real_dataME_uncut[0:NumberPerE]
        real_energyME = real_energyME[0:NumberPerE]



        fake_datasME = []
        fake_datasME_uncut = []
        latent_datasME = []
        for j in range(0, len(models)):
            temp, lat, temp_uncut = gen_from_model(model=models[j], model_type=model_types[j], 
                                       number=NumberPerE, latent_dim=latent_dims[j],
                                       device=device, E_max=e+0, E_min=e-0, model_P = models_P[j], 
                                       model_P2 = models_P2[j], dataset=(real_dataME_uncut, real_energyME))
            fake_datasME.append(temp)
            latent_datasME.append(lat)
            fake_datasME_uncut.append(temp_uncut)


        etot_RealME = getTotE(real_dataME, xbins=30, ybins=30)
        occ_realME = getOcc(real_dataME, xbins=30, ybins=30)    

        etot_fakesME, occ_fakesME = [], []


        for j in range(0, len(models)):               
            etot = getTotE(fake_datasME[j], xbins=30, ybins=30)
            etot_fakesME.append(etot)     

            occ = getOcc(fake_datasME[j], xbins=30, ybins=30)    
            occ_fakesME.append(occ)




        if (int(e*100)) == energy1:
            e_sum1_real = etot_RealME
            e_sum1_fakes = etot_fakesME
            occ1_real = occ_realME
            occ1_fakes = occ_fakesME


        if (int(e*100)) == energy2:
            e_sum2_real = etot_RealME
            e_sum2_fakes = etot_fakesME
            occ2_real = occ_realME
            occ2_fakes = occ_fakesME


        if (int(e*100)) == energy3:
            e_sum3_real = etot_RealME
            e_sum3_fakes = etot_fakesME
            occ3_real = occ_realME
            occ3_fakes = occ_fakesME



        mu, sigma, mu_err, sigma_err = fit90(etot_RealME)

        mu_real[fit_array_counter,0]        = (mu)
        mu_real_err[fit_array_counter,0]    = (mu_err)
        sigma_real[fit_array_counter,0]     = (sigma)
        sigma_real_err[fit_array_counter,0] = (sigma_err)


        for j in range(0, len(models)):
            mu, sigma, mu_err, sigma_err = fit90(etot_fakesME[j])

            mu_fake[fit_array_counter,j] =        (mu)
            mu_fake_err[fit_array_counter,j] =    (mu_err)
            sigma_fake[fit_array_counter,j] =     (sigma)
            sigma_fake_err[fit_array_counter,j] = (sigma_err)


        e_list[fit_array_counter,0] = (int(e*100))
        fit_array_counter += 1

        if abs(e-0.5) < 0.01:                          
            plt_ExampleImage(real_dataME, model_title='Geant4 {:d} GeV showers'.format(int(e*100.0)), 
                           save_title='Shower_Real_{:d}_GeV'.format(int(e*100.0)), draw_3D=True, n=1)



            for j in range(0, len(models)):
                plt_ExampleImage(fake_datasME[j], model_title=model_titles[j]+' shower {:d} GeV'.format(int(e*100.0)),
                           save_title='Shower_'+save_title_comp+save_titles[j]+'_{:d}_GeV'.format(int(e*100.0)), draw_3D=True)


    plt_singleEnergy_3ofaKind(data_real=e_sum1_real, data_real2=e_sum2_real, data_real3=e_sum3_real,
                              data_fake=e_sum1_fakes, data_fake2=e_sum2_fakes, data_fake3=e_sum3_fakes,
                              energy1=energy1, energy2=energy2, energy3=energy3,
                              model_title=model_titles, save_title=save_title_comp, showLegend=True)    


    plt_Occupancy_3ofaKind(data_real=occ1_real, data_real2=occ2_real, data_real3=occ3_real, 
                           data_fake=occ1_fakes, data_fake2=occ2_fakes, data_fake3=occ3_fakes,      
                           energy1=energy1, energy2=energy2, energy3=energy3,
                           model_title=model_titles, save_title=save_title_comp, showLegend=True)    



    ratio_sigma_real_err = np.sqrt((sigma_real_err/sigma_real)**2
                                  +(mu_real_err/mu_real)**2)*(sigma_real/mu_real)
    ratio_sigma_fake_err = np.sqrt((sigma_fake_err/sigma_fake)**2
                                  +(mu_fake_err/mu_fake)**2)*(sigma_fake/mu_fake)

    ratio_sigma2_real_err = np.sqrt((sigma_real_err/sigma_real)**2
                               +(sigma_real_err*sigma_real/(sigma_real**2))**2)
    ratio_sigma2_fake_err = np.sqrt((sigma_fake_err/sigma_real)**2
                               +(sigma_real_err*sigma_fake/(sigma_real**2))**2)



    plt_scatter_error_subplot(x_data=e_list, 
                 x_data_sub=e_list, 
                 x_data_error=None,  
                 x_data_sub_error=None, 

                 data_real=(sigma_real/mu_real), 
                 data_real_error=ratio_sigma_real_err, 
                 data_real_sub=(sigma_real-sigma_real)/sigma_real,
                 data_real_sub_error=ratio_sigma2_real_err, 

                 data_fake = (sigma_fake/mu_fake),
                 data_fake_error = ratio_sigma_fake_err,
                 data_fake_sub=(sigma_fake-sigma_real)/sigma_real,
                 data_fake_sub_error=ratio_sigma2_fake_err, 

                 data_fake2 = None, 
                 data_fake2_error = None, 
                 data_fake2_sub = None, 
                 data_fake2_sub_error = None, 

                 model_title=model_titles ,save_title=save_title_comp+"_Calo_Width_relative.png", 
                 plt_title=' ', 
                 x_title='photon energy [GeV]', x_fontsize = 18,
                 y_title='$\\frac{\\sigma_{90}}{\\mu_{90}}$', y_fontsize = 25,
                 minX=10, maxX=100, minY=0.0, maxY=0.15, yloc=0.01,
                 x_title_sub='photon energy [GeV]', x_subfontsize = 18,
                 y_title_sub='$\\frac{\\sigma_{90}-\\sigma_{90}^{\\mathrm{G4}}}{\\sigma_{90}^{\\mathrm{G4}}}$', 
                 y_subfontsize = 25,
                 minX_sub=10, maxX_sub=100, maxY_sub=0.40, minY_sub=-0.40, yloc_sub=0.05, showLegend=False)                





    ratio_mu_real_err = np.sqrt((mu_real_err/mu_real)**2
                               +(mu_real_err*mu_real/(mu_real**2))**2)
    ratio_mu_fake_err = np.sqrt((mu_fake_err/mu_real)**2
                               +(mu_real_err*mu_fake/(mu_real**2))**2)


    plt_scatter_error_subplot(x_data=e_list, 
                 x_data_sub=e_list, 
                 x_data_error=None,  
                 x_data_sub_error=None, 

                 data_real=mu_real, 
                 data_real_error=mu_real_err, 
                 data_real_sub=(mu_real-mu_real)/mu_real, 
                 data_real_sub_error=ratio_mu_real_err, 

                 data_fake = mu_fake,
                 data_fake_error = mu_fake_err,
                 data_fake_sub=(mu_fake-mu_real)/mu_real, 
                 data_fake_sub_error=ratio_mu_fake_err, 

                 data_fake2 = None, 
                 data_fake2_error = None, 
                 data_fake2_sub = None, 
                 data_fake2_sub_error = None, 

                 model_title=model_titles ,save_title=save_title_comp+"_Calo_Mean_relative.png", 
                 plt_title=' ', 
                 x_title='photon energy [GeV]',x_fontsize = 18,
                 y_title='$\\mu_{90} \\mathrm{[MeV]}$', y_fontsize = 20,
                 minX=10, maxX=100, minY=0, maxY=2000, yloc=100,
                 x_title_sub='photon energy [GeV]',  x_subfontsize = 18,
                 y_title_sub='$\\frac{\\mu_{90}-\\mu_{90}^{\\mathrm{G4}}}{\\mu_{90}^{\\mathrm{G4}}}$', 
                 y_subfontsize = 25,
                 minX_sub=10, maxX_sub=100, maxY_sub=0.045, minY_sub=-0.045, yloc_sub=0.006, showLegend=True)                



        
        
        
        
def getRealImages(filepath, number):
    tf = tf_lin_cut_F 

    dataset_physeval = HDF5Dataset(filepath, transform=tf, train_size=number)
    data = dataset_physeval.get_data_range_tf(0, number)
    ener = dataset_physeval.get_energy_range(0, number)
    
    tf = tf_lin_cut_F2
    dataset_physeval = HDF5Dataset(filepath, transform=tf, train_size=number)
    data_uncut = dataset_physeval.get_data_range_tf(0, number)
    return [data, ener, data_uncut]


def getFakeImagesVAE_ENR(model, number, E_max, E_min, latent_dim, device='cpu', thresh=0.0):
    batchsize = 100
    fake_list = []
    latent_list=[]
    fake_uncut_list = []

    model.eval()

    for i in np.arange(0, number, batchsize):
        with torch.no_grad():
            x = torch.zeros(batchsize, latent_dim, device=device)
            E = (torch.rand(batchsize, 1, device=device)*(E_max-E_min)+E_min)*100
            latent = torch.cat((x, E), dim=1)
            data = model(x=x, E_true=E, 
                            z = torch.randn(batchsize, latent_dim),  mode='decode')
            data = data.cpu().numpy()
            latent = latent.cpu().numpy()

        data_uncut = np.array(data)
        fake_uncut_list.append(data_uncut)
        data[data < global_thresh] = 0.0  
        fake_list.append(data)
        latent_list.append(latent)
        print(i)

    data_full = np.vstack(fake_list)
    latent_full = np.vstack(latent_list)
    data_uncut_full = np.vstack(fake_uncut_list)
    
    print(data_full.shape)

        
    return data_full[:,0,:,:,:],latent_full,data_uncut_full[:,0,:,:,:]


def getRadialDistribution(data, xbins=30, ybins=30, layers=30):
    current = np.reshape(data,[-1, layers, xbins,ybins])
    current_sum = np.sum(current, axis=(0,1))
 
    r_list=[]
    phi_list=[]
    e_list=[]
    n_cent_x = (xbins-1)/2.0
    n_cent_y = (ybins-1)/2.0

    for n_x in np.arange(0, xbins):
        for n_y in np.arange(0, ybins):
            if current_sum[n_x,n_y] != 0.0:
                r = np.sqrt((n_x - n_cent_x)**2 + (n_y - n_cent_y)**2)
                r_list.append(r)
                phi = np.arctan((n_x - n_cent_x)/(n_y - n_cent_y))
                phi_list.append(phi)
                e_list.append(current_sum[n_x,n_y])
                
    r_arr = np.asarray(r_list)
    phi_arr = np.asarray(phi_list)
    e_arr = np.asarray(e_list)

    return r_arr, phi_arr, e_arr

def get0Moment(x):
    n, l = x.shape
    tiles = np.tile(np.arange(l), (n,1))
    y = x * tiles
    y = y.sum(1)
    y = y/x.sum(1)
    return y

def get1Moment(x):
    i=1
    n, l = x.shape
    tiles = np.tile(np.arange(l), (n,1))
    y = x * (tiles - get0Moment(x).reshape(-1,1))**(i+1)
    y = y.sum(1)
    y = y/x.sum(1)
    return y

def getXMoment(x, i):
    if i == 0:
        return get0Moment(x)
    else:
        n, l = x.shape
        tiles = np.tile(np.arange(l), (n,1))
        y = x * (tiles - get0Moment(x).reshape(-1,1))**(i+1)
        y = y.sum(1)
        y = y/x.sum(1)
        return y



def runCorrelations(data, energy, Model_title='ML Model', save_title='ML_model', xbins=30, ybins=30, layers=30):
    #First calculate relevant parameters from shower
    import seaborn as sns
    import pandas
    #Moments data[B,Z,X,Y]
    nMax = 30000
    data = data[0:nMax]
    
    name_list = []
    data_list = []
    
    Moment_0_x = get0Moment(np.sum(data, axis=(1,3)))
    Moment_0_y = get0Moment(np.sum(data, axis=(1,2)))
    Moment_0_z = get0Moment(np.sum(data, axis=(2,3)))
    data_list.append(Moment_0_x)
    data_list.append(Moment_0_y)
    data_list.append(Moment_0_z)    
    name_list.append('$m_{1, x}$')
    name_list.append('$m_{1, y}$')
    name_list.append('$m_{1, z}$')

    Moment_1_x = get1Moment(np.sum(data, axis=(1,3)))
    Moment_1_y = get1Moment(np.sum(data, axis=(1,2)))
    Moment_1_z = get1Moment(np.sum(data, axis=(2,3)))
    data_list.append(Moment_1_x)
    data_list.append(Moment_1_y)
    data_list.append(Moment_1_z)    
    #print(Moment_1_z.shape)
    name_list.append('$m_{2, x}$')
    name_list.append('$m_{2, y}$')
    name_list.append('$m_{2, z}$')

    ecal_sum = getTotE(data)
    data_list.append(ecal_sum)    
    name_list.append('$E_{\\mathrm{vis}}$')
    #print(ecal_sum.shape)

    
    
    p_energy = energy[0:nMax, 0]
    data_list.append(p_energy)    
    name_list.append('$E_{\\mathrm{inc}}$')
    #print(p_energy)

    hits = getOcc(data)
    data_list.append(hits)    
    name_list.append('$n_{\\mathrm{hit}}$')
    #print(hits.shape)

    ratio1_total = np.sum(data[:,0:10], axis=(1,2,3))/ecal_sum
    ratio2_total = np.sum(data[:,10:20], axis=(1,2,3))/ecal_sum
    ratio3_total = np.sum(data[:,20:30], axis=(1,2,3))/ecal_sum
    data_list.append(ratio1_total)    
    data_list.append(ratio2_total)    
    data_list.append(ratio3_total)    
    #print(ratio1_total.shape)

    name_list.append('$E_{1}/E_{\\mathrm{vis}}$')
    name_list.append('$E_{2}/E_{\\mathrm{vis}}$')
    name_list.append('$E_{3}/E_{\\mathrm{vis}}$')
    
    
  
    df_data = pandas.DataFrame(data=np.vstack(data_list).transpose(), columns = name_list)
    correlations_full = df_data.corr()
    
    upper_triangle = np.ones((len(name_list),len(name_list)))
    upper_triangle[np.triu_indices(len(name_list),1)] = 0.0
    #print(upper_triangle)
    
    correlations = correlations_full.mask(upper_triangle == 0)
    
    fig_cor = plt.figure(figsize=(10,10))
    ax_cor = fig_cor.add_subplot(1,1,1)
    plt.gcf().subplots_adjust(left=0.2, bottom=0.2)

    temp1 = plt.rcParams['font.size']
    temp2 = plt.rcParams['font.family']
    
    #cmap=sns.diverging_palette(5, 250, as_cmap=True),
    #cmap=sns.color_palette(RdBu_r, as_cmap=True),
    
    plt.rcParams['font.size'] = 50 
    plt.rcParams['font.family'] = "serif"
    g = sns_plot = sns.heatmap((correlations),
        xticklabels=correlations.columns,
        yticklabels=correlations.index,
        cmap=sns.diverging_palette(5, 250, as_cmap=True),
        annot=True, ax=ax_cor, vmin=-1, vmax=1,
        annot_kws={"size": 14}, fmt=".2f", square=True, cbar=False, linewidths=0, linecolor='black'
        )
    
    ax_cor.axhline(y=0, color='k',linewidth=4)
    ax_cor.axhline(y=correlations.shape[1], color='k',linewidth=4)
    ax_cor.axvline(x=0, color='k',linewidth=4)
    ax_cor.axvline(x=correlations.shape[0], color='k',linewidth=4)

    g.tick_params(axis='both', which='both', length=0)
    g.set_xticklabels(g.get_xmajorticklabels(), fontsize = 26, rotation='vertical')
    g.set_yticklabels(g.get_ymajorticklabels(), fontsize = 26, rotation='horizontal')

    for tick in g.get_xmajorticklabels():
        tick.set_fontname("serif")

    for tick in g.get_ymajorticklabels():
        tick.set_fontname("serif")
    fig_cor.suptitle(Model_title, fontsize='29')
    fig_cor.savefig('./plots/' + save_title+'_correlations.png')

    ax_cor.patch.set_facecolor('white')
    fig_cor.patch.set_facecolor('white')

    
    plt.rcParams['font.size'] = temp1
    plt.rcParams['font.family'] = temp2

    return correlations_full, name_list




def differenceCorrelations(cor_real, cor_fake, name_list, Model_title='ML Model', save_title='ML_model', xbins=30, ybins=30, layers=30):
    #First calculate relevant parameters from shower
    import seaborn as sns
    import pandas
    correlations_full = cor_real - cor_fake
    
    upper_triangle = np.ones((len(name_list),len(name_list)))
    upper_triangle[np.triu_indices(len(name_list),1)] = 0.0
    
    correlations = correlations_full.mask(upper_triangle == 0)
    
    cor_np = np.nan_to_num(correlations.to_numpy())
    
    print(geant_label + ' - '+Model_title + ' Mean Abs', np.sum(np.abs(cor_np)))
    
    fig_cor = plt.figure(figsize=(10,10))
    ax_cor = fig_cor.add_subplot(1,1,1)
    plt.gcf().subplots_adjust(left=0.2, bottom=0.2)

    temp1 = plt.rcParams['font.size']
    temp2 = plt.rcParams['font.family']
    
    cmap = sns.diverging_palette(290, 120, as_cmap=True)
    
    plt.rcParams['font.size'] = 50 
    plt.rcParams['font.family'] = "serif"
    g = sns_plot = sns.heatmap(correlations,
        xticklabels=correlations.columns,
        yticklabels=correlations.index,
        cmap=cmap,
        annot=True, ax=ax_cor, vmin=-1, vmax=1,
        annot_kws={"size": 14}, fmt=".2f", square=True, cbar=False
        )


    g.set_xticklabels(g.get_xmajorticklabels(), fontsize = 30, rotation='vertical')
    g.set_yticklabels(g.get_ymajorticklabels(), fontsize = 30, rotation='horizontal')


    for tick in g.get_xmajorticklabels():
        tick.set_fontname("serif")

    for tick in g.get_ymajorticklabels():
        tick.set_fontname("serif")
    fig_cor.suptitle(geant_label + ' - '+Model_title, fontsize='29')

    ax_cor.patch.set_facecolor('white')
    fig_cor.patch.set_facecolor('white')
    ax_cor.axhline(y=0, color='k',linewidth=4)
    ax_cor.axhline(y=correlations.shape[1], color='k',linewidth=4)
    ax_cor.axvline(x=0, color='k',linewidth=4)
    ax_cor.axvline(x=correlations.shape[0], color='k',linewidth=4)

    g.tick_params(axis='both', which='both', length=0)

    fig_cor.savefig('./plots/' + save_title+'_correlations_diff.png')

    plt.rcParams['font.size'] = temp1
    plt.rcParams['font.family'] = temp2

    
    return correlations_full
        
def getOcc(data, xbins=30, ybins=30, layers=30):
    data = np.reshape(data,[-1, layers*xbins*ybins])
    occ_arr = (data > 0.0).sum(axis=(1))
    return occ_arr


def getTotE(data, xbins=30, ybins=30, layers=30):
    data = np.reshape(data,[-1, layers*xbins*ybins])
    etot_arr = np.sum(data, axis=(1))
    return etot_arr

def getSpinalProfile(data, xbins=30, ybins=30, layers=30):
    data = np.reshape(data,[-1, layers, xbins*ybins])
    etot_arr = np.sum(data, axis=(2))
    return etot_arr


def getHitE(data, xbins=30, ybins=30, layers=30):
    ehit_arr = np.reshape(data,[data.shape[0]*xbins*ybins*layers])
    #etot_arr = np.trim_zeros(etot_arr)
    ehit_arr = ehit_arr[ehit_arr != 0.0]
    return ehit_arr


def interval_quantile_(x, quant=0.9):
    """Calculate the shortest interval that contains the desired quantile"""
    x = np.sort(x)
    # the number of possible starting points
    n_low = int(len(x) * (1 - quant))
    # the number of events contained in the quantil
    n_quant = len(x) - n_low

    # Calculate all distances in one go
    distances = x[-n_low:] - x[:n_low]
    i_start = np.argmin(distances)

    return i_start, i_start + n_quant



def fit90(x): 
    n10percent = int(round(len(x)*0.1))
    n90percent = len(x) - n10percent
    
    start, end = interval_quantile_(x, quant=0.9)
    
    rms90 = np.std(x[start:end])
    mean90 = np.mean(x[start:end])
    mean90_err = rms90/np.sqrt(n90percent)
    rms90_err = rms90/np.sqrt(2*n90percent)   # estimator in root
    return mean90, rms90, mean90_err, rms90_err


def plt_ExampleImage(image, model_title='ML Model', save_title='ML_model', draw_3D=False, n=1):
    
    cmap = mpl.cm.viridis
    cmap.set_bad('white',1.)
    
    for k in range(0,n):
        figExIm = plt.figure(figsize=(6,6))
        axExIm1 = figExIm.add_subplot(1,1,1)
        image1 = np.sum(image[k], axis=0)
        masked_array1 = np.ma.array(image1, mask=(image1==0.0))
        im1 = axExIm1.imshow(masked_array1, filternorm=False, interpolation='none', cmap = cmap, vmin=0.01, vmax=100,
                           norm=mpl.colors.LogNorm(), origin='lower')
        figExIm.patch.set_facecolor('white')
        axExIm1.title.set_text(model_title)
        axExIm1.set_xlabel('y [cells]', family='serif')
        axExIm1.set_ylabel('x [cells]', family='serif')
        figExIm.colorbar(im1)
        plt.savefig('./plots/' + save_title+"_CollapseZ_{:d}.png".format(k))
        plt.savefig('./plots/' + save_title+"_CollapseZ_{:d}.pdf".format(k))

        figExIm = plt.figure(figsize=(6,6))
        axExIm2 = figExIm.add_subplot(1,1,1)    
        image2 = np.sum(image[k], axis=1)
        masked_array2 = np.ma.array(image2, mask=(image2==0.0))
        im2 = axExIm2.imshow(masked_array2, filternorm=False, interpolation='none', cmap = cmap, vmin=0.01, vmax=100,
                           norm=mpl.colors.LogNorm(), origin='lower') 
        figExIm.patch.set_facecolor('white')
        axExIm2.title.set_text(model_title)
        axExIm2.set_xlabel('y [cells]', family='serif')
        axExIm2.set_ylabel('z [layers]', family='serif')
        figExIm.colorbar(im2)
        plt.savefig('./plots/' + save_title+"_CollapseX_{:d}.png".format(k))
        plt.savefig('./plots/' + save_title+"_CollapseX_{:d}.pdf".format(k))

        figExIm = plt.figure(figsize=(6,6))
        axExIm3 = figExIm.add_subplot(1,1,1)
        image3 = np.sum(image[k], axis=2)
        masked_array3 = np.ma.array(image3, mask=(image3==0.0))
        im3 = axExIm3.imshow(masked_array3, filternorm=False, interpolation='none', cmap = cmap, vmin=0.01, vmax=100,
                           norm=mpl.colors.LogNorm(), origin='lower')
        figExIm.patch.set_facecolor('white')
        axExIm3.title.set_text(model_title)
        axExIm3.set_xlabel('x [cells]', family='serif')
        axExIm3.set_ylabel('z [layers]', family='serif')
        figExIm.colorbar(im3)
        plt.savefig('./plots/' + save_title+"_CollapseY_{:d}.png".format(k))
        plt.savefig('./plots/' + save_title+"_CollapseY_{:d}.pdf".format(k))
    

    figExIm = plt.figure(figsize=(6,6))
    axExIm1 = figExIm.add_subplot(1,1,1)
    image1 = np.mean(np.sum(image, axis=1), axis=0)#+1.0
    masked_array1 = np.ma.array(image1, mask=(image1==0.0))
    im1 = axExIm1.imshow(masked_array1, filternorm=False, interpolation='none', cmap = cmap, vmin=0.01, vmax=100,
                       norm=mpl.colors.LogNorm(), origin='lower')
    figExIm.patch.set_facecolor('white')
    axExIm1.title.set_text(model_title)
    axExIm1.set_xlabel('y [cells]', family='serif')
    axExIm1.set_ylabel('x [cells]', family='serif')
    figExIm.colorbar(im1)
    plt.savefig('./plots/' + save_title+"_CollapseZSum.png")

    figExIm = plt.figure(figsize=(6,6))
    axExIm2 = figExIm.add_subplot(1,1,1)    
    image2 = np.mean(np.sum(image, axis=2), axis=0)#+1.0
    masked_array2 = np.ma.array(image2, mask=(image2==0.0))
    im2 = axExIm2.imshow(masked_array2, filternorm=False, interpolation='none', cmap = cmap, vmin=0.01, vmax=100,
                       norm=mpl.colors.LogNorm(), origin='lower') 
    figExIm.patch.set_facecolor('white')
    axExIm2.title.set_text(model_title)
    axExIm2.set_xlabel('y [cells]', family='serif')
    axExIm2.set_ylabel('z [layers]', family='serif')
    figExIm.colorbar(im2)
    plt.savefig('./plots/' + save_title+"__CollapseXSum.png")
   
    figExIm = plt.figure(figsize=(6,6))
    axExIm3 = figExIm.add_subplot(1,1,1)    
    image3 = np.mean(np.sum(image, axis=3), axis=0)#+1.0
    masked_array3 = np.ma.array(image3, mask=(image3==0.0))
    im3 = axExIm3.imshow(masked_array3, filternorm=False, interpolation='none', cmap = cmap, vmin=0.01, vmax=100,
                       norm=mpl.colors.LogNorm(), origin='lower')
    figExIm.patch.set_facecolor('white')
    axExIm3.title.set_text(model_title)
    axExIm3.set_xlabel('x [cells]', family='serif')
    axExIm3.set_ylabel('z [layers]', family='serif')
    figExIm.colorbar(im3)
    plt.savefig('./plots/' + save_title+"_CollapseYSum.png")
    
    
    figExImModeCol = plt.figure(figsize=(3*5+4,5+1))
    axExIm1ModeCol = figExImModeCol.add_subplot(1,3,1)
    axExIm2ModeCol = figExImModeCol.add_subplot(1,3,2)
    axExIm3ModeCol = figExImModeCol.add_subplot(1,3,3)

    image_sum = np.mean(image, axis=0)
    
    image1 = image_sum[8]
    image2 = image_sum[15]
    image3 = image_sum[22]
    
        
    n_images = image.shape[0]
    
    masked_array1 = np.ma.array(image1, mask=(image1==0.0))
    masked_array2 = np.ma.array(image2, mask=(image2==0.0))
    masked_array3 = np.ma.array(image3, mask=(image3==0.0))
    cmap = mpl.cm.viridis
    cmap.set_bad('white',1.)

    im1 = axExIm1ModeCol.imshow(masked_array1, filternorm=False, interpolation='none', cmap = cmap, vmin=0.01, vmax=100,
                       norm=mpl.colors.LogNorm())
    im2 = axExIm2ModeCol.imshow(masked_array2, filternorm=False, interpolation='none', cmap = cmap, vmin=0.01, vmax=100,
                       norm=mpl.colors.LogNorm())
    im3 = axExIm3ModeCol.imshow(masked_array3, filternorm=False, interpolation='none', cmap = cmap, vmin=0.01, vmax=100,
                       norm=mpl.colors.LogNorm())

    figExImModeCol.patch.set_facecolor('white')
    axExIm2ModeCol.title.set_text(model_title + ' sum {0:d} images, {1:d}/27,000 pixels activte'.format(n_images, 
                                                                                    np.count_nonzero(image_sum)) )

    figExImModeCol.colorbar(im3)
    plt.savefig('./plots/' + save_title+"_ModeCollaps.png")

    
def truncate_colormap(cmap, minval=0.0, maxval=1.0, n=100):
    new_cmap = clr.LinearSegmentedColormap.from_list(
        'trunc({n},{a:.2f},{b:.2f})'.format(n=cmap.name, a=minval, b=maxval),
        cmap(np.linspace(minval, maxval, n)))
    return new_cmap


def plt_3dShowers(images, model_titles, save_titles):
    vmin = 10.0
    vmax = 0.1
    
    for ima in images:
        vmin = min(vmin, np.min(ima[np.nonzero(ima)]))
        vmax = max(vmax, np.max(ima[np.nonzero(ima)]))
        
    print('vmin, vmax', vmin, vmax)
        
    for i in range(len(images)):
        plt_3dShower(images[i], model_title=model_titles[i], save_title=save_titles[i], vmax=vmax, vmin=vmin)
    

        
def plt_3dShower(image, model_title='ML Model', save_title='ML_model', vmax=None, vmin=None):
    
    if save_title+".png" == 'Shower_Real_50_GeV.png':
        figExIm = plt.figure(figsize=(20,16))
    else:
        figExIm = plt.figure(figsize=(16,16))

    axExIm1 = figExIm.gca(projection='3d')
    image = image+0.0


    masked_array = np.ma.array(image, mask=(image==0.0))
    cmap = mpl.cm.viridis
    axExIm1.view_init(elev=20.0, azim=290.0)
    xL,yL,zL,cL = [],[],[],[]
    for index, c in np.ndenumerate(masked_array):
        (x,y,z) = index
        if c != 0:
            xtmp = x
            if xtmp%2 == 0:
                xtmp = xtmp + 0
            else:
                xtmp = xtmp - 0

            xL.append(xtmp)
            yL.append(y)
            zL.append(z)
            cL.append(c)

    cmap = mpl.cm.viridis
    cmap.set_bad('white',1.)

    xL = np.array(xL)
    yL = np.array(yL)
    zL = np.array(zL)
    cL = np.array(cL)
    figExIm.patch.set_facecolor('white')
    
    cmap = mpl.cm.jet
    my_cmap = truncate_colormap(cmap, 0.0, 0.7)
    transparent = (0.1, 0.1, 0.9, 0.0)

    axExIm1.set_xticklabels([])
    axExIm1.set_yticklabels([])
    axExIm1.set_zticklabels([])
    axExIm1.set_xticks(np.arange(0,31,1))
    axExIm1.set_yticks(np.arange(0,31,1))
    axExIm1.set_zticks(np.arange(0,31,1))    
    
    axExIm1.set_xlabel('z [layers]', family='serif', fontsize='35')
    axExIm1.set_ylabel('x [cells]', family='serif', fontsize='35')
    axExIm1.set_zlabel('y [cells]', family='serif', fontsize='35')
    
    axExIm1.set_xlim([0, 30])
    axExIm1.set_ylim([0, 30])
    axExIm1.set_zlim([0, 30])
    
    a = Arrow3D([-7, 3], [15.0, 15.0], 
                [15.0, 15.0], mutation_scale=20, 
                lw=3, arrowstyle="-|>", color="k")
    axExIm1.add_artist(a)
    

    
    plotMatrix(ax=axExIm1, x=xL, y=yL, z=zL, data=cL, cmap=my_cmap, alpha=0.7, edgecolors=transparent, vmax=vmax, vmin=vmin)    
    if save_title+".png" == 'Shower_Real_50_GeV.png':
        norm = mpl.colors.LogNorm(vmin=0.099, vmax=vmax)
        cbar = figExIm.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap=my_cmap), ax=axExIm1)
        
        cbar.ax.tick_params(labelsize='35') 

    plt.savefig('./plots/' + save_title+".png")
    

      
def plotMatrix(ax, x, y, z, data, cmap="jet", cax=None, alpha=0.1, edgecolors=None, vmax=None, vmin=None):
    # plot a Matrix 
    norm = mpl.colors.LogNorm(vmin=vmin, vmax=vmax)
    colors = lambda i : mpl.cm.ScalarMappable(norm=norm, cmap = cmap).to_rgba(data[i]) 
    norm_max = vmax


    for i, xi in enumerate(x):
        alp2 = 0.1+0.9*np.log(data[i]*10)/np.log(norm_max*10)
        plotCubeAt(pos=(x[i], y[i], z[i]), l=(0.1, 0.8, 0.8), c=colors(i), c2=colors(i), alpha=alp2,  ax=ax, edgecolors=edgecolors)


        
def plotCubeAt(pos=(0,0,0), l=(1.0,1.0,1.0), c="b", c2="k", alpha=0.1, ax=None, edgecolors=None):
    # Plotting N cube elements at position pos
    if ax !=None:
        x_range = np.array([[pos[0], pos[0]+l[0]]])
        y_range = np.array([[pos[1], pos[1]+l[1]]])
        z_range = np.array([[pos[2], pos[2]+l[2]]])
        
        z_range
        xx, yy = np.meshgrid(x_range, y_range)
        
        ax.plot_surface(xx, yy, (np.tile(z_range[:,0:1], (2, 2))), color=c, rstride=1, cstride=1, alpha=alpha, edgecolors=edgecolors)
        ax.plot_surface(xx, yy, (np.tile(z_range[:,1:2], (2, 2))), color=c, rstride=1, cstride=1, alpha=alpha, edgecolors=edgecolors)

        lw=0.5
        
        ax.plot(xs=x_range[0], ys=np.tile(y_range[0,0:1], (2)), zs=np.tile(z_range[0,0:1], (2)), lw=lw, c=c2)
        ax.plot(xs=x_range[0], ys=np.tile(y_range[0,1:2], (2)), zs=np.tile(z_range[0,0:1], (2)), lw=lw, c=c2)
        ax.plot(xs=x_range[0], ys=np.tile(y_range[0,0:1], (2)), zs=np.tile(z_range[0,1:2], (2)), lw=lw, c=c2)
        ax.plot(xs=x_range[0], ys=np.tile(y_range[0,1:2], (2)), zs=np.tile(z_range[0,1:2], (2)), lw=lw, c=c2)
        
        yy, zz = np.meshgrid(y_range, z_range)
        ax.plot_surface((np.tile(x_range[:,0:1], (2, 2))), yy, zz, color=c, rstride=1, cstride=1, alpha=alpha, edgecolors=edgecolors)
        ax.plot_surface((np.tile(x_range[:,1:2], (2, 2))), yy, zz, color=c, rstride=1, cstride=1, alpha=alpha, edgecolors=edgecolors)

        ax.plot(xs=np.tile(x_range[0,0:1], (2)), ys=y_range[0], zs=np.tile(z_range[0,0:1], (2)), lw=lw, c=c2)
        ax.plot(xs=np.tile(x_range[0,1:2], (2)), ys=y_range[0], zs=np.tile(z_range[0,0:1], (2)), lw=lw, c=c2)
        ax.plot(xs=np.tile(x_range[0,0:1], (2)), ys=y_range[0], zs=np.tile(z_range[0,1:2], (2)), lw=lw, c=c2)
        ax.plot(xs=np.tile(x_range[0,1:2], (2)), ys=y_range[0], zs=np.tile(z_range[0,1:2], (2)), lw=lw, c=c2)
        
        
        
        xx, zz = np.meshgrid(x_range, z_range)
        ax.plot_surface(xx, (np.tile(y_range[:,0:1], (2, 2))), zz, color=c, rstride=1, cstride=1, alpha=alpha, edgecolors=edgecolors)
        ax.plot_surface(xx, (np.tile(y_range[:,1:2], (2, 2))), zz, color=c, rstride=1, cstride=1, alpha=alpha, edgecolors=edgecolors)
    
        ax.plot(xs=np.tile(x_range[0,0:1], (2)), ys=np.tile(y_range[0,0:1], (2)), zs=z_range[0], lw=lw, c=c2)
        ax.plot(xs=np.tile(x_range[0,1:2], (2)), ys=np.tile(y_range[0,0:1], (2)), zs=z_range[0], lw=lw, c=c2)
        ax.plot(xs=np.tile(x_range[0,0:1], (2)), ys=np.tile(y_range[0,1:2], (2)), zs=z_range[0], lw=lw, c=c2)
        ax.plot(xs=np.tile(x_range[0,1:2], (2)), ys=np.tile(y_range[0,1:2], (2)), zs=z_range[0], lw=lw, c=c2)
        
        
        
def plotCubeAtA(pos=(0,0,0), l=(1.0,1.0,1.0), c="b", alpha=0.1, ax=None, edgecolors=None):
    # Plotting N cube elements at position pos
    if ax !=None:
        x_range = np.array([[pos[0]-l[0]/2, pos[0]+l[0]/2]])
        y_range = np.array([[pos[1]-l[1]/2, pos[1]+l[1]/2]])
        z_range = np.array([[pos[2]-l[2]/2, pos[2]+l[2]/2]])
        
        z_range
        xx, yy = np.meshgrid(x_range, y_range)
        
        ax.plot_surface(xx, yy, (np.tile(z_range[:,0:1], (2, 2))), color=c, rstride=1, cstride=1, alpha=alpha, edgecolors=edgecolors)
        ax.plot_surface(xx, yy, (np.tile(z_range[:,1:2], (2, 2))), color=c, rstride=1, cstride=1, alpha=alpha, edgecolors=edgecolors)

        yy, zz = np.meshgrid(y_range, z_range)
        ax.plot_surface((np.tile(x_range[:,0:1], (2, 2))), yy, zz, color=c, rstride=1, cstride=1, alpha=alpha, edgecolors=edgecolors)
        ax.plot_surface((np.tile(x_range[:,1:2], (2, 2))), yy, zz, color=c, rstride=1, cstride=1, alpha=alpha, edgecolors=edgecolors)

        
        xx, zz = np.meshgrid(x_range, z_range)
        ax.plot_surface(xx, (np.tile(y_range[:,0:1], (2, 2))), zz, color=c, rstride=1, cstride=1, alpha=alpha, edgecolors=edgecolors)
        ax.plot_surface(xx, (np.tile(y_range[:,1:2], (2, 2))), zz, color=c, rstride=1, cstride=1, alpha=alpha, edgecolors=edgecolors)
        
        
def plotSurfaceAt(pos=(0,0,0), l=(1.0,1.0,1.0), c="b", alpha=0.1, ax=None, edgecolors=None):
    # Plotting N cube elements at position pos
    if ax !=None:
        x_range = np.array([[pos[0]-l[0]/2, pos[0]+l[0]/2]])
        y_range = np.array([[pos[1]-l[1]/2, pos[1]+l[1]/2]])
        z_range = np.array([[pos[2]-l[2]/2, pos[2]+l[2]/2]])
        
        yy, zz = np.meshgrid(y_range, z_range)
        ax.plot_surface((np.tile(x_range[:,0:1], (2, 2))), yy, zz, color=c, rstride=1, cstride=1, alpha=alpha, edgecolors=edgecolors)
        

        
    
        
def plotCubeAtB(pos=(0,0,0), l=(1.0,1.0,1.0), c="b", alpha=0.1, ax=None, edgecolors=None):
    # Plotting N cube elements at position pos
    if ax !=None:
        x_range = np.array([[pos[0]-l[0]/2, pos[0]+l[0]/2]])
        y_range = np.array([[pos[1]-l[1]/2, pos[1]+l[1]/2]])
        z_range = np.array([[pos[2]-l[2]/2, pos[2]+l[2]/2]])
        
        z_range
        xx, yy = np.meshgrid(x_range, y_range)
        lw=0.3
        
        ax.plot(xs=x_range[0], ys=np.tile(y_range[0,0:1], (2)), zs=np.tile(z_range[0,0:1], (2)), lw=lw, c='black')
        ax.plot(xs=x_range[0], ys=np.tile(y_range[0,1:2], (2)), zs=np.tile(z_range[0,0:1], (2)), lw=lw, c='black')
        ax.plot(xs=x_range[0], ys=np.tile(y_range[0,0:1], (2)), zs=np.tile(z_range[0,1:2], (2)), lw=lw, c='black')
        ax.plot(xs=x_range[0], ys=np.tile(y_range[0,1:2], (2)), zs=np.tile(z_range[0,1:2], (2)), lw=lw, c='black')
        
        ax.plot(xs=np.tile(x_range[0,0:1], (2)), ys=y_range[0], zs=np.tile(z_range[0,0:1], (2)), lw=lw, c='black')
        ax.plot(xs=np.tile(x_range[0,1:2], (2)), ys=y_range[0], zs=np.tile(z_range[0,0:1], (2)), lw=lw, c='black')
        ax.plot(xs=np.tile(x_range[0,0:1], (2)), ys=y_range[0], zs=np.tile(z_range[0,1:2], (2)), lw=lw, c='black')
        ax.plot(xs=np.tile(x_range[0,1:2], (2)), ys=y_range[0], zs=np.tile(z_range[0,1:2], (2)), lw=lw, c='black')
                
        ax.plot(xs=np.tile(x_range[0,0:1], (2)), ys=np.tile(y_range[0,0:1], (2)), zs=z_range[0], lw=lw, c='black')
        ax.plot(xs=np.tile(x_range[0,1:2], (2)), ys=np.tile(y_range[0,0:1], (2)), zs=z_range[0], lw=lw, c='black')
        ax.plot(xs=np.tile(x_range[0,0:1], (2)), ys=np.tile(y_range[0,1:2], (2)), zs=z_range[0], lw=lw, c='black')
        ax.plot(xs=np.tile(x_range[0,1:2], (2)), ys=np.tile(y_range[0,1:2], (2)), zs=z_range[0], lw=lw, c='black')


    
    

def plt_RadialE(data_real, data_fake, energy_real, energy_fake, data_fake2=None, energy_fake2=None, 
                model_title='ML Model', model_title2=' ', save_title='ML_model',
                number=1, numberf=1, numberf2=1, showLegend=True):
    figRadE = plt.figure(figsize=(6,6))
    axRadE = figRadE.add_subplot(1,1,1)
    lightblue = (0.1, 0.1, 0.9, 0.3)

    pRadEa = axRadE.hist(data_real, bins=24, range=[0,24], density=None, 
                   weights=energy_real/(float(number)), edgecolor=color_list[0], 
                   label = "orig", linewidth=linewidth_list[0],color=fillcolor_list[0],
                   histtype='stepfilled')
    if showLegend:
        axRadE.plot((0.42, 0.48),(0.87-0.02, 0.87-0.02),linewidth=linewidth_list[0], 
                 linestyle=linestyle_list[0], transform=axRadE.transAxes, color = color_list[0]) 
        axRadE.text(0.50, 0.87, 'GEANT 4', horizontalalignment='left',verticalalignment='top', 
                 transform=axRadE.transAxes, color = color_list[0])


    for j in range(0, len(data_fake)):
        pRadEb = axRadE.hist(data_fake[j], bins=pRadEa[1], range=None, density=None, 
                       weights=energy_fake[j]/(float(numberf)), edgecolor=color_list[j+1], 
                       label = "orig", linewidth=linewidth_list[j+1], linestyle=linestyle_list[j+1],
                       histtype='step')

        if showLegend:
            axRadE.plot((0.42, 0.48),(0.87-0.02-(j+1)*0.06, 0.87-0.02-(j+1)*0.06),linewidth=linewidth_list[j+1], 
                     linestyle=linestyle_list[0], transform=axRadE.transAxes, color = color_list[j+1]) 
            axRadE.text(0.50, 0.87-(j+1)*0.06, model_title[j], horizontalalignment='left',verticalalignment='top', 
                     transform=axRadE.transAxes, color = color_list[j+1])

    axRadE.set_xlabel('radius [pixels]', family='serif')
    axRadE.set_ylabel('energy [MeV]', family='serif')
    axRadE.set_xlim([0, 24.0])
    axRadE.set_ylim([0.01, 1000])
    axRadE.xaxis.set_ticks([0.5,1.0,1.5,2.0])

    axRadE.xaxis.set_minor_locator(MultipleLocator(1))
    axRadE.xaxis.set_major_locator(MultipleLocator(5))
    axRadE.text(0.5, 0.95, 'full spectrum', horizontalalignment='left',verticalalignment='top', 
             transform=axRadE.transAxes)

    plt.subplots_adjust(left=0.18, right=0.95, top=0.95, bottom=0.18)
    plt.yscale('log')
    figRadE.patch.set_facecolor('white')

    plt.savefig('./plots/' + save_title+"_radial_E_dist.png")

    
    

def plt_SpinalE(data_real, data_fake, data_fake2=None, model_title='ML Model',
                model_title2='ML Model2', save_title='ML_model',
                number=1, numberf=1, numberf2=1, showLegend=True):
    figSpnE = plt.figure(figsize=(6,6))
    axSpnE = figSpnE.add_subplot(1,1,1)
    lightblue = (0.1, 0.1, 0.9, 0.3)
    n_layers = data_real.shape[1]
    hits = np.arange(0, n_layers)+0.5

    pSpnEa = axSpnE.hist(hits, bins=30, range=[0,30], density=None, 
                   weights=np.mean(data_real, 0), edgecolor=color_list[0], 
                   label = "orig", linewidth=linewidth_list[0],color=fillcolor_list[0],
                   histtype='stepfilled')
    if showLegend:
        axSpnE.plot((0.42, 0.48),(0.87-0.02, 0.87-0.02),linewidth=linewidth_list[0], 
                 linestyle=linestyle_list[0], transform=axSpnE.transAxes, color = color_list[0]) 
        axSpnE.text(0.50, 0.87, 'GEANT 4', horizontalalignment='left',verticalalignment='top', 
                 transform=axSpnE.transAxes, color = color_list[0])


    for j in range(0, len(data_fake)):
        pSpnEb = axSpnE.hist(hits, bins=pSpnEa[1], range=None, density=None, 
                       weights=np.mean(data_fake[j], 0), edgecolor=color_list[j+1], 
                       label = "orig", linewidth=linewidth_list[j+1], linestyle=linestyle_list[j+1],
                       histtype='step')
        if showLegend:
            axSpnE.plot((0.42, 0.48),(0.87-0.02-(j+1)*0.06, 0.87-0.02-(j+1)*0.06),linewidth=linewidth_list[j+1], 
                     linestyle=linestyle_list[0], transform=axSpnE.transAxes, color = color_list[j+1]) 
            axSpnE.text(0.50, 0.87-(j+1)*0.06, model_title[j], horizontalalignment='left',verticalalignment='top', 
                     transform=axSpnE.transAxes, color = color_list[j+1])


    axSpnE.set_xlabel('layer', family='serif')
    axSpnE.set_ylabel('energy [MeV]', family='serif')
    axSpnE.set_xlim([0, 31.0])
    axSpnE.set_ylim([1, 1000])
    axSpnE.xaxis.set_ticks([0.5,1.0,1.5,2.0])

    axSpnE.xaxis.set_minor_locator(MultipleLocator(1))
    axSpnE.xaxis.set_major_locator(MultipleLocator(5))

    axSpnE.text(0.5,
            0.95,
            'full spectrum', horizontalalignment='left',verticalalignment='top', 
             transform=axSpnE.transAxes)

    plt.subplots_adjust(left=0.18, right=0.95, top=0.95, bottom=0.18)
    plt.yscale('log')
    figSpnE.patch.set_facecolor('white')

    plt.savefig('./plots/' + save_title+"_spinal_E_dist.png")

    


def plt_singleEnergy_3ofaKind(data_real, data_real2, data_real3, 
                              data_fake, data_fake2, data_fake3, 
                              energy1, energy2, energy3,
                     energy_center=0, model_title='ML Model', model_title2=' ' ,save_title='ML_model', showLegend=True):
    figSE = plt.figure(figsize=(6,6*0.77/0.67))

    axSE = figSE.add_subplot(1,1,1)
    lightblue = (0.1, 0.1, 0.9, 0.3)
    
    maxE = 1900
    minE = 150
    ymax = 0.32
    ymin = 0.0
    bins = 100
    
    pSEa = axSE.hist(data_real, bins=bins, range=[minE, maxE], density=None, 
                   weights=np.ones_like(data_real)/(float(len(data_real))), edgecolor=color_list[0], 
                   label = "orig", linewidth=linewidth_list[0],color=fillcolor_list[0],
                   histtype='stepfilled')
    pSEd = axSE.hist(data_real2, bins=bins, range=[minE, maxE], density=None, 
                   weights=np.ones_like(data_real2)/(float(len(data_real2))), edgecolor=color_list[0], 
                   label = "orig", linewidth=linewidth_list[0],color=fillcolor_list[0],
                   histtype='stepfilled')
    pSEe = axSE.hist(data_real3, bins=bins, range=[minE, maxE], density=None, 
                   weights=np.ones_like(data_real3)/(float(len(data_real3))), edgecolor=color_list[0], 
                   label = "orig", linewidth=linewidth_list[0],color=fillcolor_list[0],
                   histtype='stepfilled')
    if showLegend:
        axSE.plot((0.42, 0.48),(0.87-0.02, 0.87-0.02),linewidth=linewidth_list[0], 
                 linestyle=linestyle_list[0], transform=axSE.transAxes, color = color_list[0]) 
        axSE.text(0.50, 0.87, geant_label, horizontalalignment='left',verticalalignment='top', 
                 transform=axSE.transAxes, color = color_list[0])


    for j in range(0, len(data_fake)):
        pSpnEb = axSE.hist(data_fake[j], bins=pSEa[1], range=None, density=None, 
                       weights=np.ones_like(data_fake[j])/(float(len(data_fake[j]))), edgecolor=color_list[j+1], 
                       label = "orig", linewidth=linewidth_list[j+1], linestyle=linestyle_list[j+1],
                       histtype='step')
        pSpnEc = axSE.hist(data_fake2[j], bins=pSEa[1], range=None, density=None, 
                       weights=np.ones_like(data_fake2[j])/(float(len(data_fake2[j]))), edgecolor=color_list[j+1], 
                       label = "orig", linewidth=linewidth_list[j+1], linestyle=linestyle_list[j+1],
                       histtype='step')
        pSpnEf = axSE.hist(data_fake3[j], bins=pSEa[1], range=None, density=None, 
                       weights=np.ones_like(data_fake3[j])/(float(len(data_fake3[j]))), edgecolor=color_list[j+1], 
                       label = "orig", linewidth=linewidth_list[j+1], linestyle=linestyle_list[j+1],
                       histtype='step')
        if showLegend:
            axSE.plot((0.42, 0.48),(0.87-0.02-(j+1)*0.06, 0.87-0.02-(j+1)*0.06),linewidth=linewidth_list[j+1], 
                     linestyle=linestyle_list[0], transform=axSE.transAxes, color = color_list[j+1]) 
            axSE.text(0.50, 0.87-(j+1)*0.06, model_title[j], horizontalalignment='left',verticalalignment='top', 
                     transform=axSE.transAxes, color = color_list[j+1])


    axSE.set_ylabel('normalized', family='serif')
    axSE.set_xlabel('visible energy [MeV]', family='serif')
    axSE.set_xlim([minE, maxE])
    axSE.set_ylim([ymin, ymax])

    axSE.xaxis.set_minor_locator(MultipleLocator(100))
    axSE.xaxis.set_major_locator(MultipleLocator(500))
    
    axSE.text((0.4*np.mean(data_real)+0.6*np.max(data_real)-minE)/(maxE-minE),
              (np.max(pSEa[0])-ymin)/(ymax-ymin),
               '{:d} GeV'.format(int(energy1)), horizontalalignment='left',verticalalignment='top', 
               transform=axSE.transAxes)
    axSE.text((0.4*np.mean(data_real2)+0.6*np.max(data_real2)-minE)/(maxE-minE),
               (np.max(pSEd[0])-ymin)/(ymax-ymin),
               '{:d} GeV'.format(int(energy2)), horizontalalignment='left',verticalalignment='top', 
               transform=axSE.transAxes)    
    axSE.text((0.4*np.mean(data_real3)+0.6*np.max(data_real3)-minE)/(maxE-minE),
               (np.max(pSEe[0])-ymin)/(ymax-ymin),
               '{:d} GeV'.format(int(energy3)), horizontalalignment='left',verticalalignment='top', 
               transform=axSE.transAxes)
    
    plt.subplots_adjust(left=0.18, right=0.95, top=0.85, bottom=0.18)
    figSE.patch.set_facecolor('white')

    plt.savefig('./plots/' + save_title+"_single_E_dist_3ofaKind.png")

    
    

def plt_HitE_uncut(data_real, data_fake, data_real_cut, data_fake_cut, data_fake2=None, 
                model_title='ML Model', model_title2='ML Model2', save_title='ML_model', 
                energy_center=None, n_bins = 200, E_max = 1000, E_min = 0.00001, y_max = 10.0, y_min = 0.00001, showLegend=True):
    figHitE = plt.figure(figsize=(6,6*0.77/0.67))
    axHitE = figHitE.add_subplot(1,1,1)
    lightblue = (0.1, 0.1, 0.9, 0.3)

    pHitEEa = axHitE.hist(data_real, bins=np.logspace(np.log10(E_min),np.log10(E_max), n_bins), range=None, density=None, 
                   weights=np.ones_like(data_real)/(float(len(data_real_cut))), edgecolor=color_list[0], 
                   label = "orig", linewidth=linewidth_list[0],color=fillcolor_list[0],
                   histtype='stepfilled')
    leg_x = 0.42
    leg_y = 0.37
    if showLegend:
        axHitE.plot((leg_x-0.08, leg_x-0.02),(leg_y-0.02, leg_y-0.02),linewidth=linewidth_list[0], 
                 linestyle=linestyle_list[0], transform=axHitE.transAxes, color = color_list[0]) 
        axHitE.text(leg_x, leg_y, geant_label, horizontalalignment='left',verticalalignment='top', 
                 transform=axHitE.transAxes, color = color_list[0])


    for j in range(0, len(data_fake)):
        pHitEEb = axHitE.hist(data_fake[j], bins=pHitEEa[1], range=None, density=None, 
                       weights=np.ones_like(data_fake[j])/(float(len(data_fake_cut[j]))), edgecolor=color_list[j+1], 
                       label = "orig", linewidth=linewidth_list[j+1], linestyle=linestyle_list[j+1],
                       histtype='step')
        if showLegend:

            axHitE.plot((leg_x-0.08, leg_x-0.02),(leg_y-0.02-(j+1)*0.06, leg_y-0.02-(j+1)*0.06),linewidth=linewidth_list[j+1], 
                     linestyle=linestyle_list[j+1], transform=axHitE.transAxes, color = color_list[j+1]) 
            axHitE.text(leg_x, leg_y-(j+1)*0.06, model_title[j], horizontalalignment='left',verticalalignment='top', 
                     transform=axHitE.transAxes, color = color_list[j+1])

    pHitEEeg = axHitE.axvspan(E_min, 0.1, facecolor='grey', alpha=0.5, hatch= "/" )

            
    new_tick_locations = np.array([0.01, .1, 1.0, 10.0])

    plt.tick_params(
        axis='x',          # changes apply to the x-axis
        which='both',      # both major and minor ticks are affected
        bottom=True,      # ticks along the bottom edge are off
        top=False,         # ticks along the top edge are off
        labelbottom=True  # labels along the bottom edge are off
    ) 

    axHitE.set_xlabel('visible cell energy [MeV]', family='serif')
    axHitE.set_ylabel('normalized', family='serif')
    axHitE.set_xlim([E_min,E_max])
    axHitE.set_ylim([y_min, y_max])
 
    axHitE.xaxis.set_major_locator(plt.NullLocator())


    plt.xscale('log')
    
    def forward(x):
        return x/0.2

    def inverse(x):
        return x*0.2

    secax = axHitE.secondary_xaxis('top', functions=(forward, inverse))
    secax.set_xlabel('visible cell energy [MIPs]')
    secax.xaxis.labelpad = 12
    
    if energy_center == None:
        axHitE.text(0.5,
                0.95,
                'full spectrum'.format(energy_center), horizontalalignment='left',verticalalignment='top', 
                 transform=axHitE.transAxes)
    else:
        axHitE.text(0.5,
                0.95,
                '{:d} GeV Photons'.format(energy_center), horizontalalignment='left',verticalalignment='top', 
                 transform=axHitE.transAxes)

    plt.subplots_adjust(left=0.18, right=0.95, top=0.85, bottom=0.18)
    # original figure space: y 0.95-0.18 = 0.77
    # new figure space: y 0.85-0.18 = 0.67
    # new figsize 6*0.77/0.64
    
    plt.yscale('log')
    plt.xscale('log')
    figHitE.patch.set_facecolor('white')

    plt.savefig('./plots/' + save_title+"_hit_E_dist.png")
    
    plt.tick_params(
        axis='x',          # changes apply to the x-axis
        which='both',      # both major and minor ticks are affected
        bottom=True,      # ticks along the bottom edge are off
        top=True,         # ticks along the top edge are off
        labelbottom=True  # labels along the bottom edge are off
    ) 

    
    
    
def plt_Occupancy_3ofaKind(data_real, data_real2, data_real3, 
                           data_fake, data_fake2, data_fake3, 
                           energy1, energy2, energy3,
                           model_title='ML Model', model_title2='ML Model 2', 
                  save_title=' ', energy_center=50, showLegend=True):
    #figOcc = plt.figure(figsize=(6,6))
    figOcc = plt.figure(figsize=(6,6*0.77/0.67))

    axOcc = figOcc.add_subplot(1,1,1)
    lightblue = (0.1, 0.1, 0.9, 0.3)

    xmin = 200
    xmax = 1400
    nbins= 100
    ymax = 0.22
    ymin = 0

    pOcca = axOcc.hist(data_real, bins=nbins, range=[xmin,xmax], density=None, 
                   weights=np.ones_like(data_real)/(float(len(data_real))), edgecolor=color_list[0], 
                   label = "orig", linewidth=linewidth_list[0],color=fillcolor_list[0],
                   histtype='stepfilled')
    pOccd = axOcc.hist(data_real2, bins=nbins, range=[xmin,xmax], density=None, 
                   weights=np.ones_like(data_real2)/(float(len(data_real2))), edgecolor=color_list[0], 
                   label = "orig", linewidth=linewidth_list[0],color=fillcolor_list[0],
                   histtype='stepfilled')
    pOcce = axOcc.hist(data_real3, bins=nbins, range=[xmin,xmax], density=None, 
                   weights=np.ones_like(data_real3)/(float(len(data_real3))), edgecolor=color_list[0], 
                   label = "orig", linewidth=linewidth_list[0],color=fillcolor_list[0],
                   histtype='stepfilled')
    if showLegend:
        axOcc.plot((0.42, 0.48),(0.87-0.02, 0.87-0.02),linewidth=linewidth_list[0], 
                 linestyle=linestyle_list[0], transform=axOcc.transAxes, color = color_list[0]) 
        axOcc.text(0.50, 0.87, geant_label, horizontalalignment='left',verticalalignment='top', 
                 transform=axOcc.transAxes, color = color_list[0])


    for j in range(0, len(data_fake)):
        pOccb = axOcc.hist(data_fake[j], bins=pOcca[1], range=None, density=None, 
                       weights=np.ones_like(data_fake[j])/(float(len(data_fake[j]))), edgecolor=color_list[j+1], 
                       label = "orig", linewidth=linewidth_list[j+1], linestyle=linestyle_list[j+1],
                       histtype='step')
        pOccb = axOcc.hist(data_fake2[j], bins=pOcca[1], range=None, density=None, 
                       weights=np.ones_like(data_fake2[j])/(float(len(data_fake2[j]))), edgecolor=color_list[j+1], 
                       label = "orig", linewidth=linewidth_list[j+1], linestyle=linestyle_list[j+1],
                       histtype='step')
        pOccf = axOcc.hist(data_fake3[j], bins=pOcca[1], range=None, density=None, 
                       weights=np.ones_like(data_fake3[j])/(float(len(data_fake3[j]))), edgecolor=color_list[j+1], 
                       label = "orig", linewidth=linewidth_list[j+1], linestyle=linestyle_list[j+1],
                       histtype='step')
        if showLegend:
            axOcc.plot((0.42, 0.48),(0.87-0.02-(j+1)*0.06, 0.87-0.02-(j+1)*0.06),linewidth=linewidth_list[j+1], 
                     linestyle=linestyle_list[0], transform=axOcc.transAxes, color = color_list[j+1]) 
            axOcc.text(0.50, 0.87-(j+1)*0.06, model_title[j], horizontalalignment='left',verticalalignment='top', 
                     transform=axOcc.transAxes, color = color_list[j+1])


    axOcc.set_xlabel('number of hits', family='serif')
    axOcc.set_ylabel('normalized', family='serif')
    axOcc.set_xlim([xmin, xmax])
    axOcc.set_ylim([ymin, ymax])

    axOcc.xaxis.set_minor_locator(MultipleLocator(50))
    axOcc.xaxis.set_major_locator(MultipleLocator(500))

    axOcc.text((0.5*np.mean(data_real)+0.5*np.max(data_real)-xmin)/(xmax-xmin),
              (np.max(pOcca[0])-ymin)/(ymax-ymin),
               '{:d} GeV'.format(int(energy1)), horizontalalignment='left',verticalalignment='top', 
               transform=axOcc.transAxes)
    axOcc.text((0.45*np.mean(data_real2)+0.55*np.max(data_real2)-xmin)/(xmax-xmin),
               ((np.max(pOccd[0])-ymin)/(ymax-ymin))*1.1,
               '{:d} GeV'.format(int(energy2)), horizontalalignment='left',verticalalignment='top', 
               transform=axOcc.transAxes)    
    axOcc.text((0.45*np.mean(data_real3)+0.55*np.max(data_real3)-xmin)/(xmax-xmin),
               (np.max(pOcce[0])-ymin)/(ymax-ymin),
               '{:d} GeV'.format(int(energy3)), horizontalalignment='left',verticalalignment='top', 
               transform=axOcc.transAxes)
        
    plt.subplots_adjust(left=0.18, right=0.95, top=0.85, bottom=0.18)
    figOcc.patch.set_facecolor('white')

    figOcc.savefig('./plots/' + save_title+"_occ_dist_3ofaKind.png")

    
    
def plt_CenterOfGravity(data_real, data_fake, data_fake2=None, model_title='ML Model',
                model_title2=' ', save_title='ML_model', energy_center=None, showLegend=True):
    figSpnE = plt.figure(figsize=(6,6))
    axSpnE = figSpnE.add_subplot(1,1,1)
    lightblue = (0.1, 0.1, 0.9, 0.3)
        

    pSpnEa = axSpnE.hist(data_real, bins=50, range=[5,25], density=None, 
                   weights=np.ones_like(data_real)/(float(len(data_real))), edgecolor=color_list[0], 
                   label = "orig", linewidth=linewidth_list[0],color=fillcolor_list[0],
                   histtype='stepfilled')
    if showLegend:
        axSpnE.plot((0.42, 0.48),(0.87-0.02, 0.87-0.02),linewidth=linewidth_list[0], 
                 linestyle=linestyle_list[0], transform=axSpnE.transAxes, color = color_list[0]) 
        axSpnE.text(0.50, 0.87, geant_label, horizontalalignment='left',verticalalignment='top', 
                 transform=axSpnE.transAxes, color = color_list[0])


    for j in range(0, len(data_fake)):
        pSpnEb = axSpnE.hist(data_fake[j], bins=pSpnEa[1], range=None, density=None, 
                       weights=np.ones_like(data_fake[j])/(float(len(data_fake[j]))), edgecolor=color_list[j+1], 
                       label = "orig", linewidth=linewidth_list[j+1], linestyle=linestyle_list[j+1],
                       histtype='step')
        if showLegend:
            axSpnE.plot((0.52, 0.58),(0.87-0.02-(j+1)*0.06, 0.87-0.02-(j+1)*0.06),linewidth=linewidth_list[j+1], 
                     linestyle=linestyle_list[0], transform=axSpnE.transAxes, color = color_list[j+1]) 
            axSpnE.text(0.60, 0.87-(j+1)*0.06, model_title[j], horizontalalignment='left',verticalalignment='top', 
                     transform=axSpnE.transAxes, color = color_list[j+1])


    axSpnE.set_xlabel('center of gravity Z [layer]', family='serif')
    axSpnE.set_ylabel('normalized', family='serif')
    axSpnE.set_xlim([8.0, 22.0])
    axSpnE.set_ylim([0, 0.15])

    axSpnE.xaxis.set_minor_locator(MultipleLocator(1))
    axSpnE.xaxis.set_major_locator(MultipleLocator(5))

    if energy_center == None:
        axSpnE.text(0.5,
                0.95,
                'full spectrum'.format(energy_center), horizontalalignment='left',verticalalignment='top', 
                 transform=axSpnE.transAxes)
    else:
        axSpnE.text(0.5,
                0.95,
                '{:d} GeV Photons'.format(energy_center), horizontalalignment='left',verticalalignment='top', 
                 transform=axSpnE.transAxes)

    plt.subplots_adjust(left=0.18, right=0.95, top=0.95, bottom=0.18)
    #plt.yscale('log')
    figSpnE.patch.set_facecolor('white')
    plt.savefig('./plots/' + save_title+"_CenterOfGravity_dist.png")


    
    
def plt_scatter_error_subplot(x_data, x_data_sub, x_data_error, x_data_sub_error, 
                        data_real, data_real_sub, data_real_error, data_real_sub_error,
                        data_fake, data_fake_sub, data_fake_error, data_fake_sub_error, 
                        data_fake2_sub = None, data_fake2 = None, data_fake2_sub_error = None, data_fake2_error = None, 
                     model_title='ML Model', model_title2='ML Model 2' ,save_title='ML_model', 
                     plt_title='scatter', x_title='x', y_title='y', maxX=1000, minX=0, maxY=100, minY=0, yloc=100,
                     x_title_sub='x', y_title_sub='y', maxX_sub=1000, minX_sub=0, maxY_sub=100, minY_sub=0, yloc_sub=100, 
                     showLegend=True, x_fontsize = 10, y_fontsize = 10, x_subfontsize = 10, y_subfontsize = 10):

    figStr, (axStr, axSub) = plt.subplots(2, 1, sharex=True, figsize=(6.5,8), dpi=96, 
                                          gridspec_kw={'height_ratios': [3, 1], 'hspace' : 0.})
    
    lightblue = (0.1, 0.1, 0.9, 0.3)
    markersize1=10
    markersize2=10
       

    pStra = axStr.errorbar(x=x_data[:,0], y=data_real[:,0], xerr=None, yerr=data_real_error[:,0], 
                   marker=marker_list[0], linestyle=linestyle_list[0], 
                   linewidth=linewidth_list[0], color=color_list[0])  
    pSuba = axSub.errorbar(x=x_data_sub[:,0], y=data_real_sub[:,0], xerr=None, yerr=data_real_sub_error[:,0], 
                   marker=marker_list[0], linestyle=linestyle_list[0], 
                   linewidth=linewidth_list[0], color=color_list[0])

    x_leg = 0.05
    y_leg = 0.95
    if showLegend:    
        axStr.plot((x_leg, x_leg+0.06),(y_leg-0.02, y_leg-0.02),linewidth=linewidth_list[0], 
                 linestyle=linestyle_list[0], transform=axStr.transAxes, color = color_list[0]) 
        axStr.text(x_leg+0.08, y_leg, geant_label, horizontalalignment='left',verticalalignment='top', 
                 transform=axStr.transAxes, color = color_list[0])


    for j in range(0, data_fake.shape[1]):
        pStrb = axStr.errorbar(x=x_data, y=data_fake[:,j], xerr=None, yerr=data_fake_error[:,j], 
                       marker=marker_list[j+1], linestyle=linestyle_list[j+1], 
                       linewidth=linewidth_list[j+1], color=color_list[j+1])
        pSubb = axSub.errorbar(x=x_data_sub, y=data_fake_sub[:,j], xerr=None, yerr=data_fake_sub_error[:,j], 
                       marker=marker_list[j+1], linestyle=linestyle_list[j+1], 
                       linewidth=linewidth_list[j+1], color=color_list[j+1])

        if showLegend:
            axStr.plot((x_leg, x_leg+0.06),(y_leg-0.02-(j+1)*0.06, y_leg-0.02-(j+1)*0.06),linewidth=linewidth_list[j+1], 
                     linestyle=linestyle_list[j+1], transform=axStr.transAxes, color = color_list[j+1]) 
            axStr.text(x_leg+0.08, y_leg-(j+1)*0.06, model_title[j], horizontalalignment='left',verticalalignment='top', 
                     transform=axStr.transAxes, color = color_list[j+1])


        
    axStr.set_ylabel(y_title, family='serif', fontsize=y_fontsize)
    axStr.set_xlabel(x_title, family='serif', fontsize=x_fontsize)
    axStr.set_xlim([minX, maxX])
    axStr.set_ylim([minY, maxY])
    
    axSub.set_ylabel(y_title_sub, family='serif', fontsize=y_subfontsize)
    axSub.set_xlabel(x_title_sub, family='serif', fontsize=x_subfontsize)
    axSub.set_xlim([minX_sub, maxX_sub])
    axSub.set_ylim([minY_sub, maxY_sub])


    axStr.xaxis.set_minor_locator(MultipleLocator(5))
    axStr.xaxis.set_major_locator(MultipleLocator(20))
    axStr.yaxis.set_minor_locator(MultipleLocator(yloc))
    axStr.yaxis.set_major_locator(MultipleLocator(yloc*5))

    axSub.xaxis.set_minor_locator(MultipleLocator(5))
    axSub.xaxis.set_major_locator(MultipleLocator(20))
    axSub.yaxis.set_minor_locator(MultipleLocator(yloc_sub))
    axSub.yaxis.set_major_locator(MultipleLocator(yloc_sub*5))

    axStr.text(0.5,
            0.95,
            plt_title, horizontalalignment='left',verticalalignment='top', 
             transform=axStr.transAxes)

    plt.subplots_adjust(left=0.30, right=0.95, top=0.95, bottom=0.18)
    axStr.patch.set_facecolor('white')
    figStr.patch.set_facecolor('white')
    plt.savefig('./plots/' + save_title)

    
    


