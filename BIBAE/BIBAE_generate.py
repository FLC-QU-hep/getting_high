import sys
import numpy as np
import torch
import torch.utils.data
import matplotlib.pyplot as plt
import matplotlib as mpl
import BIBAE_models as models
import BIBAE_functions as functions
from torch import nn


#Define model paramters 
latent_dim = 512
args = {
        'E_cond' : True,
        'latent' : latent_dim
}
cuda = torch.cuda.is_available()
device = torch.device("cuda" if cuda else "cpu")
print(device)

#create instance of BIBAE model
model_BIBAE = models.BiBAE_F_3D_LayerNorm_SmallLatent(args, device=device, z_rand=512-24,
                                             z_enc=24).to(device)   
model_BIBAE = nn.DataParallel(model_BIBAE)


#load combined BIBAE statedict
checkpoint_BIBAE = torch.load('/beegfs/desy/user/diefenbs/VAE_results/2020_KW13/3D_M_BiBAESmLVar_P_1ConvEconV2_C_ConvDiffv3_CL_Default_KLD005_MDL_PMDKS_L24_512_1MCor/'+
                              'check_3D_M_BiBAESmLVar_P_1ConvEconV2_C_ConvDiffv3_CL_Default_KLD005_MDL_PMDKS_L24_512_1MCor_39.pth', map_location=torch.device(device))

#apply statedicts to apropriate model
model_BIBAE.load_state_dict(checkpoint_BIBAE['model_state_dict']) 


batchsize=100   #number of events to generate
E_max = 100.0   #upper and lower bound of energy range
E_min = 10.0

global_thresh = 0.1 #0.5MIP cutoff

model_BIBAE.eval()

with torch.no_grad():
    #define combined latent space of noise (x) and energy label (E)
    x = torch.zeros(batchsize, latent_dim, device=device)
    E = (torch.rand(batchsize, 1, device=device)*(E_max-E_min)+E_min)*100
    latent = torch.cat((x, E), dim=1)
    
    #apply BIBAE to noise
    data = model_BIBAE(x=x, E_true=E, z = torch.randn(batchsize, latent_dim),  mode='decode')

    #port to numpy array
    data = data.view(-1, 30, 30, 30).cpu().numpy() 

#apply 0.5MIP cutoff
data[ data < global_thresh] = 0.0  
        

print(data.shape)
