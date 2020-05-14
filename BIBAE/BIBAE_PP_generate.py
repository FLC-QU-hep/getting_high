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

#create instance of BIBAE and PP model
model_BIBAE = models.BiBAE_F_3D_LayerNorm_SmallLatent(args, device=device, z_rand=512-24,
                                             z_enc=24).to(device)   
model_BIBAE = nn.DataParallel(model_BIBAE)

model_BIBAE_PP = models.PostProcess_Size1Conv_EcondV2(bias=True, out_funct='none').to(device)
model_BIBAE_PP = nn.DataParallel(model_BIBAE_PP)

#load combined BIBAE PP statedict
checkpoint_BIBAE_PP = torch.load('*path_to_saved_checkpoint*.pth', map_location=torch.device(device))

#apply statedicts to apropriate models
model_BIBAE.load_state_dict(checkpoint_BIBAE_PP['model_state_dict']) 
model_BIBAE_PP.load_state_dict(checkpoint_BIBAE_PP['model_P_state_dict']) 


batchsize=100   #number of events to generate
E_max = 100.0   #upper and lower bound of energy range
E_min = 10.0

global_thresh = 0.1 #0.5MIP cutoff

model_BIBAE.eval()
model_BIBAE_PP.eval()

with torch.no_grad():
    #define combined latent space of noise (x) and energy label (E)
    x = torch.zeros(batchsize, latent_dim, device=device)
    E = (torch.rand(batchsize, 1, device=device)*(E_max-E_min)+E_min)*100
    latent = torch.cat((x, E), dim=1)
    
    #apply BIBAE to noise
    data = model_BIBAE(x=x, E_true=E, z = torch.randn(batchsize, latent_dim),  mode='decode')
    
    #apply PP to BIBAE output
    dataPP = model_BIBAE_PP.forward(data, E)

    #port to numpy array
    data = data.view(-1, 30, 30, 30).cpu().numpy() 
    dataPP = dataPP.view(-1, 30, 30, 30).cpu().numpy() 

#apply 0.5MIP cutoff
data[ data < global_thresh] = 0.0  
dataPP[ dataPP < global_thresh] = 0.0  
        

print(dataPP.shape)
