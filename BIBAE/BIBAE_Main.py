import numpy as np
import torch
import torch.utils.data
from torch import nn, optim
from torch.nn import functional as F
from torchvision import datasets, transforms
from torchvision.utils import save_image
from data_utils.data_loader import HDF5Dataset

from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist
from torch.multiprocessing import Process

import BIBAE_models as models
import BIBAE_functions as functions
import skimage.measure
import time

import importlib
importlib.reload(models)
importlib.reload(functions)


def main(kwargs):
    L_KLD,         L_P_KLD         = 0.05,   0.0
    L_ENR_RECON,   L_P_ENR_RECON   = 0.0,    0.0001
    L_ENR_RECON_N, L_P_ENR_RECON_N = 0.0,    0.01
    L_MMD_Latent,  L_P_MMD_Latent  = 100.0,  0.0
    L_MMD_HitKS,   L_P_MMD_HitKS   = 0.0,    5.0
    L_SortMSE,     L_P_SortMSE     = 0.0,    10.0
    L_SortMAE,     L_P_SortMAE     = 0.0,    10.0

    empt = 0.0
    E_true_trans = lambda x:(x)  #100GeV -> 1.0

    L_Losses =  [L_KLD,         L_ENR_RECON,  
                 L_ENR_RECON_N, L_MMD_Latent,
                 L_MMD_HitKS,   L_SortMSE,    
                 L_SortMAE]

    L_P_Losses =  [L_P_KLD,         L_P_ENR_RECON,  
                   L_P_ENR_RECON_N, L_P_MMD_Latent,
                   L_P_MMD_HitKS,   L_P_SortMSE,    
                   L_P_SortMAE]
    
    default_params = {
        
        "model" : "VAE_ML",
        "suffix" : "_test",
        "E_cond" : False,
        "loss_List" : L_Losses,
        "lossP_List" : L_P_Losses,
        "E_true_trans" : E_true_trans,

        # IO
        "input_path"  : '/beegfs/desy/user/diefenbs/gamma-fullG_5mm_bins.hdf5',
        "output_path" : './results/',

        # False: Train; True: read weights file 
        "batch_size" : 32,
        "epochs" : 10,
        "no_cuda" : True,
        "seed" : 1,
        "log_interval" : 50,
        "train_size" : 1000,
        "shuffle" : True,
        "num_workers" : 1,
        "sample_interval" : 20,
        "save_interval": 40,       
        "continue_train":False,
        "continue_epoch":0,
        "start_PostProc_after_ep":0,
        "PostProc_pretrain":'MSE',
        "PostProc_train":'MSE',
        "latentSmL":24,
        "latent":100,        
        "lr_VAE":1e-3,
        "lr_Critic":1e-4,           
        "lr_PostProc":1e-4,
        "lr_Critic_L":1e-4,
        "gamma_VAE":1.0,
        "gamma_PostProc":1.0,        
        "gamma_Critic":1.0,        
        "gamma_Critic_L":1.0,
        "opt_VAE" :'Adam',
        "L_adv" : 1.0,
        "L_adv_L" : 1.0,
        'L_D_P' : 1.0,
        "HitMMDKS_Ker" : 50,
        "HitMMDKS_Str" : 25,
        "HitMMDKS_Cut" : 2000,       
        "HitMMD_alpha" : 500.0,
        "ENR_Cut_Cutoff" : 0.1,
        "multi_gpu":False,
        'BIBAE_Train' : True
    }



    params = {}
    for param in default_params.keys():

        if param in kwargs.keys():
            params[param] = kwargs[param]
        else:
            params[param] = default_params[param]

    cuda = not params["no_cuda"] and torch.cuda.is_available() 
    
    # create output folder if not already existing
    functions.create_output_folder(params["output_path"])

    
    torch.manual_seed(params["seed"])
    
    if cuda:
        if not params["multi_gpu"]:
            device = torch.device("cuda:0")
        if params["multi_gpu"]:
            device = torch.device("cuda:0")       

    else:
         device = torch.device("cpu")        

    print(torch.cuda.current_device()) 
    loader_params = {'shuffle': params['shuffle'], 'num_workers': params['num_workers']}


    tf_lin_F = lambda x:(x)

    if params["model"] == "3D_M_BiBAESmLVar_P_1ConvEconV2_C_ConvDiffv3_CL_Default":
        netD = models.Discriminator_F_Conv_DIRENR_Diff_v3().to(device)  
        netD_L = models.Latent_Critic().to(device)  
        model = models.BiBAE_F_3D_LayerNorm_SmallLatent(params, device=device, 
                                                         z_rand=(params["latent"]-params["latentSmL"]),
                                                         z_enc=params["latentSmL"]).to(device)   

        model_P = models.PostProcess_Size1Conv_EcondV2(bias=True, out_funct='none').to(device)
        train = functions.train_BiBAE_F_linear_PostProcess
        test = functions.test_BiBAE_F_linear_PostProcess
        LATENT_DIM = params["latent"]+1
        tf = tf_lin_F
        
        
    

        
        
    elif params["model"] == "3D_M_BiBAESmLVar_P_None_C_ConvDiffv3_CL_Default":
        netD = models.Discriminator_F_Conv_DIRENR_Diff_v3().to(device)  
        netD_L = models.Latent_Critic().to(device)  
        model = models.BiBAE_F_3D_LayerNorm_SmallLatent(params, device=device, 
                                                         z_rand=(params["latent"]-params["latentSmL"]),
                                                         z_enc=params["latentSmL"]).to(device)   

        model_P = models.PostProcess_Size1Conv_EcondV2(bias=True, out_funct='none').to(device)
        train = functions.train_BiBAE_F_linear
        test = functions.test_BiBAE_F_linear
        LATENT_DIM = params["latent"]+1
        tf = tf_lin_F     

    if params["opt_VAE"] == 'Adam':
        optimizer = optim.Adam(model.parameters(), lr=params["lr_VAE"], 
                               betas=(0.5, 0.9))
    elif  params["opt_VAE"] == 'RMSprop':
        optimizer = optim.RMSprop(model.parameters(), lr=params["lr_VAE"])
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=params["gamma_VAE"])

    
    optimizerD = optim.Adam(netD.parameters(), lr=params["lr_Critic"],
                            betas=(0.5, 0.9))
    schedulerD = optim.lr_scheduler.StepLR(optimizerD, step_size=1, gamma=params["gamma_Critic"])

    
    optimizerD_L = optim.Adam(netD_L.parameters(), lr=params["lr_Critic_L"], 
                              betas=(0.5, 0.9))        
    schedulerD_L = optim.lr_scheduler.StepLR(optimizerD_L, step_size=1, gamma=params["gamma_Critic_L"])

    
    optimizerP = optim.Adam(model_P.parameters(), lr=params["lr_PostProc"], 
                               betas=(0.5, 0.9))
    schedulerP = optim.lr_scheduler.StepLR(optimizerP, step_size=1, gamma=params["gamma_PostProc"])


    if params["multi_gpu"]:
        print(torch.cuda.device_count(), " GPUs")
        model = nn.DataParallel(model)
        model_P = nn.DataParallel(model_P)
        netD = nn.DataParallel(netD)
        netD_L = nn.DataParallel(netD_L)

    
    if params["continue_train"]:
        checkpoint = torch.load(params["output_path"] + "check_" + params["model"] + params["suffix"] + '_' +
                                str(params["continue_epoch"]) + '.pth', map_location=torch.device(device))

        model.load_state_dict(checkpoint['model_state_dict'])
        netD.load_state_dict(checkpoint['netD_state_dict'])
        netD_L.load_state_dict(checkpoint['netD_L_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        optimizerD.load_state_dict(checkpoint['optimizerD_state_dict'])
        optimizerD_L.load_state_dict(checkpoint['optimizerD_L_state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        schedulerD.load_state_dict(checkpoint['schedulerD_state_dict'])
        schedulerD_L.load_state_dict(checkpoint['schedulerD_L_state_dict'])

        try:
            model_P.load_state_dict(checkpoint['model_P_state_dict'])
            optimizerP.load_state_dict(checkpoint['optimizerP_state_dict'])
            schedulerP.load_state_dict(checkpoint['schedulerP_state_dict'])
        except:
            for i in range(0,params["continue_epoch"]):
                schedulerP.step()


    dataset = HDF5Dataset(params["input_path"], transform=tf, train_size=params["train_size"])

    dataset_train, dataset_val = torch.utils.data.random_split(dataset, 
                                                               [int(0.95*params["train_size"]), 
                                                                params["train_size"] - int(0.95*params["train_size"])])

    data_loader = torch.utils.data.DataLoader(dataset, **loader_params)

    train_loader = torch.utils.data.DataLoader(
        dataset_train, batch_size = params["batch_size"], **loader_params)

    test_loader = torch.utils.data.DataLoader(
        dataset_val, batch_size = params["batch_size"], **loader_params)

    

    print(params)
    

    if params["model"] == "3D_M_BiBAESmLVar_P_1Conv_C_ConvDiffv3_CL_Default":
        for epoch in range(1+params["continue_epoch"], params["epochs"] + 1 + params["continue_epoch"]):
            print(optimizer)
            print(optimizerP)
            print(optimizerD)
            print(optimizerD_L)

            
            time_start_ep = time.time()
            train(model=model, modelP = model_P, netD=netD, netD_L=netD_L, epoch=epoch, 
                  train_loader=train_loader, device=device, args=params,
                  optimizer=optimizer, optimizerP=optimizerP, optimizerD=optimizerD, optimizerD_L=optimizerD_L, 
                  L_Losses=params["loss_List"], L_Losses_P=params["lossP_List"], 
                  L_D=params['L_adv'], L_D_L=params['L_adv_L'], L_D_P=params["L_D_P"])
            test(model=model, netD=netD, netD_L=netD_L, epoch=epoch, 
                 test_loader=test_loader, device=device, args=params, 
                 optimizer=optimizer, L_Losses=params["loss_List"],
                 L_D=params['L_adv'], L_D_L=params['L_adv_L'])             

            scheduler.step()
            schedulerP.step()
            schedulerD.step()
            schedulerD_L.step()

            
            if epoch%params["save_interval"] == 0 or epoch == 1:
                print('Saving to ' + params["output_path"] + "check_" +
                    params["model"] + params["suffix"] + '_' + str(epoch) + '.pth')
                torch.save({
                    'model_state_dict': model.state_dict(),
                    'model_P_state_dict': model_P.state_dict(),
                    'netD_state_dict': netD.state_dict(),
                    'netD_L_state_dict': netD_L.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'optimizerP_state_dict': optimizerP.state_dict(),
                    'optimizerD_state_dict': optimizerD.state_dict(),
                    'optimizerD_L_state_dict': optimizerD_L.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict(),
                    'schedulerP_state_dict': schedulerP.state_dict(),
                    'schedulerD_state_dict': schedulerD.state_dict(),
                    'schedulerD_L_state_dict': schedulerD_L.state_dict()
                    }, 
                    params["output_path"] + "check_" +
                    params["model"] + params["suffix"] + '_' + str(epoch) + '.pth'
                )

            time_stop_ep = time.time()

            

            print("Duration of this epoch in sec: ", time_stop_ep - time_start_ep)

    elif params["model"] == "3D_M_BiBAESmLVar_P_None_C_ConvDiffv3_CL_Default":

        for epoch in range(1+params["continue_epoch"], params["epochs"] + 1 + params["continue_epoch"]):
            print(optimizer)
            print(optimizerP)
            print(optimizerD)
            print(optimizerD_L)

            
            time_start_ep = time.time()
            train(model=model, netD=netD, netD_L=netD_L, epoch=epoch, 
                  train_loader=train_loader, device=device, args=params,
                  optimizer=optimizer, optimizerD=optimizerD, optimizerD_L=optimizerD_L, 
                  L_Losses=params["loss_List"],  
                  L_D=params['L_adv'], L_D_L=params['L_adv_L'])
            test(model=model, netD=netD, netD_L=netD_L, epoch=epoch, 
                 test_loader=test_loader, device=device, args=params, 
                 optimizer=optimizer, L_Losses=params["loss_List"],
                 L_D=params['L_adv'], L_D_L=params['L_adv_L'])             

            scheduler.step()
            schedulerP.step()
            schedulerD.step()
            schedulerD_L.step()

            
            if epoch%params["save_interval"] == 0 or epoch == 1:
                print('Saving to ' + params["output_path"] + "check_" +
                    params["model"] + params["suffix"] + '_' + str(epoch) + '.pth')
                torch.save({
                    'model_state_dict': model.state_dict(),
                    'model_P_state_dict': model_P.state_dict(),
                    'netD_state_dict': netD.state_dict(),
                    'netD_L_state_dict': netD_L.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'optimizerP_state_dict': optimizerP.state_dict(),
                    'optimizerD_state_dict': optimizerD.state_dict(),
                    'optimizerD_L_state_dict': optimizerD_L.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict(),
                    'schedulerP_state_dict': schedulerP.state_dict(),
                    'schedulerD_state_dict': schedulerD.state_dict(),
                    'schedulerD_L_state_dict': schedulerD_L.state_dict()
                    }, 
                    params["output_path"] + "check_" +
                    params["model"] + params["suffix"] + '_' + str(epoch) + '.pth'
                )
                


            time_stop_ep = time.time()

            

            print("Duration of this epoch in sec: ", time_stop_ep - time_start_ep)
            
