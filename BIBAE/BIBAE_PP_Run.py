#How to run:
# 1 choose BIBAE_Run for BIBAE training or BIBAE_PP_Run for BIBAE with post processing training
# 2 modify parameters accoring to your need
# 3 launch python BIBAE_Run.py or python BIBAE_PP_Run.py 



import BIBAE_Main

#Weights of the indiviual loss terms, in ordoer:
# KLD: KL-Divergence on the latent space
# ENR_RECON: Squared error between input and reconstructed energy sum
# ENR_RECON_N: Absolute error between input and reconstructed energy sum
# MMD_Latent: Maximum Mean Discrepency on the latent space
# MMD_HitKS: Sorted Kernel MMD on the hit energy specturm
# SortMSE: Squared error between input and reconstructed energy sum

L_KLD,         L_P_KLD         = 0.05,   0.0
L_ENR_RECON,   L_P_ENR_RECON   = 0.0,    0.0 #can sometimes help for post processing, though not used for paper
L_ENR_RECON_N, L_P_ENR_RECON_N = 0.0,    0.0 #can sometimes help for post processing, though not used for paper
L_MMD_Latent,  L_P_MMD_Latent  = 100.0,  0.0
L_MMD_HitKS,   L_P_MMD_HitKS   = 0.0,    5.0
L_SortMSE,     L_P_SortMSE     = 0.0,    0.0 #can sometimes help for post processing, though not used for paper
L_SortMAE,     L_P_SortMAE     = 0.0,    0.0 #can sometimes help for post processing, though not used for paper

E_true_trans = lambda x:(x)  #transformation function for the true energy label, currently 1:1 mapping

#List with loss weights for BIBAE trainin
L_Losses =  [L_KLD,         L_ENR_RECON,  
             L_ENR_RECON_N, L_MMD_Latent,
             L_MMD_HitKS,   L_SortMSE,    
             L_SortMAE]

#List with loss weights for PP training
L_P_Losses =  [L_P_KLD,         L_P_ENR_RECON,  
               L_P_ENR_RECON_N, L_P_MMD_Latent,
               L_P_MMD_HitKS,   L_P_SortMSE,    
               L_P_SortMAE]

params = {
        #I-O parameters:
        'input_path'  : '/beegfs/desy/user/diefenbs/gamma-fullG-950kCorrected.hdf5',  #path for input file
        'output_path' : '/beegfs/desy/user/diefenbs/VAE_results/test/', #path where models are saved
        'log_interval': 1,                                              #number of iterations after which progess is printed
        'save_interval': 1,                                             #number of epochs after which wieghts are saved
        'sample_interval': 1,                                           #number of epochs after which example images are generated
        'train_size': 9450,                                             #number of events used form input (set no larger than file size)
        'shuffle': True,                                                #defines whether training files is shuffles
        'num_workers': 1,                                               #defines number of workers for data loader (bugged for > 1)
        "continue_train":False,                                         #If true network will look for saved weights in current output
        "continue_epoch":0,                                             #directory saved after 'continue_epoch' epochs and resume training
    
        #Model definitions
        'model': '3D_M_BiBAESmLVar_P_1ConvEconV2_C_ConvDiffv3_CL_Default',#picks model setup to train with, defined in Main
        'suffix': '_KLD005_MDL_L24_512_1MCor',                          #wights are saved as 'model' + 'suffix' + epoch.pth
        'E_cond': True,                                                 #Turns energy conditioning on/off
        "latentSmL":24,                                                 #Number of active latent space dimensions
        "latent":512,                                                   #Total numver of latent space dimension, if more then 'latentSmL'
                                                                        #the rest is filled with gaussian noise
        'BIBAE_Train' : True,                                           #set false to freeze BIBAE weights during PP training
        'multi_gpu': True,                                              #Turns on multi GPU capabilities, also works with 1 GPU

    
        #Training parameters
        'loss_List': L_Losses,                                          #weight list for BIBAE losses, defined above
        "lossP_List" : L_P_Losses,                                      #weight list for PP losses, defined above
        'E_true_trans': E_true_trans,                                   #transformation function for the true energy label, defined above

        'batch_size': 8,                                                #batch size
        'epochs': 1000,                                                 #maximum number of epochs
        'no_cuda': False,                                               #set true for CPU training (not recomended)
        'seed': 1,                                                      #manual seed

        "start_PostProc_after_ep":100,                                  #Number of epochs after which PP is started, before PP is 
                                                                        #trained only using MSE
        'opt_VAE' :'Adam',                                              #BIBAE optimizer, choose Adam or SGD
        'lr_VAE':1e-3*0.5,                                              #initial BIBAE learning rate
        'lr_Critic':1e-3*0.5,                                           #initial critic learning rate
        "lr_PostProc":1e-3*0.5,                                         #initial PP learning rate
        'lr_Critic_L':2.0*1e-3,                                         #initial latent critic learning rate
        'gamma_VAE':0.95,                                               #BIBAE LR decay rate after each epoch
        'gamma_Critic':0.95,                                            #critic LR decay rate after each epoch
        "gamma_PostProc":0.95,                                          #PP LR decay rate after each epoch
        'gamma_Critic_L':0.95,                                          #latent critic LR decay rate after each epoch
          
        'L_D_P' : 0.0,                                                  #loss weight of critic for post processing
        'L_adv' : 1.0,                                                  #loss weight of critic for BIBAE
        'L_adv_L' : 100.0,                                              #loss weight of latent critic for BIBAE
    
        "HitMMDKS_Ker" : 100,                                           #kernel size for the sorted kernel MMD
        "HitMMDKS_Str" : 25,                                            #stride for the sorted kernel MMD
        "HitMMDKS_Cut" : 2000,                                          #lower cutoff for the sorted kernel MMD

        'HitMMD_alpha' : 200.0                                          #parameter in Gaussian MMD kernel function 
}

BIBAE_Main.main(params)


