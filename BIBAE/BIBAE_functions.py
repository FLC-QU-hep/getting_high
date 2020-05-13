import numpy as np
import argparse
import torch
import torch.utils.data
from torch import nn, optim
from torch.nn import functional as F
from torchvision import datasets, transforms
from torchvision.utils import save_image
from torch import autograd
import os
tf_logscale_rev = lambda x:(torch.exp(x)-1.0)
tf_logscale_rev_np = lambda x:(np.exp(x)-1.0)
tf_logscale_ML = lambda x:(np.log((skimage.measure.block_reduce(x, (6,1,1), np.sum)+1.0)))



    
def calc_gradient_penalty_F_DIRENR(netD, real_data, fake_data, device, batchsize, E_true, LAMBDA = 10.0):
    alpha = torch.rand(batchsize, 1, device=device)
    alpha = alpha.expand(batchsize, 
                         int(real_data.nelement()/batchsize)).contiguous().view(batchsize, 
                         real_data.size(1), real_data.size(2), real_data.size(3), real_data.size(4))


    interpolates = alpha * real_data + ((1 - alpha) * fake_data)
    interpolates = interpolates.clone().detach().requires_grad_(True)
    disc_interpolates = netD(interpolates, E_true=E_true)

    gradients = autograd.grad(outputs=disc_interpolates, inputs=interpolates,
                            grad_outputs=torch.ones(disc_interpolates.size(), device=device),
                            create_graph=True, retain_graph=True, only_inputs=True)[0]

    gradients = gradients.view(gradients.size(0), -1)

    
    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean() * LAMBDA

    return gradient_penalty



def calc_gradient_penalty_AAE(netD, real_data, fake_data, device, batchsize, LAMBDA = 10.0):
    alpha = torch.rand(batchsize, 1, device=device)
    alpha = alpha.expand(batchsize,
                         int(real_data.nelement()/batchsize)).contiguous().view(batchsize,
                         real_data.size(1))
    
    interpolates = alpha * real_data + ((1 - alpha) * fake_data)
    interpolates = interpolates.clone().detach().requires_grad_(True) 
    disc_interpolates = netD(interpolates)

    gradients = autograd.grad(outputs=disc_interpolates, inputs=interpolates,
                            grad_outputs=torch.ones(disc_interpolates.size(),device=device),
                            create_graph=True, retain_graph=True, only_inputs=True)[0]
    gradients = gradients.view(gradients.size(0), -1)

    
    
    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean() * LAMBDA
    return gradient_penalty


###################################################################
######## BIB-AE Trainging Energy Conditinging  ##########
###################################################################
    
    
def train_BiBAE_F_linear(model, netD, netD_L, epoch, train_loader, device, args, 
                optimizer, optimizerD, optimizerD_L, L_Losses, L_D, L_D_L):
    one = torch.tensor(1.0).to(device)
    mone = (one * -1).to(device)

    E_true_trans = args['E_true_trans']
        
    model.train()
    train_loss = 0
    
    for batch_idx, (data, energy) in enumerate(train_loader):
        data = data.to(device)
        E_true = E_true_trans(energy).to(device)
        calo_size = data.size(-1)

        
        for p in netD.parameters():
            p.requires_grad = True  # to avoid computation
        for p in netD_L.parameters():
            p.requires_grad = True  # to avoid computation
  
        
        for i in range(5):
            recon_batch, mu, logvar, z = model(data, E_true)

            optimizerD.zero_grad()            
            
            
            real_data = torch.cat((data.view(-1,1,calo_size,calo_size,calo_size), 
                                   data.view(-1,1,calo_size,calo_size,calo_size)), 1)
            
            D_real = netD(real_data, E_true=E_true)
            D_real = D_real.mean()
            D_real.backward(mone)
            

            fake_data = torch.cat((data.view(-1,1,calo_size,calo_size,calo_size), 
                                   recon_batch.view(-1,1,calo_size,calo_size,calo_size)), 1)
            fake_data = fake_data.clone().detach()

            D_fake = netD(fake_data, E_true=E_true)
            D_fake = D_fake.mean()
            D_fake.backward(one)
            
            gradient_penalty = calc_gradient_penalty_F_DIRENR(netD, real_data.data, fake_data.data, device, data.size(0), 
                                                    E_true=E_true, LAMBDA = 10.0)  
            gradient_penalty.backward()

            
            optimizerD.step()
            
            optimizerD_L.zero_grad()
    
            zv = z.clone().detach()#.view(-1,1)
         
            realv = torch.randn_like(zv)
        
            D_L_real = netD_L(realv)
            D_L_real = D_L_real.mean()
            D_L_real.backward(mone)


            D_L_fake = netD_L(zv)
            D_L_fake = D_L_fake.mean()
            D_L_fake.backward(one)

            gradient_penalty_L = calc_gradient_penalty_AAE(netD_L, realv.data, zv.data, device, realv.size(0),
                                                    LAMBDA = 10.0)                                                 
            gradient_penalty_L.backward()

            optimizerD_L.step()

            
        for p in netD.parameters():
            p.requires_grad = False  # to avoid computation
        for p in netD_L.parameters():
            p.requires_grad = False  # to avoid computation
               
        recon_batch, mu, logvar, z = model(data, E_true)

        optimizer.zero_grad()
 
        sum_data = torch.sum((data.view(-1, calo_size*calo_size*calo_size)), dim=1)*1e-4 #GeV to TeV   
        sum_recon = torch.sum((recon_batch.view(-1, calo_size*calo_size*calo_size)), dim=1)*1e-4 #GeV to TeV

        fake_data = torch.cat((data.view(-1,1,calo_size,calo_size,calo_size), 
                               recon_batch.view(-1,1,calo_size,calo_size,calo_size)), 1)
        lossD = netD(fake_data, E_true=E_true)
        lossD = lossD.mean()


        lossD_L = netD_L(z)#.view(-1,1))
        lossD_L = lossD_L.mean()
        
        lossFix,weighted,unwe,name = loss_function_VAE_ENR(recon_x=recon_batch, x=data, mu=mu, logvar=logvar, 
                                                           E_pred=E_true, E_true=E_true, E_sum_pred=sum_recon, E_sum_true=sum_data, 
                                                           device=device, L_Losses = L_Losses, z=z, args=args)   
        
        loss = L_D*lossD + L_D_L*lossD_L - lossFix
        
        train_loss += loss.item()     
        
        loss.backward(mone)
        
        optimizer.step()
        if batch_idx % args["log_interval"] == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\lossD: {:.6f} lossD_L: {:.6f} Fix {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader),
                L_D*lossD.item(),L_D_L*lossD_L.item(), lossFix.item()
            ))
            line=''
    for (w, u, n) in zip(weighted, unwe, name):
        line = line + n + ' \t{:.3f} \t{:.3f} \n'.format(w,u)
    print(line)

    print('====> Epoch: {} Average loss: {:.4f}'.format(
          epoch, train_loss ))
    
    
    
def test_BiBAE_F_linear(model, netD, netD_L, epoch, test_loader, device, args, optimizer, L_Losses, L_D, L_D_L):    
    model.train()
    
    E_true_trans = args['E_true_trans']

    test_loss_sample = 0
    test_loss_recon = 0
    test_weighted_sample = torch.zeros(7)
    test_weighted_recon = torch.zeros(7)
    test_unwe_sample = torch.zeros(7)
    test_unwe_recon = torch.zeros(7)
    test_lossD_sample = 0
    test_lossD_recon = 0
    test_lossD_L_sample = 0
    test_lossD_L_recon = 0

    with torch.no_grad():
        for i, (data, energy) in enumerate(test_loader):
            data = data.to(device)
            E_true = E_true_trans(energy).to(device)
            calo_size = data.size(-1)

            
            recon_batch, mu, logvar, z = model(data, E_true)
            
            sample_batch = torch.randn(recon_batch.size(0), args["latent"]).to(device)
            sample_batch = model.forward(data, E_true, z=sample_batch, mode='decode')
            
            sum_true = torch.sum((data.view(-1, calo_size*calo_size*calo_size)), dim=1)*1e-4
            sum_recon = torch.sum((recon_batch.view(-1, calo_size*calo_size*calo_size)), dim=1)*1e-4
            sum_sample = torch.sum((sample_batch.view(-1, calo_size*calo_size*calo_size)), dim=1)*1e-4
            
            loss_recon,weighted_recon,unwe_recon,name_recon  = loss_function_VAE_ENR(recon_x=recon_batch, x=data, mu=mu, 
                                                               logvar=logvar, E_pred=E_true, E_true=E_true, E_sum_pred=sum_recon,
                                                               E_sum_true=sum_true, reduction='mean', device=device,
                                                               L_Losses = L_Losses, z=z, args=args)   
                      
            test_weighted_recon += torch.tensor(weighted_recon)
            test_unwe_recon += torch.tensor(unwe_recon)

            loss_sample,weighted_sample,unwe_sample,name_sample = loss_function_VAE_ENR(recon_x=sample_batch, x=data, 
                                                   mu=torch.zeros_like(data), logvar=torch.ones_like(data),
                                                   E_pred=E_true, E_true=E_true, E_sum_pred=sum_sample,
                                                   E_sum_true=sum_true, reduction='mean', device=device,
                                                   L_Losses = L_Losses, z=z, args=args)   
            
            test_weighted_sample += torch.tensor(weighted_sample)
            test_unwe_sample += torch.tensor(unwe_sample)
            
           
            fake_data_recon = torch.cat((data.view(-1,1,calo_size,calo_size,calo_size), 
                                         recon_batch.view(-1,1,calo_size,calo_size,calo_size)), 1)
            lossD_recon = netD(fake_data_recon, E_true=E_true)
            lossD_recon = lossD_recon.mean()
            test_lossD_recon += lossD_recon
            
            fake_data_sample = torch.cat((data.view(-1,1,calo_size,calo_size,calo_size), 
                                          sample_batch.view(-1,1,calo_size,calo_size,calo_size)), 1)
            lossD_sample = netD(fake_data_sample, E_true=E_true)
            lossD_sample= lossD_sample.mean()
            test_lossD_sample += lossD_sample
            
            lossD_L_recon = netD_L(z)#.view(-1,1))
            lossD_L_recon = lossD_L_recon.mean()
            test_lossD_L_recon += lossD_L_recon

            test_lossD_L_sample = 0

            
            test_loss_sample += loss_sample.item() - L_D * lossD_sample.item()
            test_loss_recon += (loss_recon.item() - L_D * lossD_recon.item()
                                 - L_D_L * lossD_L_recon.item())
            
            if epoch%args["sample_interval"] == 0 or epoch == 1:
                if i == 0:
                    n = min(data.size(0), 8)
                    comparison = torch.cat([data.view(args["batch_size"],1, calo_size*calo_size, calo_size)[:n],
                                          recon_batch.view(args["batch_size"],1, calo_size*calo_size, calo_size)[:n]])
                    save_image(comparison.cpu(),
                             args["output_path"]+ "reconstruction_" + args["model"] + 
                               args["suffix"] + '_' + str(epoch) + '.png', nrow=n)

                    
    test_weighted_recon=test_weighted_recon/len(test_loader.dataset)*args["batch_size"]                
    test_unwe_recon=test_unwe_recon/len(test_loader.dataset)*args["batch_size"]   
    test_weighted_sample=test_weighted_sample/len(test_loader.dataset)*args["batch_size"]
    test_unwe_sample=test_unwe_sample/len(test_loader.dataset)*args["batch_size"]
    test_lossD_sample = test_lossD_sample/len(test_loader.dataset)*args["batch_size"]
    test_lossD_recon = test_lossD_recon/len(test_loader.dataset)*args["batch_size"]
    test_lossD_L_recon = test_lossD_L_recon/len(test_loader.dataset)*args["batch_size"]
    
    line =""
    for (w, u, n) in zip(test_weighted_recon, test_unwe_recon, name_recon):
        line = line +'Recon:  '+ n + '\t{:.3f} \t{:.3f} \n'.format(w,u)
    print(line)
    print('Test Recon lossD: {:.4f} Weight {:.2f}'.format(test_lossD_recon, L_D))
    print('Test Recon lossD_L: {:.4f} Weight {:.2f}'.format(test_lossD_L_recon, L_D_L))


    line =""
    for (w, u, n) in zip(test_weighted_sample, test_unwe_sample, name_sample):
        line = line + 'Sample: ' + n + '\t{:.3f} \t{:.3f} \n'.format(w,u)
    print(line)
    print('Test Sample lossD: {:.4f} Weight {:.2f}'.format(test_lossD_sample, L_D))
    print('Test Sample lossD_L: {:.4f} Weight {:.2f}'.format(test_lossD_L_sample, L_D_L))

    for b in range(0, min(E_true.size(0), 10)):
        print('sum_sample {:.3f}  sum_recon {:.3f}  sum_true {:.3f}  E_true {:.3f}'.format(
            sum_sample[b].item(), sum_recon[b].item(), sum_true[b].item(), E_true[b].item()))
    test_loss_sample = test_loss_sample/len(test_loader.dataset)*args["batch_size"]
    test_loss_recon = test_loss_recon/len(test_loader.dataset)*args["batch_size"]

    print('====> Test Recon loss: {:.4f}'.format(test_loss_recon))
    print('====> Test Sample loss: {:.4f}'.format(test_loss_sample))
    #print(z)
    #plt.hist(z.cpu().numpy(),bins=20,density=None, weights=None, edgecolor='blue', 
    #           label = 'Latent Var ' + str(i), linewidth=1, histtype='step')
    #plt.savefig('latent_ep_{:d}.png'.format((int(epoch))))


    
###################################################################
######## BIB-AE Trainging Energy Conditinging PostProcess #########
###################################################################
    
    
def train_BiBAE_F_linear_PostProcess(model, modelP, netD, netD_L, epoch, train_loader, device, args, 
                optimizer, optimizerP, optimizerD, optimizerD_L, L_Losses,  L_Losses_P, L_D, L_D_L, L_D_P):
    one = torch.tensor(1.0).to(device)
    mone = (one * -1).to(device)

    E_true_trans = args['E_true_trans']
        
    model.train()
    train_loss = 0
    
    
    for batch_idx, (data, energy) in enumerate(train_loader):
        data = data.to(device)
        E_true = E_true_trans(energy).to(device)
        batch_size_n = data.size(0)
        calo_size = data.size(-1)

        
        if args['BIBAE_Train']:
            for p in netD.parameters():
                p.requires_grad = True  # to avoid computation
            for p in netD_L.parameters():
                p.requires_grad = True  # to avoid computation

            for i in range(5):
                recon_batch, mu, logvar, z = model(data, E_true)

                optimizerD.zero_grad()            


                real_data = torch.cat((data.view(-1,1,calo_size,calo_size,calo_size), 
                                       data.view(-1,1,calo_size,calo_size,calo_size)), 1)

                D_real = netD(real_data, E_true=E_true)
                D_real = D_real.mean()
                D_loss = (-1.0)*D_real

                fake_data = torch.cat((data.view(-1,1,calo_size,calo_size,calo_size), 
                                       recon_batch.view(-1,1,calo_size,calo_size,calo_size)), 1)
                fake_data = fake_data.clone().detach()

                D_fake = netD(fake_data, E_true=E_true)
                D_fake = D_fake.mean()
                D_loss += (1.0)*D_fake

                gradient_penalty = calc_gradient_penalty_F_DIRENR(netD, real_data.data, fake_data.data, device, data.size(0), 
                                                        E_true=E_true, LAMBDA = 10.0)  
                D_loss += (1.0)*gradient_penalty
                D_loss.backward()

                optimizerD.step()

                optimizerD_L.zero_grad()

                zv = z.clone().detach()

                realv = torch.randn_like(zv)

                D_L_real = netD_L(realv)
                D_L_real = D_L_real.mean()
                D_L_loss = (-1.0)*D_L_real

                D_L_fake = netD_L(zv)
                D_L_fake = D_L_fake.mean()
                D_L_loss += (1.0)*D_L_fake

                gradient_penalty_L = calc_gradient_penalty_AAE(netD_L, realv.data, zv.data, device, realv.size(0),
                                                        LAMBDA = 10.0)                                                 
                D_L_loss += (1.0)*gradient_penalty_L
                D_L_loss.backward()

                optimizerD_L.step()


            for p in netD.parameters():
                p.requires_grad = False  # to avoid computation
            for p in netD_L.parameters():
                p.requires_grad = False  # to avoid computation
               
        recon_batch, mu, logvar, z = model(data, E_true)

        optimizer.zero_grad()

        sum_data = torch.sum((data.view(-1, calo_size*calo_size*calo_size)), dim=1) 
        sum_recon = torch.sum((recon_batch.view(-1, calo_size*calo_size*calo_size)), dim=1) 

        fake_data = torch.cat((data.view(-1,1,calo_size,calo_size,calo_size), 
                               recon_batch.view(-1,1,calo_size,calo_size,calo_size)), 1)
        lossD = netD(fake_data, E_true=E_true)
        lossD = lossD.mean()


        lossD_L = netD_L(z)
        lossD_L = lossD_L.mean()

        lossFix,weighted,unwe,name = loss_function_VAE_ENR(recon_x=recon_batch, x=data, mu=mu, logvar=logvar, 
                                                           E_pred=E_true, E_true=E_true, E_sum_pred=sum_recon, 
                                                           E_sum_true=sum_data, 
                                                           device=device, L_Losses = L_Losses, z=z, args=args)   

        loss = L_D*lossD + L_D_L*lossD_L - lossFix

        train_loss += loss.item()     
            
        if args['BIBAE_Train']:
            loss.backward(mone)
            optimizer.step()

        
        ### PostProcess ###
        postprocess_batch = modelP(recon_batch.detach(), E_True=E_true)
        
        sum_postprocess = torch.sum((postprocess_batch.view(-1, calo_size*calo_size*calo_size)), dim=1)
        optimizerP.zero_grad()
 
        if L_D_P == 0:
            lossD_P = torch.tensor(0, device=device)
        else:
            fake_data_pp = torch.cat((postprocess_batch.view(-1,1,calo_size,calo_size,calo_size), 
                                      postprocess_batch.view(-1,1,calo_size,calo_size,calo_size)), 1)
            lossD_P = netD(fake_data_pp, E_true=E_true)
            lossD_P = lossD_P.mean()
        
        if epoch <= args['start_PostProc_after_ep']:
            #run mse loss got get postProc network to reporoduce orig showers for the first sectioin
            if args['PostProc_pretrain'] == 'MSE':
                lossFixP = F.mse_loss(postprocess_batch.view(batch_size_n, -1), 
                                      recon_batch.detach().view(batch_size_n, -1), reduction='mean')
            elif args['PostProc_pretrain'] == 'MAE_Rela':
                lossFixP =  torch.mean(torch.abs(
                              (recon_batch.view(batch_size_n, -1).detach()/sum_recon.view(-1, 1).detach()) -
                              (postprocess_batch.view(batch_size_n, -1)/sum_postprocess.view(-1, 1).detach())
                            ))*2000.0
        else:
            lossFixP,weightedP,unweP,nameP = loss_function_VAE_ENR(recon_x=postprocess_batch, x=data, mu=mu, logvar=logvar, 
                                                           E_pred=E_true, E_true=E_true, E_sum_pred=sum_postprocess, 
                                                           E_sum_true=sum_data, 
                                                           device=device, L_Losses = L_Losses_P, z=z, args=args)   

            if args['PostProc_train'] == 'MSE':
                lossFixP += F.mse_loss(postprocess_batch.view(batch_size_n, -1), 
                                       recon_batch.detach().view(batch_size_n, -1), reduction='mean')
            elif args['PostProc_train'] == 'MAE_Rela':
                lossFixP += torch.mean(torch.abs(
                              (recon_batch.view(batch_size_n, -1).detach()/sum_recon.view(-1, 1).detach()) -
                              (postprocess_batch.view(batch_size_n, -1)/sum_postprocess.view(-1, 1).detach())
                            ))*2000.0
        
        
        lossP = L_D_P*lossD_P - lossFixP
        
        lossP.backward(mone)
        optimizerP.step()

        
        
        if batch_idx % args["log_interval"] == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\lossD: {:.6f} lossD_L: {:.6f} Fix {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader),
                L_D*lossD.item(),L_D_L*lossD_L.item(), lossFix.item()
            ))
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\lossD_P: {:.6f}  FixP {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader),
                L_D_P*lossD_P.item(), lossFixP.item()
            ))
    line=''

    for (w, u, n) in zip(weighted, unwe, name):
        line = line + n + ' \t{:.3f} \t{:.3f} \n'.format(w,u)
    print(line)
    if epoch > args['start_PostProc_after_ep']:
        line=''
        for (w, u, n) in zip(weightedP, unweP, nameP):
            line = line + n + ' \t{:.3f} \t{:.3f} \n'.format(w,u)
        print(line)

 
    print('====> Epoch: {} Average loss: {:.4f}'.format(
          epoch, train_loss ))

  
    
    
      

    
def test_BiBAE_F_linear_PostProcess(model, netD, netD_L, epoch, test_loader, device, args, optimizer, L_Losses, L_D, L_D_L):    
    model.train()
    
    E_true_trans = args['E_true_trans']

    test_loss_sample = 0
    test_loss_recon = 0
    test_weighted_sample = torch.zeros(7)
    test_weighted_recon = torch.zeros(7)
    test_unwe_sample = torch.zeros(7)
    test_unwe_recon = torch.zeros(7)
    test_lossD_sample = 0
    test_lossD_recon = 0
    test_lossD_L_sample = 0
    test_lossD_L_recon = 0

    with torch.no_grad():
        for i, (data, energy) in enumerate(test_loader):
            data = data.to(device)
            E_true = E_true_trans(energy).to(device)
            calo_size = data.size(-1)
            
            recon_batch, mu, logvar, z = model(data, E_true)
            
            sample_batch = torch.randn(recon_batch.size(0), args["latent"]).to(device)
            sample_batch = model.forward(data, E_true, z=sample_batch, mode='decode')
            
            sum_true = torch.sum((data.view(-1, calo_size*calo_size*calo_size)), dim=1)*1e-4 
            sum_recon = torch.sum((recon_batch.view(-1, calo_size*calo_size*calo_size)), dim=1)*1e-4 
            sum_sample = torch.sum((sample_batch.view(-1, calo_size*calo_size*calo_size)), dim=1)*1e-4 
            
            loss_recon,weighted_recon,unwe_recon,name_recon  = loss_function_VAE_ENR(recon_x=recon_batch, x=data, mu=mu, 
                                                               logvar=logvar, E_pred=E_true, E_true=E_true, E_sum_pred=sum_recon,
                                                               E_sum_true=sum_true, reduction='mean', device=device,
                                                               L_Losses = L_Losses, z=z, args=args)   
                      
            test_weighted_recon += torch.tensor(weighted_recon)
            test_unwe_recon += torch.tensor(unwe_recon)

            loss_sample,weighted_sample,unwe_sample,name_sample = loss_function_VAE_ENR(recon_x=sample_batch, x=data, 
                                                   mu=torch.zeros_like(data), logvar=torch.ones_like(data),
                                                   E_pred=E_true, E_true=E_true, E_sum_pred=sum_sample,
                                                   E_sum_true=sum_true, reduction='mean', device=device,
                                                   L_Losses = L_Losses, z=z, args=args)   
            
            test_weighted_sample += torch.tensor(weighted_sample)
            test_unwe_sample += torch.tensor(unwe_sample)
            
           
            fake_data_recon = torch.cat((data.view(-1,1,calo_size,calo_size,calo_size), 
                                         recon_batch.view(-1,1,calo_size,calo_size,calo_size)), 1)
            lossD_recon = netD(fake_data_recon, E_true=E_true)
            lossD_recon = lossD_recon.mean()
            test_lossD_recon += lossD_recon
            
            fake_data_sample = torch.cat((data.view(-1,1,calo_size,calo_size,calo_size), 
                                          sample_batch.view(-1,1,calo_size,calo_size,calo_size)), 1)
            lossD_sample = netD(fake_data_sample, E_true=E_true)
            lossD_sample= lossD_sample.mean()
            test_lossD_sample += lossD_sample
            
            lossD_L_recon = netD_L(z)#.view(-1,1))
            lossD_L_recon = lossD_L_recon.mean()
            test_lossD_L_recon += lossD_L_recon

            test_lossD_L_sample = 0

            
            test_loss_sample += loss_sample.item() - L_D * lossD_sample.item()
            test_loss_recon += (loss_recon.item() - L_D * lossD_recon.item()
                                 - L_D_L * lossD_L_recon.item())
            
            if epoch%args["sample_interval"] == 0 or epoch == 1:
                if i == 0:
                    n = min(data.size(0), 8)
                    comparison = torch.cat([data.view(args["batch_size"],1, calo_size*calo_size, calo_size)[:n],
                                          recon_batch.view(args["batch_size"],1, calo_size*calo_size, calo_size)[:n]])
                    save_image(comparison.cpu(),
                             args["output_path"]+ "reconstruction_" + args["model"] + 
                               args["suffix"] + '_' + str(epoch) + '.png', nrow=n)

                    
    test_weighted_recon=test_weighted_recon/len(test_loader.dataset)*args["batch_size"]                
    test_unwe_recon=test_unwe_recon/len(test_loader.dataset)*args["batch_size"]   
    test_weighted_sample=test_weighted_sample/len(test_loader.dataset)*args["batch_size"]
    test_unwe_sample=test_unwe_sample/len(test_loader.dataset)*args["batch_size"]
    test_lossD_sample = test_lossD_sample/len(test_loader.dataset)*args["batch_size"]
    test_lossD_recon = test_lossD_recon/len(test_loader.dataset)*args["batch_size"]
    test_lossD_L_recon = test_lossD_L_recon/len(test_loader.dataset)*args["batch_size"]
    
    line =""
    for (w, u, n) in zip(test_weighted_recon, test_unwe_recon, name_recon):
        line = line +'Recon:  '+ n + '\t{:.3f} \t{:.3f} \n'.format(w,u)
    print(line)
    print('Test Recon lossD: {:.4f} Weight {:.2f}'.format(test_lossD_recon, L_D))
    print('Test Recon lossD_L: {:.4f} Weight {:.2f}'.format(test_lossD_L_recon, L_D_L))


    line =""
    for (w, u, n) in zip(test_weighted_sample, test_unwe_sample, name_sample):
        line = line + 'Sample: ' + n + '\t{:.3f} \t{:.3f} \n'.format(w,u)
    print(line)
    print('Test Sample lossD: {:.4f} Weight {:.2f}'.format(test_lossD_sample, L_D))
    print('Test Sample lossD_L: {:.4f} Weight {:.2f}'.format(test_lossD_L_sample, L_D_L))

    for b in range(0, min(E_true.size(0), 10)):
        print('sum_sample {:.3f}  sum_recon {:.3f}  sum_true {:.3f}  E_true {:.3f}'.format(
            sum_sample[b].item(), sum_recon[b].item(), sum_true[b].item(), E_true[b].item()))
    test_loss_sample = test_loss_sample/len(test_loader.dataset)*args["batch_size"]
    test_loss_recon = test_loss_recon/len(test_loader.dataset)*args["batch_size"]

    print('====> Test Recon loss: {:.4f}'.format(test_loss_recon))
    print('====> Test Sample loss: {:.4f}'.format(test_loss_sample))


def loss_function_VAE_ENR(recon_x, x, mu, logvar, E_pred, E_true, E_sum_pred, E_sum_true, reduction='mean', device='cpu',
              L_Losses = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0], 
                          z=None, args=None):

    B=(logvar.size(0)) #
    calo_size = x.size(-1)
    
    
    [L_KLD,         L_ENR_RECON,  
     L_ENR_RECON_N, L_MMD_Latent,
     L_MMD_HitKS,   L_SortMSE,    
     L_SortMAE] = L_Losses
 
    empt = 0
    
    
    x_sorted = x.view(B, -1)
    recon_x_sorted = recon_x.view(B, -1)
    
    x_sorted, _ = torch.sort(x_sorted, dim=1, descending=True)   
    recon_x_sorted, _ = torch.sort(recon_x_sorted, dim=1, descending=True) 

    if L_KLD == 0:
        KLD = 0
    else:
        KLD = (-0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()))/(B)

    if L_SortMSE == 0:
        SortMSE = 0
    else:  
        SortMSE = (x_sorted - recon_x_sorted).pow(2)
        SortMSE = torch.mean(SortMSE)

    if L_SortMAE == 0:
        SortMAE = 0
    else:  
        SortMAE = torch.abs(x_sorted - recon_x_sorted)
        SortMAE = torch.mean(SortMAE)
        
    if L_ENR_RECON == 0:
        ENR_RECON = 0
    else:  
        ENR_RECON = (E_sum_true - E_sum_pred).pow(2)
        ENR_RECON = torch.mean(ENR_RECON)
 
    if L_ENR_RECON_N == 0:
        ENR_RECON_N = 0
    else:  
        ENR_RECON_N = torch.abs(E_sum_true - E_sum_pred)
        ENR_RECON_N = torch.mean(ENR_RECON_N)

    if L_MMD_HitKS == 0:
        MMD_HitKS = 0
    else:
        mmd1 = mmd_hit_sortKernel(recon_x_sorted, x_sorted, kernel_size = args["HitMMDKS_Ker"], 
                                  stride = args["HitMMDKS_Str"], cutoff = args["HitMMDKS_Cut"], 
                                  alpha = args["HitMMD_alpha"])        
        MMD_HitKS = (mmd1)
        
    if L_MMD_Latent == 0:
        MMD_Latent = 0
    else:
        Bz = z.size(0)
        z_real = torch.randn_like(z)
        MMD_Latent = (mmd_latent_loss(z, z_real, alpha=1.0)/(Bz*Bz) + 
                      mmd_latent_loss(z, z_real, alpha=0.01)/(Bz*Bz))

        
    loss = (L_KLD*KLD                 + L_ENR_RECON*ENR_RECON     + 
            L_ENR_RECON_N*ENR_RECON_N + L_MMD_Latent*MMD_Latent   +
            L_MMD_HitKS*MMD_HitKS     + L_SortMSE*SortMSE         + 
            L_SortMAE*SortMAE
           )
    return loss, \
           [L_KLD*KLD,                 L_ENR_RECON*ENR_RECON,       
            L_ENR_RECON_N*ENR_RECON_N, L_MMD_Latent*MMD_Latent,
            L_MMD_HitKS*MMD_HitKS,     L_SortMSE*SortMSE,           
            L_SortMAE*SortMAE],\
           [KLD,         ENR_RECON, 
            ENR_RECON_N, MMD_Latent,
            MMD_HitKS,   SortMSE,   
            SortMAE],\
           ['KLD        ', 'ENR_RECON  ',
            'ENR_RECON_N', 'MMD_Latent ',
            'MMD_HitKS  ', 'SortMSE    ', 
            'SortMAE    ']



#####################################################
################ Losses Sub Functions ###############
#####################################################


def mmd_latent_loss(recon_z, z, alpha=1.0):     
    B = z.size(0)
    x = z
    y = recon_z
    
    xx, yy, xy = torch.mm(x,x.t()), torch.mm(y,y.t()), torch.mm(x,y.t())

    rx = (xx.diag().unsqueeze(0).expand_as(xx))
    ry = (yy.diag().unsqueeze(0).expand_as(yy))

    K = torch.exp(- alpha * (rx.t() + rx - 2*xx))
    L = torch.exp(- alpha * (ry.t() + ry - 2*yy))
    P = torch.exp(- alpha * (rx.t() + ry - 2*xy))

    return  (torch.sum(K)+torch.sum(L) - 2*torch.sum(P))



def mmd_hit_loss_cast_mean(recon_x, x, alpha=0.01): 
    B = x.size(0)
    
    x_batch = x.view(B, -1)
    y_batch = recon_x.view(B, -1)

    x = x_batch.view(B,1,-1)
    y = y_batch.view(B,1,-1)

    xx = torch.matmul(torch.transpose(x,1,2),x) 
    yy = torch.matmul(torch.transpose(y,1,2),y)
    xy = torch.matmul(torch.transpose(y,1,2),x)
    
    rx = (torch.diagonal(xx, dim1=1, dim2=2).unsqueeze(1).expand_as(xx))
    ry = (torch.diagonal(yy, dim1=1, dim2=2).unsqueeze(1).expand_as(yy))
    
    K = torch.exp(- alpha * (torch.transpose(rx,1,2) + rx - 2*xx))
    L = torch.exp(- alpha * (torch.transpose(ry,1,2) + ry - 2*yy))
    P = torch.exp(- alpha * (torch.transpose(ry,1,2) + rx - 2*xy))

    out = (torch.mean(K, (1,2))+torch.mean(L, (1,2)) - 2*torch.mean(P, (1,2)))
    
    return out



def mmd_hit_sortKernel(recon_x_sorted, x_sorted, kernel_size, stride, cutoff, alpha = 200):
    
    B = x_sorted.size(0)
    pixels = x_sorted.size(1)
    out = 0
    norm_out = 0
    
    for j in np.arange(0, min(cutoff, pixels), step = stride):
        distx = x_sorted[:, j:j+kernel_size]
        disty = recon_x_sorted[:, j:j+kernel_size]

        if j == 0:
            out = mmd_hit_loss_cast_mean(disty, distx, alpha=alpha)
        else:
            out += mmd_hit_loss_cast_mean(disty, distx, alpha=alpha)
        
        norm_out += 1
    return (torch.mean(out)/norm_out)


### FILE MANAGEMENT ###
 
# create output folder if it does not exists yet
def create_output_folder(outpath):
    if not os.path.exists(outpath):
        os.makedirs(outpath)
        print("ouput directory created in ", outpath)

    
