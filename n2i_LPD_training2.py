CUDA_LAUNCH_BLOCKING=1
### Importing packages and modules
import odl
import torch
import torch.nn as nn
import torch.optim as optim
import torch.func
from odl.contrib.torch import OperatorModule
#from odl.contrib import torch as odl_torch
#from torch.nn.utils import clip_grad_norm_
import numpy as np
from n2i_LPD_training_module_0307 import get_images #, geometry_and_ray_trafo
from utils import data_split, to_cpu, geometry_plus_ray_trafo
from LearnedPrimalDualModule import LearnedPrimalDual
import matplotlib.pyplot as plt
import os
import time
# import tensorboardX


### Check if nvidia CUDA is available and using it, if not, the CPU
device = 'cuda' if torch.cuda.is_available() else 'cpu'
# device = 'cpu'
# print(device)
# torch.cuda.empty_cache()
# writer = tensorboardX.SummaryWriter('runs/test_n2i_LPD')
path = r"C:\Users\antti\Documents\Koodit\asalline Cambridge-Autumn-School main Deep%20Learning%20Reconstructions\Deep Learning Reconstructions\walnuts"
train_batch = 2

len_images = len(os.listdir(path))
# train_batch = 2
### Using function "get_images" to import images from the path.
# list_of_test_images = list(range(0,363,5)) #range(0,363,5)
# test_amount = len(list_of_test_images)
list_of_test_images = list(range(0,len_images, 7))
images = get_images(path, amount_of_images='all', scale_number=2)
### Converting images such that they can be used in calculations
images = np.array(images, dtype='float32')
images = torch.from_numpy(images).float().to(device)
# test_images = test_images[list_of_test_images]
test_images = images[list_of_test_images]
test_amount = test_images.shape[0]
print('shape', test_images.shape)

### Using functions from "UNet_functions". Taking shape from images to produce
### odl parameters and getting Radon transform operator and its adjoint.
shape = (np.shape(test_images)[1], np.shape(test_images)[2])
domain, geometry, ray_transform, output_shape = geometry_plus_ray_trafo(min_domain_point=(-1,-1),
                                                                        max_domain_point=(1,1),
                                                                        shape=shape,
                                                                        num_of_angles=1024,
                                                                        num_of_lines=512,
                                                                        angle_interval=(0, 2*np.pi),
                                                                        line_interval=(-np.pi, np.pi),
                                                                        source_radius=2,
                                                                        detector_radius=2,
                                                                        dtype='float32')
# print(domain)
# print(geometry)
# print(ray_transform)
# print(output_shape)
fbp_operator = odl.tomo.analytic.filtered_back_projection.fbp_op(ray_transform, padding=1)

### Using odl functions to make odl operators into PyTorch modules
ray_transform_module = OperatorModule(ray_transform).to(device)
adjoint_operator_module = OperatorModule(ray_transform.adjoint).to(device)
fbp_operator_module = OperatorModule(fbp_operator).to(device)

operator_norm = odl.power_method_opnorm(ray_transform)

### Making sinograms from the images using Radon transform module
test_sinograms = ray_transform_module(test_images)
test_sinograms = torch.as_tensor(test_sinograms)

### Allocating used tensors
test_noisy_sinograms = torch.zeros((test_sinograms.shape[0], ) + output_shape).to(device)
rec_images = torch.zeros((test_sinograms.shape[0], ) + shape)
test_input_reco = torch.zeros((test_sinograms.shape[0], ) + shape)
test_target_reco = torch.zeros((test_sinograms.shape[0], ) + shape)

mean = 0
# variance = 0.005
# sigma = variance ** 0.5
percentage = 0.05
num_of_splits = 4
averaged = 3

name = 'n2i_LPD50k_005_num4_avg3_ReLU_updated.pth'

split_list = list(range(num_of_splits))
split_geom = [geometry[j::num_of_splits] for j in split_list]
operator_split = [odl.tomo.RayTransform(domain, split, impl='astra_' + device) for split in split_geom]
operator_list = [OperatorModule(operator) for operator in operator_split]
split_fbp = [odl.tomo.analytic.filtered_back_projection.fbp_op(operator, padding=1) for operator in operator_split]
fbp_list = [OperatorModule(fbp) for fbp in split_fbp]

all_arrangements = torch.zeros((test_amount, num_of_splits) + shape)



### Adding Gaussian noise to the sinograms. Here some problem solving is
### needed to make this smoother.
start = time.time()
# noise = torch.normal(mean=mean, std = torch.max(test_sinograms), size=test_sinograms.shape).to(device)
# test_noisy_sinograms = test_sinograms + noise
for k in range(test_amount):
    test_sinogram_k = test_sinograms[k,:,:].cpu().detach().numpy()
    noise = np.random.normal(mean, test_sinogram_k.max(), test_sinogram_k.shape) * percentage
    test_noisy_sinogram = test_sinogram_k + noise
    test_noisy_sinograms[k,:,:] = torch.as_tensor(test_noisy_sinogram)
end = time.time()
print('noise time', end-start)
# 0.07616


print('noise', noise.shape)
start = time.time()
test_input_reco, _, _, _, all_arrangements, all_sinograms = \
    data_split(test_amount, num_of_splits, shape, averaged, test_noisy_sinograms, fbp_list, domain, device, ray_transform)
end = time.time()
print('TIME HERE: ', end-start)
### Using FBP to get reconstructed images from noisy sinograms
rec_images = fbp_operator_module(test_noisy_sinograms)

### All the data into same device
test_sinograms = test_sinograms[:,None,:,:].cpu().detach()
test_noisy_sinograms = test_noisy_sinograms[:,None,:,:].cpu().detach()
rec_images = rec_images[:,None,:,:].cpu().detach()
test_images = test_images[:,None,:,:].cpu().detach()
test_input_reco = test_input_reco[:,None,:,:].cpu().detach()
test_target_reco = test_target_reco[:,None,:,:].cpu().detach()

### Setting UNet as model and passing it to the used device
LPD_net = LearnedPrimalDual(num_of_layers=5,
                            in_channels_primal=2,
                            in_channels_dual=3,
                            mid_channels=32,
                            out_channels=1,
                            conv_kernel_size=(3,3),
                            padding=1,
                            num_parameters=1,
                            alpha=0,
                            operator=ray_transform_module,
                            adjoint_operator=adjoint_operator_module,
                            operator_norm=operator_norm,
                            num_unrolled_iterations=10).to(device)
# ray_transform_module, adjoint_operator_module, operator_norm, n_iter=10, device=device
# print(LGS_net.device)

### Getting model parameters
LPD_parameters = list(LPD_net.parameters())

### Defining PSNR function.
def psnr(loss):
    
    psnr = 10 * np.log10(1.0 / loss+1e-10)
    
    return psnr


loss_train = nn.MSELoss()
loss_test = nn.MSELoss()

### Defining evaluation (test) function
def eval(net, g, f):

    test_loss = []
    
    ### Setting network into evaluation mode
    net.eval()
    test_loss.append(torch.sqrt(loss_test(net(g), f)).item())
    print(test_loss)
    out3 = net(g[0,None,:,:])

    return out3

### Setting up some lists used later
running_loss = []
running_test_loss = []
tensorboard_loss = []
torch.autograd.set_detect_anomaly(True)
### Defining training scheme
def train_network(net, n_train=300, batch_size=25, noise=noise): #g_train, g_test, f_train, f_test, 

    ### Defining optimizer, ADAM is used here
    optimizer = optim.Adam(LPD_parameters, lr=0.0001, betas = (0.9, 0.99)) #betas = (0.9, 0.99)
    
    ### Definign scheduler, can be used if wanted
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, n_train)
    
    ### Setting network into training mode
    net.train()
    ### Here starts the itarting in training
    for i in range(n_train):
        start = time.time()
        # print(i)
        net.train()
        if i % train_batch == 0:
            images = get_images(path, amount_of_images=train_batch, scale_number=2)
            ### Converting images such that they can be used in calculations
            images = np.array(images, dtype='float32')
            images = torch.from_numpy(images).float().to(device)
            sinograms = ray_transform_module(images)
            sinograms = torch.as_tensor(sinograms)

            ### Allocating used tensors
            noisy_sinograms = torch.zeros((sinograms.shape[0], ) + output_shape).to(device)
            rec_images = torch.zeros((sinograms.shape[0], ) + shape).to(device)
            input_reco = torch.zeros((sinograms.shape[0], ) + shape).to(device)
            target_reco = torch.zeros((sinograms.shape[0], ) + shape).to(device)

            # averaged = 1
            ### Adding Gaussian noise to the sinograms. Here some problem solving is
            ### needed to make this smoother.
            for k in range(train_batch):
                sinogram_k = sinograms[k,:,:].cpu().detach().numpy()
                noise = np.random.normal(mean, sinogram_k.max(), sinogram_k.shape) * percentage
                noisy_sinogram = sinogram_k + noise
                noisy_sinograms[k,:,:] = torch.as_tensor(noisy_sinogram)
            # start2 = time.time()
            input_reco, target_reco, sinogram_reco, target_sinogram_reco, rec_split, sinogram_split2 \
                = data_split(train_batch, num_of_splits, shape, averaged, noisy_sinograms, fbp_list, domain, device, ray_transform)
            # end2 = time.time()
            # print('TIME DATA SPLIT: ', end2-start2)

            ### Using FBP to get reconstructed images from noisy sinograms
            rec_images = fbp_operator_module(noisy_sinograms)

            ### All the data into same device
            sinograms = sinograms[:,None,:,:].cpu().detach()
            noisy_sinograms = noisy_sinograms[:,None,:,:].cpu().detach()
            rec_images = rec_images[:,None,:,:].cpu().detach()
            images = images[:,None,:,:].cpu().detach()
            input_reco = input_reco[:,None,:,:].cpu().detach()
            target_reco = target_reco[:,None,:,:].cpu().detach()
            sinogram_reco = sinogram_reco[:,None,:,:]
            target_sinogram_reco = target_sinogram_reco[:,None,:,:]
            # print('rec', rec_split.shape)
            # sinogram_split2 = sinogram_split2 [:,None,:,:]
        
        ### Taking batch size amount of data pieces from the random 
        ### permutation of all training data
        n_index = np.random.permutation(input_reco.shape[0])[:batch_size]
        g_batch = sinogram_split2[n_index,0:averaged,:,:].to(device)
        f_batch2 = rec_split[n_index,0:averaged,:,:].to(device)
        f_batch = target_reco[n_index,:,:,:].to(device)
        
        outs = net(f_batch2.float().to(device), g_batch.float().to(device), operator_list, fbp_list, domain, num_of_splits, averaged, noise)
        
        optimizer.zero_grad()
        
        ### Calculating loss of the outputs
        loss = loss_train(outs, f_batch.float())
        tensorboard_loss.append(loss.item()**0.5)
        
        ### Calculating gradient
        loss.backward()
        
        ### Here gradient clipping could be used
        torch.nn.utils.clip_grad_norm_(LPD_parameters, max_norm=1.0, norm_type=2)
        
        ### Taking optimizer step
        optimizer.step()
        scheduler.step()
        
        #running_loss.append(loss.item())

        ### Here starts the running tests
        if i % 100 == 0:
            
            net.eval()
            with torch.no_grad():
                
                outs4 = torch.zeros((test_amount, num_of_splits) + shape).to(device)
                
                for k in range(test_amount):
                
                    outs4[k,:,:,:] = net(all_arrangements[[k],:,:,:].to(device), all_sinograms[[k],:,:,:].to(device), \
                                       operator_list, fbp_list, domain, num_of_splits, averaged, noise)
               
                outs4 = torch.mean(outs4, dim=1)
                
                ### Calculating test loss with test data outputs
                test_loss = loss_test(outs4[:,None,:,:], test_images.to(device)).item()**0.5
                train_loss = loss.item() ** 0.5
                running_loss.append(train_loss)
                running_test_loss.append(test_loss)
                # print(f'step length = {step_length2}')
            
            ### Printing some data out
            if i % 100 == 0:
                print(f'Iter {i}/{n_train} Train Loss: {train_loss:.2e}, Test Loss: {test_loss:.2e}, PSNR: {psnr(test_loss**2):.2f}') #, end='\r'
                plt.figure()
                plt.subplot(1,2,1)
                plt.imshow(to_cpu(outs4[4,:,:]))
                plt.subplot(1,2,2)
                plt.imshow(to_cpu(test_images[4,0,:,:]))
                plt.show()

        end = time.time()
        print(f'TIME OF ITER #{i}: ', end-start)

    ### Plotting running loss and running test loss
    plt.figure()
    plt.semilogy(running_loss)
    plt.semilogy(running_test_loss)
    plt.show()

    return running_loss, running_test_loss, net

### Calling training function to start the naural network training
running_loss, running_test_loss, ready_to_eval = train_network(LPD_net, n_train=50001, batch_size=1, noise=noise) # g_train, g_test, f_train, f_test

torch.save(ready_to_eval.state_dict(), '/scratch2/antti/networks/thesis_networks/final_networks/' + name)