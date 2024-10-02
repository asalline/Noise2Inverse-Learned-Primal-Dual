import odl
import torch
import torch.nn as nn
import torch.optim as optim
import torch.func
import functorch
from odl.contrib.torch import OperatorModule
#from odl.contrib import torch as odl_torch
#from torch.nn.utils import clip_grad_norm_
import numpy as np
from n2i_LPD_training_module_0307 import get_images, geometry_and_ray_trafo
from LearnedPrimalDualModuleUNet import LPD
import matplotlib.pyplot as plt
import os
import time
import copy
from utils import data_split
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


shape = (np.shape(test_images)[1], np.shape(test_images)[2])
domain, geometry, ray_transform, output_shape = geometry_and_ray_trafo(setup='full', shape=shape, device=device, factor_lines = 2)
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
num_of_splits = 8
averaged = 7

name = 'n2i_LPD50k_005_num4_avg3_ReLU_updated_0210.pth'

split_list = list(range(num_of_splits))
split_geom = [geometry[j::num_of_splits] for j in split_list]
operator_split = [odl.tomo.RayTransform(domain, split, impl='astra_' + device) for split in split_geom]
operator_list = [OperatorModule(operator) for operator in operator_split]
split_fbp = [odl.tomo.analytic.filtered_back_projection.fbp_op(operator, padding=1) for operator in operator_split]
fbp_list = [OperatorModule(fbp) for fbp in split_fbp]

all_arrangements = torch.zeros((test_amount, num_of_splits) + shape)

### Adding Gaussian noise to the sinograms. Here some problem solving is
### needed to make this smoother.
for k in range(test_amount):
    test_sinogram_k = test_sinograms[k,:,:].cpu().detach().numpy()
    noise = np.random.normal(mean, test_sinogram_k.max(), test_sinogram_k.shape) * percentage
    test_noisy_sinogram = test_sinogram_k + noise
    test_noisy_sinograms[k,:,:] = torch.as_tensor(test_noisy_sinogram)

print('noise', noise.shape)
start = time.time()
test_input_reco, _, _, _, all_arrangements, all_sinograms = \
    data_split(test_amount, num_of_splits, shape, averaged, test_noisy_sinograms, fbp_list, domain, device, ray_transform)

# operator_list = []
# fbp_list = []
# for j in range(len(operator_split)):
#             operator = operator_split[j]
#             operator_list.append(OperatorModule(operator).to(device))
#             fbp_list.append(OperatorModule(odl.tomo.analytic.filtered_back_projection.fbp_op(operator, padding=1)).to(device))
            # fbp2 = OperatorModule(fbp2).to(device)

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
LPD_net = LPD(primal_in_channels=2, dual_in_channels=3, out_channels=1, first_channel=32, depth=3, conv_kernel_size=(3,3), max_pool_kernel_size=(2,2),
              up_conv_kernel_size=(2,2), padding=1, skip_connection_list=[], operator=ray_transform_module, adjoint_operator=adjoint_operator_module,
              operator_norm=operator_norm, n_iter=5, device='cuda').to(device)

# print(LGS_net.device)

### Getting model parameters
LPD_parameters = list(LPD_net.parameters())
# print(LPD_net)

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
    # nets = [net.to(device) for _ in range(test_amount)]
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
            start2 = time.time()
            input_reco, target_reco, sinogram_reco, target_sinogram_reco, rec_split, sinogram_split2\
                = data_split(train_batch, num_of_splits, shape, averaged, noisy_sinograms, fbp_list, domain, device, ray_transform)
            end2 = time.time()
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
        
        # print('g_batch', g_batch.shape)
        # net.train()
        # optimizer.zero_grad()
        ### Taking out some enhanced images
        # print('g_batch', f_batch2[0,0,:,:].shape)
        # print('f_batch2', f_batch2.shape)
        outs = net(f_batch2.float().to(device), g_batch.float().to(device), operator_list, fbp_list, domain, num_of_splits, averaged, noise)
        # print('OUTS', outs.shape)
        
        optimizer.zero_grad()
        
        
        ### Setting gradient to zero
        # optimizer.zero_grad()
        
        ### Calculating loss of the outputs
        loss = loss_train(outs, f_batch.float())
        tensorboard_loss.append(loss.item()**0.5)
        # loss = torch.from_numpy(loss)
        # writer.add_scalar('data/scalar1', tensorboard_loss[i], i)
        #loss.requires_grad=True
        
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
            # print('EVAL')
            net.eval()
            with torch.no_grad():
                # class EvalModel(torch.autograd.Function):
                #     generate_vmap_rule = False

                #     @staticmethod
                    

                # def eval_model(net, all_arrangements, all_sinograms, operator_split, domain, num_of_splits, averaged, noise):
                #     return net(all_arrangements, all_sinograms, operator_split, domain, num_of_splits, averaged, noise)
                
                # # outs5 = torch.vmap(eval_model, in_dims=(None,0,0,None,None,None,None,None))(net, all_arrangements[None,:,:,:], all_sinograms[None,:,:,:], operator_split, domain, num_of_splits, averaged, noise).generate_vmap_rule=False
                # # print(outs5.shape)
                # params, buffers = torch.func.stack_module_state(nets)
                # # base_model = copy.deepcopy(net)
                # # base_model = base_model.to('meta')
                # base_model = nets[0]
                # # @torch.no_grad
                # # def fmodel(params, buffers, all_arrangements, all_sinograms, operator_split, domain, num_of_splits, averaged, noise):
                # #     return torch.func.functional_call(base_model, (params, buffers), (all_arrangements, all_sinograms, operator_split, domain, num_of_splits, averaged, noise))
                
                # # predictions_vmap = torch.vmap(fmodel, in_dims=(None,None,0,0,None,None,None,None,None,None))(params, buffers, all_arrangements[None,:,:,:], all_sinograms[None,:,:,:], operator_split, domain, num_of_splits, averaged, noise)
                # def fmodel(all_arrangements, all_sinograms, operator_split, domain, num_of_splits, averaged, noise):
                #     return torch.func.functional_call(base_model, (all_arrangements, all_sinograms, operator_split, domain, num_of_splits, averaged, noise))
                
                # predictions_vmap = torch.vmap(eval_model, in_dims=(None,0,0,None,None,None,None,None))(net, all_arrangements[None,:,:,:], all_sinograms[None,:,:,:], operator_split, domain, num_of_splits, averaged, noise)

                # print('preds', predictions_vmap.shape)
                # outs2 = torch.zeros((test_amount, num_of_splits) + shape).to(device)
                # outs3 = torch.zeros((test_amount, ) + shape).to(device)
                outs4 = torch.zeros((test_amount, num_of_splits) + shape).to(device)
                # print('test', all_arrangements[[0],:,:,:].shape)
                # for k in range(test_amount):
                #     for j in range(num_of_splits):
                        
                #         # print('test', all_arrangements[[k],[j],:,:].shape)
                #         # outs2[k,j,:,:] = outs2[k,j,:,:] + net(all_arrangements[[k],[j],None,:,:].to(device))
                #         # outs3[k,:,:] = outs3[k,:,:] + outs2[k,j,:,:]
                #         outs3[k,:,:] = outs3[k,:,:] + net(all_arrangements[[k],[j],None,:,:].to(device))
                        
                #     outs4[k,:,:] = torch.mean(outs3[k,:,:], dim=0)
              
                # outs2 = outs2 / test_amount
                # print('hmm', torch.swapaxes(all_arrangements[[0],:,:,:], 0,1).shape)
                # start3 = time.time()
                # losses = 0
                
                # for k in range(test_amount):
                #     losses += functorch.vmap(net, in_dims=(0,0,None,None,None,None,None))(all_arrangements[[k],:,:,:][None,:,:,:].to(device), all_sinograms[[k],:,:,:][None,:,:,:].to(device), \
                #                        operator_split, domain, num_of_splits, averaged, noise)
                # # print(all_arrangements.shape)
                # # losses = functorch.vmap(net, in_dims=(0,0,None,None,None,None,None))(all_arrangements[None,:,:,:].to(device), all_sinograms[None,:,:,:].to(device), \
                # #                        operator_split, domain, num_of_splits, averaged, noise)
                # # net_list = [net]
                # print('HERE')
                # fmodel, params, buffers = functorch.combine_state_for_ensemble(net_list)
                # outputs = functorch.vmap(fmodel, in_dims=(0,None,None,None))(params, buffers, all_arrangements, all_sinograms, operator_split, domain, num_of_splits, averaged, noise).to(device)
                # outs4 = loss_test(outputs, test_images.to(device)).item()**0.5
                for k in range(test_amount):
                # for j in range(num_of_splits):
                # print('hmm2', torch.swapaxes(all_arrangements[[k],:,:,:], 0,1).shape)
                    # print('hmm2', all_arrangements[[k],j,:,:].shape)
                    # print('hmm3', all_sinograms[[k],j,:,:].shape)
                    outs4[k,:,:,:] = net(all_arrangements[[k],:,:,:].to(device), all_sinograms[[k],:,:,:].to(device), \
                                       operator_list, fbp_list, domain, num_of_splits, averaged, noise)
                    # print('outs3', outs3.shape)
                    # outs4[k,:,:] = torch.mean(outs3, dim=0)
                
                outs4 = torch.mean(outs4, dim=1)
                # end3 = time.time()
                # print('time loss: ', end3-start3)
                # print('outs4', outs4.shape)
                # print('outs2', outs2.shape)
                # print('images', test_images.shape)
                # print('4',outs4.shape)
                ### Calculating test loss with test data outputs
                start4 = time.time()
                test_loss = loss_test(outs4[:,None,:,:], test_images.to(device)).item()**0.5
                end4 = time.time()
                # print('time loss2', end4-start4)
                train_loss = loss.item() ** 0.5
                running_loss.append(train_loss)
                running_test_loss.append(test_loss)
                # print(f'step length = {step_length2}')
            
            ### Printing some data out
            if i % 100 == 0:
                print(f'Iter {i}/{n_train} Train Loss: {train_loss:.2e}, Test Loss: {test_loss:.2e}, PSNR: {psnr(test_loss**2):.2f}') #, end='\r'
                plt.figure()
                plt.subplot(1,2,1)
                plt.imshow(outs4[4,:,:].cpu().detach().numpy())
                plt.subplot(1,2,2)
                plt.imshow(test_images[4,0,:,:].cpu().detach().numpy())
                plt.show()
            # if i % 4000 == 0:
            #     plt.figure()
            #     plt.subplot(1,2,1)
            #     plt.imshow(outs4[4,:,:].cpu().detach().numpy())
            #     plt.subplot(1,2,2)
            #     plt.imshow(test_images[4,0,:,:].cpu().detach().numpy())
            #     plt.show()
        end = time.time()
        print(f'TIME OF ITER #{i}: ', end-start)

    ### After iterating taking one reconstructed image and its ground truth
    ### and showing them
    # plt.figure()
    # plt.subplot(1,2,1)
    # plt.imshow(outs[0,0,:,:].cpu().detach().numpy())
    # plt.subplot(1,2,2)
    # plt.imshow(f_batch[0,0,:,:].cpu().detach().numpy())
    # plt.show()

    ### Plotting running loss and running test loss
    plt.figure()
    plt.semilogy(running_loss)
    plt.semilogy(running_test_loss)
    plt.show()

    return running_loss, running_test_loss, net

### Calling training function to start the naural network training
running_loss, running_test_loss, ready_to_eval = train_network(LPD_net, n_train=50001, batch_size=1, noise=noise) # g_train, g_test, f_train, f_test

torch.save(ready_to_eval.state_dict(), '/scratch2/antti/networks/thesis_networks/final_networks/' + name)