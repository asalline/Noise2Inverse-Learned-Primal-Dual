import odl
from odl.contrib.torch import OperatorModule
import cv2 as cv
import numpy as np
# import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import os
import matplotlib.pyplot as plt
import time

# def get_images(path, amount_of_images='all', scale_number=1):

#     all_images = []
#     all_image_names = os.listdir(path)
#     # print(len(all_image_names))
#     if amount_of_images == 'all':
#         for name in all_image_names:
#             temp_image = cv.imread(path + '/' + name, cv.IMREAD_UNCHANGED)
#             # image = temp_image[90:410, 90:410]
#             # image = image[0:320:scale_number, 0:320:scale_number]
#             # image = temp_image[2:498, 2:498]
#             image = temp_image[59:443, 59:443]
#             image = image[0::scale_number, 0::scale_number]
#             image = image / 0.07584485627272729 #np.max(np.max(image))
#             all_images.append(image)
#     else:
#         temp_indexing = np.random.permutation(len(all_image_names))[:amount_of_images]
#         images_to_take = [all_image_names[i] for i in temp_indexing]
#         for name in images_to_take:
#             temp_image = cv.imread(path + '/' + name, cv.IMREAD_UNCHANGED)
#             # image = temp_image[90:410, 90:410]
#             # image = image[0:320:scale_number, 0:320:scale_number]
#             # image = temp_image[2:498, 2:498]
#             image = temp_image[59:443, 59:443]
#             image = image[0::scale_number, 0::scale_number]
#             image = image / 0.07584485627272729 #np.max(np.max(images))
#             all_images.append(image)
    
#     return all_images

def get_images(path, amount_of_images='all', scale_number=1):

    all_images = []
    all_image_names = os.listdir(path)
    # print(len(all_image_names))
    if amount_of_images == 'all':
        for name in all_image_names:
            temp_image = cv.imread(path + '/' + name, cv.IMREAD_UNCHANGED)
            image = temp_image[2:498, 2:498]
            image = image[0::scale_number, 0::scale_number]
            image = image / 0.07584485627272729
            all_images.append(image)
    else:
        temp_indexing = np.random.permutation(len(all_image_names))[:amount_of_images]
        images_to_take = [all_image_names[i] for i in temp_indexing]
        # print(images_to_take)
        for name in images_to_take:
            temp_image = cv.imread(path + '/' + name, cv.IMREAD_UNCHANGED)
            # print(temp_image)
            image = temp_image[2:498, 2:498]
            image = image[0::scale_number, 0::scale_number]
            image = image / 0.07584485627272729
            all_images.append(image)
    
    return all_images

# def get_images(path, amount_of_images='all', scale_number=1):

#     all_images = []
#     all_image_names = os.listdir(path)
#     # print(len(all_image_names))
#     if amount_of_images == 'all':
#         for name in all_image_names:
#             # print(path + '\\' + name)
#             temp_image = cv.imread(path + '//' + name, cv.IMREAD_UNCHANGED)
#             image = temp_image[80:420, 80:420]
#             image = image[0:340:scale_number, 0:340:scale_number]
#             image = image / 0.07584485627272729
#             all_images.append(image)
#     else:
#         temp_indexing = np.random.permutation(len(all_image_names))[:amount_of_images]
#         images_to_take = [all_image_names[i] for i in temp_indexing]
#         for name in images_to_take:
#             temp_image = cv.imread(path + '/' + name, cv.IMREAD_UNCHANGED)
#             image = temp_image[90:410, 90:410]
#             image = image[0:320:scale_number, 0:320:scale_number]
#             image = image / 0.07584485627272729
#             all_images.append(image)

#     return all_images

def geometry_and_ray_trafo(setup='full', min_domain_corner=[-1,-1], max_domain_corner=[1,1], \
                           shape=(100,100), source_radius=2, detector_radius=1, \
                           dtype='float32', device='cpu', factor_lines = 1):

    device = 'astra_' + device
    # print(device)
    domain = odl.uniform_discr(min_domain_corner, max_domain_corner, shape, dtype=dtype)

    if setup == 'full':
        angles = odl.uniform_partition(0, 2*np.pi, 1024)
        # lines = odl.uniform_partition(-1*np.pi, np.pi, 352)
        lines = odl.uniform_partition(-1*np.pi, np.pi, int(1024/factor_lines))
        geometry = odl.tomo.FanBeamGeometry(angles, lines, source_radius, detector_radius)
        # output_shape = (512, 352)
        output_shape = (1024, int(1024/factor_lines))
    elif setup == 'sparse':
        angle_measurements = 100
        line_measurements = int(1024/factor_lines)
        angles = odl.uniform_partition(0, 2*np.pi, angle_measurements)
        lines = odl.uniform_partition(-1*np.pi, np.pi, line_measurements)
        geometry = odl.tomo.FanBeamGeometry(angles, lines, source_radius, detector_radius)
        output_shape = (angle_measurements, line_measurements)
    elif setup == 'limited':
        starting_angle = 0
        final_angle = np.pi
        angles = odl.uniform_partition(starting_angle, final_angle, 1024)
        lines = odl.uniform_partition(-1*np.pi, np.pi, int(1024/factor_lines))
        geometry = odl.tomo.FanBeamGeometry(angles, lines, source_radius, detector_radius)
        output_shape = (int(1024), int(1024/factor_lines))
        
    ray_transform = odl.tomo.RayTransform(domain, geometry, impl=device)
    
    return domain, geometry, ray_transform, output_shape


# def geometry_and_ray_trafo(setup='full', min_domain_corner=[-1,-1], max_domain_corner=[1,1], \
#                            shape=(100,100), source_radius=2, detector_radius=1, \
#                            dtype='float32', device='cpu', factor_lines = 1):

#     device = 'astra_' + device
#     # print(device)
#     domain = odl.uniform_discr(min_domain_corner, max_domain_corner, shape, dtype=dtype)

#     if setup == 'full':
#         angles = odl.uniform_partition(0, 2*np.pi, 1024)
#         lines = odl.uniform_partition(-1*np.pi, np.pi, int(1028/factor_lines))
#         geometry = odl.tomo.FanBeamGeometry(angles, lines, source_radius, detector_radius)
#         output_shape = (1024, int(1028/factor_lines))
#     elif setup == 'sparse':
#         angle_measurements = 100
#         line_measurements = int(512/factor_lines)
#         angles = odl.uniform_partition(0, 2*np.pi, angle_measurements)
#         lines = odl.uniform_partition(-1*np.pi, np.pi, line_measurements)
#         geometry = odl.tomo.FanBeamGeometry(angles, lines, source_radius, detector_radius)
#         output_shape = (angle_measurements, line_measurements)
#     elif setup == 'limited':
#         starting_angle = 0
#         final_angle = np.pi * 3/4
#         angles = odl.uniform_partition(starting_angle, final_angle, 360)
#         lines = odl.uniform_partition(-1*np.pi, np.pi, int(512/factor_lines))
#         geometry = odl.tomo.FanBeamGeometry(angles, lines, source_radius, detector_radius)
#         output_shape = (int(360), int(512/factor_lines))
        
#     ray_transform = odl.tomo.RayTransform(domain, geometry, impl=device)

#     return domain, geometry, ray_transform, output_shape

def data_split2(num_of_images, num_of_splits, shape, averaged, noisy_sinograms, geometry, domain, device, ray_transform):    
    sinogram_split = torch.zeros((num_of_images, num_of_splits) + (int(noisy_sinograms.shape[1]/num_of_splits), noisy_sinograms.shape[2]))
    rec_split = torch.zeros((num_of_images, num_of_splits) + (shape))
    operator_split = []
    # # print(rec_split.shape)
    # for k in range(num_of_images):
    #     print(k)
    #     for j in range(num_of_splits):
    #         split = geometry[j::num_of_splits]
    #         # noisy_sinogram = noisy_sinogram[j,:,:]
    #         sinogram_split = noisy_sinograms[:, j::num_of_splits, :]
    #         operator_split = odl.tomo.RayTransform(domain, split, impl='astra_' + device)
    #         # print(f'during {j}',operator_split.range)
    #         split_FBP = odl.tomo.analytic.filtered_back_projection.fbp_op(operator_split, padding=1)
    #         split_FBP_module = OperatorModule(split_FBP)
    #         # reco = split_FBP_module(sinogram_split)
    #         # print('reco',reco.shape)
    #         # rec_split = split_FBP_module(sinogram_split)
    #         rec_split[k,j,:,:] = split_FBP_module(sinogram_split)[k,:,:]
    #         # print(type(split_FBP))
    #         # FBP_domain = split_FBP.domain
    #         # print(split_FBP.domain)
    #         # split_FBP = split_FBP.asarray()
    #         # sinogram_split[j,:,:] = domain.element(sinogram_split[j,:,:])
    #         # print(type(sinogram_split[j,:,:]))
    #         # rec_split[j,:,:] = split_FBP(sinogram_split[j,:,:])
            
    for j in range(num_of_splits):
        split = geometry[j::num_of_splits]
        operator_split.append(odl.tomo.RayTransform(domain, split, impl='astra_' + device))
        # print('ray', operator_split)
        # print(f'during {j}',operator_split.range)
        split_FBP = odl.tomo.analytic.filtered_back_projection.fbp_op(operator_split[-1], padding=1)
        split_FBP_module = OperatorModule(split_FBP)
        
        for k in range(num_of_images):
            # print(k)
            if k == 0:
                # print(np.shape(noisy_sinograms[k, j::num_of_splits, :]))
                # print(np.shape(sinogram_split))
                sinogram_split[k,j,:,:] = noisy_sinograms[k, j::num_of_splits, :][:,:]
                to_FBP = noisy_sinograms[:, j::num_of_splits, :]
                rec_split[k,j,:,:] = split_FBP_module(to_FBP)[k,:,:]
            else:
                sinogram_split[k,j,:,:] = noisy_sinograms[k, j::num_of_splits, :][:,:]
                to_FBP = noisy_sinograms[:, j::num_of_splits, :]
                rec_split[k,j,:,:] = split_FBP_module(to_FBP)[k,:,:]
    
    # print('after',operator_split.range)
    # split_FBP = odl.tomo.analytic.filtered_back_projection.fbp_op(operator_split, padding=1)
    # # print('asd')
    # fbp_operator_module = OperatorModule(split_FBP).to('cuda')
    # # print('fbp',split_FBP.range)
    # # split_reco = torch.zeros((num_of_splits, ) + (shape))
    # rec_split = fbp_operator_module(sinogram_split)
    # print('rec split', rec_split.shape)
    # print('noisy',np.shape(noisy_sinograms))
    input_reco = np.zeros((num_of_images, ) + shape)#.astype('float32')
    target_reco = np.zeros((num_of_images, ) + shape)#.astype('float32'))
    sinogram_reco = np.zeros((num_of_images, ) + (int(np.shape(noisy_sinograms)[1]/num_of_splits), np.shape(noisy_sinograms)[2]))
    target_sinogram_reco = np.zeros((num_of_images, ) + (int(np.shape(noisy_sinograms)[1]/num_of_splits), np.shape(noisy_sinograms)[2]))
    # print('reco',np.shape(sinogram_reco))
    # eval_reco = torch.zeros((averaged, ) + shape)
    # print('rec', rec_split.shape)
    for j in range(num_of_images):
        for k in range(averaged):
        # eval_reco[k,:,:] = rec_split[k,:,:]#.cpu().detach().numpy()
            input_reco[j,:,:] = input_reco[j,:,:] + rec_split[j,k,:,:].cpu().detach().numpy()
            sinogram_reco[j,:,:] = sinogram_reco[j,:,:] + noisy_sinograms[j, k::num_of_splits, :].cpu().detach().numpy()
            
    # print('input', input_reco.shape)
    input_reco = input_reco / averaged
    sinogram_reco = sinogram_reco / averaged
    
    if num_of_splits - averaged != 0:
        for j in range(num_of_images):
            for k in range(num_of_splits - averaged):
                target_reco[j,:,:] = target_reco[j,:,:] + rec_split[j,averaged + k,:,:].cpu().detach().numpy()
                target_sinogram_reco[j,:,:] = target_sinogram_reco[j,:,:] + noisy_sinograms[j, (averaged + k)::num_of_splits, :].cpu().detach().numpy()
    
        target_reco = target_reco / (num_of_splits - averaged)
        target_sinogram_reco = target_sinogram_reco / (num_of_splits - averaged)

# torch.as_tensor(input_reco), torch.as_tensor(target_reco)    
# input_reco, target_reco
    # print('sinogram split',np.shape(sinogram_split))
    return torch.as_tensor(input_reco), torch.as_tensor(target_reco), torch.as_tensor(sinogram_reco), \
        torch.as_tensor(target_sinogram_reco), rec_split, sinogram_split, operator_split

def data_split(num_of_images, num_of_splits, shape, averaged, noisy_sinograms, geometry, domain, device, ray_transform):
    sinogram_split = torch.zeros((num_of_images, num_of_splits) + (int(noisy_sinograms.shape[1]/num_of_splits), noisy_sinograms.shape[2])).to(device)
    sinogram_split2 = torch.zeros((num_of_images, num_of_splits) + (int(noisy_sinograms.shape[1]/num_of_splits), noisy_sinograms.shape[2])).to(device)
    rec_split = torch.zeros((num_of_images, num_of_splits) + (shape)).to(device)
    rec_split2 = torch.zeros((num_of_images, num_of_splits) + (shape)).to(device)
    # operator_split = []
    split_list = list(range(num_of_splits))
    split_geom = [geometry[j::num_of_splits] for j in split_list]
    operator_split = [odl.tomo.RayTransform(domain, split, impl='astra_' + device) for split in split_geom]
    split_fbp = [odl.tomo.analytic.filtered_back_projection.fbp_op(operator, padding=1) for operator in operator_split]
    split_FBP_module = [OperatorModule(fbp) for fbp in split_fbp]
    
    for j in range(num_of_splits):
        sinogram_split[:,j,:,:] = noisy_sinograms[:, j::num_of_splits, :][:,:,:]
        to_FBP = noisy_sinograms[:, j::num_of_splits, :]
        rec_split[:,j,:,:] = split_FBP_module[j](to_FBP)[:,:,:]

    input_reco = torch.zeros((num_of_images, ) + shape).to(device)
    target_reco = torch.zeros((num_of_images, ) + shape).to(device)
    sinogram_reco = torch.zeros((num_of_images, ) + (int(np.shape(noisy_sinograms)[1]/num_of_splits), np.shape(noisy_sinograms)[2])).to(device)
    target_sinogram_reco = torch.zeros((num_of_images, ) + (int(np.shape(noisy_sinograms)[1]/num_of_splits), np.shape(noisy_sinograms)[2])).to(device)

    for k in range(averaged):
        input_reco[:,:,:] = input_reco[:,:,:] + rec_split[:,k,:,:]
        sinogram_reco[:,:,:] = sinogram_reco[:,:,:] + noisy_sinograms[:, k::num_of_splits, :]
    
    input_reco = input_reco / averaged
    sinogram_reco = sinogram_reco / averaged
    
    if num_of_splits - averaged != 0:
        for k in range(num_of_splits - averaged):
            target_reco[:,:,:] = target_reco[:,:,:] + rec_split[:,averaged + k,:,:]
            target_sinogram_reco[:,:,:] = target_sinogram_reco[:,:,:] + noisy_sinograms[:, (averaged + k)::num_of_splits, :]
    
        target_reco = target_reco / (num_of_splits - averaged)
        target_sinogram_reco = target_sinogram_reco / (num_of_splits - averaged)

    return input_reco, target_reco, sinogram_reco, \
        target_sinogram_reco, rec_split, sinogram_split, operator_split


# def data_split(num_of_splits, shape, averaged, test_amount, noisy_sinograms, geometry, domain, device, ray_transform):    
#     sinogram_split = torch.zeros((num_of_splits, ) + (int(noisy_sinograms.shape[0]/num_of_splits), noisy_sinograms.shape[1]))
#     rec_split = torch.zeros((num_of_splits, ) + (shape))
#     for j in range(num_of_splits):
#         split = geometry[j::num_of_splits]
#         # noisy_sinogram = noisy_sinogram[j,:,:]
#         sinogram_split[j,:,:] = noisy_sinograms[j::num_of_splits, :]
#         operator_split = odl.tomo.RayTransform(domain, split, impl='astra_' + device)
#         # split_FBP = odl.tomo.analytic.filtered_back_projection.fbp_op(operator_split, padding=1)
#         # split_FBP_module = OperatorModule(split_FBP)
#         # rec_split[j,:,:] = split_FBP_module(sinogram_split[j,:,:])
#         # print(type(split_FBP))
#         # FBP_domain = split_FBP.domain
#         # print(split_FBP.domain)
#         # split_FBP = split_FBP.asarray()
#         # sinogram_split[j,:,:] = domain.element(sinogram_split[j,:,:])
#         # print(type(sinogram_split[j,:,:]))
#         # rec_split[j,:,:] = split_FBP(sinogram_split[j,:,:])
    
#     # print('split shape',sinogram_split.shape)
#     split_FBP = odl.tomo.analytic.filtered_back_projection.fbp_op(operator_split, padding=1)
#     # print('asd')
#     fbp_operator_module = OperatorModule(split_FBP).to('cuda')
#     # print('fbp',split_FBP.range)
#     # split_reco = torch.zeros((num_of_splits, ) + (shape))
#     rec_split = fbp_operator_module(sinogram_split)
#     # print('rec split', rec_split.shape)
#     input_reco = np.zeros(shape)
#     target_reco = np.zeros(shape)
#     # eval_reco = torch.zeros((averaged, ) + shape)
#     for k in range(averaged):
#         # eval_reco[k,:,:] = rec_split[k,:,:]#.cpu().detach().numpy()
#         input_reco = input_reco + rec_split[k,:,:].cpu().detach().numpy()
        
#     input_reco = input_reco / averaged
    
#     for k in range(num_of_splits - averaged):
#         target_reco = target_reco + rec_split[averaged + k,:,:].cpu().detach().numpy()
    
#     target_reco = target_reco / (num_of_splits - averaged)
    
#     return input_reco, target_reco, sinogram_split, operator_split


class LPD_step(nn.Module):
    def __init__(self, operator, adjoint_operator, operator_norm, device='cuda'):
        super().__init__()
        
        self.operator = operator
        self.adjoint_operator = adjoint_operator
        self.operator_norm = operator_norm
        self.device = device

        ### Primal block of the network
        self.primal_step = nn.Sequential(
            nn.Conv2d(in_channels=2, out_channels=32, kernel_size=(3,3), padding=1),
            nn.PReLU(num_parameters=32, init=0),
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(3,3), padding=1),
            nn.PReLU(num_parameters=32, init=0),
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(3,3), padding=1),
            nn.PReLU(num_parameters=32, init=0),
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(3,3), padding=1),
            nn.PReLU(num_parameters=32, init=0),
            # nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(3,3), padding=1),
            # nn.PReLU(num_parameters=32, init=0),
            nn.Conv2d(in_channels=32, out_channels=1, kernel_size=(3,3), padding=1)
            )

        ### Dual block of the network
        self.dual_step = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=32, kernel_size=(3,3), padding=1),
            nn.PReLU(num_parameters=32, init=0),
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(3,3), padding=1),
            nn.PReLU(num_parameters=32, init=0),
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(3,3), padding=1),
            nn.PReLU(num_parameters=32, init=0),
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(3,3), padding=1),
            nn.PReLU(num_parameters=32, init=0),
            # nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(3,3), padding=1),
            # nn.PReLU(num_parameters=32, init=0),
            nn.Conv2d(in_channels=32, out_channels=1, kernel_size=(3,3), padding=1)
            )

        self.to(device)
        # self.final_relu = nn.ReLU()

    ### Must needed forward function
    def forward(self, f, g, g_expand, h, operator_split, domain, num_of_splits, averaged, device='cuda'):
        # print('g', g.shape)
        # start2 = time.time()
        f = f.clone()
        h = h.clone()
        f_sinograms = torch.zeros((num_of_splits-averaged, g.shape[1]) + (g.shape[2], g.shape[3])).to(device)
        adjoint_eval = torch.zeros((g.shape[1], ) + (f.shape[2], f.shape[3])).to(device)
        u = torch.zeros((averaged, 3) + (g_expand.shape[0], g_expand.shape[1])).to(device)
        
        # print('f_sinos', f_sinograms.shape)
        # print('f', f.shape)
        for j in range(f.shape[1]):
            operator = operator_split[j]
            operator = OperatorModule(operator).to(device)
            
            f_sinograms[0,j,:,:] = operator(f[:,j,:,:])
        
        # sino_reco = torch.zeros((int(g.shape[2]*g.shape[1]), g.shape[3])).to(device)
        # sino_reco = []
        # sino_reco2 = torch.zeros((int(g.shape[2]*g.shape[1]), g.shape[3])).to(device)
        # print('f sinos', f_sinograms.shape)
        # print(g.shape)
        # print('avg', averaged)
        # iter = 0
        # # start1 = time.time()
        # for k in range(g.shape[2]):
        #     for j in range(g.shape[1]):
        #         sino_reco2[iter, :] = f_sinograms[0,j,k,:]
        #         iter = iter + 1
        # end1 = time.time()
        # print(end1-start1)
        
        # start2 = time.time()
        # print('g', g.shape)
        # print('g_exp', g_expand.shape)
        # print(f_sinograms.shape[2])
        start = time.time()
        with torch.no_grad():
            if num_of_splits == 4:
                # print('g', g.shape)
                # sino_reco = torch.zeros(g_expand.shape).to(device)
                # print('f', f_sinograms.shape)
                # plt.figure()
                # plt.imshow(f_sinograms[0,0,:,:].cpu().detach().numpy())
                # plt.show()
                # sino_reco = []
                sino_reco = torch.Tensor().to(device)
                # print(sino_reco)
                # for k in range(g.shape[2]):
                if g.shape[1] < 4:
                    # sino_reco = torch.cat([f_sinograms[0,0,k,:],f_sinograms[0,1,k,:],f_sinograms[0,2,k,:]], dim=0)
                    # sino_reco = [None] * f_sinograms.shape[2] * 3
                    for k in range(g.shape[2]):
                        sino_reco = torch.cat([sino_reco,f_sinograms[0,0,k,:],f_sinograms[0,1,k,:],f_sinograms[0,2,k,:]], dim=-1)
                        # print(sino_reco[])
                    # print(sino_reco.shape)
                    sino_reco = torch.reshape(sino_reco, shape = (3*int((f_sinograms.shape[2])), int((f_sinograms[0,0,0,:].size(dim=0)))))
                    # # print(k)
                    #     sino_reco = torch.cat([sino_reco, f_sinograms[0,0,k,:]], dim=0)
                    #     print(sino_reco)
                    #     sino_reco = torch.cat([sino_reco, f_sinograms[0,1,k,:]], dim=0)
                    #     print(sino_reco)
                    #     sino_reco = torch.cat([sino_reco, f_sinograms[0,2,k,:]], dim=0)
                        # print(sino_reco)
                        # sino_reco.append(f_sinograms[0,0,k,:].cpu().detach().numpy())
                        # sino_reco.append(f_sinograms[0,1,k,:].cpu().detach().numpy())
                        # sino_reco.append(f_sinograms[0,2,k,:].cpu().detach().numpy())
                        # sino_reco[128*3*k:128*(3*k+1)-1] = (f_sinograms[0,0,k,:].cpu().detach().numpy())
                        # sino_reco[128*(3*k+1):128*(3*k+2)-1] = (f_sinograms[0,1,k,:].cpu().detach().numpy())
                        # sino_reco[128*(3*k+2):128*(3*k+3)-1] = (f_sinograms[0,2,k,:].cpu().detach().numpy())
                    # sino_reco[3*k,:] = f_sinograms[0,0,k,:]
                    # sino_reco[3*k+1,:] = f_sinograms[0,1,k,:]
                    # sino_reco[3*k+2,:] = f_sinograms[0,2,k,:]
                else:
                    # sino_reco = torch.cat([f_sinograms[0,0,k,:],f_sinograms[0,1,k,:],f_sinograms[0,2,k,:],f_sinograms[0,3,k,:]], dim=0)
                    # print('here')
                    # # sino_reco = [0] * f_sinograms.shape[2] * 4
                    for k in range(g.shape[2]):
                        sino_reco = torch.cat([sino_reco, f_sinograms[0,0,k,:],f_sinograms[0,1,k,:],f_sinograms[0,2,k,:],f_sinograms[0,3,k,:]], dim=-1)
                    # print(sino_reco.shape)
                    sino_reco = torch.reshape(sino_reco, shape = (4*int((f_sinograms.shape[2])), int((f_sinograms[0,0,0,:].size(dim=0)))))
                    # sino_reco = torch.reshape(sino_reco, shape = (int((f_sinograms[0,0,0,:].size(dim=0))), 4))
                    #     sino_reco = torch.cat([sino_reco, f_sinograms[0,0,k,:]], dim=0)
                    #     sino_reco = torch.cat([sino_reco, f_sinograms[0,1,k,:]], dim=0)
                    #     sino_reco = torch.cat([sino_reco, f_sinograms[0,2,k,:]], dim=0)
                    #     sino_reco = torch.cat([sino_reco, f_sinograms[0,3,k,:]], dim=0)
                        # sino_reco.append(f_sinograms[0,0,k,:].cpu().detach().numpy())
                        # sino_reco.append(f_sinograms[0,1,k,:].cpu().detach().numpy())
                        # sino_reco.append(f_sinograms[0,2,k,:].cpu().detach().numpy())
                        # sino_reco.append(f_sinograms[0,3,k,:].cpu().detach().numpy()) 
                    # sino_reco[4*k,:] = f_sinograms[0,0,k,:]
                    # sino_reco[4*k+1,:] = f_sinograms[0,1,k,:]
                    # sino_reco[4*k+2,:] = f_sinograms[0,2,k,:]
                    # sino_reco[4*k+3,:] = f_sinograms[0,3,k,:]
            
            if num_of_splits == 2:
                sino_reco = []
                for k in range(g.shape[2]):
                    if g.shape[1] < 2:
                        sino_reco.append(f_sinograms[0,0,k,:].cpu().detach().numpy())
                    else:
                        sino_reco.append(f_sinograms[0,0,k,:].cpu().detach().numpy())
                        sino_reco.append(f_sinograms[0,1,k,:].cpu().detach().numpy()) 
                    
            
            # print((sino_reco))
            # sino_reco = np.array(sino_reco)
            # print('fshape', f_sinograms[0,0,0,:].shape)
            # print(sino_reco)
            # sino_reco = torch.reshape(sino_reco, shape = (int((f_sinograms[0,0,0,:].size(dim=0))), 3))
            # sino_reco = torch.from_numpy(sino_reco).to(device)
            # print('sino reco', sino_reco.shape)
        end = time.time()

        # plt.figure()
        # plt.imshow(sino_reco.cpu().detach().numpy())
        # plt.show()


        # print('ELAPSED TIME', end-start)        
        # print('h', h.shape)
        # print('sino reco', sino_reco[None,None,:,:].shape)
        # # print('f_sino', f_sinograms.shape)
        # print('g expand', g_expand.shape)
        # print('g', g.shape)
        # print('u', u.shape)
        # for k in range(f_sinograms.shape[0]):
        # print(h.shape, sino_reco.shape, g_expand.shape)
        u[0,:,:,:] = torch.cat([h[None,None,:,:], sino_reco[None,None,:,:], g_expand[None,None,:,:]], dim=1)
        # print('u2', u.shape)
        h[:,:] = h[:,:] + self.dual_step(u[0,:,:,:][None,:,:,:]).to(device)
            # h[0,k,:,:] = h[0,k,:,:].clone()
        # h = h + self.dual_step(u)

        # print('h',h.shape)
        # plt.figure()
        # plt.imshow(h.cpu().detach().numpy())
        # plt.show()
        
        for j in range(g.shape[1]):
            operator = operator_split[j]
            # adjoint_operator = OperatorModule(operator.adjoint).to(device)
            fbp2 = odl.tomo.analytic.filtered_back_projection.fbp_op(operator, padding=1)
            # adjoint_operator = operator.adjoint
            fbp2 = OperatorModule(fbp2).to(device)
            # print('adjoint', adjoint_operator(h[:,k,:,:]).shape)
            # adjoint_eval[j,:,:] = fbp2(h[j::int(g.shape[1]),:][None,None,:,:])
            # adjoint_eval[j,:,:] = adjoint_operator(h[j::int(g.shape[1]),:][None,None,:,:])
            # adjoint_eval[0,:,:] = adjoint_eval[0,:,:] + fbp2(h[j::int(g.shape[1]),:][None,None,:,:])
            adjoint_eval[j,:,:] = fbp2(h[j::int(g.shape[1]),:][None,None,:,:])
        
        # adjoint_eval = adjoint_eval / ((num_of_splits)**2)
        # adjoint_eval = adjoint_eval / ((num_of_splits-averaged)**2)
        # print('asjoint_eval', adjoint_eval.shape)
        # print('f', f.shape)
        adjoint_eval = torch.mean(adjoint_eval, dim=0)
        # print('asjoint_eval', adjoint_eval.shape)
        u = torch.zeros((averaged, 2) + (f.shape[2], f.shape[3])).to(device)
        f = torch.mean(f, dim=1)
        # print('asjoint_eval', adjoint_eval.shape)
        # print('f', f.shape)
        u[0,:,:,:] = torch.cat([f, adjoint_eval[None,:,:]], dim=0)
        # print(u.shape)
        f[0,:,:] = f[0,:,:] + self.primal_step(u[0,:,:,:][None,:,:,:]).to(device)
        # plt.figure()
        # plt.imshow(adjoint_eval[:,:].cpu().detach().numpy())
        # plt.show()
        # if g.shape[1] < num_of_splits:
            
        #     u = torch.zeros((5, 2) + (f.shape[2], f.shape[3])).to(device)
        #     zero_add = torch.zeros((f.shape[2], f.shape[3])).to(device)
        #     f = torch.mean(f, dim=1)
        #     # print('u', u.shape)
        # # for k in range(adjoint_eval.shape[0]):
        #     # print(f.shape, adjoint_eval.shape)
        #     # print(f[:,None,:,:].shape, adjoint_eval[:,None,:,:].shape)
        # # print(f[:,k].shape, adjoint_eval[:,k].shape)
        #     u = torch.cat([f[None,:,:,:], adjoint_eval[None,:,:,:], zero_add[None,None,:,:]], dim=1)
        #     # print(u.shape)
        #     f[0,:,:] = f[0,:,:] + self.primal_step(u).to(device)
        
        # else:
        #     u = torch.zeros((5, 2) + (f.shape[2], f.shape[3])).to(device)
        #     f = torch.mean(f, dim=1)
        #     # print('u', u.shape)
        # # for k in range(adjoint_eval.shape[0]):
        #     # print(f.shape, adjoint_eval.shape)
        #     # print(f[:,None,:,:].shape, adjoint_eval[:,None,:,:].shape)
        # # print(f[:,k].shape, adjoint_eval[:,k].shape)
        #     u = torch.cat([f[None,:,:,:], adjoint_eval[None,:,:,:]], dim=1)
        #     # print(u.shape)
        #     f[0,:,:] = f[0,:,:] + self.primal_step(u).to(device)
        
        # f[0,k,:,:] = f[0,k,:,:].clone()
        # print('u2', u.shape)
        # f = f + self.primal_step(u)
        # print('f', f.shape)
        # f = self.final_relu(f)
        # h = self.final_relu(h)
        # f = torch.mean(f, dim=0)
        # ### Dual iterate happens here
        # f_sinogram = self.operator(f) / self.operator_norm
        # u = torch.cat([h, f_sinogram, g / self.operator_norm], dim=1)
        # h = h + self.dual_step(u)

        # ### Primal iterate happens here
        # adjoint_eval = self.adjoint_operator(h) / self.operator_norm
        # u = torch.cat([f, adjoint_eval], dim=1)
        # f = f + self.primal_step(u)
        # print('f', f.shape)
        # print('h', h.shape)
        # return torch.mean(f, dim=1), torch.mean(h, dim=1)
        # print('step')
        # end2 = time.time()
        # print(end2-start2)
        return f[None,:,:,:], h
        
class LPD(nn.Module):
    def __init__(self, operator, adjoint_operator, operator_norm, n_iter, device='cuda'):
        super().__init__()
        
        self.operator = operator
        self.adjoint_operator = adjoint_operator
        self.operator_norm = operator_norm
        self.n_iter = n_iter
        self.device = device
        self.final_relu = nn.ReLU()

        ### Initializing the parameters for every unrolled iteration step.
        for k in range(self.n_iter):
            step = LPD_step(operator, adjoint_operator, operator_norm, device=self.device)
            setattr(self, f'step{k}', step)
            
    def forward(self, f, g, operator_split, domain, num_of_splits, averaged, noise, device='cuda'):
        print('F', f.shape)
        print('G', g.shape)
        print('num', num_of_splits)
        print('avg', averaged)
        ### Initializing "h" as a zero matrix
        # g_expand = 
        g_expand = torch.Tensor().to(device)
        # print('dev',g_expand.device)
        # print('f sinos', f_sinograms.shape)
        # print('g_forw', g.shape)
        # iter = 0
        # for k in range(g.shape[2]):
        #     for j in range(g.shape[1]):
        #         g_expand[iter, :] = g[0,j,k,:]
        #         iter = iter + 1
        # print('g',g.shape)
        if num_of_splits == 4:
            if g.shape[1] < 4:
                for k in range(g.shape[2]):
                    g_expand = torch.cat([g_expand,g[0,0,k,:],g[0,1,k,:],g[0,2,k,:]], dim=-1)
                g_expand = torch.reshape(g_expand, shape = (3*int((g.shape[2])), int((g[0,0,0,:].size(dim=0)))))
            else:
                for k in range(g.shape[2]):
                    g_expand = torch.cat([g_expand,g[0,0,k,:],g[0,1,k,:],g[0,2,k,:],g[0,3,k,:]], dim=-1)
                g_expand = torch.reshape(g_expand, shape = (4*int((g.shape[2])), int((g[0,0,0,:].size(dim=0)))))
            # for k in range(g.shape[2]):
            #     if g.shape[1] < 4:

            #         g_expand.append(g[0,0,k,:].cpu().detach().numpy())
            #         g_expand.append(g[0,1,k,:].cpu().detach().numpy())
            #         g_expand.append(g[0,2,k,:].cpu().detach().numpy())
            #     else:
            #         g_expand.append(g[0,0,k,:].cpu().detach().numpy())
            #         g_expand.append(g[0,1,k,:].cpu().detach().numpy())
            #         g_expand.append(g[0,2,k,:].cpu().detach().numpy())
            #         g_expand.append(g[0,3,k,:].cpu().detach().numpy()) 
        
        if num_of_splits == 2:
            for k in range(g.shape[2]):
                if g.shape[1] < 2:
                    g_expand.append(g[0,0,k,:].cpu().detach().numpy())
                else:
                    g_expand.append(g[0,0,k,:].cpu().detach().numpy())
                    g_expand.append(g[0,1,k,:].cpu().detach().numpy()) 
                
        # g_expand = torch.as_tensor(np.array(g_expand)).to(self.device)

        # plt.figure()
        # plt.imshow(g_expand.cpu().detach().numpy())
        # plt.show()
        h = torch.zeros(g_expand.shape).to(self.device)
        # print('dev', h.device)
        # print('g', g.shape)
        # h = torch.as_tensor(noise, dtype=torch.float32)[0:256,:].to(self.device)
        # h = torch.as_tensor(np.random.normal(0, 0.065, size=(g.shape[2], g.shape[3])), dtype=torch.float32).to(self.device)
        # h = h[None,None,:,:]
        # print('h', h.shape)
        

        ### Here happens the unrolled iterations
        for k in range(self.n_iter):
            step = getattr(self, f'step{k}')
            # print('iter', k)
            f, h = step(f, g, g_expand, h, operator_split, domain, num_of_splits, averaged, device='cuda')
            
        f = self.final_relu(f)
        return f
