import astra
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from UNetModule import UNet
import cv2 as cv
import random
from torch.utils.tensorboard import SummaryWriter
from utils import to_cpu

astra.set_gpu_index(2)
device = 'cuda:2'

writer = SummaryWriter()
num_of_images = 10
test_list = []
sample = random.sample(range(1000,1100), num_of_images)
for k in sample:
    test_list.append('0' + str(k))

print(test_list)

num_of_splits = 4
averaged = 3

name = 'CWI_data_test_0412.pth'



# reco_path = '/scratch2/antti/dataset_lion/all_rec/'+'slice'+ slice_num +'/mode1/reconstruction.tif'
# sino_path = '/scratch2/antti/dataset_lion/all_sinograms/'+'slice'+ slice_num +'/mode1/sinogram.tif'
# dark_path = '/scratch2/antti/dataset_lion/all_sinograms/'+'slice'+ slice_num +'/mode1/dark.tif'
# flat1_path = '/scratch2/antti/dataset_lion/all_sinograms/'+'slice'+ slice_num +'/mode1/flat1.tif'
# flat2_path = '/scratch2/antti/dataset_lion/all_sinograms/'+'slice'+ slice_num +'/mode1/flat2.tif'

test_recos = []
test_sinograms = []
test_darks = []
test_flats = []

for slice_num in test_list:
        reco_path = '/scratch2/antti/dataset_lion/all_rec/'+'slice'+ slice_num +'/mode2/reconstruction.tif'
        sino_path = '/scratch2/antti/dataset_lion/all_sinograms/'+'slice'+ slice_num +'/mode1/sinogram.tif'
        dark_path = '/scratch2/antti/dataset_lion/all_sinograms/'+'slice'+ slice_num +'/mode1/dark.tif'
        flat1_path = '/scratch2/antti/dataset_lion/all_sinograms/'+'slice'+ slice_num +'/mode1/flat1.tif'
        flat2_path = '/scratch2/antti/dataset_lion/all_sinograms/'+'slice'+ slice_num +'/mode1/flat2.tif'
        test_recos.append(cv.imread(reco_path, cv.IMREAD_UNCHANGED)[0::2, 0::2]) #[0::2, 0::2]
        test_sinograms.append(cv.imread(sino_path, cv.IMREAD_UNCHANGED))
        test_darks.append(cv.imread(dark_path, cv.IMREAD_UNCHANGED))
        flat1 = cv.imread(flat1_path, cv.IMREAD_UNCHANGED)
        flat2 = cv.imread(flat2_path, cv.IMREAD_UNCHANGED)
        test_flats.append(np.mean(np.array([flat1, flat2]), axis=0))

test_sinograms = np.ascontiguousarray(test_sinograms)

test_recos = torch.as_tensor(test_recos).to(device)

n_lines = 478
num_lines = 1800

# # rec = cv.imread(rec_path, cv.IMREAD_UNCHANGED)
# reco = cv.imread(reco_path, cv.IMREAD_UNCHANGED)[0::2, 0::2]
# sinogram = cv.imread(sino_path, cv.IMREAD_UNCHANGED)
# dark = cv.imread(dark_path, cv.IMREAD_UNCHANGED)
# flat1 = cv.imread(flat1_path, cv.IMREAD_UNCHANGED)
# flat2 = cv.imread(flat2_path, cv.IMREAD_UNCHANGED)
# flat = np.mean(np.array([flat1, flat2]), axis=0)
# sinogram =  np.ascontiguousarray(sinogram)

# Change data type of the giles from uint16 to float32
test_darks = np.asarray(test_darks)
test_flats = np.asarray(test_flats)
test_sinograms = test_sinograms.astype('float32')
test_darks = test_darks.astype('float32')
test_flats = test_flats.astype('float32')

# Down-sample the sinogram as well as the dark and flat field
# for i in np.arange(sino_dims[0]):
test_sinograms = (test_sinograms[:,0::2,0::4]+test_sinograms[:,0::2,1::4])
test_darks = (test_darks[:,0::2,0::4]+test_darks[:,0::2,1::4])
test_flats = (test_flats[:,0::2,0::4]+test_flats[:,0::2,1::4])
# dark = (dark[0::2,0::4]+dark[0::2,1::4])
# flat = (flat[0::2,0::4]+flat[0::2,1::4])

datas = test_sinograms - test_darks
datas = datas/(test_flats-test_darks)
print(datas.shape)
# Exclude last projection if desired.
# if excludeLastPro:
    # data = data[0:-1,:]
detPix = 0.0748
corr = np.array([1.00, 0.0])
# Create detector shift via linear grid interpolation.
if slice_num in range(1,2830+1) or slice_num in range(5521,5870+1):
    detShift = corr[0] * detPix
else:
    detShift = corr[1] * detPix

detGrid = np.arange(0,n_lines) * detPix
detGridShifted = detGrid + detShift
detShiftCorr = interp1d(detGrid, datas, kind='linear', bounds_error=False, fill_value='extrapolate')
datas = detShiftCorr(detGridShifted)

# Clip the data on the lower end to 1e-6 to avoid division by zero in next step.
datas = datas.clip(1e-6, None)
# print("Values have been clipped from [", np.min(data), ",", np.max(data),"] to [1e-6,None]")

# Take negative log.
datas = np.log(datas)
datas = np.negative(datas)
datas = np.ascontiguousarray(datas)

# Create array that stores the used projection angles.
angles = np.linspace(0,2*np.pi, num_lines) # 3601 = full width of sinograms.

# Apply exclusion of last projection if desired.
# if excludeLastPro:
#     angles = angles[0:-1]
#     print('Excluded last projection.')
# binning = 1 # Manual selection of detector pixel binning after acqusisition.
# excludeLastPro = True # Exclude last projection angle which is often the same as the first one.
angSubSamp = 1 # Define a sub-sampling factor in angular direction.
# (all reference reconstructions are computed with full angular resolution).
maxAng = 360
# Apply angular subsampling.
datas = datas[0::angSubSamp,:]
angles = angles[0::angSubSamp]
angInd = np.where(angles<=(maxAng/180*np.pi))
angles = angles[angInd]
datas = datas[:,:(angInd[-1][-1]+1),:]
# datas = np.asarray(datas)
# datas = torch.as_tensor(datas, dtype=torch.float)
print(np.shape(datas))

detSubSamp = 2
binning = 1
detPixSz = detSubSamp * binning * detPix
SOD = 431.019989 
SDD = 529.000488

# Scale factor calculation.
# ASTRA assumes that the voxel size is 1mm.
# For this to be true we need to calculate a scale factor for the geometry.
# This can be done by first calculating the 'true voxel size' via the intersect theorem
# and then scaling the geometry accordingly.

# Down-sampled width of the detector.
nPix = n_lines
det_width = detPixSz * nPix

# Physical width of the field of view in the measurement plane via intersect theorem.
FOV_width = det_width * SOD/SDD
print('Physical width of FOV (in mm):', FOV_width)

# True voxel size with a given number of voxels to be used.
nVox = 512
voxSz = FOV_width / nVox
print('True voxel size (in mm) for', nVox, 'voxels to be used:', voxSz)

# Scaling the geometry accordingly.
scaleFactor = 1./voxSz
print('Self-calculated scale factor:', scaleFactor)
SDD = SDD * scaleFactor
SOD = SOD * scaleFactor
detPixSz = detPixSz * scaleFactor
recSz = (512, 512) #(2048,2048)

volGeo = astra.create_vol_geom(recSz[0], recSz[1])
projGeo = astra.create_proj_geom('fanflat', detPixSz, nPix, np.linspace(0, 2*np.pi, num_lines), SOD, SDD - SOD)
proj_id = astra.create_projector('cuda', projGeo, volGeo)
# print(projGeo['ProjectionAngles'])
proj_names = {'proj1', 'proj2', 'proj3','proj4'}
angles = list(np.linspace(0, 2*np.pi, num_lines))
print(np.shape(angles))
num_of_splits = 4
split_list = list(range(4))
split_geom = [angles[j::num_of_splits] for j in split_list]
# print(split_geom)
print(np.shape(split_geom))
# for k in range(num_of_splits):
#     print(astra.create_proj_geom('fanflat', det_pix_size, 1912, split_geom[k], SOD, SDD - SOD))
# test_ang= list(np.linspace(0,16,16))
# print(test_ang[0::4])
# test_split = [test_ang[j:-1:num_of_splits] for j in split_list]
# print(test_split)


split_projGeo = [astra.create_proj_geom('fanflat', detPixSz, nPix, split, SOD, SDD - SOD) for split in split_geom]
# split_geom = [projGeo['ProjectionAngles'][j::num_of_splits] for j in split_list]
# print(split_geom)
# print(split_projGeo)
operator_split = [astra.create_projector('cuda', split_Proj, volGeo) for split_Proj in split_projGeo]
operator_list = [operator for operator in operator_split]
split_fbp = [astra.astra_dict('FBP_CUDA') for k, operator in enumerate(operator_split)]
# fbp_list = [OperatorModule(fbp) for fbp in split_fbp]
# print([projGeo['ProjectionAngles'][j::num_of_splits] for j in split_list])
# sino_id, sinogram = astra.create_sino(rec, proj_id)
# sino_id = astra.data2d.create('-sino', projGeo, data)

# print(type(split_projGeo))
# print(split_projGeo)
# print(type(sino))
# print('data', datas.shape)
# sinogram_list = [[]]
# # print(data[1::4].shape)
# for k in range(num_of_images):
#     for j in range(num_of_splits):
#         sinogram_list[k] += [astra.data2d.create('-sino', split_projGeo[j], datas[k, j::num_of_splits, :])]
test_sinogram_split = np.zeros((num_of_images, num_of_splits) + (int(datas.shape[1]/num_of_splits), datas.shape[2]))
test_sinogram_split = []
for k in range(num_of_images):
    for j in range(num_of_splits):
        test_sinogram_split += [astra.data2d.create('-sino', split_projGeo[j], datas[k, j::num_of_splits, :])]
# print(torch.as_tensor(sinogram_list).shape)
# print(sinogram_list[0])
# print(split_projGeo[0])
# idx = 0
# for count, sino_id in enumerate(test_sinogram_split):
#     print(idx, count%4, sino_id)
#     if count%num_of_splits == 3:
#         idx += 1
#         # print(idx)
    

rec_id = astra.data2d.create('-vol', volGeo)
print('test', type(test_sinogram_split[0]))
recos = torch.zeros((num_of_images, num_of_splits) + (recSz))
# for j in range(num_of_images):
idx = 0
for count, sino_id in enumerate(test_sinogram_split):
    count = count%num_of_splits
    cfg = split_fbp[count]
    cfg['ProjectorId'] = astra.create_projector('cuda', split_projGeo[count], volGeo)
    cfg['ReconstructionDataId'] = rec_id
    cfg['ProjectionDataId'] = sino_id #test_sinogram_split[j*num_of_splits+k]
    cfg['option'] = { 'FilterType': 'Ram-Lak' }
    
    alg_id = astra.algorithm.create(cfg)
    astra.algorithm.run(alg_id)
    recos[idx,count,:,:] = torch.as_tensor(np.maximum([astra.data2d.get(rec_id)], 0))
    astra.algorithm.delete(alg_id)
    astra.data2d.delete(sino_id)
    
    if count%num_of_splits == (num_of_splits-1):
        idx += 1

test_rec_images = recos.to(device)
print('RECOS', recos.shape)
# astra.algorithm.delete(alg_id)
astra.data2d.delete(rec_id)
# astra.data2d.delete(test_sinogram_split)
astra.projector.delete(proj_id)
# print('data', datas.shape)
# test_rec_images = torch.as_tensor(reco_list, dtype=torch.float)[:,None,:,:].to(device)
# print(len(split_projGeo[0]['ProjectionAngles']))
# print('rec', test_rec_images.shape)
# volGeo = astra.create_vol_geom(recSz[0], recSz[1])
# projGeo = astra.create_proj_geom('fanflat', detPixSz, nPix, np.linspace(0, 2*np.pi, num_lines), SOD, SDD - SOD)
# recID = astra.data2d.create('-vol', volGeo)
# # sinoID = astra.data2d.create('-sino', projGeo, datas)
# # projID   = astra.create_projector('cuda', projGeo, volGeo)
# # A = astra.OpTomo(projID)



# # plt.show()
# test_recos = torch.as_tensor(test_recos).float().to(device)
# test_recos = test_recos[None,:,:]
# rec_images = np.zeros(shape=(datas.shape[0], test_recos.shape[2], test_recos.shape[3]))
# print(rec_images.shape)

# plt.figure()
# plt.imshow(to_cpu(test_rec_images[0,0,:,:]))
# plt.show()

# sino_ids = []
# rec_ids = []
# for i in range(datas.shape[0]):
#     sino_ids.append(astra.data2d.create('-sino', projGeo, datas[i]))
#     rec_ids.append(astra.data2d.create('-vol', volGeo))


# for k in range(datas.shape[0]):
    
#     # sino_id = astra.data2d.create('-sino', projGeo, datas[k])
#     # rec_id = astra.data2d.create('-vol', volGeo)
#     cfg = astra.astra_dict('FBP_CUDA')
#     cfg['ReconstructionDataId'] = rec_ids[k]
#     cfg['ProjectionDataId'] = sino_ids[k]
#     cfg['option'] = { 'FilterType': 'Ram-Lak' }
    
#     alg_id = astra.algorithm.create(cfg)
#     astra.algorithm.run(alg_id)

#     rec_images[k] = astra.data2d.get(rec_ids[k]) #[0::2, 0::2]
#     rec_images[k] = np.maximum(rec_images[k], 0)
#     astra.algorithm.delete(alg_id)
#     # astra.data2d.delete(rec_id)
#     # astra.data2d.delete(sino_id)
#     # plt.figure()
#     # plt.imshow(rec_images[k])
#     # plt.show()
#     # astra.projector.delete(proj_id)
#     # rec_images[k] = np.maximum(rec_images[k], 0)
# # sinograms = ray_transform_module(images)
# # noisy_sinograms = torch.zeros((sinograms.shape[0], ) + output_shape)
# # rec_images = torch.zeros((sinograms.shape[0], ) + shape)

# ### Adding Gaussian noise to the sinograms
# # for k in range(np.shape(sinograms)[0]):
# #     sinogram_k = sinograms[k,:,:].cpu().detach().numpy()
# #     noise = np.random.normal(mean, sinogram_k.max(), sinogram_k.shape) * percentage
# #     noisy_sinogram = sinogram_k + noise
# #     noisy_sinograms[k,:,:] = torch.as_tensor(noisy_sinogram)

# # # scheduler.step()
# # rec_images = fbp_operator_module(noisy_sinograms)

# ### All the data into same device
# datas = torch.as_tensor(datas[None,None,:,:], dtype=torch.float).to(device)
# # noisy_sinograms = noisy_sinograms[:,None,:,:].to(device)
# test_rec_images = torch.as_tensor(rec_images[:,None,:,:], dtype=torch.float).to(device)
# # reco = torch.as_tensor(reco[None,:,:], dtype=torch.float).to(device)
# # test_rec_images[0,:,:,:] = test_rec_images[0,:,:,:]*(1/torch.max(test_rec_images[0,:,:,:]))
# plt.figure()
# plt.imshow(to_cpu(test_rec_images[0,0,:,:]))
# plt.show()
# unet = UNet(in_channels=1,
#             out_channels=1, 
#             first_channel=64, 
#             depth=4, 
#             conv_kernel_size=(3,3),
#             max_pool_kernel_size=(2,2), 
#             up_conv_kernel_size=(2,2), 
#             padding=1, 
#             skip_connection_list=[]).to(device)

unet = UNet(in_channels=1,
            out_channels=1,
            first_channel=64,
            depth=4,
            conv_kernel_size=(3,3),
            max_pool_kernel_size=(2,2),
            up_conv_kernel_size=(2,2),
            padding=1, 
            skip_connection_list=[]).to(device)

print(unet)

### Getting model parameters
unet_parameters = list(unet.parameters())

### Defining PSNR function
def psnr(max_val, loss):
    
    psnr = 10 * np.log10((max_val**2) / (loss+1e-10))
    
    return psnr

loss_train = nn.MSELoss()
loss_test = nn.MSELoss()

### Setting up some lists used later
running_loss = []
running_test_loss = []
tensorboard_loss = []

### Defining training scheme
def train_network(net, n_train=50000, batch_size=4):

    ### Defining optimizer, ADAM is used here
    optimizer = optim.Adam(unet_parameters, lr=0.001) #betas = (0.9, 0.99)
    
    ### Definign scheduler, can be used if wanted
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, n_train)

    ### Here starts the itarting in training
    for i in range(n_train):
        # print(i)
        # images = get_images('/scratch2/antti/summer2023/usable_walnuts', amount_of_images=1, scale_number=2)
        slice_num = '0' + str(np.random.randint(1101,5000))
        # slice_num = '0' + str(np.random.randint(1000,5000))
        # reco_path = '/scratch2/antti/dataset_lion/all_rec/'+'slice'+ slice_num +'/mode1/reconstruction.tif'
        sino_path = '/scratch2/antti/dataset_lion/all_sinograms/'+'slice'+ slice_num +'/mode1/sinogram.tif'
        dark_path = '/scratch2/antti/dataset_lion/all_sinograms/'+'slice'+ slice_num +'/mode1/dark.tif'
        flat1_path = '/scratch2/antti/dataset_lion/all_sinograms/'+'slice'+ slice_num +'/mode1/flat1.tif'
        flat2_path = '/scratch2/antti/dataset_lion/all_sinograms/'+'slice'+ slice_num +'/mode1/flat2.tif'

        # rec = cv.imread(rec_path, cv.IMREAD_UNCHANGED)
        # reco = cv.imread(reco_path, cv.IMREAD_UNCHANGED)[0::2, 0::2]
        sinogram = cv.imread(sino_path, cv.IMREAD_UNCHANGED)
        dark = cv.imread(dark_path, cv.IMREAD_UNCHANGED)
        flat1 = cv.imread(flat1_path, cv.IMREAD_UNCHANGED)
        flat2 = cv.imread(flat2_path, cv.IMREAD_UNCHANGED)
        flat = np.mean(np.array([flat1, flat2]), axis=0)
        sinogram =  np.ascontiguousarray(sinogram)

        # Change data type of the giles from uint16 to float32
        sinogram = sinogram.astype('float32')
        dark = dark.astype('float32')
        flat = flat.astype('float32')

        # Down-sample the sinogram as well as the dark and flat field
        # for i in np.arange(sino_dims[0]):
        sinogram = (sinogram[0::2,0::4]+sinogram[0::2,1::4])
        dark = (dark[0::2,0::4]+dark[0::2,1::4])
        flat = (flat[0::2,0::4]+flat[0::2,1::4])
        data = sinogram - dark
        data = data/(flat-dark)

        # Exclude last projection if desired.
        # if excludeLastPro:
            # data = data[0:-1,:]
        detPix = 0.0748
        corr = np.array([1.00, 0.0])
        # Create detector shift via linear grid interpolation.
        if slice_num in range(1,2830+1) or slice_num in range(5521,5870+1):
            detShift = corr[0] * detPix
        else:
            detShift = corr[1] * detPix

        detGrid = np.arange(0,n_lines) * detPix
        detGridShifted = detGrid + detShift
        detShiftCorr = interp1d(detGrid, data, kind='linear', bounds_error=False, fill_value='extrapolate')
        data = detShiftCorr(detGridShifted)

        # Clip the data on the lower end to 1e-6 to avoid division by zero in next step.
        data = data.clip(1e-6, None)
        # print("Values have been clipped from [", np.min(data), ",", np.max(data),"] to [1e-6,None]")

        # Take negative log.
        data = np.log(data)
        data = np.negative(data)
        data = np.ascontiguousarray(data)

        # Create array that stores the used projection angles.
        angles = np.linspace(0,2*np.pi, num_lines) # 3601 = full width of sinograms.

        # Apply exclusion of last projection if desired.
        # if excludeLastPro:
        #     angles = angles[0:-1]
        #     print('Excluded last projection.')
        # binning = 1 # Manual selection of detector pixel binning after acqusisition.
        # excludeLastPro = True # Exclude last projection angle which is often the same as the first one.
        angSubSamp = 1 # Define a sub-sampling factor in angular direction.
        # (all reference reconstructions are computed with full angular resolution).
        # maxAng = 360
        # Apply angular subsampling.
        data = data[0::angSubSamp,:]
        # angles = angles[0::angSubSamp]
        # angInd = np.where(angles<=(maxAng/180*np.pi))
        # angles = angles[angInd]
        data = data[:(angInd[-1][-1]+1),:]
        # plt.figure()
        # plt.imshow(data[:,:])
        # plt.show()

        # print('data', data.shape)
        sinogram_split = np.zeros((num_of_images, num_of_splits) + (int(data.shape[0]/num_of_splits), data.shape[1]))
        sinogram_split = []
        for j in range(num_of_splits):
            sinogram_split += [astra.data2d.create('-sino', split_projGeo[j], data[j::num_of_splits, :])]
        # print(torch.as_tensor(sinogram_list).shape)
        # print(sinogram_list[0])
        # print('sino1', sinogram_split)
        # print(split_projGeo[0])
        rec_id = astra.data2d.create('-vol', volGeo)
        # print('test', type(test_sinogram_split[0]))
        reco_list = []
        # print('reco1', rec_id)
        
        recos = torch.zeros((1, num_of_splits) + (recSz))
        # for j in range(num_of_images):
        for count, sino_id in enumerate(sinogram_split):
            count = count%4
            cfg = split_fbp[count]
            cfg['ProjectorId'] = astra.create_projector('cuda', split_projGeo[count], volGeo)
            cfg['ReconstructionDataId'] = rec_id
            cfg['ProjectionDataId'] = sino_id #test_sinogram_split[j*num_of_splits+k]
            cfg['option'] = { 'FilterType': 'Ram-Lak' }

            alg_id = astra.algorithm.create(cfg)
            astra.algorithm.run(alg_id)
            recos[0,count,:,:] = torch.as_tensor(np.maximum([astra.data2d.get(rec_id)], 0))
            astra.algorithm.delete(alg_id)
            astra.data2d.delete(sino_id)
        
        # for k in range(num_of_splits):
        #     cfg = split_fbp[k]
        #     cfg['ProjectorId'] = astra.create_projector('cuda', split_projGeo[k], volGeo)
        #     cfg['ReconstructionDataId'] = rec_id
        #     cfg['ProjectionDataId'] = sinogram_split[k]
        #     cfg['option'] = { 'FilterType': 'Ram-Lak' }
        
        #     alg_id = astra.algorithm.create(cfg)
        #     astra.algorithm.run(alg_id)
        #     reco_list += [astra.data2d.get(rec_id)]

        reco_list = np.maximum(reco_list, 0)
        # print('data', datas.shape)
        rec_images = torch.as_tensor(recos, dtype=torch.float).to(device)
        # print('HERE', np.shape(data))
        # plt.show()
        # reco = torch.from_numpy(reco).float().to(device)
        # reco = reco[None,:,:]
        # sino_id = astra.data2d.create('-sino', projGeo, data)
        # rec_id = astra.data2d.create('-vol', volGeo)
        # cfg = astra.astra_dict('FBP_CUDA')
        # cfg['ReconstructionDataId'] = rec_id
        # cfg['ProjectionDataId'] = sino_id
        # cfg['option'] = { 'FilterType': 'Ram-Lak' }
        
        # alg_id = astra.algorithm.create(cfg)
        # astra.algorithm.run(alg_id)
        
        # rec_images = astra.data2d.get(rec_id)#[0::2, 0::2]
        # rec_images = np.maximum(rec_images, 0)
        # sinograms = ray_transform_module(images)
        # noisy_sinograms = torch.zeros((sinograms.shape[0], ) + output_shape)
        # rec_images = torch.zeros((sinograms.shape[0], ) + shape)
        
        ### Adding Gaussian noise to the sinograms
        # for k in range(np.shape(sinograms)[0]):
        #     sinogram_k = sinograms[k,:,:].cpu().detach().numpy()
        #     noise = np.random.normal(mean, sinogram_k.max(), sinogram_k.shape) * percentage
        #     noisy_sinogram = sinogram_k + noise
        #     noisy_sinograms[k,:,:] = torch.as_tensor(noisy_sinogram)
        
        # # scheduler.step()
        # rec_images = fbp_operator_module(noisy_sinograms)
        
        ### All the data into same device
        data = torch.as_tensor(data[None,None,:,:], dtype=torch.float).to(device)
        # noisy_sinograms = noisy_sinograms[:,None,:,:].to(device)
        # rec_images = torch.as_tensor(rec_images[None,None,:,:], dtype=torch.float).to(device)
        # reco = torch.as_tensor(reco[None,:,:], dtype=torch.float).to(device)
        
        ### Taking batch size amount of data pieces from the random 
        ### permutation of all training data
        n_index = 0#np.random.permutation(sinograms.shape[0])[:batch_size]
        # print('AAAAA', rec_images.shape)
        g_batch = data#[n_index,:,:,:]
        f_batch = rec_images[0,0,:,:]
        f_batch2 = torch.sum(rec_images[0,1::,:,:], dim=0)/averaged #*(1/torch.max(rec_images))#[n_index]
        # print('here', f_batch2.shape, f_batch.shape)
        # print(f_batch.shape)
        # print(f_batch2.shape)
        
        # plt.figure()
        # plt.subplot(1,2,1)
        # plt.imshow(f_batch2[0,0,:,:].cpu().detach().numpy())
        # plt.subplot(1,2,2)
        # plt.imshow(f_batch[0,0,:,:].cpu().detach().numpy())
        # plt.show()
        
        net.train()
        # print('f_batch2', f_batch2.shape)
        ### Evaluating the network which produces reconstructed images.
        outs = net(f_batch2[None,None,:,:])#[0,0,0::2,0::2][None,None,:,:]
        # outs = outs*(1/torch.max(outs))
        # print('outs final', torch.max(outs), torch.min(outs))
        # print('f_batch', torch.max(f_batch), torch.min(f_batch))
        # print('OUTS', outs.shape)
        # print(f_batch.shape)
        
        # if i % 10 == 0:
        #     print('outs', torch.max(outs), torch.min(outs))
        #     plt.figure()
        #     plt.subplot(1,2,1)
        #     plt.imshow(outs[0,0,:,:].cpu().detach().numpy())
        #     plt.subplot(1,2,2)
        #     plt.imshow(f_batch[0,0,:,:].cpu().detach().numpy())
        #     plt.show()
            
        optimizer.zero_grad()
        
        ### Calculating loss of the outputs
        loss = loss_train(f_batch[None,None,:,:].to(device), outs)
        writer.add_scalar('Train loss', loss, i)
        tensorboard_loss.append(loss.item()**0.5)
        
        ### Calculating gradient
        loss.backward()
        
        ### Here gradient clipping could be used
        # torch.nn.utils.clip_grad_norm_(unet_parameters, max_norm=1.0, norm_type=2)
        
        ### Taking optimizer step
        optimizer.step()
        scheduler.step()
        
        astra.algorithm.delete(alg_id)
        astra.data2d.delete(rec_id)
        astra.data2d.delete(sinogram_split)
        # astra.data2d.delete(phantom_id)
        astra.projector.delete(proj_id)
        
        # print('sino2', sinogram_split)
        # print('reco2', rec_id)
        # del images, sinograms, noisy_sinograms, rec_images, g_batch, f_batch2, outs
        # gc.collect()
        ### Here starts the running tests
        if i % 100 == 0:
            
            
            ### Using predetermined test data to see how the outputs are
            ### in our neural network
            net.eval()
            with torch.no_grad():
                outs4 = torch.zeros((num_of_images, ) + (test_recos.shape[1], test_recos.shape[2])).to(device)
                # print('outs4', outs4.shape)
                # print('test', all_arrangements[[0],:,:,:].shape)
                # for k in range(test_amount):
                #     for j in range(num_of_splits):
                        
                #         # print('test', all_arrangements[[k],[j],:,:].shape)
                #         # outs2[k,j,:,:] = outs2[k,j,:,:] + net(all_arrangements[[k],[j],None,:,:].to(device))
                #         # outs3[k,:,:] = outs3[k,:,:] + outs2[k,j,:,:]
                #         outs3[k,:,:] = outs3[k,:,:] + net(all_arrangements[[k],[j],None,:,:].to(device))
                        
                #     outs4[k,:,:] = torch.mean(outs3[k,:,:], dim=0)
              
                # outs2 = outs2 / test_amount
                # print('test', test_rec_images.shape)
                for k in range(num_of_images):
                    outs3 = net(torch.swapaxes(test_rec_images[[k],:,:,:], 0,1).to(device))
                    # print(outs3.shape)
                    outs4[k,:,:] = torch.mean(outs3, dim=0)
                
                # print(test_rec_images.shape)
                # outs2 = net(test_rec_images)
                # outs2 = outs2*(1/torch.max(outs2))
                # print('outs4', outs4.shape)
                # print('test rec', test_recos.shape)
                # print(torch.swapaxes(test_recos, axis0=0, axis1=1).shape)
                ### Calculating test loss with test data outputs
                test_loss = loss_test(test_recos, outs4).item()
            train_loss = loss.item()
            running_loss.append(train_loss)
            running_test_loss.append(test_loss)
            # print('outs max min', torch.max(outs4[0,:,:]), torch.min(outs4[0,:,:]))
            # print('truth max min', torch.max(test_recos), torch.min(test_recos))
            writer.add_scalar('test_loss loss', test_loss, i)
            
            ### Printing some data out
            if i % 500 == 0:
                value = (torch.max(test_recos[0,:,:]) - torch.min(test_recos[0,:,:])).item()
                # print('value', value)
                print(f'Iter {i}/{n_train} Train Loss: {train_loss:.2e}, Test Loss: {test_loss:.2e}, PSNR: {psnr(value, test_loss):.2f}') #, end='\r'
                # print(f'Step lenght: {step_len[0]}')
                plt.figure()
                plt.subplot(1,2,1)
                plt.imshow(outs4[0,:,:].cpu().detach().numpy())
                plt.subplot(1,2,2)
                plt.imshow(test_recos[0,:,:].cpu().detach().numpy())
                plt.show()
                
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
running_loss, running_test_loss, net = train_network(unet, n_train=50001, \
                                                         batch_size=1)
writer.flush()
writer.close()

torch.save(net.state_dict(), '/scratch2/antti/Noise2Inverse-Learned-Primal-Dual/final_networks/' + name)