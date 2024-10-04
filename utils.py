import torch
import numpy as np
import odl
from odl.contrib.torch import OperatorModule

def compute_sinogram(input_tensor: torch.Tensor,
                     operator):
    
    sinogram = operator(input_tensor)

    return sinogram

def expand_sinogram(split_sinogram: torch.Tensor,
                    num_of_splits: int,
                    device='cuda') -> torch.Tensor:
    
    expanded_sinogram = torch.Tensor().to(device)
    if split_sinogram.shape[1] < num_of_splits:
        expanded_sinogram = torch.cat([expanded_sinogram, torch.cat([split_sinogram[0,0:num_of_splits-1,:,:]])], dim=-1)
        expanded_sinogram = torch.reshape(expanded_sinogram, shape=((num_of_splits-1)*split_sinogram.shape[2], split_sinogram.shape[-1]))
    else:
        expanded_sinogram = torch.cat([expanded_sinogram, split_sinogram[0,0:num_of_splits,:,:]], dim=-1)
        expanded_sinogram = torch.reshape(expanded_sinogram, shape=((num_of_splits)*split_sinogram.shape[2], split_sinogram.shape[-1]))

    return expanded_sinogram

def to_cpu(input_tensor: torch.Tensor) -> np.array:
    return input_tensor.cpu().detach().numpy()

def data_split(num_of_images, num_of_splits, shape, averaged, noisy_sinograms, split_FBP_module, domain, device, ray_transform):
    sinogram_split = torch.zeros((num_of_images, num_of_splits) + (int(noisy_sinograms.shape[1]/num_of_splits), noisy_sinograms.shape[2])).to(device)
    # sinogram_split2 = torch.zeros((num_of_images, num_of_splits) + (int(noisy_sinograms.shape[1]/num_of_splits), noisy_sinograms.shape[2])).to(device)
    rec_split = torch.zeros((num_of_images, num_of_splits) + (shape)).to(device)
    # rec_split2 = torch.zeros((num_of_images, num_of_splits) + (shape)).to(device)
    # operator_split = []
    # split_list = list(range(num_of_splits))
    # split_geom = [geometry[j::num_of_splits] for j in split_list]
    # operator_split = [odl.tomo.RayTransform(domain, split, impl='astra_' + device) for split in split_geom]
    # split_fbp = [odl.tomo.analytic.filtered_back_projection.fbp_op(operator, padding=1) for operator in operator_split]
    # split_FBP_module = [OperatorModule(fbp) for fbp in split_fbp]
    
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
        target_sinogram_reco, rec_split, sinogram_split


def geometry_plus_ray_trafo(min_domain_point: tuple[int, int],
                            max_domain_point: tuple[int, int],
                            shape: tuple[int,int],
                            num_of_angles: int,
                            num_of_lines: int,
                            angle_interval: tuple[float, float],
                            line_interval: tuple[float, float],
                            source_radius: float,
                            detector_radius: float,
                            dtype: str,
                            device='cuda'
                            ):
    
    device = 'astra_' + device

    domain = odl.uniform_discr(min_pt=min_domain_point,
                               max_pt=max_domain_point,
                               shape=shape,
                               dtype=dtype)
    
    angles = odl.uniform_partition(angle_interval[0],
                                   angle_interval[1],
                                   num_of_angles)
    
    lines = odl.uniform_partition(line_interval[0],
                                  line_interval[1],
                                  num_of_lines)
    
    geometry = odl.tomo.FanBeamGeometry(angles,
                                        lines,
                                        source_radius,
                                        detector_radius)
    
    output_shape = (num_of_angles, num_of_lines)

    ray_transform = odl.tomo.RayTransform(domain, geometry, impl=device)
    
    return domain, geometry, ray_transform, output_shape

