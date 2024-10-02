import torch
import torch.nn as nn
from UNetModule import UNet
import odl
from odl.contrib.torch import OperatorModule
import time
import matplotlib.pyplot as plt
from utils import expand_sinogram

class PrimalUNet(nn.Module):
    '''
    ResNet architecture for the primal step in Learned Primal-Dual algorithm.
    '''
    def __init__(
        self,
        in_channels:int,
        out_channels:int,
        first_channel:int,
        depth:int,
        conv_kernel_size:tuple,
        max_pool_kernel_size:tuple,
        up_conv_kernel_size:tuple,
        padding:int,
        skip_connection_list:list) -> None:
        super().__init__()
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.first_channel = first_channel
        self.depth = depth
        self.conv_kernel_size = conv_kernel_size
        self.max_pool_kernel_size = max_pool_kernel_size
        self.up_conv_kernel_size = up_conv_kernel_size
        self.padding = padding
        self.skip_connection_list = skip_connection_list
        
        # self.primal_resnet = nn.ModuleList()
        
        self.primal_unet = UNet(in_channels=self.in_channels,
                                out_channels=self.out_channels,
                                first_channel=self.first_channel,
                                depth=self.depth,
                                conv_kernel_size=self.conv_kernel_size,
                                max_pool_kernel_size=self.max_pool_kernel_size,
                                up_conv_kernel_size=self.up_conv_kernel_size,
                                padding=self.padding,
                                skip_connection_list=self.skip_connection_list)
                        
    def forward(self,
                input_tensor:torch.Tensor) -> torch.Tensor:
        output_tensor = self.primal_unet(input_tensor)
        return output_tensor
    
class DualUNet(nn.Module):
    '''
    ResNet architecture for the dual step in Learned Primal-Dual algorithm.
    '''
    def __init__(
        self,
        in_channels:int,
        out_channels:int,
        first_channel:int,
        depth:int,
        conv_kernel_size:tuple,
        max_pool_kernel_size:tuple,
        up_conv_kernel_size:tuple,
        padding:int,
        skip_connection_list:list) -> None:
        super().__init__()
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.first_channel = first_channel
        self.depth = depth
        self.conv_kernel_size = conv_kernel_size
        self.max_pool_kernel_size = max_pool_kernel_size
        self.up_conv_kernel_size = up_conv_kernel_size
        self.padding = padding
        self.skip_connection_list = skip_connection_list
        
        # self.primal_resnet = nn.ModuleList()
        
        self.dual_unet = UNet(in_channels=self.in_channels,
                                out_channels=self.out_channels,
                                first_channel=self.first_channel,
                                depth=self.depth,
                                conv_kernel_size=self.conv_kernel_size,
                                max_pool_kernel_size=self.max_pool_kernel_size,
                                up_conv_kernel_size=self.up_conv_kernel_size,
                                padding=self.padding,
                                skip_connection_list=self.skip_connection_list)

    def forward(self, input_tensor:torch.Tensor) -> torch.Tensor:
        output_tensor = self.dual_unet(input_tensor)
        return output_tensor
    
class LearnedPrimalDualStep(nn.Module):
    '''
    One unrolled iteration (step) of the Learned Primal-Dual algorithm.
    '''
    def __init__(
        self, 
        primal_in_channels:int,
        dual_in_channels:int,
        out_channels:int,
        first_channel:int,
        depth:int,
        conv_kernel_size:tuple,
        max_pool_kernel_size:tuple,
        up_conv_kernel_size:tuple,
        padding:int,
        skip_connection_list:list,
        operator,
        adjoint_operator,
        operator_norm:float) -> None:
        super().__init__()

        self.alpha_1 = torch.nn.Parameter(torch.zeros([1,1,1,1]), requires_grad=True)
        self.alpha_2 = torch.nn.Parameter(torch.zeros([1,1,1,1]), requires_grad=True)
        
        self.primal_in_channels = primal_in_channels
        self.dual_in_channels = dual_in_channels
        self.out_channels = out_channels
        self.first_channel = first_channel
        self.depth = depth
        self.conv_kernel_size = conv_kernel_size
        self.max_pool_kernel_size = max_pool_kernel_size
        self.up_conv_kernel_size = up_conv_kernel_size
        self.padding = padding
        self.skip_connection_list = skip_connection_list
        self.operator = operator
        self.adjoint_operator = adjoint_operator
        self.operator_norm = operator_norm
        
        self.primal_unet = PrimalUNet(in_channels=self.primal_in_channels,
                                out_channels=self.out_channels,
                                first_channel=self.first_channel,
                                depth=self.depth,
                                conv_kernel_size=self.conv_kernel_size,
                                max_pool_kernel_size=self.max_pool_kernel_size,
                                up_conv_kernel_size=self.up_conv_kernel_size,
                                padding=self.padding,
                                skip_connection_list=self.skip_connection_list)
        
        self.dual_unet = DualUNet(in_channels=self.dual_in_channels,
                                out_channels=self.out_channels,
                                first_channel=self.first_channel,
                                depth=self.depth,
                                conv_kernel_size=self.conv_kernel_size,
                                max_pool_kernel_size=self.max_pool_kernel_size,
                                up_conv_kernel_size=self.up_conv_kernel_size,
                                padding=self.padding,
                                skip_connection_list=self.skip_connection_list)
        
        # self.to('cuda')
        
    def forward(self, f, g, g_expand, h, operator_split, fbp_list, domain, num_of_splits, averaged, device='cuda'):
        f_orig = f.clone()
        # print('exp', g_expand.shape)
        f = f.clone()
        h = h.clone()
        f_sinograms = torch.zeros((num_of_splits-averaged, g.shape[1]) + (g.shape[2], g.shape[3])).to(device)
        adjoint_eval = torch.zeros((g.shape[1], ) + (f.shape[2], f.shape[3])).to(device)
        u = torch.zeros((1, 3) + (g_expand.shape[0], g_expand.shape[1])).to(device)
        start = time.time()
        for j in range(f.shape[1]):
            f_sinograms[0,j,:,:] = operator_split[j](f[:,j,:,:])
        end = time.time()

        # print('op time', end-start)


        start = time.time()
        with torch.no_grad():
            # print('num', num_of_splits)
            # print(g.shape)
            start = time.time()
            sino_reco = expand_sinogram(f_sinograms, num_of_splits, device=device)

                    
        end = time.time()
        # print('EXPAND TIME', end-start)
        # print(h.shape)
        # print(sino_reco.shape)
        # print(g_expand.shape)
        u[0,:,:,:] = torch.cat([h[None,None,:,:], sino_reco[None,None,:,:], g_expand[None,None,:,:]], dim=1)
        h[:,:] = h[:,:] + self.alpha_1*self.dual_unet(u[0,:,:,:][None,:,:,:])[0,0,:,:].to(device)
        
        for j in range(g.shape[1]):
            # operator = operator_split[j]
            # fbp2 = odl.tomo.analytic.filtered_back_projection.fbp_op(operator, padding=1)
            # fbp2 = OperatorModule(fbp2).to(device)
            adjoint_eval[j,:,:] = fbp_list[j](h[j::int(g.shape[1]),:][None,None,:,:])
        
        adjoint_eval = torch.mean(adjoint_eval, dim=0)
        u = torch.zeros((averaged, 2) + (f.shape[2], f.shape[3])).to(device)
        f = torch.mean(f, dim=1)
        u[0,:,:,:] = torch.cat([f, adjoint_eval[None,:,:]], dim=0)
        f[0,:,:] = f[0,:,:] + self.alpha_2*self.primal_unet(u[0,:,:,:][None,:,:,:])[0,0,:,:].to(device)
        # plt.figure()
        # plt.imshow(f[0,:,:].cpu().detach().numpy())
        # plt.show()

        return f[None,:,:,:], h
        
    # def forward(self, input_tensor:torch.Tensor, input_tensor_2:torch.Tensor, input_tensor_3:torch.Tensor) -> tuple:
    #     sinogram = self.operator(input_tensor) / self.operator_norm
    #     update = torch.cat([input_tensor_3, sinogram, input_tensor_2], dim=1)
    #     input_tensor_3 = input_tensor_3 + self.dual_resnet(update)
    #     adjoint_evaluation = self.adjoint_operator(input_tensor_3) / self.operator_norm
    #     update = torch.cat([input_tensor, adjoint_evaluation], dim=1)
    #     input_tensor = input_tensor + self.primal_resnet(update)
        
    #     return input_tensor, input_tensor_3
    
class LPD(nn.Module):
    def __init__(self,
                 primal_in_channels:int,
                 dual_in_channels:int,
                 out_channels:int,
                 first_channel:int,
                 depth,conv_kernel_size:tuple,
                 max_pool_kernel_size:tuple,
                 up_conv_kernel_size:tuple,
                 padding:int,
                 skip_connection_list:list,
                 operator:odl.tomo,
                 adjoint_operator:odl.tomo,
                 operator_norm:float,
                 n_iter:int,
                 device='cuda'):
        super().__init__()
        
        self.primal_in_channels = primal_in_channels
        self.dual_in_channels = dual_in_channels
        self.out_channels = out_channels
        self.first_channel = first_channel
        self.depth = depth
        self.conv_kernel_size = conv_kernel_size
        self.max_pool_kernel_size = max_pool_kernel_size
        self.up_conv_kernel_size = up_conv_kernel_size
        self.padding = padding
        self.skip_connection_list = skip_connection_list
        self.operator = operator
        self.adjoint_operator = adjoint_operator
        self.operator_norm = operator_norm
        self.n_iter = n_iter
        self.device = device
        self.final_relu = nn.ReLU()

        ### Initializing the parameters for every unrolled iteration step.
        for k in range(self.n_iter):
            step = LearnedPrimalDualStep(self.primal_in_channels,
                                         self.dual_in_channels,
                                         self.out_channels,
                                         self.first_channel,
                                         self.depth,
                                         self.conv_kernel_size,
                                         self.max_pool_kernel_size,
                                         self.up_conv_kernel_size,
                                         self.padding,
                                         self.skip_connection_list,
                                         self.operator,
                                         self.adjoint_operator,
                                         self.operator_norm)
            setattr(self, f'step{k}', step)
            
    def forward(self, f, g, operator_split, fbp_list, domain, num_of_splits, averaged, noise, device='cuda'):
        # print('F', f.shape)
        # print('G', g.shape)
        # print('num', num_of_splits)
        # print('avg', averaged)

        ### Initializing "h" as a zero matrix
        g_expand = expand_sinogram(g,num_of_splits,device=device)
        # g_expand = torch.Tensor().to(device)
                
        h = torch.zeros(g_expand.shape).to(self.device)
        for k in range(self.n_iter):
            step = getattr(self, f'step{k}')
            f, h = step(f, g, g_expand, h, operator_split, fbp_list, domain, num_of_splits, averaged, device='cuda')
            
        f = self.final_relu(f)
        return f