import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np

class DoubleConvAndReLU(nn.Module):
    '''
    Class for double convolutions (2D) and ReLUs that are used in the standard U-Net architecture.
    '''
    def __init__(
        self, 
        in_channels:int, 
        out_channels:int, 
        kernel_size:tuple, 
        padding:int) -> None:
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.padding = padding

        self.double_conv_and_ReLU = nn.Sequential(
            nn.Conv2d(in_channels=self.in_channels, 
                      out_channels=self.out_channels, 
                      kernel_size=self.kernel_size, 
                      padding=self.padding),
            nn.ReLU(),
            nn.Conv2d(in_channels=self.out_channels, 
                      out_channels=self.out_channels, 
                      kernel_size=self.kernel_size, 
                      padding=self.padding),
            nn.ReLU()
        )

    def forward(self, input_tensor:torch.Tensor) -> torch.Tensor:
        return self.double_conv_and_ReLU(input_tensor)
    
class UNet(nn.Module):
    '''
    The U-Net architecture module. Very generalized and gives a modularity for the depth.
    This version goes from one input channel to 64 in the first convolution.
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

        self.alpha = nn.Parameter(torch.zeros(1,1,1,1), requires_grad=True)

        self.max_pools = nn.ModuleList()
        self.encoder_block_convs = nn.ModuleList()
        self.up_convolutions = nn.ModuleList()
        self.decoder_block_convs = nn.ModuleList()
        
        self.up_conv_channel = self.first_channel*self.depth**2*2
        self.up_conv_down_channel = self.up_conv_channel // 2
        self.decoder_channel = self.first_channel*self.depth**2*2
        self.decoder_down_channel = self.decoder_channel // 2
        
        self.relu = nn.ReLU()

        for k in range(self.depth):
            if k == 0:
                self.encoder_block_convs.append(DoubleConvAndReLU(in_channels = self.in_channels, 
                                                                  out_channels=self.first_channel, 
                                                                  kernel_size=self.conv_kernel_size, 
                                                                  padding=self.padding))
                # self.decoder_block_convs.append(DoubleConvAndReLU(in_channels=self.decoder_channel, 
                #                                                   out_channels=self.decoder_down_channel, 
                #                                                   kernel_size=self.conv_kernel_size, 
                #                                                   padding=self.padding))
                # self.up_convolutions.append(nn.ConvTranspose2d(in_channels=self.up_conv_channel, 
                #                                                out_channels=self.up_conv_down_channel, 
                #                                                kernel_size=self.up_conv_kernel_size, 
                #                                                stride=2, 
                #                                                padding=0))
                self.max_pools.append(nn.MaxPool2d(kernel_size=self.max_pool_kernel_size))
                self.out_channels = int(self.first_channel / 2)
                self.in_channels = self.first_channel
                self.up_conv_channel = int(self.up_conv_channel / 2)
                self.up_conv_down_channel = self.up_conv_down_channel // 2
                self.decoder_channel = int(self.decoder_channel / 2)
                self.decoder_down_channel = self.decoder_down_channel // 2
                self.first_channel = int(self.first_channel / 2)
            elif k < self.depth-1:
                self.encoder_block_convs.append(DoubleConvAndReLU(in_channels=self.in_channels, 
                                                                  out_channels=int(self.in_channels * 2), 
                                                                  kernel_size=self.conv_kernel_size, 
                                                                  padding=self.padding))
                # self.decoder_block_convs.append(DoubleConvAndReLU(in_channels=self.decoder_channel, 
                #                                                   out_channels=self.decoder_down_channel, 
                #                                                   kernel_size=self.conv_kernel_size, 
                #                                                   padding=self.padding))
                # self.up_convolutions.append(nn.ConvTranspose2d(in_channels=self.up_conv_channel, 
                #                                                out_channels=self.up_conv_down_channel, 
                #                                                kernel_size=self.up_conv_kernel_size, 
                #                                                stride=2, 
                #                                                padding=0))
                self.max_pools.append(nn.MaxPool2d(kernel_size=self.max_pool_kernel_size))
                self.in_channels = int(self.in_channels * 2)
                self.out_channels = int(self.out_channels / 2)
                self.up_conv_channel = int(self.up_conv_channel / 2)
                self.up_conv_down_channel = self.up_conv_down_channel // 2
                self.first_channel = int(self.first_channel) / 2
                self.decoder_channel = int(self.decoder_channel / 2)
                self.decoder_down_channel = self.decoder_down_channel // 2
            else:
                self.encoder_block_convs.append(DoubleConvAndReLU(in_channels=self.in_channels, 
                                                                  out_channels=int(self.in_channels * 2), 
                                                                  kernel_size=self.conv_kernel_size, 
                                                                  padding=self.padding))
                # self.decoder_block_convs.append(nn.Conv2d(in_channels=self.decoder_channel, 
                #                                           out_channels=1, 
                #                                           kernel_size=self.conv_kernel_size, 
                #                                           padding=self.padding))
                self.in_channels = self.in_channels*2
                self.out_channels = int(self.in_channels / 2)
        
        # self.in_channels =self.first_channel * 2*(depth-1)
        # self.out_channels = int(self.in_channels // 2)
        
        for k in range(depth):
            if k == 0:
                self.decoder_block_convs.append(DoubleConvAndReLU(in_channels=self.in_channels, 
                                                                  out_channels=self.out_channels, 
                                                                  kernel_size=self.conv_kernel_size, 
                                                                  padding=self.padding))
                self.up_convolutions.append(nn.ConvTranspose2d(in_channels=self.in_channels, 
                                                               out_channels=self.out_channels, 
                                                               kernel_size=self.up_conv_kernel_size, 
                                                               stride=2, 
                                                               padding=0))
                self.in_channels = self.out_channels
                self.out_channels = int(self.out_channels / 2)
            elif k < self.depth-1:
                self.decoder_block_convs.append(DoubleConvAndReLU(in_channels=self.in_channels, 
                                                                  out_channels=self.out_channels, 
                                                                  kernel_size=self.conv_kernel_size, 
                                                                  padding=self.padding))
                self.up_convolutions.append(nn.ConvTranspose2d(in_channels=self.in_channels, 
                                                               out_channels=self.out_channels, 
                                                               kernel_size=self.up_conv_kernel_size, 
                                                               stride=2, 
                                                               padding=0))
                self.in_channels = self.out_channels
                self.out_channels = int(self.out_channels / 2)
            else:
                self.decoder_block_convs.append(nn.Conv2d(in_channels=self.in_channels, 
                                                          out_channels=1, 
                                                          kernel_size=self.conv_kernel_size, 
                                                          padding=self.padding))

    def forward(
        self, 
        input_tensor:torch.Tensor) -> torch.Tensor:
        # print('enc', self.encoder_block_convs)
        # print('max', self.max_pools)
        # print('up', self.up_convolutions)
        # print('dual', self.decoder_block_convs)
        # print('inp', input_tensor.shape)
        residual_connection_tensor = input_tensor#.clone()
        # print('input', torch.max(residual_connection_tensor), torch.min(residual_connection_tensor))
        # print('alpha', self.alpha)
        # plt.figure()
        # plt.imshow(residual_connection_tensor[0,0,:,:].cpu().detach().numpy())
        # plt.show()
        

        for k in range(self.depth-1):
            # print('input1', torch.max(input_tensor), torch.min(input_tensor))
            input_tensor = self.encoder_block_convs[k](input_tensor)
            # print('input1', torch.max(input_tensor), torch.min(input_tensor))
            self.skip_connection_list.append(input_tensor)
            input_tensor = self.max_pools[k](input_tensor)
            # print('input2', torch.max(input_tensor), torch.min(input_tensor))

        input_tensor = self.encoder_block_convs[-1](input_tensor)
        # print('input2', torch.max(input_tensor), torch.min(input_tensor))
        # plt.figure()
        # plt.imshow(input_tensor[0,0,:,:].cpu().detach().numpy())
        # plt.show()
        for i in range(self.depth-1):
            input_tensor = self.up_convolutions[i](input_tensor)
            # print('outs3', torch.max(input_tensor), torch.min(input_tensor))
            input_tensor = torch.cat([input_tensor, self.skip_connection_list[-1]], dim=1)
            self.skip_connection_list.pop()
            input_tensor = self.decoder_block_convs[i](input_tensor)
            # print('outs4', torch.max(input_tensor), torch.min(input_tensor))
        
        # print('input2', torch.max(input_tensor), torch.min(input_tensor))
        # plt.figure()
        # plt.imshow(input_tensor[0,0,:,:].cpu().detach().numpy())
        # plt.show()
        # print('outs1', torch.max(input_tensor), torch.min(input_tensor))
        input_tensor = input_tensor*(1/torch.max(input_tensor))
        input_tensor = self.decoder_block_convs[-1](input_tensor)
        # print('input2', torch.max(input_tensor), torch.min(input_tensor))
        # plt.figure()
        # plt.imshow(input_tensor[0,0,:,:].cpu().detach().numpy())
        # plt.show()
        # print('input', torch.max(residual_connection_tensor), torch.min(residual_connection_tensor))
        # print('outs2', torch.max(input_tensor), torch.min(input_tensor))
        # print(input_tensor.shape)

        input_tensor = input_tensor + self.alpha*residual_connection_tensor
        # print('input2', torch.max(input_tensor), torch.min(input_tensor))
        # _, ax = plt.subplots(1,2)
        # ax[0].imshow(input_tensor[0,0,:,:].cpu().detach().numpy())
        # ax[1].imshow((self.alpha*residual_connection_tensor)[0,0,:,:].cpu().detach().numpy())
        # plt.show()
        # print(self.alpha)
        # input_tensor = input_tensor + residual_connection_tensor
        # print('outs3', torch.max(self.alpha*residual_connection_tensor), torch.min(self.alpha*residual_connection_tensor))

        return input_tensor #self.relu(input_tensor)

# def double_conv_and_ReLU(in_channels, out_channels):
#     list_of_operations = [
#         nn.Conv2d(in_channels, out_channels, kernel_size=(3,3), padding=1),
#         nn.ReLU(),
#         nn.Conv2d(out_channels, out_channels, kernel_size=(3,3), padding=1),
#         nn.ReLU()
#     ]

#     return nn.Sequential(*list_of_operations)


# class encoding(nn.Module):
#     def __init__(self, in_channels):
#         super().__init__()

#         ### Defining instance variables
#         self.in_channels = in_channels

#         self.convs_and_relus1 = double_conv_and_ReLU(self.in_channels, out_channels=64)
#         self.maxpool1 = nn.MaxPool2d(kernel_size=(2,2))
#         self.convs_and_relus2 = double_conv_and_ReLU(in_channels=64, out_channels=128)
#         self.maxpool2 = nn.MaxPool2d(kernel_size=(2,2))
#         self.convs_and_relus3 = double_conv_and_ReLU(in_channels=128, out_channels=256)
#         self.maxpool3 = nn.MaxPool2d(kernel_size=(2,2))
#         self.convs_and_relus4 = double_conv_and_ReLU(in_channels=256, out_channels=512)
#         # self.maxpool4 = nn.MaxPool2d(kernel_size=(2,2))
#         # self.convs_and_relus5 = double_conv_and_ReLU(in_channels=512, out_channels=1024)
#         # self.convs_and_relus5 = double_conv_and_ReLU(in_channels=256, out_channels=512)

#     ### Must have forward function. Follows skip connecting UNet architechture
#     def forward(self, g):
#         g_start = g
#         encoding_features = []
#         g = self.convs_and_relus1(g)
#         encoding_features.append(g)
#         g = self.maxpool1(g)
#         g = self.convs_and_relus2(g)
#         encoding_features.append(g)
#         g = self.maxpool2(g)
#         g = self.convs_and_relus3(g)
#         encoding_features.append(g)
#         g = self.maxpool3(g)
#         g = self.convs_and_relus4(g)
#         # print('here', g.shape)
#         # encoding_features.append(g)
#         # g = self.maxpool4(g)
#         # g = self.convs_and_relus5(g)
#         # print('encoding g shape', g.shape)
#         # print('enc2', encoding_features[-1].shape)
#         # g = self.maxpool4(g)
#         # g = self.convs_and_relus5(g)
#         # print('g3', g.shape)

#         return g, encoding_features, g_start

# ### Class for decoding part of the UNet. This is the part of the UNet which
# ### goes back up with transpose of the convolution
# class decoding(nn.Module):
#     def __init__(self, out_channels):
#         super().__init__()

#         ### Defining instance variables
#         self.out_channels = out_channels

#         # self.transpose0 = nn.ConvTranspose2d(in_channels=1024, out_channels=512, kernel_size=(2,2), stride=2, padding=0, output_padding=1)
#         # self.convs_and_relus0 = double_conv_and_ReLU(in_channels=1024, out_channels=512)
#         self.transpose1 = nn.ConvTranspose2d(in_channels=512, out_channels=256, kernel_size=(2,2), stride=2, padding=0)
#         self.convs_and_relus1 = double_conv_and_ReLU(in_channels=512, out_channels=256)
#         self.transpose2 = nn.ConvTranspose2d(in_channels=256, out_channels=128, kernel_size=(2,2), stride=2, padding=0)
#         self.convs_and_relus2 = double_conv_and_ReLU(in_channels=256, out_channels=128)
#         self.transpose3 = nn.ConvTranspose2d(in_channels=128, out_channels=64, kernel_size=(2,2), stride=2, padding=0)
#         self.convs_and_relus3 = double_conv_and_ReLU(in_channels=128, out_channels=64)
#         # self.transpose4 = nn.ConvTranspose2d(in_channels=64, out_channels=1, kernel_size=(2,2), stride=2, padding=0)
#         # self.convs_and_relus4 = double_conv_and_ReLU(in_channels=64, out_channels=32)
#         self.final_conv = nn.Conv2d(in_channels=64, out_channels=self.out_channels, kernel_size=(3,3), padding=1)
#         self.final_relu = nn.ReLU()

#     ### Must have forward function. Follows skip connecting UNet architechture
#     def forward(self, g, encoding_features, g_start):
#         # print('g shape', g.shape)
#         # g = self.transpose0(g)
#         # # # print('g shape', g.shape)
#         # # # print('enc shape', encoding_features[-1].shape)
#         # g = torch.cat([g, encoding_features[-1]], dim=1)
#         # encoding_features.pop()
#         # g = self.convs_and_relus0(g)
#         g = self.transpose1(g)
#         # print('g2', g.shape)
#         # print('enc', encoding_features[-1].shape)
#         g = torch.cat([g, encoding_features[-1]], dim=1)
#         encoding_features.pop()
#         g = self.convs_and_relus1(g)
#         # print(g.shape)
#         g = self.transpose2(g)
#         g = torch.cat([g, encoding_features[-1]], dim=1)
#         encoding_features.pop()
#         g = self.convs_and_relus2(g)
#         g = self.transpose3(g)
#         g = torch.cat([g, encoding_features[-1]], dim=1)
#         encoding_features.pop()
#         g = self.convs_and_relus3(g)
#         # g = self.transpose4(g)
#         # g = torch.cat([g, encoding_features[-1]], dim=1)
#         # encoding_features.pop()
#         # g = self.convs_and_relus4(g)
#         # print(g.shape)
#         g = self.final_conv(g)
#         # print(g.shape)
#         g = g_start + g
#         # g = self.final_relu(g)

#         return g

# ### Class for the UNet model itself
# class UNet(nn.Module):
#     def __init__(self, in_channels, out_channels):
#         super().__init__()
        
#         ### Defining instance variables
#         self.in_channels = in_channels
#         self.out_channels = out_channels

#         self.encoder = encoding(self.in_channels)
#         self.decoder = decoding(self.out_channels)

#     ### Must have forward function. Calling encoder and deoder classes here
#     ### and making the whole UNet model
#     def forward(self, g):
        
#         g, encoding_features, g_start = self.encoder(g)
#         g = self.decoder(g, encoding_features, g_start)

#         return g

# class DoubleConvAndReLU(nn.Module):
#     '''
#     Class for double convolutions (2D) and ReLUs that are used in the standard U-Net architecture.
#     '''
#     def __init__(
#         self, 
#         in_channels:int, 
#         out_channels:int, 
#         kernel_size:tuple, 
#         padding:int) -> None:
#         super().__init__()

#         self.in_channels = in_channels
#         self.out_channels = out_channels
#         self.kernel_size = kernel_size
#         self.padding = padding

#         self.double_conv_and_ReLU = nn.Sequential(
#             nn.Conv2d(in_channels=self.in_channels, 
#                       out_channels=self.out_channels, 
#                       kernel_size=self.kernel_size, 
#                       padding=self.padding),
#             nn.ReLU(),
#             nn.Conv2d(in_channels=self.out_channels, 
#                       out_channels=self.out_channels, 
#                       kernel_size=self.kernel_size, 
#                       padding=self.padding),
#             nn.ReLU()
#         )

#     def forward(self, input_tensor:torch.Tensor) -> torch.Tensor:
#         return self.double_conv_and_ReLU(input_tensor)
    
# class UNet(nn.Module):
#     '''
#     The U-Net architecture module. Very generalized and gives a modularity for the depth.
#     This version goes from one input channel to 64 in the first convolution.
#     '''
#     def __init__(
#         self, 
#         in_channels:int,
#         out_channels:int,
#         first_channel:int,
#         depth:int,
#         conv_kernel_size:tuple, 
#         max_pool_kernel_size:tuple, 
#         up_conv_kernel_size:tuple, 
#         padding:int, 
#         skip_connection_list:list) -> None:
#         super().__init__()

#         self.in_channels = in_channels
#         self.out_channels = out_channels
#         self.first_channel = first_channel
#         self.depth = depth
#         self.conv_kernel_size = conv_kernel_size
#         self.max_pool_kernel_size = max_pool_kernel_size
#         self.up_conv_kernel_size = up_conv_kernel_size
#         self.padding = padding
#         self.skip_connection_list = skip_connection_list

#         self.alpha = nn.Parameter(torch.zeros(1,1,1,1), requires_grad=True)

#         self.max_pools = nn.ModuleList()
#         self.encoder_block_convs = nn.ModuleList()
#         self.up_convolutions = nn.ModuleList()
#         self.decoder_block_convs = nn.ModuleList()
        
#         self.up_conv_channel = self.first_channel * (2*(depth-1))
#         self.up_conv_down_channel = self.up_conv_channel // 2
#         self.decoder_channel = self.first_channel * (2*(depth-1))
#         self.decoder_down_channel = self.decoder_channel // 2

#         for k in range(self.depth):
#             if k == 0:
#                 self.encoder_block_convs.append(DoubleConvAndReLU(in_channels = self.in_channels, 
#                                                                   out_channels=self.first_channel, 
#                                                                   kernel_size=self.conv_kernel_size, 
#                                                                   padding=self.padding))
#                 self.decoder_block_convs.append(DoubleConvAndReLU(in_channels=self.decoder_channel, 
#                                                                   out_channels=self.decoder_down_channel, 
#                                                                   kernel_size=self.conv_kernel_size, 
#                                                                   padding=self.padding))
#                 self.up_convolutions.append(nn.ConvTranspose2d(in_channels=self.up_conv_channel, 
#                                                                out_channels=self.up_conv_down_channel, 
#                                                                kernel_size=self.up_conv_kernel_size, 
#                                                                stride=2, 
#                                                                padding=0))
#                 self.max_pools.append(nn.MaxPool2d(kernel_size=self.max_pool_kernel_size))
#                 self.out_channels = int(self.first_channel / 2)
#                 self.in_channels = self.first_channel
#                 self.up_conv_channel = int(self.up_conv_channel / 2)
#                 self.up_conv_down_channel = self.up_conv_down_channel // 2
#                 self.decoder_channel = int(self.decoder_channel / 2)
#                 self.decoder_down_channel = self.decoder_down_channel // 2
#                 self.first_channel = int(self.first_channel / 2)
#             elif k < self.depth-1:
#                 self.encoder_block_convs.append(DoubleConvAndReLU(in_channels=self.in_channels, 
#                                                                   out_channels=int(self.in_channels * 2), 
#                                                                   kernel_size=self.conv_kernel_size, 
#                                                                   padding=self.padding))
#                 self.decoder_block_convs.append(DoubleConvAndReLU(in_channels=self.decoder_channel, 
#                                                                   out_channels=self.decoder_down_channel, 
#                                                                   kernel_size=self.conv_kernel_size, 
#                                                                   padding=self.padding))
#                 self.up_convolutions.append(nn.ConvTranspose2d(in_channels=self.up_conv_channel, 
#                                                                out_channels=self.up_conv_down_channel, 
#                                                                kernel_size=self.up_conv_kernel_size, 
#                                                                stride=2, 
#                                                                padding=0))
#                 self.max_pools.append(nn.MaxPool2d(kernel_size=self.max_pool_kernel_size))
#                 self.in_channels = int(self.in_channels * 2)
#                 self.out_channels = int(self.out_channels / 2)
#                 # self.up_conv_channel = int(self.up_conv_channel / 2)
#                 self.first_channel = int(self.first_channel) / 2
#                 self.decoder_channel = int(self.decoder_channel / 2)
#                 self.decoder_down_channel = self.decoder_down_channel // 2
#             else:
#                 self.encoder_block_convs.append(DoubleConvAndReLU(in_channels=self.in_channels, 
#                                                                   out_channels=int(self.in_channels * 2), 
#                                                                   kernel_size=self.conv_kernel_size, 
#                                                                   padding=self.padding))
#                 self.decoder_block_convs.append(nn.Conv2d(in_channels=self.decoder_channel, 
#                                                           out_channels=1, 
#                                                           kernel_size=self.conv_kernel_size, 
#                                                           padding=self.padding))

#     def forward(
#         self, 
#         input_tensor:torch.Tensor) -> torch.Tensor:
#         # print('enc', self.encoder_block_convs)
#         # print('max', self.max_pools)
#         # print('up', self.up_convolutions)
#         # print('dual', self.decoder_block_convs)
        
#         residual_connection_tensor = input_tensor.clone()

#         for k in range(self.depth-1):
#             input_tensor = self.encoder_block_convs[k](input_tensor)
#             self.skip_connection_list.append(input_tensor)
#             input_tensor = self.max_pools[k](input_tensor)

#         input_tensor = self.encoder_block_convs[-1](input_tensor)

#         for i in range(self.depth-1):
#             input_tensor = self.up_convolutions[i](input_tensor)
#             input_tensor = torch.cat([input_tensor, self.skip_connection_list[-1]], dim=1)
#             self.skip_connection_list.pop()
#             input_tensor = self.decoder_block_convs[i](input_tensor)

#         input_tensor = self.decoder_block_convs[-1](input_tensor)

#         return input_tensor + self.alpha*residual_connection_tensor
