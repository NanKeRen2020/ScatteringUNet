#!/usr/bin/env python3
#coding=utf-8

#============================#
#Program:Model.py
#       
#Date:20-4-16
#Author:liheng
#Version:V1.0
#============================#

import layers
import torch
import torch.nn as nn
import torch.nn.functional as F

import sys
import os
sys.path.append("/home/nankeren/AI_projects/WaveletScattering/wavelet-scattering-generative-model-main")
from istc import ISTC
# from istc_l0 import ISTC
# from phase_scattering2d_torch import ScatteringTorch2D_wph
from phase_scattering2d_torch17kymatio02 import ScatteringTorch2D_wph
# from wavelet_scattering_network import wavelet_scattering_size



def wavelet_scattering_size(J, max_order = 1, L=8, A=4):
    order0_size = 1
    order1_size = L * J * A
    output_size = order0_size + order1_size

    if max_order == 2:
        order2_size = L ** 2 * J * (J - 1) // 2
        output_size += order2_size
    return output_size


def double_conv(in_channels, out_channels):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, 3, padding=1),
        nn.ReLU(inplace=True),
        nn.Conv2d(out_channels, out_channels, 3, padding=1),
        nn.ReLU(inplace=True)
    )   


class UNet(nn.Module):

    def __init__(self, in_channel, out_channel):
        super(UNet, self).__init__()
                
        self.dconv_down1 = double_conv(in_channel, 64)
        self.dconv_down2 = double_conv(64, 128)
        self.dconv_down3 = double_conv(128, 256)
        self.dconv_down4 = double_conv(256, 512)        

        self.maxpool = nn.MaxPool2d(2)
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)        
        
        self.dconv_up3 = double_conv(256 + 512, 256)
        self.dconv_up2 = double_conv(128 + 256, 128)
        self.dconv_up1 = double_conv(128 + 64, 64)
        
        self.conv_last = nn.Conv2d(64, out_channel, 1)
        
        
    def forward(self, x):
        conv1 = self.dconv_down1(x)
        x = self.maxpool(conv1)

        conv2 = self.dconv_down2(x)
        x = self.maxpool(conv2)
        
        conv3 = self.dconv_down3(x)
        x = self.maxpool(conv3)   
        
        x = self.dconv_down4(x)
        
        x = self.upsample(x)        
        x = torch.cat([x, conv3], dim=1)
        
        x = self.dconv_up3(x)
        x = self.upsample(x)        
        x = torch.cat([x, conv2], dim=1)       

        x = self.dconv_up2(x)
        x = self.upsample(x)        
        x = torch.cat([x, conv1], dim=1)   
        
        x = self.dconv_up1(x)
        
        out = self.conv_last(x)
        
        return out



class Model(torch.nn.Module):
    def __init__(self):
        super(Model,self).__init__()
        # encoder
        self.encoder1 = self.EncoderBlock(1,16)
        self.encoder2 = self.EncoderBlock(16,32)
        self.encoder3 = self.EncoderBlock(32,64)
        self.encoder4 = self.EncoderBlock(64,96)

        # decoder
        self.decode4 = self.DecodeBlock(96,64)
        self.decode3 = self.DecodeBlock(128,32)
        self.decode2 = self.DecodeBlock(64,16)
        self.decode1 = self.DecodeBlock(32,16)

        self.res_conv = layers.conv2d(16,11,3,1)

    def EncoderBlock(self,in_channels, out_channels, t=6):
        return torch.nn.Sequential(layers.sepconv2d(in_channels,out_channels,3,2,False),
                                   layers.InvertedResidual(out_channels,out_channels,t=t,s=1))
    def DecodeBlock(self,in_channels,out_channels,kernel_size=3,bias=True):
        """

        :param in_channels:
        :param out_channels:
        :param kernel_size:
        :param bias:
        :return:
        """
        return torch.nn.Sequential(
            # conv1x1
            nn.Conv2d(in_channels,in_channels//4,1,bias=bias),
            nn.BatchNorm2d(in_channels//4),
            nn.ReLU6(),

            #deconv 3X3
            nn.ConvTranspose2d(in_channels//4,in_channels//4,kernel_size,stride=2,padding=1,output_padding=1,bias=bias),
            nn.BatchNorm2d(in_channels//4),
            nn.ReLU6(),

            # conv1x1
            nn.Conv2d(in_channels//4,out_channels,1,bias=bias),
            nn.BatchNorm2d(out_channels),
            nn.ReLU6())

    def forward(self, x):
        #encode stage
        e1 = self.encoder1(x) # [B,16,32,48]
        e2 = self.encoder2(e1) # [B,32,16,24]
        e3 = self.encoder3(e2) # [B,64,8,12]
        e4 = self.encoder4(e3) # [B,96,4,6]

        #decode stage
        d4 = torch.cat((self.decode4(e4),e3),dim=1) # [B,64+64,8,12]
        d3 = torch.cat((self.decode3(d4),e2),dim=1) #[B,32+32,16,24]
        d2 = torch.cat((self.decode2(d3),e1),dim=1) #[B,16+16,32,48]
        d1 = self.decode1(d2) #[B,16,64,96]

        #res
        res = self.res_conv(d1) #[B,11,64,96]
        return res

class ModelScattering(torch.nn.Module):
    def __init__(self):
        super(ModelScattering, self).__init__()
        # encoder
        order = 1
        Shape=(64, 80)
        
        in_channel = 1

        self.encoder1 = ScatteringTorch2D_wph(J=1, shape=Shape, L=8, A=4, max_order=order,
                                            backend='torch_skcuda').to('cuda')
        self.encoder1 = nn.Sequential(self.encoder1, nn.Flatten(1, 2))

        self.encoder2 = ScatteringTorch2D_wph(J=2, shape=Shape, L=8, A=4, max_order=order,
                                            backend='torch_skcuda').to('cuda')
        self.encoder2 = nn.Sequential(self.encoder2, nn.Flatten(1, 2))


        self.encoder3 = ScatteringTorch2D_wph(J=3, shape=Shape, L=8, A=4, max_order=order,
                                            backend='torch_skcuda').to('cuda')
        self.encoder3 = nn.Sequential(self.encoder3, nn.Flatten(1, 2))

        encoder1_size = wavelet_scattering_size(1, order)*in_channel
        encoder2_size = wavelet_scattering_size(2, order)*in_channel
        encoder3_size = wavelet_scattering_size(3, order)*in_channel
        
        # self.project1 = nn.Conv2d(encoder1_size, 16, kernel_size=1, stride=1, padding=0, bias=False)

        # self.project2 = nn.Conv2d(encoder2_size, 32, kernel_size=1, stride=1, padding=0, bias=False)

        # self.project3 = nn.Conv2d(encoder3_size, 128, kernel_size=1, stride=1, padding=0, bias=False)

        # self.istc = ISTC(128, dictionary_size=128*8, n_iterations=12,
        #     lambda_0=0.3, lambda_star=0.05, lambda_star_lb=0.05,
        #     grad_lambda_star=True, epsilon_lambda_0=1.0,
        #     output_rec=True, use_W=False)
        # self.encoder4 = ScatteringTorch2D_wph(J=4, shape=(64, 80), L=8, A=4, max_order=2,
        #                                     backend='torch_skcuda').to('cuda')
        # self.encoder4 = nn.Sequential(self.encoder4, nn.Flatten(1, 2))


        # # decoder
        # self.decode4 = self.DecodeBlock(96,64)
        self.decode3 = self.DecodeBlock(encoder3_size + encoder2_size, 64)
        self.decode2 = self.DecodeBlock(64 + encoder1_size,32)
        self.decode1 = self.DecodeBlock(32,16)

        self.res_conv = layers.conv2d(16,11,3,1)

    def EncoderBlock(self,in_channels, out_channels, t=6):
        return torch.nn.Sequential(layers.sepconv2d(in_channels,out_channels,3,2,False),
                                   layers.InvertedResidual(out_channels,out_channels,t=t,s=1))
    def DecodeBlock(self,in_channels,out_channels,kernel_size=3,bias=True):
        """

        :param in_channels:
        :param out_channels:
        :param kernel_size:
        :param bias:
        :return:
        """
        return torch.nn.Sequential(
            # conv1x1
            nn.Conv2d(in_channels,in_channels//4,1,bias=bias),
            nn.BatchNorm2d(in_channels//4),
            nn.ReLU6(),

            #deconv 3X3
            nn.ConvTranspose2d(in_channels//4,in_channels//4,kernel_size,stride=2,padding=1,output_padding=1,bias=bias),
            nn.BatchNorm2d(in_channels//4),
            nn.ReLU6(),

            # conv1x1
            nn.Conv2d(in_channels//4,out_channels,1,bias=bias),
            nn.BatchNorm2d(out_channels),
            nn.ReLU6())

    def forward(self, x):
        #encode stage
        e1 = self.encoder1(x) 
        # e1 = self.project1(e1) # [B,16,32,48]

        e2 = self.encoder2(x) 
        # e2 = self.project2(e2) # [B,32,16,24]

        e3 = self.encoder3(x) 
        # e3 = self.project3(e3) # [B,64,8,12]
        # e4 = self.encoder4(x) # [B,96,4,6]
        
        # e3 = self.istc(e3)
        #decode stage
        # d4 = torch.cat((self.decode4(e4),e3),dim=1) # [B,64+64,8,12]
        d3 = torch.cat((self.decode3(e3),e2),dim=1) #[B,32+32,16,24]
        d2 = torch.cat((self.decode2(d3),e1),dim=1) #[B,16+16,32,48]
        d1 = self.decode1(d2) #[B,16,64,96]

        #res
        res = self.res_conv(d1) #[B,11,64,96]
        return res

class ModelScatteringSparse(torch.nn.Module):
    def __init__(self):
        super(ModelScatteringSparse, self).__init__()
        # encoder
        order = 1
        Shape=(64, 80)

        self.encoder1 = ScatteringTorch2D_wph(J=1, shape=Shape, L=8, A=4, max_order=order,
                                            backend='torch_skcuda').to('cuda')
        self.encoder1 = nn.Sequential(self.encoder1, nn.Flatten(1, 2))

        self.encoder2 = ScatteringTorch2D_wph(J=2, shape=Shape, L=8, A=4, max_order=order,
                                            backend='torch_skcuda').to('cuda')
        self.encoder2 = nn.Sequential(self.encoder2, nn.Flatten(1, 2))


        self.encoder3 = ScatteringTorch2D_wph(J=3, shape=Shape, L=8, A=4, max_order=order,
                                            backend='torch_skcuda').to('cuda')
        self.encoder3 = nn.Sequential(self.encoder3, nn.Flatten(1, 2))

        self.project1 = nn.Conv2d(wavelet_scattering_size(1, order), 16, kernel_size=1, stride=1, padding=0, bias=False)

        self.project2 = nn.Conv2d(wavelet_scattering_size(2, order), 32, kernel_size=1, stride=1, padding=0, bias=False)

        self.project3 = nn.Conv2d(wavelet_scattering_size(3, order), 128, kernel_size=1, stride=1, padding=0, bias=False)

        self.istc1 = ISTC(16, dictionary_size=16*8, n_iterations=12,
            lambda_0=0.3, lambda_star=0.05, lambda_star_lb=0.05,
            grad_lambda_star=True, epsilon_lambda_0=1.0,
            output_rec=True, use_W=False)

        self.istc2 = ISTC(32, dictionary_size=32*8, n_iterations=12,
            lambda_0=0.3, lambda_star=0.05, lambda_star_lb=0.05,
            grad_lambda_star=True, epsilon_lambda_0=1.0,
            output_rec=True, use_W=False)

        self.istc3 = ISTC(128, dictionary_size=128*8, n_iterations=12,
            lambda_0=0.3, lambda_star=0.05, lambda_star_lb=0.05,
            grad_lambda_star=True, epsilon_lambda_0=1.0,
            output_rec=True, use_W=False)
        # self.encoder4 = ScatteringTorch2D_wph(J=4, shape=(64, 80), L=8, A=4, max_order=2,
        #                                     backend='torch_skcuda').to('cuda')
        # self.encoder4 = nn.Sequential(self.encoder4, nn.Flatten(1, 2))


        # # decoder
        # self.decode4 = self.DecodeBlock(96,64)
        self.decode3 = self.DecodeBlock(128,32)
        self.decode2 = self.DecodeBlock(64,16)
        self.decode1 = self.DecodeBlock(32,16)

        self.res_conv = layers.conv2d(16,11,3,1)

    def EncoderBlock(self,in_channels, out_channels, t=6):
        return torch.nn.Sequential(layers.sepconv2d(in_channels,out_channels,3,2,False),
                                   layers.InvertedResidual(out_channels,out_channels,t=t,s=1))
    def DecodeBlock(self,in_channels,out_channels,kernel_size=3,bias=True):
        """

        :param in_channels:
        :param out_channels:
        :param kernel_size:
        :param bias:
        :return:
        """
        return torch.nn.Sequential(
            # conv1x1
            nn.Conv2d(in_channels,in_channels//4,1,bias=bias),
            nn.BatchNorm2d(in_channels//4),
            nn.ReLU6(),

            #deconv 3X3
            nn.ConvTranspose2d(in_channels//4,in_channels//4,kernel_size,stride=2,padding=1,output_padding=1,bias=bias),
            nn.BatchNorm2d(in_channels//4),
            nn.ReLU6(),

            # conv1x1
            nn.Conv2d(in_channels//4,out_channels,1,bias=bias),
            nn.BatchNorm2d(out_channels),
            nn.ReLU6())

    def forward(self, x):
        #encode stage
        e1 = self.encoder1(x) 
        e1 = self.project1(e1) # [B,16,32,48]
        e1 = self.istc1(e1)

        e2 = self.encoder2(x) 
        e2 = self.project2(e2) # [B,32,16,24]
        e2 = self.istc2(e2)

        e3 = self.encoder3(x) 
        e3 = self.project3(e3) # [B,64,8,12]
        e3 = self.istc3(e3)
        
        # e4 = self.encoder4(x) # [B,96,4,6]

        # for train
        if self.training:
            rand_batch = torch.normal(mean=0, std=1, size=e3.shape) 
            rand_batch = rand_batch.to('cuda')
            e3 = e3 + rand_batch        
       
        #decode stage
        # d4 = torch.cat((self.decode4(e4),e3),dim=1) # [B,64+64,8,12]
        d3 = torch.cat((self.decode3(e3),e2),dim=1) #[B,32+32,16,24]
        d2 = torch.cat((self.decode2(d3),e1),dim=1) #[B,16+16,32,48]
        d1 = self.decode1(d2) #[B,16,64,96]

        #res
        res = self.res_conv(d1) #[B,11,64,96]
        return res

class UNetScattering(nn.Module):
    
    def __init__(self, in_channel, n_class):
        super(UNetScattering, self).__init__()

        order = 1
        Shape=(64, 80)

        self.encoder1 = ScatteringTorch2D_wph(J=1, shape=Shape, L=8, A=4, max_order=order,
                                            backend='torch_skcuda')
        self.encoder1 = nn.Sequential(self.encoder1, nn.Flatten(1, 2)).to('cuda')

        self.encoder2 = ScatteringTorch2D_wph(J=2, shape=Shape, L=8, A=4, max_order=order,
                                            backend='torch_skcuda')
        self.encoder2 = nn.Sequential(self.encoder2, nn.Flatten(1, 2)).to('cuda')


        self.encoder3 = ScatteringTorch2D_wph(J=3, shape=Shape, L=8, A=4, max_order=order,
                                            backend='torch_skcuda')
        self.encoder3 = nn.Sequential(self.encoder3, nn.Flatten(1, 2)).to('cuda')


        # self.project1 = nn.Conv2d(wavelet_scattering_size(1, order), 16, kernel_size=1, stride=1, padding=0, bias=False)

        # self.project2 = nn.Conv2d(wavelet_scattering_size(2, order), 32, kernel_size=1, stride=1, padding=0, bias=False)

        # self.project3 = nn.Conv2d(wavelet_scattering_size(3, order), 128, kernel_size=1, stride=1, padding=0, bias=False)

        encoder1_size = wavelet_scattering_size(1, order)*in_channel
        encoder2_size = wavelet_scattering_size(2, order)*in_channel
        encoder3_size = wavelet_scattering_size(3, order)*in_channel
       
        self.istc1 = ISTC(encoder3_size, dictionary_size=encoder3_size*8, n_iterations=12,
            lambda_0=0.3, lambda_star=0.05, lambda_star_lb=0.05,
            grad_lambda_star=True, epsilon_lambda_0=1.0,
            output_rec=True, use_W=False)

        self.istc2 = ISTC(encoder2_size, dictionary_size=encoder2_size*8, n_iterations=12,
            lambda_0=0.3, lambda_star=0.05, lambda_star_lb=0.05,
            grad_lambda_star=True, epsilon_lambda_0=1.0,
            output_rec=True, use_W=False)

        self.istc3 = ISTC(encoder1_size, dictionary_size=encoder1_size*8, n_iterations=12,
            lambda_0=0.3, lambda_star=0.05, lambda_star_lb=0.05,
            grad_lambda_star=True, epsilon_lambda_0=1.0,
            output_rec=True, use_W=False)

        # self.dconv_down1 = double_conv(3, 64)
        # self.dconv_down2 = double_conv(64, 128)
        # self.dconv_down3 = double_conv(128, 256)
        # self.dconv_down4 = double_conv(256, 512)        

        self.maxpool = nn.MaxPool2d(2)
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)        
        
        self.dconv_up3 = double_conv(encoder2_size + encoder3_size, 128)
        self.dconv_up2 = double_conv(encoder1_size + 128, 64)
        # self.dconv_up1 = double_conv(128, 64)
        
        self.conv_last = nn.Conv2d(64, n_class, 1)
        
        
    def forward(self, x):
        # conv1 = self.dconv_down1(x)
        # x = self.maxpool(conv1)

        # conv2 = self.dconv_down2(x)
        # x = self.maxpool(conv2)
        
        # conv3 = self.dconv_down3(x)
        # x = self.maxpool(conv3)   
        
        # x = self.dconv_down4(x)
        x = x.contiguous()
        e1 = self.encoder1(x) 
        # e1 = self.project1(e1) # [B,16,32,48]
        # e1 = self.istc1(e1)

        e2 = self.encoder2(x) 
        # e2 = self.project2(e2) # [B,32,16,24]
        # e2 = self.istc2(e2)

        e3 = self.encoder3(x) 
        # e3 = self.project3(e3) # [B,64,8,12]
        # e3 = self.istc3(e3)

        x = self.upsample(e3)        
        x = torch.cat([x, e2], dim=1)
        
        x = self.dconv_up3(x)
        x = self.upsample(x)        
        x = torch.cat([x, e1], dim=1)       

        x = self.dconv_up2(x)
        x = self.upsample(x)        
        # x = torch.cat([x, conv1], dim=1)   
        
        # x = self.dconv_up1(x)
        
        out = self.conv_last(x)
        
        return out



class UNetScatteringDynamic(nn.Module):
    
    def __init__(self, in_channel, out_channel):
        super(UNetScatteringDynamic, self).__init__()

        self.order = 1
        Shape=(64, 80)

        # self.project1 = nn.Conv2d(wavelet_scattering_size(1, order), 16, kernel_size=1, stride=1, padding=0, bias=False)

        # self.project2 = nn.Conv2d(wavelet_scattering_size(2, order), 32, kernel_size=1, stride=1, padding=0, bias=False)

        # self.project3 = nn.Conv2d(wavelet_scattering_size(3, order), 128, kernel_size=1, stride=1, padding=0, bias=False)

        encoder1_size = wavelet_scattering_size(1, self.order)*in_channel
        encoder2_size = wavelet_scattering_size(2, self.order)*in_channel
        encoder3_size = wavelet_scattering_size(3, self.order)*in_channel
       
        self.istc1 = ISTC(encoder3_size, dictionary_size=encoder3_size*8, n_iterations=12,
            lambda_0=0.3, lambda_star=0.05, lambda_star_lb=0.05,
            grad_lambda_star=True, epsilon_lambda_0=1.0,
            output_rec=True, use_W=False)

        self.istc2 = ISTC(encoder2_size, dictionary_size=encoder2_size*8, n_iterations=12,
            lambda_0=0.3, lambda_star=0.05, lambda_star_lb=0.05,
            grad_lambda_star=True, epsilon_lambda_0=1.0,
            output_rec=True, use_W=False)

        self.istc3 = ISTC(encoder1_size, dictionary_size=encoder1_size*8, n_iterations=12,
            lambda_0=0.3, lambda_star=0.05, lambda_star_lb=0.05,
            grad_lambda_star=True, epsilon_lambda_0=1.0,
            output_rec=True, use_W=False)

        # self.dconv_down1 = double_conv(3, 64)
        # self.dconv_down2 = double_conv(64, 128)
        # self.dconv_down3 = double_conv(128, 256)
        # self.dconv_down4 = double_conv(256, 512)        

        self.maxpool = nn.MaxPool2d(2)
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)        
        
        self.dconv_up3 = double_conv(encoder2_size + encoder3_size, 128)
        self.dconv_up2 = double_conv(encoder1_size + 128, 64)
        # self.dconv_up1 = double_conv(128, 64)
        
        self.conv_last = nn.Conv2d(64, out_channel, 1)

        self.encoder1 = None
        self.encoder2 = None
        self.encoder3 = None

        
    def forward(self, x):
        # conv1 = self.dconv_down1(x)
        # x = self.maxpool(conv1)

        # conv2 = self.dconv_down2(x)
        # x = self.maxpool(conv2)
        
        # conv3 = self.dconv_down3(x)
        # x = self.maxpool(conv3)   
        
        # x = self.dconv_down4(x)
        x = x.contiguous()
        shape_size = [x.shape[2], x.shape[3]]
        if self.encoder1 is None: 
            self.encoder1 = ScatteringTorch2D_wph(J=1, shape=shape_size, L=8, A=4, max_order=self.order,
                                                backend='torch_skcuda')
            self.encoder1 = nn.Sequential(self.encoder1, nn.Flatten(1, 2)).to('cuda')

        if self.encoder2 is None: 

            self.encoder2 = ScatteringTorch2D_wph(J=2, shape=shape_size, L=8, A=4, max_order=self.order,
                                                backend='torch_skcuda')
            self.encoder2 = nn.Sequential(self.encoder2, nn.Flatten(1, 2)).to('cuda')

        if self.encoder3 is None: 

            self.encoder3 = ScatteringTorch2D_wph(J=3, shape=shape_size, L=8, A=4, max_order=self.order,
                                                backend='torch_skcuda')
            self.encoder3 = nn.Sequential(self.encoder3, nn.Flatten(1, 2)).to('cuda')



        e1 = self.encoder1(x) 
        # e1 = self.project1(e1) # [B,16,32,48]
        # e1 = self.istc1(e1)

        e2 = self.encoder2(x) 
        # e2 = self.project2(e2) # [B,32,16,24]
        # e2 = self.istc2(e2)

        e3 = self.encoder3(x) 
        # e3 = self.project3(e3) # [B,64,8,12]
        # e3 = self.istc3(e3)

        x = self.upsample(e3)        
        x = torch.cat([x, e2], dim=1)
        
        x = self.dconv_up3(x)
        x = self.upsample(x)        
        x = torch.cat([x, e1], dim=1)       

        x = self.dconv_up2(x)
        x = self.upsample(x)        
        # x = torch.cat([x, conv1], dim=1)   
        
        # x = self.dconv_up1(x)
        
        out = self.conv_last(x)
        
        return out



class CrossEntropyLoss2d(nn.Module):
    """
    defines a cross entropy loss for 2D images
    """
    def __init__(self, weight=None, ignore_label= 255):
        """
        :param weight: 1D weight vector to deal with the class-imbalance
        Obtaining log-probabilities in a neural network is easily achieved by adding a LogSoftmax layer in the last layer of your network.
        You may use CrossEntropyLoss instead, if you prefer not to add an extra layer.
        """
        super().__init__()

        #self.loss = nn.NLLLoss2d(weight, ignore_index=255)
        # self.loss = nn.NLLLoss(weight)
        self.loss = nn.CrossEntropyLoss(weight).to('cuda')

    def forward(self, outputs, targets):
        # return self.loss(F.log_softmax(outputs, 1), targets)
        return self.loss(outputs,targets)


if __name__ == '__main__':
    from torchstat import stat

    # initial model
    model = Model()

    input_data = torch.ones([5, 1, 64, 96], dtype=torch.float32)  # [B,C,H,W]

    stat(model,(1,64,96))

    exit(0)


    # initialize the optimizer
    optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

    # print the model's state_dict
    print("model's state_dict:")
    for param_tensor in model.state_dict():
        print(param_tensor, '\t', model.state_dict()[param_tensor].size())

    print("\noptimizer's state_dict")
    for var_name in optimizer.state_dict():
        print(var_name, '\t', optimizer.state_dict()[var_name])