import copy
import math

import torch.nn.functional as F
from torch import nn
import torch
import numpy as np
import scipy.io as sio
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from torchvision.transforms import Compose, RandomCrop, ToTensor
from math import sqrt
# from models.models_others import SoftAttn
from models.models_others import SoftAttn, LAConv2D, LACRB, ChannelAttention, SpatialAttention
from models.Inv_block import InvBlock, DenseBlock,UNetConvBlock


class Generator(nn.Module):
    def __init__(self, scale_factor):
        upsample_block_num = int(math.log(scale_factor, 2))

        super(Generator, self).__init__()
        self.blk_9_16_3 = nn.Sequential(
            # nn.Conv2d(2, 64, kernel_size=9, padding=4),
            nn.Conv2d(5, 30, kernel_size=3, padding=1),
            nn.PReLU()
        )

        self.blk_8_16_3 = nn.Sequential(
            # nn.Conv2d(2, 64, kernel_size=9, padding=4),
            nn.Conv2d(4, 30, kernel_size=3, padding=1),
            nn.PReLU()
        )
        # self.blk_17_24_3 = nn.Sequential(
        #     # nn.Conv2d(2, 64, kernel_size=9, padding=4),
        #     nn.Conv2d(17, 24, kernel_size=3, padding=1),
        #     nn.PReLU()
        # )
        # self.blk1 = nn.Sequential(
        #     # nn.Conv2d(2, 64, kernel_size=9, padding=4),
        #     nn.Conv2d(17, 48, kernel_size=3, padding=1),
        #     nn.PReLU()
        # )
        self.blk2 = nn.Sequential(
            # nn.Conv2d(2, 64, kernel_size=9, padding=4),
            nn.Conv2d(60, 30, kernel_size=3, padding=1),
            nn.PReLU()
        )
        # self.blk_9_16_2 = nn.Sequential(
        #     # nn.Conv2d(2, 64, kernel_size=9, padding=4),
        #     nn.Conv2d(9, 16, kernel_size=3, padding=1),
        #     nn.PReLU()
        # )
        # self.blk_9_16_3 = nn.Sequential(
        #     # nn.Conv2d(2, 64, kernel_size=9, padding=4),
        #     nn.Conv2d(9, 16, kernel_size=3, padding=1),
        #     nn.PReLU()
        # )
        # self.blk_1_36 = nn.Sequential(
        #     # nn.Conv2d(2, 64, kernel_size=9, padding=4),
        #     nn.Conv2d(1, 36, kernel_size=3, padding=1),
        #     nn.PReLU()
        # )
        ##### up sampling #############
        self.up_sample4 = UpsampleBLock(4, 4)  ### for 4 band 4, for 8 band 8 ,in_channels, up_scale
        self.up_sample2 = UpsampleBLock(4, 2)  ### for 4 band 4, for 8 band 8
        # self.up_sample3 = UpsampleBLock(9, 2)  ### for 4 band 4, for 8 band 8
        # self.up_sample2_2 = UpsampleBLock(8, 2)  ### for 4 band 4, for 8 band 8
        # self.block2 = ResidualBlock(64)
        # self.subpixel_up = nn.PixelShuffle(2)  # up sampling
        # self.downscale = nn.MaxPool2d(kernel_size=2, stride=2)  # down sampling
        ################
        # self.lu_block1 = Exp_block(24, 48)
        # # # self.lu_block2 = Exp_block(72, 48)
        # # # self.lu_block2 = Exp_block(30, 48)
        # self.lu_block3 = Exp_block(48, 36)        #
        # self.blk_1 = ConvLayer(24, 48, 3, last=nn.ReLU)
        # self.blk_2 = ConvLayer(48, 36, 3, last=nn.PReLU)
        # self.blk_3 = ConvLayer(36, 8, 3, last=nn.ReLU)
        # self.conv1 = nn.Conv2d(in_channels=48, out_channels=48, kernel_size=3, stride=1, padding=1,
        #                        bias=True)  # change in-channel from 16 to 8
        # self.conv2 = nn.Conv2d(in_channels=48, out_channels=36, kernel_size=3, stride=1, padding=2,
        #                        dilation=2, bias=True)  # change in-channel from 16 to 8
        # self.conv3 = nn.Conv2d(in_channels=36, out_channels=24, kernel_size=3, stride=1, padding=2,
        #                        dilation=2, bias=True)  # change in-channel from 16 to 8
        self.conv6 = nn.Conv2d(in_channels=30, out_channels=4, kernel_size=3, stride=1, padding=1,
                               bias=True)  # change out as 4   or   8
        # self.selayer1 = SELayer(48)
        # self.selayer2 = SELayer(48)
        # self.blk_1_24_3 = ConvLayer(1, 24, 3, last=nn.ReLU)
        # self.conv6 = nn.Conv2d(in_channels=48, out_channels=8, kernel_size=5, stride=1, padding=2,
        #                        bias=True)  # change out as 4   or   8
        #####################
        # self.shallow1 = nn.Conv2d(in_channels=16, out_channels=64, kernel_size=9, stride=1, padding=4,
        #                           bias=True)
        # self.shallow2 = nn.Conv2d(in_channels=64, out_channels=32, kernel_size=1, stride=1, padding=0, bias=True)
        # self.shallow3 = nn.Conv2d(in_channels=32, out_channels=8, kernel_size=5, stride=1, padding=2,
        #                           bias=True)
        # self.direconv1 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1, bias=True)
        # self.direconv2 = nn.Conv2d(in_channels=32, out_channels=8, kernel_size=1, stride=1, padding=0, bias=True)
        # self.relu = nn.ReLU(inplace=True)
        # self.mulnet = mscb2()
        self.lu_block1 = Exp_block(60)
        # self.lu_lac = LAConv2D(48)
        self.lu_block2 = Exp_block(30)
        # self.lu_lacrb1 = LACRB_lu(32)
        # self.lu_lacrb2 = LACRB_lu(32)
        # self.lu_lacrb3 = LACRB_lu(32)
        # self.lu_block3 = Exp_block(24)
        ####################
        # self.blk_5_32_3 = ConvLayer(5, 32, 3, last=nn.LeakyReLU)
        # self.blk_4_32_3 = ConvLayer(4, 32, 3, last=nn.LeakyReLU)
        # self.blk_9_32_3 = ConvLayer(9, 32, 3, last=nn.LeakyReLU)
        # self.blk_32_4_1 = ConvLayer(32, 4, 1, last=nn.LeakyReLU)
        # self.blk_32_4_3 = ConvLayer(32, 4, 3, last=nn.LeakyReLU)
        # self.blk_16_32_5 = ConvLayer(16, 32, 5, last=nn.LeakyReLU)
        # self.blk_17_24_3 = ConvLayer(17, 24, 3, last=nn.LeakyReLU)
        # self.blk_9_24_3 = ConvLayer(9, 24, 3, last=nn.LeakyReLU)
        # self.blk_32_64_3 = ConvLayer(32, 64, 3, last=nn.LeakyReLU)
        # self.blk_64_32_3 = ConvLayer(64, 32, 3, last=nn.LeakyReLU)

        # self.blk_32_16_3 = ConvLayer(32, 16, 3, last=nn.LeakyReLU)
        # self.blk_32_32_3 = ConvLayer(32, 32, 3, last=nn.LeakyReLU)
        # self.blk_32_32_5 = ConvLayer(32, 32, 5, last=nn.LeakyReLU)
        # self.blk_32_32_7 = ConvLayer(32, 32, 7, last=nn.LeakyReLU)
        # self.blk_16_16_7 = ConvLayer(16, 16, 7, last=nn.LeakyReLU)
        ####################
        # self.block3 = ResidualBlock(64)
        # self.blk_32_64_7 = ConvLayer(32, 64, 7, last=nn.LeakyReLU)
        # self.blk_64_128_7 = ConvLayer(64, 128, 7, last=nn.LeakyReLU)
        # self.blk_res_64 = ResidualBlock(64)
        ################
        # self.se_layer_32 = SELayer(32)
        # self.se_layer_16 = SELayer(16)
        # self.blk_res_32 = ResidualBlock(32)
        # self.blk_res_64 = ResidualBlock(64)
        #
        # self.blk_res_16 = ResidualBlock(16)
        # # self.blk_64_64_5 = ConvLayer(64, 64, 5, last = nn.LeakyReLU)
        # # self.blk_64_32_5 = ConvLayer(64, 32, 5, last=nn.LeakyReLU)
        # self.blk_32_16_5 = ConvLayer(32, 16, 5, last=nn.LeakyReLU)
        # self.blk_16_16_3 = nn.Sequential(
        #     nn.Conv2d(16, 16, kernel_size=3, padding=1),
        #     nn.PReLU()
        # )
        #
        # blk_16_4_3 = [nn.Conv2d(16, 4, kernel_size=3, padding=1)]
        # self.blk_16_4_3 = nn.Sequential(*blk_16_4_3)
        #
        # blk_16_8_3 = [nn.Conv2d(16, 8, kernel_size=3, padding=1)]
        # self.blk_16_8_3 = nn.Sequential(*blk_16_8_3)
        #
        # self.se = SELayer(64, )

    #######################
    def forward(self, ms_up, ms_org, pan):
        # data_cat = torch.cat([ms_up, pan], dim=1)
        # temp = data_cat[:, 0:3, :,:]
        # pan4 = pan.repeat(1, 4, 1, 1)
        # pan_d = nn.functional.interpolate(pan, scale_factor=0.5)
        # ms_up2 = nn.functional.interpolate(ms_org, scale_factor=2)
        # detail = pan4 - ms_up
        # ms_org_1 = self.blk_4_32_3(ms_org)
        ms_org_up = self.up_sample4(ms_org)  ## in_channels, in_channels * up_scale ** 2
        ms_up2 = self.up_sample2(ms_org)

        data1 = torch.cat([ms_org_up, pan], dim=1)
        # ms_up4 = self.up_sample2(ms_up2)
        pan_conv = self.blk_9_16_3(data1)
        ms_up_conv = self.blk_8_16_3(ms_up)
        # pand_d = self.downscale(pan)

        out1 = torch.cat([ms_up_conv,  pan_conv], dim=1)

        out2 = self.lu_block1(out1)

        out3 = self.blk2(out2)
        out3 = self.lu_block2(out3)
        # out3 = self.lu_lacrb2(out2)
        # out3 = self.lu_lacrb2(out3)

        # out1 = self.blk1(data)
        # out1 = self.conv1(out1)

        # out1 = torch.cat([ms_up, pan], dim=1)
        # out1_conv = self.blk_9_16_1(out1)
        #
        # out2 = torch.cat([ms_org_up, pan], dim=1)
        # out2_conv = self.blk_9_16_1(out2)
        #
        # out3 = torch.cat([ms_up4, pan], dim=1)
        # # out3 = self.up_sample3(out3)
        # out3_conv = self.blk_9_16_1(out3)

        # out3 = torch.cat([out1_conv, out2_conv, out3_conv], dim=1)   # 48
        # out3 = torch.cat([out1_conv, out2_conv, out3_conv], dim=1)  # 48

        # conv1 = self.conv1(out3)   #48
        # out_b1 = self.lu_block1(out1)  ## 48

        # out_b1 = self.lu_lacrb(out_b1)
        # out_b1 = self.selayer1(out_b1)
        # out_b1 = out_b1 + out3

        # conv4 = self.blk2(out3)  ##36
        # conv2 = pan_conv -conv2
        # conv2 = self.lu_lacrb1(conv2)
        # conv2 = self.lu_block2(conv2)  ##36

        # out_b3 = self.lu_lacrb1(conv2)
        # out_b3 = self.lu_lacrb2(out_b3)
        # out1 = conv2 + out_b1
        # out_b2 = self.lu_block2(conv2)  ##36
        # out_b2 = self.selayer1(out_b2)
        # out_b2 = out_b2 + out3

        # conv3 = self.conv3(out_b2)  ##24
        # out_b3 = self.lu_block3(conv3)  ##24

        # out_b3 = self.lu_block1(out_b2)  ##36
        # out_b3 = self.selayer1(out_b3)

        # conv2 = self.blk_2(out_b2)
        out8 = self.conv6(out3)
        # out_f = (torch.tanh(out8) + 1) / 2

        # pan_24 = self.blk_1_24_3(pan)

        # up_sample2 = UpsampleBLock(4,2).cuda()
        # down_sample2 = UpsampleBLock(4,0.5)

        # ms_up2 = nn.functional.interpolate(ms_org, scale_factor=2, mode='nearest')
        # pan_d2 = nn.functional.interpolate(pan, scale_factor=0.5, mode='nearest')
        # pan_d4 = pan_d2.repeat(1, 4, 1, 1)
        # # ms_up2 = up_sample2(ms_org)
        # # pan_d2 = down_sample2(pan)

        # out_d_5 = torch.cat([ms_up, pan], dim=1)
        # ms_up_1 = self.blk_5_16_3(out_d_5)
        # out_d_32 = self.blk_5_32_3(out_d_5)
        # out_d_32 = self.se_layer_32(out_d_32)
        # out_d_16 = self.blk_32_16_3(out_d_32)
        # out_d_4 = self.blk_16_4_3(out_d_16)
        # out_d_4 = torch.add(pan_d4, out_d_4)

        # out_d_4 = self.mulnet(ms_up_1)
        # out_d_4 = torch.add(ms_up, out_d_4)
        # out_d_4 = (torch.tanh(out_d_4) + 1) / 2

        # detail_16 = self.blk_4_16_3(detail)
        # ms_16 = self.blk_4_16_3(ms_up)
        # out32 = torch.cat([detail_16, ms_16], dim=1)
        # ms_org_16 = self.blk_4_16_3(ms_up2)
        # pan_16 = self.blk_4_16_3(pan4)

        # ms_up2 = self.subpixel_up(ms_org_16)
        # ms_up2 = self.blk_4_16_3(ms_up2)
        # pan_d = self.downscale(pan_16)

        # out5 = torch.cat([ms_up2, pan_d], dim=1)
        # out32 = self.blk_5_32_3(out5)
        # out32 = self.se_layer_32(out32)
        # out16 = self.blk_32_16_3(out32)
        # out_4 = self.blk_16_4_3(out16)

        ########## second step ######################
        # out_4_up2 = nn.functional.interpolate(out_d_4, scale_factor=2)
        # out_16 = self.blk_4_16_3(out_4)
        # ms_up_2 = self.subpixel_up(out16)
        # out9 = torch.cat([ms_up_2, ms_up, pan], dim=1)
        ########### original #########################
        # out5 = torch.cat([ms_up, ms_org_up, pan], dim=1)
        # ms_up_2 = self.blk_17_24_3(out5)

        # out_48 = self.lu_block1(ms_up_2)
        # out_24 = self.blk_48_24_3(out_48)
        # out_24 = out_24 + pan_24
        # # out_add_24 = torch.add(out_24, ms_up_2)
        # out_48 = self.lu_block1(out_24)
        # out_8 = self.conv6(out_48)

        # ms_up_2 = self.blk_9_32_3(out5)
        #
        # out6 = torch.cat([ms_org_up, pan], dim=1)
        # ms_up_3 = self.blk_5_16_3(out6)

        # ms_up_4 = torch.cat([ms_up_2, ms_up_3], dim=1)
        # ms_up_2 = self.blk_16_32_5(ms_up_2)

        # out_8 = self.mulnet(ms_up_2)
        out_f = out8 + ms_up

        # out32 = self.blk_5_32_3(out5)
        # out32 = self.se_layer_32(out32)
        # out32 = self.blk_res_32(out32)
        # out16 = self.blk_5_16_3(out5)
        # out4 = self.blk_32_4_3(out32)

        # out_cat8 = torch.cat([ms_up, out4], dim=1)

        # out16 = self.blk_5_16_3(out5)
        # out16 = self.blk_res_16(out16)

        # out5 = torch.cat([out_4, pan], dim=1)
        # out16 = self.blk_5_16_3(out5)
        # out_4 = self.mulnet(out16)

        # # out16 = self.blk_res_16(out16)
        # out32 = self.blk_16_32_5(out16)
        # out32 = self.blk_5_32_3(out5)
        # # out64 = self.blk_32_64_3(out32)
        # # out64 = self.se(out64)
        # # out64 = self.blk_res_64(out64)
        #
        # # out32 = self.blk_64_32_3(out64)
        # # out32 = torch.add(out32, detail32)
        # # out_32 = self.blk_5_32_3(data_cat)
        # # out_32 = self.blk_4_32_3(detail)
        # out_32 = self.se_layer_32(out32)
        # out_32 = self.blk_res_32(out_32)
        #
        # # out_32 = self.blk_res_32(out_32)
        # out_16 = self.blk_32_16_5(out_32)
        # out_16 = self.blk_res_16(out_16)
        # out_4 = self.blk_16_4_3(out_16)
        # out_4 = torch.add(ms_up, out_4)

        # out_1 = (torch.tanh(out_4) + 1) / 2
        # out16 = self.blk_5_16_3(out5)

        # out_f = torch.add(ms_up, out_m)
        # out_f[out_f>1] = 1
        # pan32 = self.blk_4_32_3(pan4)
        # pan_32 = self.se_layer_32(pan32)
        # pan_16 = self.blk_32_16_3(pan_32)
        # out_4 = self.blk_16_4_3(pan_16)
        # out2 = out1 + out_4

        # ### 编码先 32 通道，解码要16通道
        # ms_32 = self.blk_4_32_3(ms_up)
        # pan_32 = self.blk_4_32_3(pan4)
        # # pan_res1 = self.blk_res_32(pan_32)
        # # pan_res2 = self.blk_res_32(pan_res1)
        #
        # # detail_32 = self.blk_4_32_3(detail)
        # # out_32 = self.blk_5_32_3(data_cat)
        #
        # pan_32_3 = self.blk_32_32_5(pan_32)
        # # out_32_5 = self.blk_32_32_5(out_32)
        #
        # out_32_7 = self.blk_32_32_7(pan_32_3)
        # #
        # out_32_5 = self.se_layer_32(out_32_7)
        # # # out_32_3 = self.blk_32_32_3(out_32_5)
        #
        # out_32_3 = torch.add(out_32_5, ms_32)  # add pan information
        # # out_32_3 = torch.add(pan_32_3, out_32_3)  # add pan information
        # # out_32_5 = self.blk_32_32_5(out_32_7)
        #
        # out_16_3 = self.blk_32_16_3(out_32_3)
        # # out_16_3 = out_16_3 + ms_16
        # out_4_1 = self.blk_16_4_3(out_16_3)
        # # out1 = nn.Tanh(out_4_1)
        # out_4_1 = ms_up + out_4_1
        # out_f = (torch.tanh(out_f) + 1) / 2

        # mtf = MTF_Kenels(sate)
        # out2 = mtf(out_f)
        # out_d = nn.functional.interpolate(out2, scale_factor=0.25, mode='nearest')
        # out2 = ms + out1
        # out2 = torch.add(ms, out1)
        return out_f
        # return out_f, out2, out_d


class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(channels)
        self.prelu = nn.PReLU()  # 负数部分的参数会变
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x):
        residual = self.conv1(x)
        residual = self.bn1(residual)
        residual = self.prelu(residual)
        residual = self.conv2(residual)
        residual = self.bn2(residual)

        return x + residual


class ConvLayer(nn.Module):
    def __init__(self, in_channels=1, out_channels=16, kernel_size=5, last=nn.ReLU):
        super().__init__()
        if kernel_size == 5:
            padding = 2
        elif kernel_size == 7:
            padding = 3
        elif kernel_size == 3:
            padding = 1
        else:
            padding = 0
        self.main = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=1, padding=padding),
            nn.BatchNorm2d(out_channels),
            last()
        )

    def forward(self, x):
        out = self.main(x)
        return out


# 通道注意力
class SELayer(nn.Module):
    def __init__(self, channel, reduction_ratio=4):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, int(channel / reduction_ratio)),
            nn.ReLU(inplace=True),
            nn.Linear(int(channel / reduction_ratio), channel),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y


class UpsampleBLock(nn.Module):
    def __init__(self, in_channels, up_scale):
        super(UpsampleBLock, self).__init__()
        self.conv = nn.Conv2d(in_channels, in_channels * up_scale ** 2, kernel_size=3, padding=1)
        self.pixel_shuffle = nn.PixelShuffle(up_scale)
        self.prelu = nn.PReLU()

    def forward(self, x):
        x = self.conv(x)
        x = self.pixel_shuffle(x)
        x = self.prelu(x)
        return x


class MTF_Kenels(nn.Module):
    def __init__(self, sate, channels=8):
        super(MTF_Kenels, self).__init__()
        self.sate = sate
        self.channels = channels
        if sate == 'ik':
            ms_kernel_name = './kernels/IK_ms_kernel.mat'  # read the corresponding multispectral kernel (WorldView-3
        if sate == 'pl':
            ms_kernel_name = './kernels/none_ms_kernel.mat'  # read the corresponding multispectral kernel (WorldView-3
        if sate == 'wv3_8':
            ms_kernel_name = './kernels/WV3_ms_kernel.mat'  # read the corresponding multispectral kernel (WorldView-3

        # (7x7x8x8), QuickBird and GaoFen-2 (7x7x4x4))
        ms_kernel = sio.loadmat(ms_kernel_name)
        ms_kernel = ms_kernel['ms_kernel_raw'][...]
        kernel = np.array(ms_kernel, dtype=np.float32)

        # kernel = [[0.00078633, 0.00655965, 0.01330373, 0.00655965, 0.00078633],
        #           [0.00655965, 0.05472157, 0.11098164, 0.05472157, 0.00655965],
        #           [0.01330373, 0.11098164, 0.22508352, 0.11098164, 0.01330373],
        #           [0.00655965, 0.05472157, 0.11098164, 0.05472157, 0.00655965],
        #           [0.00078633, 0.00655965, 0.01330373, 0.00655965, 0.00078633]]
        # kernel = torch.FloatTensor(kernel).unsqueeze(1)
        kernel = ToTensor()(kernel).unsqueeze(1)
        if torch.cuda.is_available():
            kernel = kernel.cuda()
        # kernel = torch.FloatTensor(kernel)
        # kernel = np.repeat(kernel, self.channels, axis=0)
        self.weight = nn.Parameter(data=kernel, requires_grad=False)

    def __call__(self, x):
        # x = F.conv2d(x.unsqueeze(0), self.weight, padding=2, groups=self.channels)
        x = F.pad(x, (3, 3, 3, 3), mode='replicate')
        x = F.conv2d(x, self.weight, groups=self.channels)  # for 4 * 1 * 7 * 7 conv, padding should be 3
        return x


def variance_scaling_initializer(tensor):
    from scipy.stats import truncnorm

    def truncated_normal_(tensor, mean=0, std=1):
        with torch.no_grad():
            size = tensor.shape
            tmp = tensor.new_empty(size + (4,)).normal_()
            valid = (tmp < 2) & (tmp > -2)
            ind = valid.max(-1, keepdim=True)[1]
            tensor.data.copy_(tmp.gather(-1, ind).squeeze(-1))
            tensor.data.mul_(std).add_(mean)
            return tensor

    def variance_scaling(x, scale=1.0, mode="fan_in", distribution="truncated_normal", seed=None):
        fan_in, fan_out = torch.nn.init._calculate_fan_in_and_fan_out(x)
        if mode == "fan_in":
            scale /= max(1., fan_in)
        elif mode == "fan_out":
            scale /= max(1., fan_out)
        else:
            scale /= max(1., (fan_in + fan_out) / 2.)
        if distribution == "normal" or distribution == "truncated_normal":
            # constant taken from scipy.stats.truncnorm.std(a=-2, b=2, loc=0., scale=1.)
            stddev = math.sqrt(scale) / .87962566103423978
        # print(fan_in,fan_out,scale,stddev)#100,100,0.01,0.1136
        truncated_normal_(x, 0.0, stddev)
        return x / 10 * 1.28

    variance_scaling(tensor)

    return tensor


def init_weights(*modules):
    for module in modules:
        for m in module.modules():
            if isinstance(m, nn.Conv2d):
                # try:
                #     import tensorflow as tf
                #     tensor = tf.get_variable(shape=m.weight.shape, initializer=tf.variance_scaling_initializer(seed=1))
                #     m.weight.data = tensor.eval()
                # except:
                #     print("try error, run variance_scaling_initializer")
                # variance_scaling_initializer(m.weight)
                variance_scaling_initializer(m.weight)
                # nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1.0)
                nn.init.constant_(m.bias, 0.0)
            elif isinstance(m, nn.Linear):
                # variance_scaling_initializer(m.weight)
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)


class Exp_block(nn.Module):
    def __init__(self, out_channels):
        super(Exp_block, self).__init__()
        # self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1,
        #                        bias=True)  # change in-channel from 16 to 8

        self.conv2_1 = nn.Conv2d(in_channels=out_channels, out_channels=out_channels // 3, kernel_size=3, stride=1,
                                 padding=1, bias=True)
        self.conv2_2 = nn.Conv2d(in_channels=out_channels, out_channels=out_channels // 3, kernel_size=3, stride=1,
                                 padding=2, dilation=2, bias=True)
        self.conv2_3 = nn.Conv2d(in_channels=out_channels, out_channels=out_channels // 3, kernel_size=3, stride=1,
                                 padding=3, dilation=3, bias=True)

        # self.conv3_1 = LACRB_lu(out_channels )
        # self.conv3_3 = LACRB_lu(out_channels // 3)
        self.conv3_1 = AdapConv(out_channels, out_channels, 3, 1, 1, use_bias=True)
        self.conv3_2 = AdapConv(out_channels, out_channels, 3, 1, 1, use_bias=True)
        # self.conv3_1 = LACRB(out_channels)
        # self.conv3_2 = LAConv2D(out_channels, out_channels // 3, 3, 1, 1, dilation=2, use_bias=True)
        # self.conv3_3 = LAConv2D(out_channels, out_channels // 3, 3, 1, 1, dilation=3, use_bias=True)
        # self.selayer = SELayer(out_channels)
        # self.attantion = SoftAttn(out_channels)
        # self.conv3_1 = SoftAttn(out_channels)
        # self.conv2_4 = nn.Conv2d(in_channels=out_channels, out_channels=out_channels // 4, kernel_size=9, stride=1,
        #                          padding=4, bias=True)

        self.relu = nn.ReLU(inplace=True)
        # init_weights(self.conv1, self.conv2_1, self.conv2_2, self.conv2_3, self.conv3, self.conv4_1, self.conv4_2,
        #              self.conv4_3, self.conv5)
        init_weights(self.conv2_1, self.conv2_2, self.conv2_3)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, sqrt(2. / n))

    def forward(self, x):
        # out1 = self.relu(self.conv1(x))
        # out1 = self.selayer(out1)
        out1 = x
        out21 = self.conv2_1(out1)
        out22 = self.conv2_2(out1)
        out23 = self.conv2_3(out1)

        # out31 = self.conv3_1(out21)
        # out32 = self.conv3_2(out22)
        # out33 = self.conv3_3(out23)

        # out24 = self.conv2_4(out1)
        # out2 = torch.cat([out21, out22], 1)
        out2 = torch.cat([out21, out22, out23], 1)
        # out2 = torch.cat([out31, out32, out33], 1)
        out2 = self.conv3_1(out2)
        out2 = self.conv3_2(out2)
        # out2 = self.selayer(out2)
        # out2 = self.conv3_1(out2)
        # out2 = self.selayer(out2)
        # out2 = self.attantion(out2)
        # out2 = self.att(out2)
        out2 = self.relu(torch.add(out2, out1))

        # out2 = self.conv3_1(out2)
        #
        return out2


class mscb2(nn.Module):
    def __init__(self):
        super(mscb2, self).__init__()

        self.lu_block1 = Exp_block(24, 48)
        # self.lu_block2 = Exp_block(48, 60)
        # self.lu_block2 = Exp_block(48, 60)
        self.lu_block3 = Exp_block(48, 36)

        # self.conv1 = nn.Conv2d(in_channels=16, out_channels=48, kernel_size=7, stride=1, padding=3,
        #                        bias=True)  # change in-channel from 16 to 8
        # # self.selayer = SELayer(48)
        # self.conv2_1 = nn.Conv2d(in_channels=48, out_channels=32, kernel_size=3, stride=1, padding=1, bias=True)
        # self.conv2_2 = nn.Conv2d(in_channels=48, out_channels=32, kernel_size=5, stride=1, padding=2, bias=True)
        # # self.conv2_3 = nn.Conv2d(in_channels=48, out_channels=16, kernel_size=7, stride=1, padding=3, bias=True)
        #
        # self.conv_lu = nn.Conv2d(in_channels=48, out_channels=64, kernel_size=3, stride=1, padding=1, bias=True)
        # self.conv_lu_1 = nn.Conv2d(in_channels=6, out_channels=64, kernel_size=3, stride=1, padding=1, bias=True)
        # self.conv_lu_2 = nn.Conv2d(in_channels=48, out_channels=64, kernel_size=5, stride=1, padding=2, bias=True)
        #
        # self.conv3 = nn.Conv2d(in_channels=64, out_channels=24, kernel_size=3, stride=1, padding=1, bias=True)
        # self.conv4_1 = nn.Conv2d(in_channels=24, out_channels=12, kernel_size=3, stride=1, padding=1, bias=True)
        # self.conv4_2 = nn.Conv2d(in_channels=24, out_channels=12, kernel_size=5, stride=1, padding=2, bias=True)
        # self.conv4_3 = nn.Conv2d(in_channels=36, out_channels=12, kernel_size=7, stride=1, padding=3, bias=True)

        # self.conv5 = nn.Conv2d(in_channels=30, out_channels=16, kernel_size=5, stride=1, padding=2,
        #                        bias=True)  # change out as 4
        self.conv6 = nn.Conv2d(in_channels=36, out_channels=8, kernel_size=5, stride=1, padding=2,
                               bias=True)  # change out as 4   or   8
        # self.shallow1 = nn.Conv2d(in_channels=16, out_channels=64, kernel_size=9, stride=1, padding=4,
        #                           bias=True)
        # self.shallow2 = nn.Conv2d(in_channels=64, out_channels=32, kernel_size=1, stride=1, padding=0, bias=True)
        # self.shallow3 = nn.Conv2d(in_channels=32, out_channels=8, kernel_size=5, stride=1, padding=2,
        #                           bias=True)
        # self.direconv1 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1, bias=True)
        # self.direconv2 = nn.Conv2d(in_channels=32, out_channels=8, kernel_size=1, stride=1, padding=0, bias=True)
        self.relu = nn.ReLU(inplace=True)
        # init_weights(self.conv1, self.conv2_1, self.conv2_2, self.conv2_3, self.conv3, self.conv4_1, self.conv4_2,
        #              self.conv4_3, self.conv5)
        init_weights(self.conv6)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, sqrt(2. / n))

    def make_layer(self, block, num_of_layer):
        layers = []
        for _ in range(num_of_layer):
            layers.append(block())
        return nn.Sequential(*layers)

    def forward(self, x):
        out1 = self.lu_block1(x)
        # out2 = self.lu_block2(out1)
        out3 = self.lu_block3(out1)
        # out3 = self.lu_block3(out2)
        # out5 = self.conv5(out2)
        out6 = self.conv6(out3)

        # out1 = self.relu(self.conv1(x))
        # # out1 = self.selayer(out1)
        # out21 = self.conv2_1(out1)
        # out22 = self.conv2_2(out1)
        # # out23 = self.conv2_3(out1)
        # out2 = torch.cat([out21, out22], 1)
        # # out2 = torch.cat([out21, out22, out23], 1)
        # out2 = self.relu(torch.add(out2, out1))
        #
        # out3 = self.relu(self.conv3(out2))
        # # out3 = self.conv3(out2)
        # out41 = self.conv4_1(out3)
        # out42 = self.conv4_2(out3)
        # # out43 = self.conv4_3(out3)
        # # out4 = torch.cat([out41, out42, out43], 1)
        # out4 = torch.cat([out41, out42], 1)
        # out4 = self.relu(torch.add(out4, out3))
        # out5 = self.conv5(out4)
        return out6


class AdapConv(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1, groups=1, use_bias=False):
        super(AdapConv, self).__init__()
        self.in_planes = in_planes
        self.out_planes = out_planes
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups
        self.bias = use_bias
        self.ch_att = ChannelAttention(kernel_size ** 2)  # change

        # Generating local adaptive weights
        self.attention1 = nn.Sequential(
            nn.Conv2d(in_planes, kernel_size ** 2, kernel_size, stride, padding),
            # nn.ReLU(inplace=True),
            # nn.Conv2d(kernel_size ** 2, kernel_size ** 2, 1),
            # nn.ReLU(inplace=True),
            # nn.Conv2d(kernel_size ** 2, kernel_size ** 2, 1),  # changed
            # SoftAttn(kernel_size ** 2)
            InvBlock(UNetConvBlock, kernel_size ** 2, kernel_size ** 2 // 2)
            # nn.Sigmoid()  # changed
        )  # b,9,H,W È«Í¨µÀÏñËØ¼¶ºË×¢ÒâÁ¦
        # self.attention2=nn.Sequential(
        #     nn.Conv2d(in_planes,(kernel_size**2)*in_planes,kernel_size, stride, padding,groups=in_planes),
        #     nn.ReLU(inplace=True),
        #     nn.Conv2d((kernel_size**2)*in_planes,(kernel_size**2)*in_planes,1,groups=in_planes),
        #     nn.Sigmoid()
        # ) #b,9n,H,W µ¥Í¨µÀÏñËØ¼¶ºË×¢ÒâÁ¦
        # self.spatt = SpatialAttention(3)
        # if use_bias == True:  # Global local adaptive weights
        #     self.attention3 = nn.Sequential(  # like channel attention
        #         nn.AdaptiveAvgPool2d(1),
        #         nn.Conv2d(in_planes, out_planes, 1),
        #         nn.ReLU(inplace=True),
        #         nn.Conv2d(out_planes, out_planes, 1)  # change
        #     )  # b,m,1,1 Í¨µÀÆ«ÖÃ×¢ÒâÁ¦

        conv1 = nn.Conv2d(in_planes, out_planes, kernel_size, stride, padding, dilation, groups)
        self.weight = conv1.weight  # m, n, k, k

    def forward(self, x):
        (b, n, H, W) = x.shape
        m = self.out_planes
        k = self.kernel_size
        n_H = 1 + int((H + 2 * self.padding - k) / self.stride)
        n_W = 1 + int((W + 2 * self.padding - k) / self.stride)
        atw1 = self.attention1(x)  # b,k*k,n_H,n_W
        # atw2=self.attention2(x) #b,n*k*k,n_H,n_W

        atw1 = atw1.permute([0, 2, 3, 1])  # b,n_H,n_W,k*k
        atw1 = atw1.unsqueeze(3).repeat([1, 1, 1, n, 1])  # b,n_H,n_W,n,k*k
        atw1 = atw1.view(b, n_H, n_W, n * k * k)  # b,n_H,n_W,n*k*k

        # atw2=atw2.permute([0,2,3,1]) #b,n_H,n_W,n*k*k

        atw = atw1  # *atw2 #b,n_H,n_W,n*k*k
        atw = atw.view(b, n_H * n_W, n * k * k)  # b,n_H*n_W,n*k*k
        atw = atw.permute([0, 2, 1])  # b,n*k*k,n_H*n_W

        kx = F.unfold(x, kernel_size=k, stride=self.stride, padding=self.padding)  # b,n*k*k,n_H*n_W
        atx = atw * kx  # b,n*k*k,n_H*n_W

        atx = atx.permute([0, 2, 1])  # b,n_H*n_W,n*k*k
        atx = atx.view(1, b * n_H * n_W, n * k * k)  # 1,b*n_H*n_W,n*k*k

        w = self.weight.view(m, n * k * k)  # m,n*k*k
        w = w.permute([1, 0])  # n*k*k,m
        y = torch.matmul(atx, w)  # 1,b*n_H*n_W,m
        y = y.view(b, n_H * n_W, m)  # b,n_H*n_W,m
        # if self.bias == True:
        #     bias = self.attention3(x)  # b,m,1,1
        #     bias = bias.view(b, m).unsqueeze(1)  # b,1,m
        #     bias = bias.repeat([1, n_H * n_W, 1])  # b,n_H*n_W,m
        #     y = y + bias  # b,n_H*n_W,m

        y = y.permute([0, 2, 1])  # b,m,n_H*n_W
        y = F.fold(y, output_size=(n_H, n_W), kernel_size=1)  # b,m,n_H,n_W
        return y


class LACRB_lu(nn.Module):
    def __init__(self, in_planes):
        super(LACRB_lu, self).__init__()
        self.conv1 = AdapConv(in_planes, in_planes, 3, 1, 1, use_bias=True)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv2 = AdapConv(in_planes, in_planes, 3, 1, 1, use_bias=True)

    def forward(self, x):
        res = self.conv1(x)
        res = self.relu1(res)
        res = self.conv2(res)
        x = x + res
        return x