import torch
import torch.nn as nn
from models.models_others import ChannelAttention, SpatialAttention, Eca_layer
import torch.nn.functional as F
from models.model_4_awf import AdapConv
from models.WSDformer_model import WSDLayer
from models.Inv_block import InvBlock, DenseBlock,UNetConvBlock


class Dense_Block(nn.Module):
    def __init__(self, in_channels, growth_rate, n_layers):
        super(Dense_Block, self).__init__()
        self.layers = nn.ModuleList()
        for i in range(n_layers):
            self.layers.append(nn.Sequential(
                nn.BatchNorm2d(in_channels),
                nn.ReLU(inplace=True),
                nn.Conv2d(in_channels, growth_rate, kernel_size=3, padding=1),
                nn.BatchNorm2d(growth_rate),
                nn.ReLU(inplace=True),
                nn.Conv2d(growth_rate, growth_rate, kernel_size=1)
            ))
            in_channels += growth_rate

    def forward(self, x):
        for layer in self.layers:
            x = torch.cat([x, layer(x)], dim=1)
        return x


class ChannelAttention2(nn.Module):
    def __init__(self, in_channels, reduction_ratio=4):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Conv2d(in_channels, in_channels // reduction_ratio, 1, bias=False)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Conv2d(in_channels // reduction_ratio, in_channels, 1, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        y = self.avg_pool(x)
        y = self.fc1(y)
        y = self.relu(y)
        y = self.fc2(y)
        y = self.sigmoid(y)
        return x * y


class AttentionModule(nn.Module):
    def __init__(self, in_channels, out_channels, attention_type):
        super(AttentionModule, self).__init__()
        if attention_type == 'pixel':
            self.attention = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1),
                nn.Conv2d(out_channels, in_channels, kernel_size=1),
                nn.Sigmoid()
            )
        elif attention_type == 'channel':
            self.attention = nn.Sequential(
                nn.AdaptiveAvgPool2d(1),
                nn.Conv2d(in_channels, in_channels, kernel_size=1),
                nn.Sigmoid()
            )
        elif attention_type == 'spatial':
            self.attention = nn.Sequential(
                nn.Conv2d(in_channels, 1, kernel_size=1),
                nn.Sigmoid()
            )
        else:
            raise ValueError('Unknown attention type')

    def forward(self, x):
        attention = self.attention(x)
        return x * attention


class ThreeBranchModule(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ThreeBranchModule, self).__init__()
        self.branch1 = AttentionModule(in_channels, out_channels, 'pixel')  ## out_channels is only a middle para
        # self.branch2 = AttentionModule(in_channels, 'channel')
        self.branch2 = ChannelAttention(in_channels)
        self.branch3 = SpatialAttention(in_channels)
        # self.branch3 = AttentionModule(in_channels, 'spatial')
        # self.branch4 = DenseBlock(in_channels, growth_rate, n_layers)

        self.conv1 = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(in_channels * 2, in_channels * 2, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(in_channels * 3, in_channels * 3, kernel_size=3, padding=1)
        # self.conv4 = nn.Conv2d(in_channels*4, in_channels*4, kernel_size=3)

    def forward(self, x):
        branch1 = self.branch1(x)
        out1 = self.conv1(branch1)
        branch2 = self.branch2(x)
        branch2 = x * branch2
        out2 = torch.cat([out1, branch2], dim=1)
        out2 = self.conv2(out2)
        branch3 = self.branch3(x)
        branch3 = x * branch3
        out3 = torch.cat([out2, branch3], dim=1)
        out3 = self.conv3(out3)
        # branch4 = self.branch4(x)
        return out3


class FourBranchModule(nn.Module):
    def __init__(self, in_channels, out_channels, growth_rate, n_layers):
        super(FourBranchModule, self).__init__()
        self.branch1 = AttentionModule(in_channels, out_channels, 'pixel')
        # self.branch2 = AttentionModule(in_channels, 'channel')
        self.branch2 = ChannelAttention(in_channels)
        self.branch3 = SpatialAttention(in_channels)
        # self.branch3 = AttentionModule(in_channels, 'spatial')
        self.dense = Dense_Block(in_channels * 3, growth_rate, n_layers)

        self.conv1 = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(in_channels * 2, in_channels * 2, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(in_channels * 3, in_channels * 3, kernel_size=3, padding=1)
        # self.conv4 = nn.Conv2d(in_channels*4, in_channels*4, kernel_size=3)

    def forward(self, x):
        branch1 = self.branch1(x)
        out1 = self.conv1(branch1)
        branch2 = self.branch2(x)
        branch2 = x * branch2
        out2 = torch.cat([out1, branch2], dim=1)
        out2 = self.conv2(out2)
        branch3 = self.branch3(x)
        branch3 = x * branch3
        out3 = torch.cat([out2, branch3], dim=1)
        out3 = self.conv3(out3)
        dense_out = self.dense(out3)
        return dense_out
        # return torch.cat([out3, branch4], dim=1)


class FeaEx(nn.Module):
    def __init__(self, in_channels, out_channel):
        super(FeaEx, self).__init__()
        # Define number of input channels
        self.in_channels = in_channels

        # First level convolutions
        self.conv_16_1 = nn.Conv2d(in_channels=self.in_channels, out_channels=out_channel, kernel_size=3, padding=1)
        self.bn_16_1 = nn.BatchNorm2d(out_channel)
        self.LeakyReLU = nn.LeakyReLU(negative_slope=0.0)

    def forward(self, x):
        out1 = self.LeakyReLU(self.bn_16_1(self.conv_16_1(x)))
        return out1


class Decoder(nn.Module):
    def __init__(self, in_channels):
        super(Decoder, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, in_channels // 2, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(in_channels // 2, in_channels // 4, kernel_size=3, dilation=2, padding=2)
        self.conv3 = nn.Conv2d(in_channels // 2, in_channels // 4, kernel_size=3, dilation=3, padding=3)
        # self.conv2 = nn.Conv2d(in_channels // 2, in_channels // 4, kernel_size=5, padding=2)
        # self.conv3 = nn.Conv2d(in_channels // 2, in_channels // 4, kernel_size=7, padding=3)
        self.conv4 = nn.Conv2d(in_channels // 2, in_channels // 4, kernel_size=3, padding=1)
        self.conv5 = nn.Conv2d(in_channels // 2, in_channels // 4, kernel_size=1)

        self.conv6 = nn.Conv2d(in_channels, in_channels // 2, kernel_size=1)
        # self.conv4 = nn.Conv2d(in_channels // 2, outchanels, kernel_size=3, padding=1)
        self.CA = ChannelAttention2(in_channels)

    def forward(self, x):
        x1 = F.leaky_relu(self.conv1(x))

        x2 = F.leaky_relu(self.conv2(x1))
        x3 = F.leaky_relu(self.conv3(x1))
        x4 = F.leaky_relu(self.conv4(x1))
        x5 = F.leaky_relu(self.conv5(x1))

        x4 = torch.cat([x3, x2, x4, x5], dim=1)
        x5 = self.CA(x4)
        x6 = x + x5
        x7 = self.conv6(x6)

        # x = self.conv4(x4)
        return x7


class MainNet(nn.Module):

    def __init__(self):
        super(MainNet, self).__init__()
        num_channel = 8 # changed to 4
        num_feature = 16
        num_dec = num_feature * 3
        self.org = FeaEx(num_channel + 1, num_feature)
        # self.org2 = FeaEx(num_channel + 1, num_feature)
        self.multi_bch1 = ThreeBranchModule(num_feature, 8)
        # self.multi_bch1 = FourBranchModule(num_feature, 8, 8,2)
        # self.multi_bch2 = ThreeBranchModule(num_feature, 8)
        self.adapconv1 = AdapConv(num_feature, kernel_size=3, padding=1, out_planes=num_feature)

        # self.spa_process = nn.Sequential(InvBlock(DenseBlock, num_feature, num_feature),
        #                                  nn.Conv2d(2 * channels, channels, 1, 1, 0))
        self.spa_inv = InvBlock(UNetConvBlock, num_feature, num_feature//2)

        self.adapconv2 = AdapConv(num_dec, kernel_size=3, padding=1, out_planes=num_dec)
        self.attlayer = WSDLayer(
            dim=num_feature,
            depth=1,
            num_heads=4,
            input_resolution=(80, 80),
            local_ws=4,

        )
        self.attlayer2 = WSDLayer(
            dim=num_dec,
            depth=1,
            num_heads=4,
            input_resolution=(80, 80),
            local_ws=5,
        )
        # self.conv_att = nn.Conv2d(num_feature, num_dec, kernel_size=3, padding=1)
        self.fc = nn.Linear(num_feature, num_dec, bias=False)

        # self.dense1 = DenseBlock(num_feature * 3, 8, 1)
        # self.dense2 = DenseBlock(num_feature * 3, 8, 1)
        self.decoder = Decoder(num_dec)
        self.conv4 = nn.Conv2d(num_dec // 2, num_channel, kernel_size=3, padding=1)



    def forward(self, ms_up, ms_org, pan):
        # pan_d = F.interpolate(pan, scale_factor=(1 / 4, 1 / 4), mode='bilinear')
        # ms_d = F.interpolate(ms_org, scale_factor=(1 / 4, 1 / 4), mode='bilinear')
        # ms_d_up = F.interpolate(ms_d, scale_factor=(4, 4), mode='bilinear')

        UP_LRHSI = ms_up
        sz = UP_LRHSI.size(2)
        Input = torch.cat((UP_LRHSI, pan), 1)
        # Input_d = torch.cat((ms_d_up, pan_d), 1)

        f1 = self.org(Input)
        f1_aconv = self.adapconv1(f1)
        f1_att = self.attlayer(f1)
        f1_out = f1_att * f1_aconv
        f1_out = f1_out + f1

        # f1_out = self.spa_inv(f1_out)


        # f1_d = self.org2(Input_d)

        # B, C, H, W = f1_out.shape
        # f1_fc = f1_out.flatten(2).transpose(1, 2).contiguous()
        # f1_fc = self.fc(f1_fc)
        # f1_fc = f1_fc.transpose(1, 2).view(B, C * 3, H, W)

        # att1_out = self.conv_att(f1_out)
        # f1_out = f1_out + att1_out

        out1 = self.multi_bch1(f1_out)
        f2_aconv = self.adapconv2(out1)
        f2_att = self.attlayer2(f2_aconv)
        out2 = f2_att * f2_aconv
        out2 = out1 + out2

        # out1_d1 = F.interpolate(out1, scale_factor=(1 / 4, 1 / 4), mode='bilinear')
        # out1_d = self.multi_bch2(f1_d)
        # out1_d_up = F.interpolate(out1_d, scale_factor=(4, 4), mode='bilinear')

        # out1 = out1 + out1_d_up
        # out1_d = out1_d + out1_d1

        # out1_dense = self.dense1(out2)
        # out1_dense_d = self.dense2(out1_d)
        # out1_dense_up = F.interpolate(out1_dense_d, scale_factor=(4, 4), mode='bilinear')

        # out_dense = out1_dense + out1_dense_up

        out_de = self.decoder(out2)
        Highpass = self.conv4(out_de)
        output = Highpass + UP_LRHSI
        return output
