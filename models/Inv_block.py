from .modules import InvertibleConv1x1
import torch
import torch.nn as nn
import torch.nn.init as init

class Eca_layer(nn.Module):
    """Constructs a ECA module.
    Args:
        channel: Number of channels of the input feature map
        k_size: Adaptive selection of kernel size
    """

    def __init__(self, channel, k_size=7):
        super(Eca_layer, self).__init__()
        # self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)  # change
        self.conv = nn.Conv1d(1, 1, kernel_size=k_size, padding=(k_size - 1) // 2, bias=False)
        # self.conv1 = nn.Conv2d(2, 1, kernel_size=k_size, padding=(k_size - 1) // 2, bias=False)  # 保持卷积前后H、W不变 change
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # x: input features with shape [b, c, h, w]
        b, c, h, w = x.size()

        # feature descriptor on the global spatial information
        # y1 = self.avg_pool(x)
        ########## change
        y2 = self.max_pool(x)  # change
        y =  y2  # change

        # x = torch.cat([y1, y2], dim=1)
        # print(x.shape)
        # y = self.conv1(x)

        # Two different branches of ECA module
        y = self.conv(y.squeeze(-1).transpose(-1, -2)).transpose(-1, -2).unsqueeze(-1)

        # Multi-scale information fusion
        y = self.sigmoid(y)

        return y*x

        # return x * y.expand_as(x)


class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ration=16):
        super(ChannelAttention, self).__init__()

        '''
        AdaptiveAvgPool2d():自适应平均池化
                            不需要自己设置kernelsize stride等
                            只需给出输出尺寸即可
        '''
        k_size = 7
        self.avg_pool = nn.AdaptiveAvgPool2d(1)  # 通道数不变，H*W变为1*1
        self.max_pool = nn.AdaptiveMaxPool2d(1)  #

        self.fc1 = nn.Conv2d(in_planes, in_planes // 4, 1, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Conv2d(in_planes // 4, in_planes, 1, bias=False)

        # self.conv = nn.Conv1d(1, 1, kernel_size=k_size, padding=(k_size - 1) // 2, bias=False) # change
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        # print(avg_out.shape)
        # 两层神经网络共享
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        # print(avg_out.shape)
        # print(max_out.shape)
        out = avg_out + max_out

        # out = self.conv(out.squeeze(-1).transpose(-1, -2)).transpose(-1, -2).unsqueeze(-1) # change
        # print(out.shape)
        return self.sigmoid(out)


''''
空间注意力模块
        先分别进行一个通道维度的最大池化和平均池化得到两个H x W x 1，
        然后两个描述拼接在一起，然后经过一个7*7的卷积层，激活函数为sigmoid，得到权重Ms

'''


class SpatialAttention(nn.Module):
    def __init__(self, in_planes, kernel_size=7):
        super(SpatialAttention, self).__init__()

        assert kernel_size in (3, 7), " kernel size must be 3 or 7"
        padding = 3 if kernel_size == 7 else 1
        # avg 和 max 两个描述，叠加 共两个通道。
        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)  # 保持卷积前后H、W不变
        self.sigmoid = nn.Sigmoid()

        # self.avgpool = nn.AvgPool2d(2, stride=2)  #### change ##############
        # self.conv2 = nn.Conv2d(in_planes, 1, kernel_size, padding=padding, bias=False)  #### change ##############

    def forward(self, x):
        # (b, n, H, W) = x.shape
        # y1 = self.avgpool(x)  #### change ##############
        # y2 = self.conv2(y1)  # change  1 channel
        # t = nn.Upsample(scale_factor=2)  #### change ##############
        # y2 = t(y2)  #### change ##############

        # egg：input: 1 , 3 * 2 * 2  avg_out :
        avg_out = torch.mean(x, dim=1, keepdim=True)  # 通道维度的平均池化
        # 注意 torch.max(x ,dim = 1) 返回最大值和所在索引，是两个值  keepdim = True 保持维度不变（求max的这个维度变为1），不然这个维度没有了
        max_out, _ = torch.max(x, dim=1, keepdim=True)  # 通道维度的最大池化
        # print(avg_out.shape)
        # print(max_out.shape)
        y = torch.cat([avg_out, max_out], dim=1)
        # print(x.shape)
        y = self.conv1(y)
        # x = x + y2  # change
        return self.sigmoid(y)*x

class UNetConvBlock(nn.Module):
    def __init__(self, in_size, out_size, d, relu_slope=0.1):
        super(UNetConvBlock, self).__init__()
        self.identity = nn.Conv2d(in_size, out_size, 1, 1, 0)

        self.conv_1 = nn.Conv2d(in_size, out_size, kernel_size=3, dilation=d, padding=d, bias=True)
        self.relu_1 = nn.LeakyReLU(relu_slope, inplace=False)
        self.conv_2 = nn.Conv2d(out_size, out_size, kernel_size=3, dilation=d, padding=d, bias=True)
        self.relu_2 = nn.LeakyReLU(relu_slope, inplace=False)

    def forward(self, x):
        out = self.relu_1(self.conv_1(x))
        out = self.relu_2(self.conv_2(out))
        out += self.identity(x)

        return out

def initialize_weights_xavier(net_l, scale=1):
    if not isinstance(net_l, list):
        net_l = [net_l]
    for net in net_l:
        for m in net.modules():
            if isinstance(m, nn.Conv2d):
                init.xavier_normal_(m.weight)
                m.weight.data *= scale  # for residual block
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                init.xavier_normal_(m.weight)
                m.weight.data *= scale
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias.data, 0.0)

def initialize_weights(net_l, scale=1):
    if not isinstance(net_l, list):
        net_l = [net_l]
    for net in net_l:
        for m in net.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, a=0, mode='fan_in')
                m.weight.data *= scale  # for residual block
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                init.kaiming_normal_(m.weight, a=0, mode='fan_in')
                m.weight.data *= scale
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias.data, 0.0)

class DenseBlock(nn.Module):
    def __init__(self, channel_in, channel_out, d = 1, init='xavier', gc=8, bias=True):
        super(DenseBlock, self).__init__()
        self.conv1 = UNetConvBlock(channel_in, gc, d)
        self.conv2 = UNetConvBlock(gc, gc, d)
        self.conv3 = nn.Conv2d(channel_in + 2 * gc, channel_out, 3, 1, 1, bias=bias)
        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

        if init == 'xavier':
            initialize_weights_xavier([self.conv1, self.conv2, self.conv3], 0.1)
        else:
            initialize_weights([self.conv1, self.conv2, self.conv3], 0.1)
        # initialize_weights(self.conv5, 0)

    def forward(self, x):
        x1 = self.lrelu(self.conv1(x))
        x2 = self.lrelu(self.conv2(x1))
        x3 = self.lrelu(self.conv3(torch.cat((x, x1, x2), 1)))

        return x3


class InvBlock(nn.Module):
    def __init__(self, subnet_constructor, channel_num, channel_split_num, d = 1, clamp=0.8):
        super(InvBlock, self).__init__()
        # channel_num: 3
        # channel_split_num: 1

        self.split_len1 = channel_split_num  # 1
        self.split_len2 = channel_num - channel_split_num  # 2

        self.clamp = clamp

        # self.F = subnet_constructor(self.split_len2, self.split_len1, d)
        # self.G = subnet_constructor(self.split_len1, self.split_len2, d)
        self.F = nn.Sequential(SpatialAttention(self.split_len2),
                               nn.Conv2d(self.split_len2, self.split_len1, kernel_size=3, padding=1))
        self.G = nn.Sequential(Eca_layer(self.split_len1),
                               nn.Conv2d(self.split_len1, self.split_len2, kernel_size=3, padding=1))

        self.H = subnet_constructor(self.split_len1, self.split_len2, d)

        in_channels = channel_num
        self.invconv = InvertibleConv1x1(in_channels, LU_decomposed=True)
        self.flow_permutation = lambda z, logdet, rev: self.invconv(z, logdet, rev)

    def forward(self, x, rev=False):
        # if not rev:
        # invert1x1conv
        x, logdet = self.flow_permutation(x, logdet=0, rev=False)

        # split to 1 channel and 2 channel.
        x1, x2 = (x.narrow(1, 0, self.split_len1), x.narrow(1, self.split_len1, self.split_len2))

        y1 = x1 + self.F(x2)  # 1 channel
        self.s = self.clamp * (torch.sigmoid(self.H(y1)) * 2 - 1)
        y2 = x2.mul(torch.exp(self.s)) + self.G(y1)  # 2 channel
        out = torch.cat((y1, y2), 1)

        return out