import torch
import torch.nn as nn
import math
import torch.nn.functional as F
from ..utils import Conv_BN_ReLU

class FPEM_v2(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(FPEM_v2, self).__init__()
        planes = out_channels
        self.dwconv3_1 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, groups=planes, bias=False)
        self.smooth_layer3_1 = Conv_BN_ReLU(planes, planes)
        self.Tucker_3_1 = Tucker(planes, planes, e=0.75)
        # self.Conv_3_1 = Conv_BN_ReLU(planes, planes, kernel_size=3, padding=1)

        self.dwconv2_1 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, groups=planes, bias=False)
        self.smooth_layer2_1 = Conv_BN_ReLU(planes, planes)
        self.Tucker_2_1 = Tucker(planes, planes, e=0.75)
        # self.Conv_2_1 = Conv_BN_ReLU(planes, planes, kernel_size=3, padding=1)

        self.dwconv1_1 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, groups=planes, bias=False)
        self.smooth_layer1_1 = Conv_BN_ReLU(planes, planes)
        self.Tucker_1_1 = Tucker(planes, planes, e=0.75)
        # self.Conv_1_1 = Conv_BN_ReLU(planes, planes, kernel_size=3, padding=1)



        self.dwconv2_2 = nn.Conv2d(planes, planes, kernel_size=3, stride=2, padding=1, groups=planes, bias=False)
        self.smooth_layer2_2 = Conv_BN_ReLU(planes, planes)
        self.Tucker_2_2 = Tucker(planes, planes, e=0.75, stride=2)
        # self.Conv_2_2 = Conv_BN_ReLU(planes, planes, kernel_size=3, stride=2, padding=1)
        # self.Conv_2_2 = nn.Conv2d(planes, planes, kernel_size=3, stride=2, padding=1)

        self.dwconv3_2 = nn.Conv2d(planes, planes, kernel_size=3, stride=2, padding=1, groups=planes, bias=False)
        self.smooth_layer3_2 = Conv_BN_ReLU(planes, planes)
        self.Tucker_3_2 = Tucker(planes, planes, e=0.75, stride=2)
        # self.Conv_3_2 = Conv_BN_ReLU(planes, planes, kernel_size=3, stride=2, padding=1)
        # self.Conv_3_2 = nn.Conv2d(planes, planes, kernel_size=3, stride=2, padding=1)

        self.dwconv4_2 = nn.Conv2d(planes, planes, kernel_size=3, stride=2, padding=1, groups=planes, bias=False)
        self.smooth_layer4_2 = Conv_BN_ReLU(planes, planes)
        self.Tucker_4_2 = Tucker(planes, planes, e=0.75, stride=2)
        # self.Conv_4_2 = Conv_BN_ReLU(planes, planes, kernel_size=3, stride=2, padding=1)
        # self.Conv_4_2 = nn.Conv2d(planes, planes, kernel_size=3, stride=2, padding=1)

    def _upsample_add(self, x, y):
        _, _, H, W = y.size()
        # return F.upsample(x, size=(H, W), mode='bilinear') + y
        return F.interpolate(x, size=(H, W), mode='bilinear') + y

    def forward(self, f1, f2, f3, f4):
        # f3_ = self.smooth_layer3_1(self.dwconv3_1(self._upsample_add(f4, f3)))
        # f2_ = self.smooth_layer2_1(self.dwconv2_1(self._upsample_add(f3_, f2)))
        # f1_ = self.smooth_layer1_1(self.dwconv1_1(self._upsample_add(f2_, f1)))

        f3_ = self.Tucker_3_1(self._upsample_add(f4, f3))
        f2_ = self.Tucker_2_1(self._upsample_add(f3_, f2))
        f1_ = self.Tucker_1_1(self._upsample_add(f2_, f1))

        # f3_ = self.Conv_3_1(self._upsample_add(f4, f3))
        # f2_ = self.Conv_2_1(self._upsample_add(f3_, f2))
        # f1_ = self.Conv_1_1(self._upsample_add(f2_, f1))
        

        # f2_ = self.smooth_layer2_2(self.dwconv2_2(self._upsample_add(f2_, f1_)))
        # f3_ = self.smooth_layer3_2(self.dwconv3_2(self._upsample_add(f3_, f2_)))
        # f4_ = self.smooth_layer4_2(self.dwconv4_2(self._upsample_add(f4, f3_)))

        f2_ = self.Tucker_2_2(self._upsample_add(f2_, f1_))
        f3_ = self.Tucker_3_2(self._upsample_add(f3_, f2_))
        f4_ = self.Tucker_4_2(self._upsample_add(f4, f3_))

        # f2_ = self.Conv_2_2(self._upsample_add(f2_, f1_))
        # f3_ = self.Conv_3_2(self._upsample_add(f3_, f2_))
        # f4_ = self.Conv_4_2(self._upsample_add(f4, f3_))

        f1 = f1 + f1_
        f2 = f2 + f2_
        f3 = f3 + f3_
        f4 = f4 + f4_

        return f1, f2, f3, f4

class Mish(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        device = x.device
        x = x * (torch.tanh(F.softplus(x)))
        x.to(device)
        return x

class Tucker(nn.Module):

    def __init__(self, c1, c2, k=3, s=0.25, e=0.75, stride=1):
        super(Tucker, self).__init__()

        sc1 = int(s*c1)
        ec2 = int(e*c2)

        self.conv1 = nn.Conv2d(c1, sc1, 1, stride)
        self.bn1 = nn.BatchNorm2d(sc1)
        self.act1 = Mish()

        self.conv2 = nn.Conv2d(sc1, ec2, k, padding=1)
        self.bn2 = nn.BatchNorm2d(ec2)
        self.act2 = Mish()

        self.conv3 = nn.Conv2d(ec2, c2, 1)
        self.bn3 = nn.BatchNorm2d(c2)
        self.act3 = Mish()

    def forward(self, x):

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.act1(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.act2(x)

        x = self.conv3(x)
        x = self.bn3(x)
        x = self.act3(x)

        return x  

class Fused_IBN(nn.Module):

    def __init__(self, c1, c2, k=3, s=8, stride=1):
        super(Fused_IBN, self).__init__()

        sc1 = int(s*c1)

        self.conv1 = nn.Conv2d(c1, sc1, k, stride, padding=1)
        self.bn1 = nn.BatchNorm2d(sc1)
        self.act1 = Mish()

        self.conv2 = nn.Conv2d(sc1, c2, 1)
        self.bn2 = nn.BatchNorm2d(c2)
        self.act2 = Mish()

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.act1(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.act2(x)

        return x

'''
class PSConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, dilation=1, parts=4, bias=False):
        super(PSConv2d, self).__init__()
        self.gwconv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, dilation, dilation, groups=parts, bias=bias)
        self.gwconv_shift = nn.Conv2d(in_channels, out_channels, kernel_size, stride, 2 * dilation, 2 * dilation, groups=parts, bias=bias)
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=bias)

        def backward_hook(grad):
            out = grad.clone()
            out[self.mask] = 0
            return out

        self.mask = torch.zeros(self.conv.weight.shape).byte().cuda()
        _in_channels = in_channels // parts
        _out_channels = out_channels // parts
        for i in range(parts):
            self.mask[i * _out_channels: (i + 1) * _out_channels, i * _in_channels: (i + 1) * _in_channels, : , :] = 1
            self.mask[(i + parts//2)%parts * _out_channels: ((i + parts//2)%parts + 1) * _out_channels, i * _in_channels: (i + 1) * _in_channels, :, :] = 1
        self.conv.weight.data[self.mask] = 0
        self.conv.weight.register_hook(backward_hook)

    def forward(self, x):
        x1, x2 = x.chunk(2, dim=1)
        x_shift = self.gwconv_shift(torch.cat((x2, x1), dim=1))
        return self.gwconv(x) + self.conv(x) + x_shift

'''
