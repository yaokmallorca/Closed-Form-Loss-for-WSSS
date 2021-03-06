import torch.nn as nn
import math
import torch
import numpy as np
import torch.nn.functional as F
affine_par = True


def outS(i):
    i = int(i)
    i = (i+1)/2
    i = int(np.ceil((i+1)/2.0))
    i = (i+1)/2
    return i
def conv3x3(in_planes, out_planes, stride=1):
    "3x3 convolution with padding"
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes, affine = affine_par)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes, affine = affine_par)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1,  dilation_ = 1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, stride=stride, bias=False) # change
        self.bn1 = nn.BatchNorm2d(planes,affine = affine_par)
        padding = 1
        if dilation_ == 2:
            padding = 2
        elif dilation_ == 4:
            padding = 4
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, # change
                               padding=padding, bias=False, dilation = dilation_)
        self.bn2 = nn.BatchNorm2d(planes,affine = affine_par)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4, affine = affine_par)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride



    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out

class Classifier_Module(nn.Module):

    def __init__(self,dilation_series,padding_series,NoLabels):
        super(Classifier_Module, self).__init__()
        self.conv2d_list = nn.ModuleList()
        for dilation,padding in zip(dilation_series,padding_series):
            self.conv2d_list.append(nn.Conv2d(2048,NoLabels,kernel_size=3,stride=1, padding =padding, dilation = dilation,bias = True))

        for m in self.conv2d_list:
            m.weight.data.normal_(0, 0.01)


    def forward(self, x):
        out = self.conv2d_list[0](x)
        for i in range(len(self.conv2d_list)-1):
            out += self.conv2d_list[i+1](x)
        return out

class ConvBNReLU(nn.Module):
    def __init__(self,ch_in,ch_out):
        super(conv_block,self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(ch_in, ch_out, kernel_size=1,stride=1,padding=0,bias=True),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True)
        )

    def forward(self,x):
        x = self.conv(x)
        return x

class ResNet(nn.Module):
    def __init__(self, block, layers,NoLabels,Reconstruct=False):
        self.inplanes = 64
        super(ResNet, self).__init__()
        self.Reconstruct = Reconstruct

        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64,affine = affine_par)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1, ceil_mode=True)
        # print("___________ make layer 1 _____________")
        self.layer1 = self._make_layer(block, 64, layers[0])
        # print("___________ make layer 2 _____________")
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        # print("___________ make layer 3 _____________")
        self.layer3 = self._make_layer(block, 256, layers[2], stride=1, dilation__ = 2)
        # print("___________ make layer 4 _____________")
        self.layer4 = self._make_layer(block, 512, layers[3], stride=1, dilation__ = 4)
        # self.fc_out = nn.Linear()

        self.layer5 = self._make_pred_layer(Classifier_Module, [6,12,18,24],[6,12,18,24],NoLabels)
        if self.Reconstruct:
            self.conv_rec = nn.Conv2d(2, 3, kernel_size=1, stride=1, padding=0,
                               bias=False)
        # self.conv1x1 = nn.Conv2d(1, 1, kernel_size=1, stride=1, padding=0, bias=False)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, 0.01)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1,dilation__ = 1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion or dilation__ == 2 or dilation__ == 4:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion,affine = affine_par),
            )
        layers = []
        layers.append(block(self.inplanes, planes, stride,dilation_=dilation__, downsample = downsample ))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes,planes,dilation_=dilation__))
        return nn.Sequential(*layers)

    def _make_pred_layer(self,block, dilation_series, padding_series,NoLabels):
        return block(dilation_series,padding_series,NoLabels)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        # print("layer1: ", x.size())
        x = self.layer2(x)
        # print("layer2: ", x.size())
        x = self.layer3(x)
        # print("layer3: ", x.size())
        x = self.layer4(x)
        # print("layer4: ", x.size())
        x = self.layer5(x)
        # print("layer5: ", x.size())
        x = F.upsample_bilinear(x,scale_factor=2)[:,:,:-1,:-1]
        # print("x: ", x.size())
        x = F.upsample_bilinear(x,scale_factor=2)[:,:,:-1,:-1]
        # print("x: ", x.size())
        if self.Reconstruct:
            x_seg = F.upsample_bilinear(x,scale_factor=2)[:,:,:-1,:-1]
            x_rec = self.conv_rec(x)
            x_rec = F.upsample_bilinear(x_rec,scale_factor=2)[:,:,:-1,:-1]
            return x_seg, x_rec
        else:
            x_seg = F.upsample_bilinear(x,scale_factor=2)[:,:,:-1,:-1]
            # x_seg = self.conv1x1(x)
            return x_seg

def ResDeeplab(NoLabels=1, Reconstruct=False):
    return ResNet(Bottleneck,[3, 4, 23, 3],NoLabels,Reconstruct)
