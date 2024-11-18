from pyclbr import Class
import torch
import torch.nn as nn


class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DoubleConv, self).__init__()
        channels = int(out_channels / 2)
        if in_channels > out_channels:
            channels = int(in_channels / 2)
        layers = [
            nn.Conv3d(in_channels, channels, kernel_size=3, stride=1, padding=1),
            nn.InstanceNorm3d(channels),
            nn.ReLU(True),
            nn.Conv3d(channels, out_channels, kernel_size=3, stride=1, padding=1),
            nn.InstanceNorm3d(out_channels),
            nn.ReLU(True)
        ]
        self.double_conv = nn.Sequential(*layers)

    def forward(self, x):
        return self.double_conv(x)

class DownSampling(nn.Module):  
    def __init__(self, in_channels, out_channels):
        super(DownSampling, self).__init__()
        self.maxpool_to_conv = nn.Sequential(
            nn.MaxPool3d(kernel_size=2, stride=2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_to_conv(x)

class UpSampling(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(UpSampling, self).__init__()
        # if bilinear:
        #     self.up = nn.Upsample(scale_factor=2, mode='trilinear', align_corners=True)
        # else:
        self.up = nn.ConvTranspose3d(in_channels, out_channels, kernel_size=2, stride=2)
        self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, inputs1, inputs2):
        inputs1 = self.up(inputs1)

        outputs = torch.cat([inputs1, inputs2], dim=1)
        outputs = self.conv(outputs)
        return outputs

class LastConv(nn.Module):
    def __init__(self, in_channels, out_channels ):
        super(LastConv, self).__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size=1 )

    def forward(self, x):
        return self.conv(x)

class UNet3D(nn.Module):
    def __init__(self, in_channels=1, num_classes=1):
        super(UNet3D, self).__init__()
        self.in_channels = in_channels

        self.inputs = DoubleConv(in_channels, 16)
        self.down_1 = DownSampling(16, 32)
        self.down_2 = DownSampling(32, 64)
        self.down_3 = DownSampling(64, 128)

        self.up_1 = UpSampling(128,64)
        self.up_2 = UpSampling(64, 32)
        self.up_3 = UpSampling(32, 16)
        self.outputs = LastConv(16, num_classes)
        self.tanh = nn.Tanh()
        # self.mlp1=MLP(in_channels=320)

    def forward(self, x):   #4,1,32,160,160
        # down 部分
        x1 = self.inputs(x)  #4,16,32,160,160
        x2 = self.down_1(x1) #4,32,16,80,80
        x3 = self.down_2(x2) #4,64,8,40,40
        x4 = self.down_3(x3) #4,128,4,20,20

        x5 = self.up_1(x4, x3) #4,64,4,16,16
        x6 = self.up_2(x5, x2)  #4,32,8,32,32
        x7 = self.up_3(x6, x1)  #4,16,16,64,64
        x_rec = self.outputs(x7) #4,1,16,64,64
        x_rec = self.tanh(x_rec)

        return x_rec
    
class extract_feature(nn.Module):
    def __init__(self, in_channels=1, num_classes=1):
        super(extract_feature, self).__init__()
        self.in_channels = in_channels

        self.inputs = DoubleConv(in_channels, 16)
        self.down_1 = DownSampling(16, 32)
        self.down_2 = DownSampling(32, 64)
        self.down_3 = DownSampling(64, 128)

        self.up_1 = UpSampling(128,64)
        self.up_2 = UpSampling(64, 32)
        self.up_3 = UpSampling(32, 16)
        self.outputs = LastConv(16, num_classes)
        self.tanh = nn.Tanh()
        # self.mlp1=MLP(in_channels=320)

    def forward(self, x):   #4,1,32,160,160
        # down 部分
        feature_list=[]
        x1 = self.inputs(x)  #4,16,32,160,160
        feature_list.append(x1)
        x2 = self.down_1(x1) #4,32,16,80,80
        feature_list.append(x2)
        x3 = self.down_2(x2) #4,64,8,40,40
        feature_list.append(x3)
        x4 = self.down_3(x3) #4,128,4,20,20
        feature_list.append(x4)

        return feature_list
