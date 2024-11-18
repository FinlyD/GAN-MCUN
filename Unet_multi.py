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

        self.up_1 = UpSampling(640, 320)
        self.up_2 = UpSampling(320, 160)
        self.up_3 = UpSampling(160, 80)
        self.outputs = LastConv(80, num_classes)
        self.tanh = nn.Tanh()

    def forward(self, x, y, z, m, n):  
        # down 部分
        x1 = self.inputs(x)  #batchsize, 1,32,256,192
        x2 = self.down_1(x1) 
        x3 = self.down_2(x2) #b, 64, 8, 32,32   524288   192 256  
        x4 = self.down_3(x3) #b, 128,4, 16, 16  131072   192 256  393216

        y1 = self.inputs(y)  
        y2 = self.down_1(y1) 
        y3 = self.down_2(y2) 
        y4 = self.down_3(y3) 

        z1 = self.inputs(z)  
        z2 = self.down_1(z1) 
        z3 = self.down_2(z2) 
        z4 = self.down_3(z3) 

        m1 = self.inputs(m)  
        m2 = self.down_1(m1) 
        m3 = self.down_2(m2) 
        m4 = self.down_3(m3) 

        n1 = self.inputs(n)  
        n2 = self.down_1(n1) 
        n3 = self.down_2(n2) 
        n4 = self.down_3(n3) 

        cat1=torch.cat([x1,y1,z1,m1,n1], dim=1)
        cat2=torch.cat([x2,y2,z2,m2,n2], dim=1)
        cat3=torch.cat([x3,y3,z3,m3,n3], dim=1)
        cat4=torch.cat([x4,y4,z4,m4,n4], dim=1)  # 3, 640, 4,32,24


        x5 = self.up_1(cat4, cat3) 
        x6 = self.up_2(x5, cat2)  
        x7 = self.up_3(x6, cat1)  
        x_rec = self.outputs(x7) 
        x_rec = self.tanh(x_rec)

        return x_rec
