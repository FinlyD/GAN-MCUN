import torch
import torch.nn as nn
import torch.nn.functional as F

import torch
import torch.nn as nn
import torch.nn.functional as F

class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels, batch_norm):
        super(DoubleConv, self).__init__()
        self.double_conv = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.InstanceNorm3d(out_channels),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.InstanceNorm3d(out_channels),
            nn.LeakyReLU(0.2, inplace=True),
            
        )

    def forward(self, x):
        x = self.double_conv(x)
        return x

class DownSampling(nn.Module):
    def __init__(self, in_channels, out_channels, batch_norm):
        super(DownSampling, self).__init__()
        self.conv = DoubleConv(in_channels, out_channels, batch_norm)
        self.pool = nn.MaxPool3d(2)

    def forward(self, x):
        x = self.conv(x)
        x = self.pool(x)
        return x

class Discriminator3D(nn.Module):
    def __init__(self, in_channels=1):
        super(Discriminator3D, self).__init__()
        self.model = nn.Sequential(
            DownSampling(1, 8, batch_norm=False),   #两卷积+池化
            DownSampling(8, 16, batch_norm=False),  
            DoubleConv(16, 32, batch_norm=False)) #两个卷积
        self.global_pool=nn.AdaptiveAvgPool3d((1,1,1))
        # self.fc1 = nn.Linear(6400, 1024)
        # self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(32,2)
           

    def forward(self, x):   #4,1,32,160,160
        x = self.model(x)  #4,32,4,80,80
        # x_view = x.view(-1, 64 * 2 * 10 * 10)
        x_pool = self.global_pool(x)  #32,1,1,1
        x_view = x_pool.view(x_pool.size(0), -1)
        # fc1 = self.fc1(x_view)
        # drop_out = self.dropout(fc1)
        fc2= self.fc2(x_view)
        out = torch.sigmoid(fc2)

        return out



       