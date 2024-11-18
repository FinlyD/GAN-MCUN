import torch
import torch.nn as nn


class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels, bath_normal=False):
        super(DoubleConv, self).__init__()
        channels = int(out_channels / 2)
        if in_channels > out_channels:
            channels = int(in_channels / 2)

        layers = [
            # in_channels：输入通道数
            # channels：输出通道数
            # kernel_size：卷积核大小
            # stride：步长
            # padding：边缘填充
            nn.Conv3d(in_channels, channels, kernel_size=3, stride=1, padding=1),
            nn.ReLU(True),

            nn.Conv3d(channels, out_channels, kernel_size=3, stride=1, padding=1),
            nn.ReLU(True)
        ]
        if bath_normal:  # 如果要添加BN层
            layers.insert(1, nn.BatchNorm3d(channels))
            layers.insert(len(layers) - 1, nn.BatchNorm3d(out_channels))

        # 构造序列器
        self.double_conv = nn.Sequential(*layers)

    def forward(self, x):
        return self.double_conv(x)

class DownSampling(nn.Module):
    def __init__(self, in_channels, out_channels, batch_normal=False):
        super(DownSampling, self).__init__()
        self.maxpool_to_conv = nn.Sequential(
            nn.MaxPool3d(kernel_size=2, stride=2),
            DoubleConv(in_channels, out_channels, batch_normal)
        )

    def forward(self, x):
        return self.maxpool_to_conv(x)

class UpSampling(nn.Module):
    def __init__(self, in_channels, out_channels, batch_normal=False, bilinear=True):
        super(UpSampling, self).__init__()
        if bilinear:
            # 采用双线性插值的方法进行上采样
            self.up = nn.Upsample(scale_factor=2, mode='trilinear', align_corners=True)
        else:
            # 采用反卷积进行上采样
            self.up = nn.ConvTranspose3d(in_channels, int(in_channels // 2), kernel_size=2, stride=2)
        self.conv = DoubleConv(in_channels + int(in_channels / 2), out_channels, batch_normal)

    # inputs1：上采样的数据（对应图中黄色箭头传来的数据）
    # inputs2：特征融合的数据（对应图中绿色箭头传来的数据）
    def forward(self, inputs1, inputs2):
        # 进行一次up操作
        inputs1 = self.up(inputs1)

        # 进行特征融合
        outputs = torch.cat([inputs1, inputs2], dim=1)
        outputs = self.conv(outputs)
        return outputs

class LastConv(nn.Module):
    def __init__(self, in_channels, out_channels ):
        super(LastConv, self).__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size=1 )

    def forward(self, x):
        return self.conv(x)

class MLP(nn.Module):

    def __init__(self,in_channels):
        super(MLP, self).__init__()

        self.model = nn.Sequential(
            nn.Linear(in_channels, 64),
            nn.LeakyReLU(inplace=True),
            nn.Linear(64, 64),
            nn.LeakyReLU(inplace=True),
            nn.Linear(64, 5),
            nn.LeakyReLU(inplace=True)
        )

    def forward(self, x):
        x = self.model(x)

        return x

class UNet3D(nn.Module):
    def __init__(self, in_channels=1, num_classes=1, batch_normal=False, bilinear=True):
        super(UNet3D, self).__init__()
        self.in_channels = in_channels
        self.batch_normal = batch_normal
        self.bilinear = bilinear

        self.inputs = DoubleConv(in_channels, 8, self.batch_normal)
        self.down_1 = DownSampling(8, 16, self.batch_normal)
        self.down_2 = DownSampling(16, 32, self.batch_normal)
        self.down_3 = DownSampling(32, 64, self.batch_normal)

        self.up_1 = UpSampling(320, 160, self.batch_normal, self.bilinear)
        self.up_2 = UpSampling(160, 80, self.batch_normal, self.bilinear)
        self.up_3 = UpSampling(80, 40, self.batch_normal, self.bilinear)
        self.outputs = LastConv(40, num_classes)

        self.mlp1 = MLP(in_channels=320)

    def forward(self, x, y, z, m, n):  #4，1，80，128，128
        # down 部分
        x1 = self.inputs(x)
        x2 = self.down_1(x1)
        x3 = self.down_2(x2)
        x4 = self.down_3(x3)

        # down 部分
        y1 = self.inputs(y)
        y2 = self.down_1(y1)
        y3 = self.down_2(y2)
        y4 = self.down_3(y3)

        # down 部分
        z1 = self.inputs(z)
        z2 = self.down_1(z1)
        z3 = self.down_2(z2)
        z4 = self.down_3(z3)

        # down 部分
        m1 = self.inputs(m)
        m2 = self.down_1(m1)
        m3 = self.down_2(m2)
        m4 = self.down_3(m3)

        # down 部分
        n1 = self.inputs(n)
        n2 = self.down_1(n1)
        n3 = self.down_2(n2)
        n4 = self.down_3(n3)

        cat1 = torch.cat([x1, y1, z1, m1, n1], dim=1)
        # mlp1=self.mlp1(cat1)
        cat2 = torch.cat([x2, y2, z2, m2, n2], dim=1)
        cat3 = torch.cat([x3, y3, z3, m3, n3], dim=1)
        cat4 = torch.cat([x4, y4, z4, m4, n4], dim=1)    #4，320，10，16，16
        # up部分


        x5 = self.up_1(cat4, cat3)
        x6 = self.up_2(x5, cat2)
        x7 = self.up_3(x6, cat1)   #4，40，80，128，128
        x = self.outputs(x7)   #4，1，80，128，128

        return x
