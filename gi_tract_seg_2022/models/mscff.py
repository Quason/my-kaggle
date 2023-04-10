import torch
import torch.nn as nn


class ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels, pad, dilation):
        super().__init__()
        self.conv_block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3,
                      stride=1, padding=pad, dilation=dilation),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),

            nn.Conv2d(out_channels, out_channels, 3,
                      stride=1, padding=pad, dilation=dilation),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )
        self.conv_skip = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3,
                      stride=1, padding=pad, dilation=dilation),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x):
        return self.conv_block(x) + self.conv_skip(x)


class Down(nn.Module):
    def __init__(self, in_channels, out_channels, pad=1, dilation=1):
        super().__init__()
        self.pooling = nn.MaxPool2d(2)
        self.cbrr_down = ResBlock(in_channels, out_channels, pad=pad, dilation=dilation)
        self.dilated = dilation

    def forward(self, x):
        if self.dilated == 1:
            y = self.pooling(x)
            y = self.cbrr_down(y)
        else:
            y = self.cbrr_down(x)
        return y


class Up(nn.Module):
    def __init__(self, in_channels, out_channels, pad=1, dilation=1):
        super().__init__()
        self.cbrr_up = ResBlock(in_channels, out_channels, pad=pad, dilation=dilation)
        self.deconv = nn.ConvTranspose2d(in_channels, in_channels, 2, stride=2)
        self.dilated = dilation

    def forward(self, x1, x2):
        x = torch.cat([x2, x1], dim=1)
        if self.dilated == 1:
            y = self.deconv(x)
            y = self.cbrr_up(y)
        else:
            y = self.cbrr_up(x)
        return y


class Concat(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, stride=1, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x1, x2):
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class MSC(nn.Module):
    ''' multi-scale concat module
    '''
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv_deconv2 = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, 3, stride=1, padding=1),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(in_channels, in_channels, 2, stride=2)
        )
        self.conv_deconv4 = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, 3, stride=1, padding=1),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(in_channels, in_channels, 2, stride=4, output_padding=2)
        )
        self.conv_deconv8 = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, 3, stride=1, padding=1),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(in_channels, in_channels, 2, stride=8, output_padding=6)
        )
        self.conv = nn.Conv2d(in_channels*6, out_channels, 3, stride=1, padding=1)

    def forward(self, x1, x2, x3, x4, x5, x6):
        x1 = self.conv_deconv8(x1)
        x2 = self.conv_deconv8(x2)
        x3 = self.conv_deconv8(x3)
        x4 = self.conv_deconv4(x4)
        x5 = self.conv_deconv2(x5)
        x = torch.cat([x1, x2, x3, x4, x5, x6], dim=1)
        return self.conv(x)


class MSCFF(nn.Module):
    def __init__(self, in_channels, out_classes):
        super().__init__()
        self.inc = ResBlock(in_channels, 64, pad=1, dilation=1)
        self.down1 = Down(64, 64, pad=1, dilation=1)
        self.down2 = Down(64, 64, pad=1, dilation=1)
        self.down3 = Down(64, 64, pad=1, dilation=1)
        self.down4 = Down(64, 64, pad=2, dilation=2)
        self.down5 = Down(64, 64, pad=4, dilation=4)
        self.mic = ResBlock(64, 64, pad=1, dilation=1)
        self.up5 = Up(128, 64, pad=4, dilation=4)
        self.up4 = Up(128, 64, pad=2, dilation=2)
        self.up3 = Up(128, 64, pad=1, dilation=1)
        self.up2 = Up(128, 64, pad=1, dilation=1)
        self.up1 = Up(128, 64, pad=1, dilation=1)
        self.concat = Concat(64, 64)
        self.msc = MSC(64, out_classes)
        self.n_classes = out_classes
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        i0 = self.inc(x)
        d1 = self.down1(i0)
        d2 = self.down2(d1)
        d3 = self.down3(d2)
        d4 = self.down4(d3)
        d5 = self.down5(d4)
        m = self.mic(d5)
        u5 = self.up5(m, d5)
        u4 = self.up4(u5, d4)
        u3 = self.up3(u4, d3)
        u2 = self.up2(u3, d2)
        u1 = self.up1(u2, d1)
        y = self.msc(m, u5, u4, u3, u2, u1)
        y = self.softmax(y)
        return y


if __name__ == '__main__':
    import torch
    input = torch.rand((10, 1, 256, 256))
    net = MSCFF(1, 4)
    pred = net(input)
    print(pred.shape)
