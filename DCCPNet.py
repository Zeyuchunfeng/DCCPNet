import torch.nn as nn
import torch.nn.functional as F
import torch
from torchsummary import summary


# MS branch--MCR block
class MS_ResNetBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(MS_ResNetBlock, self).__init__()
        self.conv = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, padding=1)
        self.block = nn.Sequential(
            nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=1),
            nn.ReLU(inplace=True),

            nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=7, padding=3),
            nn.ReLU(inplace=True),

            nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=1),
        )

    def forward(self, x):
        x_ = F.relu(self.conv(x))
        return F.relu(self.block(x_) + x_)


# PAN branch--MCR block
class PAN_ResNetBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(PAN_ResNetBlock, self).__init__()
        self.conv = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, padding=1)
        self.block = nn.Sequential(
            nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=1),
            nn.ReLU(inplace=True),

            nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=7, padding=3),
            nn.ReLU(inplace=True),

            nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=1),
        )

    def forward(self, x):
        x_ = F.relu(self.conv(x))
        return F.relu(self.block(x_) + x_)


# Image reconstruction branch--MCR block
class ResNetBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ResNetBlock, self).__init__()
        self.conv = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, padding=1)
        self.block = nn.Sequential(
            nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=1),
            nn.ReLU(inplace=True),

            nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=7, padding=3),
            nn.ReLU(inplace=True),

            nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=1),
        )

    def forward(self, x):
        x_ = F.relu(self.conv(x))
        return F.relu(self.block(x_) + x_)


# The proposed DCCPNet model
class DCCPNet(nn.Module):
    def __init__(self, ms_channels, out_channels):
        super(DCCPNet, self).__init__()
        self.ms_conv = nn.Conv2d(in_channels=ms_channels, out_channels=out_channels, kernel_size=3, padding=1)
        self.pan_conv = nn.Conv2d(in_channels=1, out_channels=out_channels, kernel_size=3, padding=1)

        self.ms_R1 = MS_ResNetBlock(in_channels=out_channels, out_channels=out_channels)
        self.ms_R2 = MS_ResNetBlock(in_channels=2 * out_channels, out_channels=out_channels)
        self.ms_R3 = MS_ResNetBlock(in_channels=2 * out_channels, out_channels=out_channels)
        self.ms_R4 = MS_ResNetBlock(in_channels=2 * out_channels, out_channels=out_channels)

        self.pan_R1 = PAN_ResNetBlock(in_channels=out_channels, out_channels=out_channels)
        self.pan_R2 = PAN_ResNetBlock(in_channels=2 * out_channels, out_channels=out_channels)
        self.pan_R3 = PAN_ResNetBlock(in_channels=2 * out_channels, out_channels=out_channels)
        self.pan_R4 = PAN_ResNetBlock(in_channels=2 * out_channels, out_channels=out_channels)

        self.fused_R1 = ResNetBlock(in_channels=2 * out_channels, out_channels=out_channels)
        self.fused_R2 = ResNetBlock(in_channels=out_channels, out_channels=out_channels)

        self.re_conv = nn.Conv2d(in_channels=out_channels, out_channels=ms_channels, kernel_size=3, padding=1)

    def forward(self, ms, pan):
        ms_ = F.relu(self.ms_conv(ms))
        pan_ = F.relu(self.pan_conv(pan))
        ms1 = self.ms_R1(ms_)
        pan1 = self.pan_R1(pan_)

        # channel cross-concatenation
        ms2 = self.ms_R2(torch.cat([ms1, pan1], dim=1))
        pan2 = self.pan_R2(torch.cat([pan1, ms1], dim=1))
        ms3 = self.ms_R3(torch.cat([ms2, pan2], dim=1))
        pan3 = self.pan_R3(torch.cat([pan2, ms2], dim=1))
        ms4 = self.ms_R4(torch.cat([ms3, pan3], dim=1))
        pan4 = self.pan_R4(torch.cat([pan3, ms3], dim=1))

        out = self.fused_R1(torch.cat([ms4, pan4], dim=1))
        out = self.fused_R2(out)
        out = self.re_conv(out)
        return out + ms

    def initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight.data, mode='fan_in', nonlinearity='relu')

            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight.data, 1)
                nn.init.constant_(m.bias.data, 0)

            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight.data, 0, 1)


if __name__ == '__main__':
    model = DCCPNet(ms_channels=4, out_channels=32)  # ms_channels = 4 or 8
    print("===> Parameter numbers : %.2fM" % (sum(p.numel() for p in model.parameters()) / 1000000.0))
    ms = torch.randn(1, 4, 32, 32)
    pan = torch.randn(1, 1, 32, 32)
    out = model(ms, pan)
    summary(model, input_size=[(4, 256, 256), (1, 256, 256)], device="cpu")
    print(out.shape)
