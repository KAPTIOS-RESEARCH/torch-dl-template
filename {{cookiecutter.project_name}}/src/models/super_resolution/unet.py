import torch
from torch import nn
import torch.nn.functional as F

class ConvBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x)

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ResidualBlock, self).__init__()
        self.body = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        )
        self.res_scale = 0.1
        
    def forward(self, x):
        res = self.body(x).mul(self.res_scale)
        res += x
        return res

class Encoder(nn.Module):
    def __init__(self, in_channels, out_channels) -> None:
        super().__init__()
        self.encoder = nn.Sequential(
            nn.MaxPool2d(2),
            ConvBlock(in_channels, out_channels)
        )

    def forward(self, x):
        x = self.encoder(x)
        return x

class Decoder(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Decoder, self).__init__()
        self.transpose = nn.ConvTranspose2d(in_channels, in_channels, kernel_size=2, stride=2)
        self.conv = ConvBlock(in_channels + out_channels, out_channels)

    def forward(self, x, skip):
        x = self.transpose(x)
        x = torch.cat([x, skip], dim=1)
        x = self.conv(x)
        return x

class UpConvBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.layers = nn.Sequential(
            nn.ConvTranspose2d(in_channels, in_channels, kernel_size=2, stride=2),
            ConvBlock(in_channels, out_channels)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x)

class SRUNet(nn.Module):
    def __init__(self):
        super(SRUNet, self).__init__()

        self.upsampling = UpConvBlock(64, 32)
        self.in_conv = ConvBlock(1, 64)
        
        self.enc_1 = Encoder(64, 128)
        self.enc_2 = Encoder(128, 256)
        self.enc_3 = Encoder(256, 512)
        self.enc_4 = Encoder(512, 512)

        self.dec_1 = Decoder(512, 512)
        self.dec_2 = Decoder(512, 256)
        self.dec_3 = Decoder(256, 128)
        self.dec_4 = Decoder(128, 64)
        self.dec_5 = Decoder(64, 32)
        
        self.out_conv = nn.Conv2d(32, 1, kernel_size=1)

    def forward(self, x):
        x1 = self.in_conv(x)
        x0 = self.upsampling(x1)
        
        x2 = self.enc_1(x1)
        x3 = self.enc_2(x2)
        x4 = self.enc_3(x3)
        x5 = self.enc_4(x4)

        x = self.dec_1(x5, x4)
        x = self.dec_2(x, x3)
        x = self.dec_3(x, x2)
        x = self.dec_4(x, x1)
        x = self.dec_5(x, x0)
        
        x = self.out_conv(x)
        return x