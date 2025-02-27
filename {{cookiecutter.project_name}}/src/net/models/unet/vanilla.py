import torch
from torch import nn
from torch.nn import functional as F
from src.net.utils.layers.conv import *

class VanillaUNet(nn.Module):
    def __init__(self, in_channels=1, out_channels=1, features=[64, 128, 256, 512], drop_prob: float = 0.0):
        super(VanillaUNet, self).__init__()
        self.encoder = nn.ModuleList()
        self.decoder = nn.ModuleList()

        # Encoder
        for feature in features:
            self.encoder.append(DoubleConvBlock(in_channels, feature, drop_prob=drop_prob))
            in_channels = feature

        # Bottleneck
        self.bottleneck = DoubleConvBlock(features[-1], features[-1] * 2, drop_prob=drop_prob)

        # Decoder
        for feature in reversed(features):
            self.decoder.append(TransposeConvBlock(feature * 2, feature))
            self.decoder.append(DoubleConvBlock(feature * 2, feature, drop_prob=drop_prob))

        # Final layer
        self.final_conv = nn.Conv2d(features[0], out_channels, kernel_size=1)

    def forward(self, x):
        skip_connections = []

        # Encoder forward
        for layer in self.encoder:
            x = layer(x)
            skip_connections.append(x)
            x = F.avg_pool2d(x, kernel_size=2)

        # Bottleneck
        x = self.bottleneck(x)

        # Decoder forward
        skip_connections = skip_connections[::-1]
        for idx in range(0, len(self.decoder), 2):
            x = self.decoder[idx](x)
            skip_connection = skip_connections[idx // 2]

            if x.shape != skip_connection.shape:
                x = nn.functional.interpolate(x, size=skip_connection.shape[2:])

            concat_skip = torch.cat((skip_connection, x), dim=1)
            x = self.decoder[idx + 1](concat_skip)

        return self.final_conv(x)