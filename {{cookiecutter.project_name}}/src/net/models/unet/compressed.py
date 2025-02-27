import torch
from torch import nn
from src.net.utils.layers import *
from torch.nn import functional as F

class CompressedUNET(nn.Module):
    def __init__(self, in_channels=1, out_channels=1, features=[64, 128, 256, 512]):
        super(CompressedUNET, self).__init__()
        self.encoder = nn.ModuleList()
        self.decoder = nn.ModuleList()

        # Encoder
        for feature in features:
            self.encoder.append(DoubleDSConv(in_channels, feature))
            in_channels = feature

        # Bottleneck
        self.bottleneck = DoubleDSConv(features[-1], features[-1] * 2)

        # Decoder
        for feature in reversed(features):
            self.decoder.append(UpsampleDSConv(feature * 2, feature))
            self.decoder.append(DepthwiseSeparableConv(feature * 2, feature))

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