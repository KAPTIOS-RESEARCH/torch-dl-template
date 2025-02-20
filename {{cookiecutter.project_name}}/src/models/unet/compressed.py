import torch
from torch import nn
from torch.nn import functional as F

class DepthwiseSeparableConv(nn.Module):
    """Depthwise separable convolution block."""
    def __init__(self, in_channels, out_channels, drop_prob):
        super().__init__()
        self.depthwise = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1, groups=in_channels, bias=False)
        self.pointwise = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        self.norm = nn.InstanceNorm2d(out_channels)
        self.activation = nn.LeakyReLU(0.2, inplace=True)
        self.dropout = nn.Dropout2d(drop_prob)

    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        x = self.norm(x)
        x = self.activation(x)
        return self.dropout(x)

class CompressedUNet(nn.Module):
    """Compressed U-Net with depthwise separable convolutions."""
    def __init__(self, in_channels, out_channels, start_feature_maps=32, num_pool_layers=4, drop_prob=0.1):
        super().__init__()

        self.down_sample_layers = nn.ModuleList()
        self.up_transpose_conv = nn.ModuleList()
        self.up_conv = nn.ModuleList()

        ch = start_feature_maps
        self.down_sample_layers.append(DepthwiseSeparableConv(in_channels, ch, drop_prob))
        for _ in range(num_pool_layers - 1):
            self.down_sample_layers.append(DepthwiseSeparableConv(ch, ch * 2, drop_prob))
            ch *= 2

        self.bottleneck = DepthwiseSeparableConv(ch, ch * 2, drop_prob)

        for _ in range(num_pool_layers - 1):
            self.up_transpose_conv.append(nn.ConvTranspose2d(ch * 2, ch, kernel_size=2, stride=2, bias=False))
            self.up_conv.append(DepthwiseSeparableConv(ch * 2, ch, drop_prob))
            ch //= 2

        self.up_transpose_conv.append(nn.ConvTranspose2d(ch * 2, ch, kernel_size=2, stride=2, bias=False))
        self.up_conv.append(nn.Sequential(
            DepthwiseSeparableConv(ch * 2, ch, drop_prob),
            nn.Conv2d(ch, out_channels, kernel_size=1)
        ))

    def forward(self, x):
        skip_connections = []

        for layer in self.down_sample_layers:
            x = layer(x)
            skip_connections.append(x)
            x = F.avg_pool2d(x, kernel_size=2)

        x = self.bottleneck(x)

        for up_transpose, up_conv in zip(self.up_transpose_conv, self.up_conv):
            skip = skip_connections.pop()
            x = up_transpose(x)
            x = self._pad_if_needed(x, skip)
            x = torch.cat([x, skip], dim=1)
            x = up_conv(x)

        return x

    @staticmethod
    def _pad_if_needed(x, ref):
        diff_h = ref.shape[-2] - x.shape[-2]
        diff_w = ref.shape[-1] - x.shape[-1]
        if diff_h != 0 or diff_w != 0:
            x = F.pad(x, [0, diff_w, 0, diff_h], mode='reflect')
        return x
