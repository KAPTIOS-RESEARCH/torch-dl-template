import torch
from torch import nn
from torch.nn import functional as F

class UNet(nn.Module):
    """
    PyTorch implementation of U-Net for biomedical image segmentation.
    Reference: Ronneberger et al., 2015 (MICCAI).
    """

    def __init__(self, in_channels: int, out_channels: int, start_feature_maps: int = 32, 
                 num_pool_layers: int = 4, drop_prob: float = 0.0):
        """
        Args:
            in_channels (int): Number of input channels.
            out_channels (int): Number of output channels.
            start_feature_maps (int): Base number of feature channels.
            num_pool_layers (int): Number of downsampling/upsampling layers.
            drop_prob (float): Dropout probability.
        """
        super().__init__()

        self.down_sample_layers = nn.ModuleList()
        self.up_transpose_conv = nn.ModuleList()
        self.up_conv = nn.ModuleList()

        # Downsampling path
        ch = start_feature_maps
        self.down_sample_layers.append(ConvBlock(in_channels, ch, drop_prob))
        for _ in range(num_pool_layers - 1):
            self.down_sample_layers.append(ConvBlock(ch, ch * 2, drop_prob))
            ch *= 2

        # Bottleneck
        self.bottleneck = ConvBlock(ch, ch * 2, drop_prob)

        # Upsampling path
        for _ in range(num_pool_layers - 1):
            self.up_transpose_conv.append(TransposeConvBlock(ch * 2, ch))
            self.up_conv.append(ConvBlock(ch * 2, ch, drop_prob))
            ch //= 2

        # Final upsampling and output layer
        self.up_transpose_conv.append(TransposeConvBlock(ch * 2, ch))
        self.up_conv.append(nn.Sequential(
            ConvBlock(ch * 2, ch, drop_prob),
            nn.Conv2d(ch, out_channels, kernel_size=1)
        ))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of U-Net.

        Args:
            x (torch.Tensor): Input tensor of shape (N, in_channels, H, W).

        Returns:
            torch.Tensor: Output tensor of shape (N, out_channels, H, W).
        """
        skip_connections = []

        # Downsampling
        for layer in self.down_sample_layers:
            x = layer(x)
            skip_connections.append(x)
            x = F.avg_pool2d(x, kernel_size=2)

        # Bottleneck
        x = self.bottleneck(x)

        # Upsampling
        for up_transpose, up_conv in zip(self.up_transpose_conv, self.up_conv):
            skip = skip_connections.pop()
            x = up_transpose(x)
            x = self._pad_if_needed(x, skip)
            x = torch.cat([x, skip], dim=1)
            x = up_conv(x)

        return x

    @staticmethod
    def _pad_if_needed(x: torch.Tensor, ref: torch.Tensor) -> torch.Tensor:
        """
        Pads x to match the spatial dimensions of ref if needed.

        Args:
            x (torch.Tensor): Tensor to pad.
            ref (torch.Tensor): Reference tensor with target size.

        Returns:
            torch.Tensor: Padded tensor.
        """
        diff_h = ref.shape[-2] - x.shape[-2]
        diff_w = ref.shape[-1] - x.shape[-1]

        if diff_h != 0 or diff_w != 0:
            x = F.pad(x, [0, diff_w, 0, diff_h], mode='reflect')
        return x


class ConvBlock(nn.Module):
    """
    Two-layer convolution block with InstanceNorm, LeakyReLU, and Dropout.
    """

    def __init__(self, in_channels: int, out_channels: int, drop_prob: float):
        """
        Args:
            in_channels (int): Number of input channels.
            out_channels (int): Number of output channels.
            drop_prob (float): Dropout probability.
        """
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.InstanceNorm2d(out_channels),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout2d(drop_prob),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.InstanceNorm2d(out_channels),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout2d(drop_prob),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x)


class TransposeConvBlock(nn.Module):
    """
    Transpose convolution block for upsampling with InstanceNorm and LeakyReLU.
    """

    def __init__(self, in_channels: int, out_channels: int):
        """
        Args:
            in_channels (int): Number of input channels.
            out_channels (int): Number of output channels.
        """
        super().__init__()
        self.layers = nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2, bias=False),
            nn.InstanceNorm2d(out_channels),
            nn.LeakyReLU(0.2, inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x)