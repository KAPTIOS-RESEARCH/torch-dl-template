from torch import nn

class DepthwiseSeparableConv(nn.Module):
    """
    Depthwise Separable Convolution block.
    This convolution block performs a DS Convolution followed by batch normalization and ReLU activation.
    """
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super(DepthwiseSeparableConv, self).__init__()

        self.depthwise = nn.Conv2d(in_channels, in_channels, kernel_size=kernel_size, 
                                   stride=stride, padding=padding, groups=in_channels, bias=False)
        
        self.pointwise = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)

        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        x = self.bn(x)
        return self.relu(x)
