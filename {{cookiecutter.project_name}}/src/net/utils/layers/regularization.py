from torch import nn

class ICLayer(nn.Module):
    
    def __init__(self, out_channels, drop_prob = 0.01):
        super(ICLayer, self).__init__()
        self.ic = nn.Sequential(
            nn.InstanceNorm2d(out_channels),
            nn.Dropout(p = drop_prob)
        )
        
    def forward(self, x):
        return self.ic(x)