import torch.nn.functional as F
import torch

class VAELoss(torch.nn.Module):

    def __init__(self, kld_weight):
        super(VAELoss, self).__init__()
        self.kld_weight = kld_weight

    def forward(self, recon_x, x, mu, logvar):
        BCE = F.mse_loss(recon_x, x)
        KLD = torch.mean(-0.5 * torch.sum(1 + logvar - mu ** 2 - logvar.exp(), dim = 1), dim = 0)
        return BCE + self.kld_weight * KLD