import torch.nn.functional as F
import torch
from torchmetrics.image import StructuralSimilarityIndexMeasure
from src.utils.device import get_available_device

class VAELoss(torch.nn.Module):

    def __init__(self, kld_weight):
        super(VAELoss, self).__init__()
        self.kld_weight = kld_weight

    def forward(self, recon_x, x, mu, logvar):
        BCE = F.mse_loss(recon_x, x)
        KLD = torch.mean(-0.5 * torch.sum(1 + logvar - mu ** 2 - logvar.exp(), dim = 1), dim = 0)
        return BCE + self.kld_weight * KLD
    
class FastMRILoss(torch.nn.Module):
    def __init__(self, lambda_l1=1.0, lambda_ssim=0.1):
        super(FastMRILoss, self).__init__()
        self.l1_loss = torch.nn.L1Loss()
        self.ssim_loss = StructuralSimilarityIndexMeasure(1.0).to(get_available_device())
        self.lambda_l1 = lambda_l1
        self.lambda_ssim = lambda_ssim

    def forward(self, recon, target):
        l1 = self.l1_loss(recon, target)
        ssim = 1 - self.ssim_loss(recon, target)

        total_loss = (
            self.lambda_l1 * l1 +
            self.lambda_ssim * ssim
        )
        
        return total_loss