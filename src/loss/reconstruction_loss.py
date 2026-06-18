import torch
from torch import nn

class ReconstructionLoss(nn.Module):
    """
    MSE + LPIPS loss
    """
    def __init__(self, lpips_weight=0.1):
        super().__init__()
        self.mse = nn.MSELoss()
        self.lpips_weight = lpips_weight
        import lpips
        self.lpips_fn = lpips.LPIPS(net='vgg')
        for p in self.lpips_fn.parameters():
            p.requires_grad = False

    def forward(self, reconstruction, lensed, **batch):
        r = reconstruction.contiguous()
        l = lensed.contiguous()
        mse_val = self.mse(r, l)
        lpips_val = self.lpips_fn(r * 2 - 1, l * 2 - 1).mean()
        loss = mse_val + self.lpips_weight * lpips_val
        return {"loss": loss, "mse_loss": mse_val, "lpips_loss": lpips_val}
