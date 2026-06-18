import torch
from torch import nn
from src.model.drunet import DRUNet
from src.model.le_admm import LeADMM

class ModularLeADMM(nn.Module):
    """
    Modular Le-ADMM with optional DRUNet prepost-processors.
    """
    def __init__(
        self,
        n_iter=5,
        init_mu=1e-4,
        init_tau=2e-4,
        use_pre=True,
        use_post=True,
        drunet_channels=(32, 64, 116, 128),
        drunet_n_res_blocks=3,
    ):
        super().__init__()
        self.use_pre = use_pre
        self.use_post = use_post
        if use_pre:
            self.pre_processor = DRUNet(in_channels=3, out_channels=3, channels=drunet_channels, n_res_blocks=drunet_n_res_blocks)
            nn.init.zeros_(self.pre_processor.tail.weight)
            nn.init.zeros_(self.pre_processor.tail.bias)
        else:
            self.pre_processor = None
        self.le_admm = LeADMM(n_iter=n_iter, init_mu=init_mu, init_tau=init_tau)
        if use_post:
            self.post_processor = DRUNet(in_channels=3, out_channels=3, channels=drunet_channels, n_res_blocks=drunet_n_res_blocks)
            nn.init.zeros_(self.post_processor.tail.weight)
            nn.init.zeros_(self.post_processor.tail.bias)
        else:
            self.post_processor = None

    def forward(self, lensless, mask, **batch):
        if self.use_pre:
            y = lensless + self.pre_processor(lensless)
        else:
            y = lensless
        admm_out = self.le_admm(y, mask)["reconstruction"]
        if self.use_post:
            reconstruction = torch.clamp(admm_out + self.post_processor(admm_out), 0.0, 1.0)
        else:
            reconstruction = admm_out
        return {"reconstruction": reconstruction}

    def __str__(self):
        all_parameters = sum([p.numel() for p in self.parameters()])
        trainable_parameters = sum([p.numel() for p in self.parameters() if p.requires_grad])
        result_info = super().__str__()
        result_info += f"\nAll parameters: {all_parameters}"
        result_info += f"\nTrainable parameters: {trainable_parameters}"
        return result_info
