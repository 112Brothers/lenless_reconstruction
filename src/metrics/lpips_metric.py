import torch
from src.metrics.base_metric import BaseMetric

class LPIPSMetric(BaseMetric):
    def __init__(self, device, *args, **kwargs):
        super().__init__(*args, **kwargs)
        import lpips
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.metric = lpips.LPIPS(net='vgg').to(device)
        self.metric.eval()

    def __call__(self, reconstruction, lensed, **kwargs):
        r = reconstruction.contiguous() * 2 - 1
        l = lensed.contiguous() * 2 - 1
        return self.metric(r, l).mean().item()
