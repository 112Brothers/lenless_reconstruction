import torch
from src.metrics.base_metric import BaseMetric


class MSEMetric(BaseMetric):
    def __init__(self, device, *args, **kwargs):
        super().__init__(*args, **kwargs)
        from torchmetrics import MeanSquaredError
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.metric = MeanSquaredError().to(device)

    def __call__(self, reconstruction, lensed, **kwargs):
        return self.metric(reconstruction.contiguous(), lensed.contiguous()).item()
