import torch
from torch import nn

from src.model.fft_utils import (
    compute_d_otf,
    compute_fft_shape,
    crop_from_fft,
    finite_diff,
    finite_diff_adjoint,
    pad_to_fft,
    psf_to_otf,
    soft_threshold,
)


class ADMM(nn.Module):
    """
    Basic ADMM with fixed params
    """

    def __init__(self, n_iter=100, mu=1e-4, tau=2e-4):
        super().__init__()
        self.n_iter = n_iter
        self.mu = mu
        self.tau = tau
        self.psf_scale = 5.0

    def forward(self, lensless, mask, **batch):
        B, C, H, W = lensless.shape
        device = lensless.device
        mu1 = mu2 = mu3 = self.mu
        tau = self.tau
        fft_shape = compute_fft_shape(H, W)
        if mask.ndim == 3:
            mask = mask.unsqueeze(1)
        psf_p = mask * self.psf_scale
        otf = psf_to_otf(psf_p, fft_shape)
        otf_conj = torch.conj(otf)
        otf_abs_sq = torch.abs(otf) ** 2

        # Finite diff
        dx_otf, dy_otf = compute_d_otf(fft_shape, device)
        dx_abs_sq = torch.abs(dx_otf) ** 2
        dy_abs_sq = torch.abs(dy_otf) ** 2

        y_padded = pad_to_fft(lensless, fft_shape)
        ones_y = torch.ones(B, C, H, W, device=device)
        ctc = pad_to_fft(ones_y, fft_shape)
        denom = mu1 * otf_abs_sq + mu2 * (dx_abs_sq + dy_abs_sq) + mu3 + 1e-8
        x = torch.zeros(B, C, fft_shape[0], fft_shape[1], device=device)
        v = torch.zeros_like(x)
        ux_var = torch.zeros_like(x)
        uy_var = torch.zeros_like(x)
        w = torch.zeros_like(x)
        alpha1 = torch.zeros_like(x)
        alpha2x = torch.zeros_like(x)
        alpha2y = torch.zeros_like(x)
        alpha3 = torch.zeros_like(x)
        for _ in range(self.n_iter):
            dx, dy = finite_diff(x)
            ux_var = soft_threshold(dx + alpha2x / mu2, tau)
            uy_var = soft_threshold(dy + alpha2y / mu2, tau)
            Hx_padded = torch.fft.irfft2(otf * torch.fft.rfft2(x), s=fft_shape)
            v = (alpha1 + mu1 * Hx_padded + y_padded) / (ctc + mu1)
            w = torch.clamp(x + alpha3 / mu3, min=0.0)
            rhs_tv = finite_diff_adjoint(ux_var - alpha2x / mu2, uy_var - alpha2y / mu2)
            rhs_spatial = (mu1 * torch.fft.irfft2(otf_conj * torch.fft.rfft2(v - alpha1 / mu1), s=fft_shape) + mu2 * rhs_tv + mu3 * (w - alpha3 / mu3))
            X = torch.fft.rfft2(rhs_spatial) / denom
            x_new = torch.fft.irfft2(X, s=fft_shape)
            Hx_new = torch.fft.irfft2(otf * torch.fft.rfft2(x_new), s=fft_shape)
            dx_new, dy_new = finite_diff(x_new)
            alpha1 = alpha1 + mu1 * (Hx_new - v)
            alpha2x = alpha2x + mu2 * (dx_new - ux_var)
            alpha2y = alpha2y + mu2 * (dy_new - uy_var)
            alpha3 = alpha3 + mu3 * (x_new - w)
            x = x_new
        reconstruction = crop_from_fft(x, (H, W))
        reconstruction = torch.clamp(reconstruction, min=0.0)
        recon_flat = reconstruction.flatten(1)
        recon_max = recon_flat.max(dim=1).values.clamp(min=1e-8)
        recon_max = recon_max[:, None, None, None]
        reconstruction = reconstruction / recon_max
        return {"reconstruction": reconstruction}

    def __str__(self):
        all_parameters = sum([p.numel() for p in self.parameters()])
        trainable_parameters = sum([p.numel() for p in self.parameters() if p.requires_grad])
        result_info = super().__str__()
        result_info += f"\nAll parameters: {all_parameters}"
        result_info += f"\nTrainable parameters: {trainable_parameters}"
        return result_info
