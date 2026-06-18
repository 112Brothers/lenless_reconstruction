import torch
import torch.nn.functional as F

# size utils 

def next_fast_fft_size(n):
    while True:
        m = n
        for p in [2, 3, 5]:
            while m % p == 0:
                m //= p
        if m == 1:
            return n
        n += 1

def compute_fft_shape(H, W, factor=2):
    fft_H = next_fast_fft_size(factor * H)
    fft_W = next_fast_fft_size(factor * W)
    return fft_H, fft_W

def pad_to_fft(x, fft_shape):
    _, _, H, W = x.shape
    fft_H, fft_W = fft_shape
    pad_top = (fft_H - H) // 2
    pad_bottom = fft_H - H - pad_top
    pad_left = (fft_W - W) // 2
    pad_right = fft_W - W - pad_left
    return F.pad(x, (pad_left, pad_right, pad_top, pad_bottom), mode="constant", value=0)

def crop_from_fft(x, target_shape):
    _, _, fft_H, fft_W = x.shape
    H, W = target_shape
    start_h = (fft_H - H) // 2
    start_w = (fft_W - W) // 2
    return x[:, :, start_h:start_h + H, start_w:start_w + W]

def psf_to_otf(psf, fft_shape):
    psf_padded = pad_to_fft(psf, fft_shape)
    psf_padded = torch.fft.ifftshift(psf_padded, dim=(-2, -1))
    otf = torch.fft.rfft2(psf_padded)
    return otf

def compute_d_otf(fft_shape, device):
    fft_H, fft_W = fft_shape
    dx_kernel = torch.zeros(1, 1, fft_H, fft_W, device=device)
    dx_kernel[0, 0, 0, 0] = -1.0
    dx_kernel[0, 0, 0, 1] = 1.0
    dx_otf = torch.fft.rfft2(dx_kernel)
    dy_kernel = torch.zeros(1, 1, fft_H, fft_W, device=device)
    dy_kernel[0, 0, 0, 0] = -1.0
    dy_kernel[0, 0, 1, 0] = 1.0
    dy_otf = torch.fft.rfft2(dy_kernel)
    return dx_otf, dy_otf

def finite_diff(x):
    dx = torch.roll(x, shifts=-1, dims=-1) - x
    dy = torch.roll(x, shifts=-1, dims=-2) - x
    return dx, dy

def finite_diff_adjoint(dx, dy):
    adj_dx = torch.roll(dx, shifts=1, dims=-1) - dx
    adj_dy = torch.roll(dy, shifts=1, dims=-2) - dy
    return adj_dx + adj_dy

def soft_threshold(x, threshold):
    return torch.sign(x) * torch.clamp(torch.abs(x) - threshold, min=0)
