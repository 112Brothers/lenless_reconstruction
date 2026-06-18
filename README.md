# Lensless Camera Reconstruction

Implementation of ADMM algorithms for lensless imaging based on:

- [Le-ADMM (Monakhova et al., 2019)](https://arxiv.org/abs/1908.11502)
- [Modular Le-ADMM (Bezzam et al., 2025)](https://arxiv.org/abs/2502.01102)

## Models

| Model | Description |
|-------|-------------|
| ADMM-100 | Standard ADMM with fixed parameters (μ=1e-4, τ=2e-4) |
| Le-ADMM-20 | Unrolled ADMM with learnable μ_i, τ_i per iteration |
| Modular pre+post | Le-ADMM-5 + U-Net pre + U-Net post |
| Modular pre only | Le-ADMM-5 + U-Net pre |
| Modular post only | Le-ADMM-5 + U-Net post |

## Installation

```bash
git clone https://github.com/112Brothers/lenless.git
cd lenless
pip install -r requirements.txt
```

## Dataset

[DigiCam-Mirflickr-MultiMask-10K](https://huggingface.co/datasets/bezzam/DigiCam-Mirflickr-MultiMask-10K)


Custom directory format:
```
data_dir/
├── lensless/   *.png
├── masks/      *.npy
└── lensed/     *.png  (optional, ground truth)
```



## Demo

See `demo.ipynb`

