"""
Calculate metrics between ground truth and reconstructed images.
Usage: python calculate_metrics.py --gt_dir path/to/lensed --pred_dir path/to/reconstructions

Note: gt images are resized to match pred spatial dimensions if they differ
(e.g. DigiCam lensed images are up to 500x500 while reconstructions are capped at MAX_SIZE=270).
"""
import argparse
from pathlib import Path
import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image
from torchmetrics.image import PeakSignalNoiseRatio, StructuralSimilarityIndexMeasure
import lpips

def load_image(path):
    img = np.array(Image.open(path).convert("RGB")).astype(np.float32) / 255.0
    return torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0)  # (1, C, H, W)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gt_dir", required=True)
    parser.add_argument("--pred_dir", required=True)
    args = parser.parse_args()
    gt_dir = Path(args.gt_dir)
    pred_dir = Path(args.pred_dir)
    gt_files = sorted(gt_dir.glob("*.png"))
    psnr_fn = PeakSignalNoiseRatio(data_range=1.0)
    ssim_fn = StructuralSimilarityIndexMeasure(data_range=1.0)
    lpips_fn = lpips.LPIPS(net='vgg')
    results = {"psnr": [], "ssim": [], "lpips": [], "mse": []}
    n_matched = 0
    for gt_path in gt_files:
        pred_path = pred_dir / gt_path.name
        if not pred_path.exists():
            continue
        n_matched += 1
        gt = load_image(gt_path)
        pred = load_image(pred_path)
        if gt.shape[-2:] != pred.shape[-2:]:
            gt = F.interpolate(gt, size=pred.shape[-2:], mode="bilinear", align_corners=False)
        results["psnr"].append(psnr_fn(pred, gt).item())
        results["ssim"].append(ssim_fn(pred, gt).item())
        results["lpips"].append(lpips_fn(pred * 2 - 1, gt * 2 - 1).mean().item())
        results["mse"].append(torch.nn.functional.mse_loss(pred, gt).item())
    if n_matched == 0:
        print(f"No matching files found between {gt_dir} and {pred_dir}")
        return
    print(f"Evaluated {n_matched} image pairs")
    print(f"{'Metric':<10} {'Mean':>10} {'Std':>10}")
    print("-" * 32)
    for name, vals in results.items():
        vals = np.array(vals)
        print(f"{name.upper():<10} {vals.mean():>10.4f} {vals.std():>10.4f}")

if __name__ == "__main__":
    main()