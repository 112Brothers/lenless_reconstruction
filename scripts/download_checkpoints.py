import os
from pathlib import Path
import requests


def download_from_url(url, dest_path):
    dest_path = Path(dest_path)
    dest_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"Downloading {url} to {dest_path}...")
    response = requests.get(url, stream=True)
    response.raise_for_status()

    with open(dest_path, 'wb') as f:
        for chunk in response.iter_content(chunk_size=8192):
            f.write(chunk)

    print(f"Downloaded to {dest_path}")


def main():
    #ADMM100 has no train params
    checkpoints = {
        "le_admm20": "https://huggingface.co/alexok2006/lenless/resolve/main/model_leadmm20.pth",
        "modular_pre_post": "https://huggingface.co/alexok2006/lenless/resolve/main/model_best.pth",
        "modular_pre": "https://huggingface.co/alexok2006/lenless/resolve/main/model_pre.pth",
        "modular_post": "https://huggingface.co/alexok2006/lenless/resolve/main/model_post.pth",
    }

    save_dir = Path("saved")
    save_dir.mkdir(exist_ok=True, parents=True)

    for name, url in checkpoints.items():
        dest = save_dir / f"{name}.pth"
        if dest.exists():
            print(f"Already exists: {dest}, skipping.")
            continue
        download_from_url(url, dest)

    print("\nDone downloading checkpoints.")


if __name__ == "__main__":
    main()
