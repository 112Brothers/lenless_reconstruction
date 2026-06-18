import warnings
from pathlib import Path

import hydra
import torch
from hydra.utils import instantiate, to_absolute_path
from omegaconf import OmegaConf

from src.datasets.data_utils import get_dataloaders
from src.trainer import Inferencer
from src.utils.init_utils import set_random_seed

warnings.filterwarnings("ignore", category=UserWarning)


@hydra.main(version_base=None, config_path="src/configs", config_name="inference_lensless")
def main(config):
    set_random_seed(config.inferencer.seed)
    if config.inferencer.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = config.inferencer.device
    config_mutable = OmegaConf.to_container(config, resolve=True)
    save_path_str = to_absolute_path(config.inferencer.save_path)
    if config.inferencer.get("from_pretrained") is not None:
        from_pretrained_abs = to_absolute_path(config.inferencer.from_pretrained)
        OmegaConf.update(config, "inferencer.from_pretrained", from_pretrained_abs)
    if hasattr(config.datasets, "test") and hasattr(config.datasets.test, "data_dir"):
        if config.datasets.test.data_dir is not None:
            abs_data_dir = to_absolute_path(config.datasets.test.data_dir)
            OmegaConf.update(config, "datasets.test.data_dir", abs_data_dir)
    dataloaders, batch_transforms = get_dataloaders(config, device)
    model = instantiate(config.model).to(device)
    print(model)
    metrics = instantiate(config.metrics)
    save_path = Path(save_path_str)
    save_path.mkdir(exist_ok=True, parents=True)
    skip_model_load = config.inferencer.get("skip_model_load", False)
    inferencer = Inferencer(
        model=model,
        config=config,
        device=device,
        dataloaders=dataloaders,
        batch_transforms=batch_transforms,
        save_path=save_path,
        metrics=metrics,
        skip_model_load=skip_model_load,
    )
    logs = inferencer.run_inference()
    for part in logs.keys():
        for key, value in logs[part].items():
            full_key = part + "_" + key
            print(f"    {full_key:15s}: {value}")

if __name__ == "__main__":
    main()
