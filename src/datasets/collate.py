import torch
import torch.nn.functional as F

def collate_fn(dataset_items: list[dict]):
    result_batch = {}
    MAX_SIZE = 270
    orig_h, orig_w = dataset_items[0]["lensless"].shape[1:]
    target_h, target_w = orig_h, orig_w
    if target_h > MAX_SIZE or target_w > MAX_SIZE:
        scale = MAX_SIZE / max(target_h, target_w)
        target_h = int(target_h * scale)
        target_w = int(target_w * scale)
    else:
        scale = 1.0
    lensless_list = []
    for item in dataset_items:
        img = item["lensless"]
        if img.shape[1:] != (target_h, target_w):
            img = F.interpolate(
                img.unsqueeze(0), size=(target_h, target_w), mode="bilinear", align_corners=False
            ).squeeze(0)
        lensless_list.append(img)
    result_batch["lensless"] = torch.stack(lensless_list)
    mask_list = []
    for item in dataset_items:
        mask = item["mask"]
        mask_h, mask_w = mask.shape
        if scale != 1.0:
            new_mask_h = max(1, int(round(mask_h * scale)))
            new_mask_w = max(1, int(round(mask_w * scale)))
            mask = F.interpolate(mask.unsqueeze(0).unsqueeze(0), size=(new_mask_h, new_mask_w), mode="bilinear", align_corners=False).squeeze(0).squeeze(0)
        mask_sum = mask.sum().clamp(min=1e-8)
        mask = mask / mask_sum
        mask_list.append(mask)
    max_mh = max(m.shape[0] for m in mask_list)
    max_mw = max(m.shape[1] for m in mask_list)
    padded_masks = []
    for mask in mask_list:
        mh, mw = mask.shape
        if mh != max_mh or mw != max_mw:
            pad_bottom = max_mh - mh
            pad_right = max_mw - mw
            mask = F.pad(mask, (0, pad_right, 0, pad_bottom), value=0.0)
            mask_sum = mask.sum().clamp(min=1e-8)
            mask = mask / mask_sum
        padded_masks.append(mask)
    result_batch["mask"] = torch.stack(padded_masks)
    # groundtruth
    if "lensed" in dataset_items[0]:
        lensed_list = []
        for item in dataset_items:
            img = item["lensed"]
            if img.shape[1:] != (target_h, target_w):
                img = F.interpolate(
                    img.unsqueeze(0), size=(target_h, target_w), mode="bilinear", align_corners=False
                ).squeeze(0)
            lensed_list.append(img)
        result_batch["lensed"] = torch.stack(lensed_list)
    if "image_id" in dataset_items[0]:
        result_batch["image_id"] = [item["image_id"] for item in dataset_items]
    return result_batch
