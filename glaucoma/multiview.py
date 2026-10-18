"""Original-resolution views and image-supervised attention MIL (opt-in)."""
from __future__ import annotations

import random

import numpy as np
import torch
from PIL import Image, ImageEnhance, ImageOps
from torch import nn

from .data import ImageDataset, decode_image, resize_image

VIEW_VERSION = "native-fov-grid-mil-v2"


def augment_view(image, parameters):
    # Source coordinates identify the original FOV region, before view augmentation.
    mirror, angle, brightness, color = parameters
    if mirror:
        image = ImageOps.mirror(image)
    image = image.rotate(angle, Image.Resampling.BILINEAR, fillcolor=(0, 0, 0))
    image = ImageEnhance.Brightness(image).enhance(brightness)
    return ImageEnhance.Color(image).enhance(color)


def view_boxes(width, height):
    """Four overlapping quadrants cover the FOV; a smaller centre adds scale."""
    side = min(width, height)
    boxes = []
    for fraction, cx, cy in ((.65, .325, .325), (.65, .675, .325),
                             (.65, .325, .675), (.65, .675, .675), (.4, .5, .5)):
        length = max(1, round(side * fraction))
        x = min(max(0, round(cx * width - length / 2)), width - length)
        y = min(max(0, round(cy * height - length / 2)), height - length)
        boxes.append((x, y, x + length, y + length))
    return boxes


def tensor_image(image, size, augmentation=None):
    return tensor_pixels(resize_image(image, size), augmentation)


def tensor_pixels(array, augmentation=None):
    if augmentation is not None:
        array = np.array(augment_view(Image.fromarray(array), augmentation), dtype=np.uint8, copy=True)
    return torch.from_numpy(np.array(array, copy=True)).permute(2, 0, 1).float().div_(255)


def native_pixels(image, size, local=True, include_global=True):
    """Fixed uint8 views, always sampled before any resize or random augmentation."""
    global_image = resize_image(image, size) if include_global else None
    if not local:
        return global_image
    width, height = image.size
    patches, coordinates, mask, status = [], [], [], []
    for box in view_boxes(width, height):
        coordinates.append([box[0]/width, box[1]/height, box[2]/width, box[3]/height])
        try:
            crop = image.crop(box)
            preview = crop.copy()
            preview.thumbnail((64, 64))
            valid = bool((np.asarray(preview).max(axis=2) > 10).mean() >= .25)
            patches.append(resize_image(crop, size) if valid else np.zeros((size, size, 3), np.uint8))
            mask.append(valid)
            status.append(0 if valid else 1)
        except (OSError, ValueError):
            # Global decode errors still fail loudly; only local extraction can fall back.
            patches.append(np.zeros((size, size, 3), np.uint8))
            mask.append(False)
            status.append(2)
    return {"global": global_image, "local": np.stack(patches),
            "coordinates": np.array(coordinates, dtype=np.float32),
            "mask": np.array(mask, dtype=np.bool_), "status": np.array(status, dtype=np.int8)}


def pixels_to_views(pixels, augmentation=None):
    if not isinstance(pixels, dict):
        return tensor_pixels(pixels, augmentation)
    return {"global": tensor_pixels(pixels["global"], augmentation),
            "local": torch.stack([tensor_pixels(p, augmentation) for p in pixels["local"]]),
            "coordinates": torch.from_numpy(pixels["coordinates"].copy()),
            "mask": torch.from_numpy(pixels["mask"].copy()),
            "status": torch.from_numpy(pixels["status"].copy())}


def native_views(image, size, local=True, augmentation=None):
    return pixels_to_views(native_pixels(image, size, local), augmentation)


class NativeViewDataset(ImageDataset):
    def __init__(self, rows, config, age_transform=None, augment=False, cache=None):
        super().__init__(rows, config["size"], age_transform, augment)
        if config.get("view_version") != VIEW_VERSION:
            raise ValueError("Unknown native-view preprocessing version")
        self.local = bool(config.get("multiview", False))
        self.view_paths = cache
        self.view_arrays = {}

    def __getstate__(self):
        state = super().__getstate__()
        state["view_arrays"] = {}
        return state

    def __getitem__(self, index):
        row = self.rows[index]
        # Explicitly disable JPEG draft downsampling for both P1 and P2.
        if self.view_paths:
            for key, path in self.view_paths.items():
                if key not in self.view_arrays:
                    self.view_arrays[key] = np.load(path, mmap_mode="r", allow_pickle=False)
            pixels = np.array(self.view_arrays["global"][index], copy=True)
            if self.local:
                record = self.view_arrays["local"][index]
                pixels = {"global": pixels, "local": record["pixels"], "coordinates": record["coordinates"],
                          "status": record["status"], "mask": record["status"] == 0}
        else:
            image = decode_image(row["path"], self.size, full_decode=True)
            pixels = native_pixels(image, self.size, self.local)
        augmentation = (random.random() < .5, random.uniform(-15, 15),
                        random.uniform(.9, 1.1), random.uniform(.9, 1.1)) if self.augment else None
        return (pixels_to_views(pixels, augmentation), torch.from_numpy(self.ages[index]),
                torch.tensor(float(row["label"])), index)


def image_dataset(rows, config, age=None, augment=False, cache=None):
    if config.get("multiview") or config.get("native_resolution"):
        if cache is not None and not isinstance(cache, dict):
            raise ValueError("Resized pixel caches cannot supply native-resolution crops")
        return NativeViewDataset(rows, config, age, augment, cache)
    return ImageDataset(rows, config["size"], age, augment, cache)


def move_images(images, device):
    if isinstance(images, dict):
        return {key: value.to(device, non_blocking=True) for key, value in images.items()}
    return images.to(device, non_blocking=True)


def image_count(images):
    return len(images["global"]) if isinstance(images, dict) else len(images)


def view_diagnostics(images, counts):
    if isinstance(images, dict):
        counts["images"] += image_count(images)
        counts["global_fallback_images"] += int((~images["mask"].any(dim=1)).sum())
        counts["invalid_patches"] += int((~images["mask"]).sum())
        counts["extraction_failures"] += int((images["status"] == 2).sum())


def empty_diagnostics():
    return dict(images=0, global_fallback_images=0, invalid_patches=0, extraction_failures=0)


def flip_images(images):
    if not isinstance(images, dict):
        return images.flip(-1)
    # As in training, flip each view while keeping its original source-region tag.
    return {**images, "global": images["global"].flip(-1), "local": images["local"].flip(-1)}


class AttentionMIL(nn.Module):
    def __init__(self, dimension, use_age):
        super().__init__()
        self.position = nn.Linear(4, dimension, bias=False)
        self.attention = nn.Sequential(nn.Linear(dimension, 128), nn.Tanh(), nn.Linear(128, 1))
        self.fusion = nn.Sequential(nn.Linear(dimension * 2 + (3 if use_age else 0), 128),
                                    nn.GELU(), nn.Linear(128, 1))
        # P1 warm start is an exact initial global path, even with valid local views.
        nn.init.zeros_(self.fusion[-1].weight)
        nn.init.zeros_(self.fusion[-1].bias)

    def forward(self, global_features, local_features, coordinates, mask, age):
        with torch.autocast(global_features.device.type, enabled=False):
            local_features = local_features.float()
            mask = mask.bool()
            positioned = local_features + self.position(coordinates.float())
            scores = self.attention(positioned).squeeze(-1).masked_fill(~mask, -1e9)
            weights = scores.softmax(dim=1) * mask
            weights = weights / weights.sum(dim=1, keepdim=True).clamp_min(1e-12)
            pooled = (weights.unsqueeze(-1) * positioned).sum(dim=1)
            inputs = [global_features.float(), pooled]
            if age is not None:
                inputs.append(age.float())
            residual = self.fusion(torch.cat(inputs, dim=1)).squeeze(-1)
            # Empty bags neither change the global logit nor propagate a local gradient.
            return residual * mask.any(dim=1), weights
