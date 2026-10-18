from __future__ import annotations

import hashlib
import os
from pathlib import Path
from types import MethodType

import torch
from torch import nn
import timm

from .common import ROOT, read_json, sha256, write_json

WEIGHTS = ROOT / "weights"
SPECS = {
    "green": {"timm": "vit_small_patch14_reg4_dinov2", "size": 392, "dim": 384,
              "mean": [0.5] * 3, "std": [0.5] * 3},
    "effnet": {"timm": "tf_efficientnetv2_s.in1k", "size": 384, "dim": 1280,
               "mean": [0.5] * 3, "std": [0.5] * 3},
    "rfm": {"timm": "vit_base_patch16_dinov3.lvd1689m", "size": 336, "dim": 1536,
            "mean": [0.4271227, 0.24723781, 0.13661437],
            "std": [0.26101578, 0.16189588, 0.09271526]},
}
SOURCES = {
    "green": {"url": "https://github.com/justinengelmann/RETFound_Green/releases/download/v0.1/retfoundgreen_statedict.pth",
              "filename": "retfoundgreen_statedict.pth", "expected_bytes": 87396676,
              "source": "https://github.com/justinengelmann/RETFound_Green",
              "license": "https://raw.githubusercontent.com/justinengelmann/RETFound_Green/main/LICENSE"},
    "rfm": {"url": "https://drive.google.com/uc?export=download&id=1YqWkop3vk4EajrdZ_KApsmg0QhGIhVVk",
            "filename": "rfm-1.3.pt", "source": "https://github.com/ok1zjf/RFM",
            "license": "https://raw.githubusercontent.com/ok1zjf/RFM/main/LICENSE-DINOv3.md"},
}


def download_weights(name):
    """Only public official weights; never execute downloaded Python or pickle objects."""
    import requests
    from html.parser import HTMLParser
    WEIGHTS.mkdir(parents=True, exist_ok=True)
    if name == "effnet":
        # timm handles the official HF model card and caches; export a plain state dict.
        path = WEIGHTS / "efficientnetv2_s_in1k.pth"
        if not path.exists():
            model = timm.create_model(SPECS[name]["timm"], pretrained=True, num_classes=0)
            torch.save(model.state_dict(), path)
        metadata = {"source": "https://huggingface.co/timm/tf_efficientnetv2_s.in1k",
                    "license": "https://huggingface.co/timm/tf_efficientnetv2_s.in1k/blob/main/README.md"}
    else:
        spec = SOURCES[name]
        path = WEIGHTS / spec["filename"]
        metadata = dict(spec)
        if not path.exists():
            with requests.Session() as session:
                response = session.get(spec["url"], stream=True, timeout=(20, 180))
                response.raise_for_status()
                if "text/html" in response.headers.get("Content-Type", ""):
                    if name != "rfm":
                        raise RuntimeError("Received HTML instead of model weights")
                    class Form(HTMLParser):
                        action = None
                        fields = {}
                        def handle_starttag(self, tag, attrs):
                            a = dict(attrs)
                            if tag == "form" and a.get("id") == "download-form":
                                self.action = a.get("action")
                            if tag == "input" and "name" in a:
                                self.fields[a["name"]] = a.get("value", "")
                    form = Form()
                    form.feed(response.text)
                    if not form.action or not form.action.startswith("https://drive.usercontent.google.com/"):
                        raise RuntimeError("Google Drive download unavailable; no login or permission bypass attempted")
                    response = session.get(form.action, params=form.fields, stream=True, timeout=(20, 180))
                    response.raise_for_status()
                if "text/html" in response.headers.get("Content-Type", ""):
                    raise RuntimeError("Weight download returned an HTML error")
                temp = path.with_suffix(".part")
                with temp.open("wb") as f:
                    for block in response.iter_content(1024 * 1024):
                        f.write(block)
                if temp.stat().st_size < 1_000_000 or ("expected_bytes" in spec and temp.stat().st_size != spec["expected_bytes"]):
                    raise RuntimeError("Incomplete weight download; original file was not replaced")
                os.replace(temp, path)
                license_response = session.get(spec["license"], timeout=30)
                license_response.raise_for_status()
                (WEIGHTS / f"{name}_LICENSE.txt").write_text(license_response.text, encoding="utf-8")
    metadata.update(bytes=path.stat().st_size, sha256=sha256(path))
    write_json(WEIGHTS / f"{name}_source.json", metadata)
    return path


def _rope_on_device(self, feat_shape, no_aug=False):
    # Adapted from RFM official model.py (MIT); see SOURCES.md.
    from timm.layers.pos_embed_sincos import make_coords_dinov3
    coords = make_coords_dinov3(*feat_shape, normalize_coords=self.normalize_coords,
                               grid_indexing=self.grid_indexing, grid_offset=self.grid_offset,
                               device=next(self.buffers()).device)
    if not no_aug:
        coords = self._apply_coord_augs(coords)
    sin, cos = self._get_pos_embed_from_coords(coords)
    return torch.cat([sin, cos], dim=-1)


def _load_backbone(backbone, path, name):
    state = torch.load(path, map_location="cpu", weights_only=True)
    for key in ("model_state_dict", "model", "teacher", "student"):
        if key in state and isinstance(state[key], dict):
            state = state[key]
            break
    cleaned = {}
    for key, value in state.items():
        for prefix in ("module.", "_orig_mod.", "backbone.", "backbone."):
            if key.startswith(prefix):
                key = key[len(prefix):]
        if isinstance(value, torch.Tensor):
            cleaned[key] = value
    expected = backbone.state_dict()
    # RFM's final norm is intentionally absent: forward reads only pre-norm block 8.
    allowed_missing = {k for k in expected if name == "rfm" and k.startswith("norm.")}
    used = {k: v for k, v in cleaned.items() if k in expected}
    missing = set(expected) - set(used) - allowed_missing
    if missing:
        raise RuntimeError(f"Pretrained backbone incompletely restored: {sorted(missing)[:12]}")
    backbone.load_state_dict(used, strict=False)


class FundusModel(nn.Module):
    def __init__(self, config, pretrained_path=None):
        super().__init__()
        self.config = dict(config)
        name = config["model"]
        spec = SPECS[name]
        self.kind = name
        self.size = int(config.get("size", spec["size"]))
        options = {"pretrained": False, "num_classes": 0}
        if name != "effnet":
            options.update(img_size=self.size, drop_path_rate=0.0)
        self.backbone = timm.create_model(spec["timm"], **options)
        if name == "rfm":
            if timm.__version__ != "1.0.24":
                raise RuntimeError("RFM adapter requires verified timm==1.0.24")
            self.backbone.rope._create_embed = MethodType(_rope_on_device, self.backbone.rope)
            self.backbone.blocks = nn.Sequential(*list(self.backbone.blocks.children())[:9])
        if pretrained_path is not None:
            _load_backbone(self.backbone, pretrained_path, name)
        if name == "green":
            # Official recipe: create in token mode, load norm, THEN switch pooling.
            self.backbone.global_pool = "avg"
            if not isinstance(self.backbone.norm, nn.LayerNorm) or not isinstance(self.backbone.fc_norm, nn.Identity):
                raise RuntimeError("Green pretrained token-normalization order was changed")
        self.register_buffer("mean", torch.tensor(spec["mean"]).reshape(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor(spec["std"]).reshape(1, 3, 1, 1))
        self.use_age = config.get("use_age", True)
        self.feature_dim = spec["dim"]
        dimension = self.feature_dim + (3 if self.use_age else 0)
        if config.get("head_standardize", False):
            self.register_buffer("head_mean", torch.zeros(dimension))
            self.register_buffer("head_scale", torch.ones(dimension))
        if config.get("head", "linear") == "linear":
            self.head = nn.Linear(dimension, 1)
        else:
            self.head = nn.Sequential(nn.Linear(dimension, 128), nn.GELU(), nn.Dropout(0.2), nn.Linear(128, 1))

    def features(self, images):
        x = (images - self.mean) / self.std
        if self.kind == "rfm":
            intermediates = self.backbone.forward_intermediates(x, intermediates_only=True,
                return_prefix_tokens=True, indices=[8], output_fmt="NLC", stop_early=True)
            patches, prefix = intermediates[-1]
            return torch.cat([prefix[:, 0], patches.mean(dim=1)], dim=1).float()
        return self.backbone(x).float()

    def classify(self, features, age):
        x = torch.cat([features, age], dim=1) if self.use_age else features
        # Folded StandardScaler coefficients can be large; keep the tiny head FP32.
        with torch.autocast(x.device.type, enabled=False):
            x = x.float()
            if self.config.get("head_standardize", False):
                x = (x - self.head_mean) / self.head_scale
            return self.head(x).squeeze(-1)

    def forward(self, images, age):
        return self.classify(self.features(images), age)

    def configure_trainable(self, tail_blocks):
        self.backbone.requires_grad_(False)
        if tail_blocks == -1:
            self.backbone.requires_grad_(True)
        elif tail_blocks:
            if self.kind == "effnet":
                raise ValueError("CNN uses full fine-tuning (-1) or frozen features (0)")
            for block in list(self.backbone.blocks.children())[-tail_blocks:]:
                block.requires_grad_(True)
            if self.kind == "green":
                self.backbone.norm.requires_grad_(True)
        self.head.requires_grad_(True)
        return self


def save_checkpoint(path, model, config, age_transform, metadata):
    state = {k: v.detach().cpu() for k, v in model.state_dict().items()}
    payload = {"format_version": 1, "config": config, "age_transform": age_transform.to_dict(),
               "state_dict": state, "metadata": metadata}
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(".tmp")
    torch.save(payload, temp)
    os.replace(temp, path)


def load_checkpoint(path, device="cpu"):
    from .data import AgeTransform
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    if checkpoint.get("format_version") != 1:
        raise ValueError("Unsupported checkpoint")
    model = FundusModel(checkpoint["config"])
    model.load_state_dict(checkpoint["state_dict"], strict=True)
    return model.to(device).eval(), AgeTransform(**checkpoint["age_transform"]), checkpoint
