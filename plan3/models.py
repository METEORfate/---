from __future__ import annotations

import copy
import math

import torch
from torch import nn
from torch.nn import functional as F

from glaucoma.models import FundusModel

ARCHITECTURES = {"dinov3": "vit_large_patch16_dinov3.lvd1689m",
                 "retfound_dinov2": "vit_large_patch14_dinov2",
                 "convnext": "convnext_large.fb_in22k_ft_in1k"}


def clean_state(state):
    for key in ("model_state_dict", "teacher", "model", "state_dict", "student"):
        if key in state and isinstance(state[key], dict):
            state = state[key]
            break
    result = {}
    for key, value in state.items():
        for prefix in ("module.", "_orig_mod.", "backbone."):
            if key.startswith(prefix):
                key = key[len(prefix):]
        if isinstance(value, torch.Tensor):
            result[key] = value
    return result


def load_backbone(backbone, state):
    """Strict tensor coverage with an explicit positional-grid resize for resolution upgrades."""
    from timm.layers import resample_abs_pos_embed
    expected = backbone.state_dict()
    state = clean_state(state)
    if "pos_embed" in state and state["pos_embed"].shape != expected.get("pos_embed", torch.empty(0)).shape:
        target = expected["pos_embed"]
        prefix = getattr(backbone, "num_prefix_tokens", 1)
        if getattr(backbone, "no_embed_class", False):
            prefix = 0
        grid = backbone.patch_embed.grid_size
        state["pos_embed"] = resample_abs_pos_embed(state["pos_embed"], grid, num_prefix_tokens=prefix)
        if state["pos_embed"].shape != target.shape:
            raise ValueError("Unsupported positional embedding migration")
    used = {k: v for k, v in state.items() if k in expected}
    if set(used) != set(expected) or any(used[k].shape != v.shape for k, v in expected.items()):
        missing = sorted(set(expected)-set(used))
        raise ValueError(f"Incomplete/incompatible pretrained encoder: {missing[:10]}")
    backbone.load_state_dict(used, strict=True)


class Encoder(nn.Module):
    def __init__(self, config, weight=None, source=None):
        super().__init__()
        self.config = dict(config)
        self.kind = config["model"]
        if self.kind in ("rfm", "effnet", "green"):
            self.base = FundusModel({**config, "multiview": False}, weight)
            self.dim = self.base.feature_dim
            self.base.head = nn.Identity()
        else:
            import timm
            spec = source or config.get("source_spec")
            if not spec or spec.get("architecture") != ARCHITECTURES.get(self.kind):
                raise ValueError("Challenge encoder needs its verified architecture and normalization manifest")
            options = dict(pretrained=False, num_classes=0)
            if self.kind != "convnext":
                options.update(img_size=config["size"], dynamic_img_size=True, dynamic_img_pad=True)
            self.backbone = timm.create_model(spec["architecture"], **options)
            self.dim = self.backbone.num_features
            self.register_buffer("mean", torch.tensor(spec["mean"]).view(1, 3, 1, 1))
            self.register_buffer("std", torch.tensor(spec["std"]).view(1, 3, 1, 1))
            if weight is not None:
                load_backbone(self.backbone, torch.load(weight, map_location="cpu", weights_only=True))
            if config.get("gradient_checkpointing", True):
                self.backbone.set_grad_checkpointing(True)

    def forward(self, image):
        if hasattr(self, "base"):
            base = self.base
            x = (image-base.mean)/base.std
            if self.kind == "rfm":
                # RFM reads pre-norm block 8. Do not append random untrained blocks.
                patches, prefix = base.backbone.forward_intermediates(x, intermediates_only=True,
                    return_prefix_tokens=True, indices=[8], output_fmt="NLC", stop_early=True)[-1]
                pooled = torch.cat([prefix[:, 0], patches.mean(1)], 1)
                tokens = torch.cat([prefix[:, :1].expand(-1, patches.shape[1], -1), patches], 2)
            else:
                features = base.backbone.forward_features(x)
                pooled = base.backbone.forward_head(features)
                tokens = features.flatten(2).transpose(1, 2) if features.ndim == 4 else features[:, base.backbone.num_prefix_tokens:]
        else:
            features = self.backbone.forward_features((image-self.mean)/self.std)
            pooled = self.backbone.forward_head(features)
            tokens = features.flatten(2).transpose(1, 2) if features.ndim == 4 else features[:, self.backbone.num_prefix_tokens:]
        # Bound attention memory without destroying the original-resolution encoder input.
        if tokens.shape[1] > 64:
            tokens = F.adaptive_avg_pool1d(tokens.transpose(1, 2), 64).transpose(1, 2)
        return pooled.float(), tokens.float()


class CrossBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.norm = nn.LayerNorm(256)
        self.attention = nn.MultiheadAttention(256, 4, batch_first=True)
        self.ff = nn.Sequential(nn.LayerNorm(256), nn.Linear(256, 1024), nn.GELU(), nn.Linear(1024, 256))

    def forward(self, query, memory, padding):
        x = self.norm(query)
        query = query + self.attention(x, memory, memory, key_padding_mask=padding, need_weights=False)[0]
        return query + self.ff(query)


class Classifier(nn.Module):
    def __init__(self, config, encoder=None):
        super().__init__()
        self.config = dict(config)
        self.encoder = encoder or Encoder(config)
        dim = self.encoder.dim
        self.view = config.get("view", "global")
        if self.view not in ("global", "local", "probability", "concat", "cross"):
            raise ValueError("Unknown classifier view")
        self.global_head = nn.Linear(dim+3, 1)
        self.local_head = nn.Linear(dim+3, 1)
        if self.view in ("concat", "cross"):
            self.project = nn.Linear(dim, 256)
            self.coordinates = nn.Linear(5, 256)
            self.blocks = nn.ModuleList([CrossBlock(), CrossBlock()]) if self.view == "cross" else nn.ModuleList()
            self.residual = nn.Linear((256 if self.view == "cross" else dim*2)+3, 1)
            nn.init.zeros_(self.residual.weight)
            nn.init.zeros_(self.residual.bias)

    def forward(self, views, age):
        g, gt = self.encoder(views["global"])
        gl = self.global_head(torch.cat([g, age], 1)).squeeze(1)
        if self.view == "global":
            return {"logits": gl, "global": gl}
        b, v = views["mask"].shape
        valid = views["mask"].flatten().nonzero(as_tuple=True)[0]
        local = g.new_zeros(b*v, g.shape[1])
        # Match token count by using the first valid local view; never encode invalid zero ROIs.
        parts, token_parts, index_parts = [], [], []
        if isinstance(views["local"], (list, tuple)):
            for view, images in enumerate(views["local"]):
                chosen = views["mask"][:, view].nonzero(as_tuple=True)[0]
                for ix in chosen.split(int(self.config.get("local_chunk_size", 1))):
                    if len(ix):
                        p, t = self.encoder(images[ix])
                        parts.append(p)
                        token_parts.append(t)
                        index_parts.append(ix*v+view)
        else:
            for ix in valid.split(int(self.config.get("local_chunk_size", 1))):
                if len(ix):
                    p, t = self.encoder(views["local"].flatten(0, 1)[ix])
                    parts.append(p)
                    token_parts.append(t)
                    index_parts.append(ix)
        if parts:
            valid = torch.cat(index_parts)
            local = local.index_copy(0, valid, torch.cat(parts))
            token_count = min(t.shape[1] for t in token_parts)
            tokens = torch.cat([F.adaptive_avg_pool1d(t.transpose(1, 2), token_count).transpose(1, 2) for t in token_parts])
        else:
            tokens = g.new_zeros(0, 1, g.shape[1])
        mask = views["mask"]
        any_valid = mask.any(1)
        local = local.view(b, v, -1)
        lp = (local*mask.unsqueeze(2)).sum(1)/mask.sum(1).clamp_min(1).unsqueeze(1)
        ll = self.local_head(torch.cat([lp, age], 1)).squeeze(1)
        if self.view == "local":
            logits = torch.where(any_valid, ll, gl)
        elif self.view == "probability":
            probability = (gl.sigmoid()+ll.sigmoid())/2
            logits = torch.where(any_valid, torch.logit(probability.clamp(1e-6, 1-1e-6)), gl)
        else:
            if self.view == "concat":
                joint = torch.cat([g, lp], 1)
            else:
                n = tokens.shape[1]
                lt = g.new_zeros(b*v, n, 256)
                if len(valid):
                    projected = self.project(tokens) + self.coordinates(views["coordinates"].flatten(0, 1)[valid]).unsqueeze(1)
                    lt = lt.index_copy(0, valid, projected)
                memory = torch.cat([self.project(gt), lt.view(b, v*n, 256)], 1)
                padding = torch.cat([torch.zeros(b, gt.shape[1], dtype=torch.bool, device=g.device),
                                     (~mask).unsqueeze(2).expand(-1, -1, n).reshape(b, -1)], 1)
                query = self.project(gt)
                for block in self.blocks:
                    query = block(query, memory, padding)
                joint = query.mean(1)
            logits = gl + self.residual(torch.cat([joint, age], 1)).squeeze(1)*any_valid
        return {"logits": logits, "global": gl, "local": ll, "local_valid": any_valid}


def classification_loss(result, labels, ranking_weight=0.):
    loss = F.binary_cross_entropy_with_logits(result["logits"].float(), labels)
    if "local" in result:
        loss = loss + .25*F.binary_cross_entropy_with_logits(result["global"].float(), labels)
        valid = result["local_valid"]
        if valid.any():
            loss = loss + .25*F.binary_cross_entropy_with_logits(result["local"][valid].float(), labels[valid])
    p, n = result["logits"][labels == 1], result["logits"][labels == 0]
    if ranking_weight and p.numel() and n.numel():
        loss = loss + ranking_weight*F.softplus(n[None, :]-p[:, None]).mean()
    return loss


class EMA:
    def __init__(self, model, decay=.999):
        self.model = copy.deepcopy(model).eval().requires_grad_(False)
        self.decay, self.updates = decay, 0

    @torch.no_grad()
    def update(self, model, successful=True):
        if not successful:
            return
        for key, value in self.model.state_dict().items():
            source = model.state_dict()[key]
            if value.is_floating_point():
                value.lerp_(source, 1-self.decay)
            else:
                value.copy_(source)
        self.updates += 1


class Locator(nn.Module):
    def __init__(self, encoder):
        super().__init__()
        self.encoder = encoder
        self.head = nn.Linear(encoder.dim, 5)

    def forward(self, image):
        output = self.head(self.encoder(image)[0])
        corners = output[:, :4].sigmoid()
        low, high = torch.minimum(corners[:, :2], corners[:, 2:]), torch.maximum(corners[:, :2], corners[:, 2:])
        return torch.cat([low, high], 1), output[:, 4]


class Distillation(nn.Module):
    """Image-level DINO-style continuation, not a reproduction of full DINOv3 pretraining."""
    def __init__(self, encoder, output_dim=1024):
        super().__init__()
        self.encoder = encoder
        self.head = nn.Sequential(nn.Linear(encoder.dim, 512), nn.GELU(), nn.Linear(512, output_dim, bias=False))
        self.register_buffer("center", torch.zeros(1, output_dim))

    def forward(self, image):
        feature = F.normalize(self.encoder(image)[0], dim=1)
        return self.head(feature)


def dino_loss(student, teacher, center, student_temperature=.1, teacher_temperature=.04):
    terms = []
    for i, target in enumerate(teacher):
        target = ((target.detach()-center)/teacher_temperature).softmax(1)
        for j, prediction in enumerate(student):
            if i != j:
                terms.append(-(target*F.log_softmax(prediction/student_temperature, 1)).sum(1).mean())
    if not terms:
        raise ValueError("Self-distillation requires at least two corresponding image views")
    return torch.stack(terms).mean()
