from __future__ import annotations

import csv
import math
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps

from glaucoma.common import digest, read_json, sha256, write_csv, write_json
from glaucoma.data import crop_fov, resize_image
from . import INPUT_VERSION

FIELDS = ["Image_name", "width", "height", "x_min", "y_min", "x_max", "y_max", "status"]
SPACE = "exif_transposed_original_pixels_xyxy_half_open"


def original(path):
    with Image.open(path) as source:
        image = ImageOps.exif_transpose(source).convert("RGB")
        image.load()
    return image


def fov_geometry(image):
    # Run the existing crop on a coordinate image as well, preserving its exact policy.
    preview = image.copy()
    preview.thumbnail((256, 256), Image.Resampling.BILINEAR)
    a = np.asarray(preview).max(2)
    mask = a > 10
    w, h = image.size
    box = (0, 0, w, h)
    if mask.mean() >= .25 and np.percentile(a, 90) >= 30:
        ys, xs = np.nonzero(mask)
        pw, ph = preview.size
        pad = max(pw, ph) * .025
        x0, x1 = max(0, xs.min()-pad), min(pw, xs.max()+1+pad)
        y0, y1 = max(0, ys.min()-pad), min(ph, ys.max()+1+pad)
        if (x1-x0)*(y1-y0) >= pw*ph*.35:
            box = (int(x0/pw*w), int(y0/ph*h), math.ceil(x1/pw*w), math.ceil(y1/ph*h))
    return image.crop(box), box


def map_box(box, crop, target_size):
    """Original xyxy to the letterboxed global image; ROI itself uses original pixels."""
    x0, y0, x1, y1 = crop
    scale = min(target_size/(x1-x0), target_size/(y1-y0))
    dx, dy = (target_size-(x1-x0)*scale)/2, (target_size-(y1-y0)*scale)/2
    return [(box[0]-x0)*scale+dx, (box[1]-y0)*scale+dy,
            (box[2]-x0)*scale+dx, (box[3]-y0)*scale+dy]


def import_annotations(source, specification, rows, folds, audit, output, minimum=1000, scope="full1000"):
    """Explicit CSV mapping or COCO import. Never infer orientation/coordinate space."""
    spec = read_json(specification)
    if spec.get("coordinate_space") != SPACE or not spec.get("version"):
        raise ValueError(f"Annotation specification must declare version and coordinate_space={SPACE}")
    if isinstance(minimum, bool) or minimum < 1:
        raise ValueError("minimum must be positive")
    source, output = Path(source), Path(output)
    if (output / "acceptance.json").exists():
        raise ValueError("Annotation version already exists; use a new output directory")
    if spec.get("format") == "csv":
        with source.open(encoding="utf-8-sig", newline="") as f:
            raw = list(csv.DictReader(f))
        mapping = spec.get("columns", {k: k for k in FIELDS})
        records = [{key: r[mapping[key]] for key in FIELDS} for r in raw]
    elif spec.get("format") == "coco":
        coco = read_json(source)
        by_id = {}
        for a in coco["annotations"]:
            if a.get("category_id") != spec["disc_category_id"]:
                continue
            if a["image_id"] in by_id:
                raise ValueError("Multiple optic disc boxes for one image")
            by_id[a["image_id"]] = a["bbox"]
        records = []
        for item in coco["images"]:
            if item["id"] not in by_id:
                raise ValueError("COCO image without box needs an explicit unlocatable status via CSV")
            x, y, w, h = by_id[item["id"]]
            records.append(dict(zip(FIELDS, [item["file_name"], item["width"], item["height"],
                                                x, y, x+w, y+h, "valid"])))
    else:
        raise ValueError("Supported annotation formats: explicit CSV mapping or COCO")
    lookup = {r["name"]: (r, int(fold)) for r, fold in zip(rows, folds)}
    seen, normalized, hashes = set(), [], {}
    usable_folds, labels = set(), {0: 0, 1: 0}
    for record in records:
        name, status = record["Image_name"], record["status"]
        status = spec.get("statuses", {}).get(status, status)
        if name not in lookup or name in seen:
            raise ValueError(f"Not a unique train Image_name: {name}")
        if status not in ("valid", "unclear", "unlocatable"):
            raise ValueError(f"Unknown annotation status: {status}")
        seen.add(name)
        row, fold = lookup[name]
        image = original(row["path"])
        width, height = int(record["width"]), int(record["height"])
        if (width, height) != image.size:
            raise ValueError(f"EXIF-original dimensions differ: {name}")
        box = [record[k] for k in FIELDS[3:7]]
        if status == "valid":
            box = list(map(float, box))
            if (not all(math.isfinite(v) for v in box) or
                    not (0 <= box[0] < box[2] <= width and 0 <= box[1] < box[3] <= height)):
                raise ValueError(f"Invalid original-coordinate box: {name}")
            usable_folds.add(fold)
            labels[row["label"]] += 1
        else:
            if any(str(v).strip() for v in box):
                raise ValueError("Unclear/unlocatable annotations must leave box coordinates empty")
            box = [""] * 4
        normalized.append(dict(zip(FIELDS, [name, width, height, *box, status])))
        hashes[name] = sha256(row["path"])
    write_csv(output / "boxes.csv", normalized, FIELDS)
    accepted = (sum(labels.values()) >= minimum and usable_folds == {0, 1, 2} and min(labels.values()) > 0)
    manifest = {"format": "plan3-annotations-v1", "accepted": accepted, "version": spec["version"],
                "coordinate_space": SPACE, "input_version": INPUT_VERSION, "minimum_valid": minimum,
                "valid": sum(labels.values()), "annotated": len(records), "labels": labels,
                "folds": sorted(usable_folds), "boxes_sha256": sha256(output / "boxes.csv"),
                "source_sha256": sha256(source), "specification": spec, "image_sha256": hashes,
                "data_fingerprint": audit["data_fingerprint"], "fold_fingerprint": audit["fold_fingerprint"],
                "acceptance_scope": scope,
                "reason": None if accepted else "Insufficient valid boxes, class coverage or fold coverage"}
    write_json(output / "acceptance.json", manifest)
    return manifest


def convert_labelme(folder, rows, output):
    """Export only anonymous image keys, dimensions, optic-disc geometry and usability."""
    output = Path(output)
    files = sorted(Path(folder).glob("*.json"))
    if not files:
        raise ValueError("No LabelMe JSON files found")
    by_name = {r["name"].casefold(): r for r in rows}
    records, inventory, seen = [], [], set()
    for path in files:
        item = read_json(path)
        basename = str(item.get("imagePath", "")).replace("\\", "/").split("/")[-1]
        row = by_name.get(basename.casefold())
        if row is None or row["name"] in seen or path.stem.casefold() != Path(basename).stem.casefold():
            raise ValueError(f"Unmatched, duplicate or inconsistent LabelMe train key: {path.name}")
        seen.add(row["name"])
        with Image.open(row["path"]) as image:
            if image.getexif().get(274, 1) != 1:
                raise ValueError("LabelMe raw-image EXIF orientation needs explicit coordinate conversion")
            if image.size != (item.get("imageWidth"), item.get("imageHeight")):
                raise ValueError(f"LabelMe dimensions do not match original image: {path.name}")
        shapes = item.get("shapes", [])
        if len(shapes) != 1 or shapes[0].get("label") != "optic_disc" or shapes[0].get("shape_type") != "rectangle":
            raise ValueError(f"Expected one optic_disc rectangle: {path.name}")
        points = shapes[0].get("points", [])
        if len(points) != 2 or any(len(p) != 2 for p in points):
            raise ValueError("LabelMe rectangle must have two corners")
        flags = item.get("flags", {})
        if any(type(flags.get(k, False)) is not bool for k in ("unlocalizable", "boundary_unclear")):
            raise ValueError("LabelMe usability flags must be boolean")
        status = "unlocatable" if flags.get("unlocalizable") else "unclear" if flags.get("boundary_unclear") else "valid"
        x0, y0 = map(float, points[0])
        x1, y1 = map(float, points[1])
        box = [min(x0, x1), min(y0, y1), max(x0, x1), max(y0, y1)] if status == "valid" else [""]*4
        records.append(dict(zip(FIELDS, [row["name"], item["imageWidth"], item["imageHeight"], *box, status])))
        inventory.append({"filename": path.name, "sha256": sha256(path)})
    write_csv(output / "source_boxes.csv", records, FIELDS)
    spec = {"format": "csv", "version": "labelme-"+digest(inventory)[:12], "coordinate_space": SPACE}
    write_json(output / "source_spec.json", spec)
    write_json(output / "source_inventory.json", {"source_format": "LabelMe", "files": inventory,
               "fingerprint": digest(inventory), "uploaded_fields": FIELDS,
               "removed": ["imagePath", "imageData", "description", "group_id", "mask", "other_metadata"]})
    return output / "source_boxes.csv", output / "source_spec.json", digest(inventory)


def require_annotations(folder, audit):
    folder = Path(folder)
    report = read_json(folder / "acceptance.json")
    if (not report.get("accepted") or report.get("input_version") != INPUT_VERSION or
            report.get("coordinate_space") != SPACE or report["boxes_sha256"] != sha256(folder / "boxes.csv") or
            any(report[k] != audit[k] for k in ("data_fingerprint", "fold_fingerprint"))):
        raise ValueError("Annotations are not accepted or their provenance changed; formal training is blocked")
    with (folder / "boxes.csv").open(encoding="utf-8", newline="") as f:
        return report, {r["Image_name"]: r for r in csv.DictReader(f)}


def roi_crop(image, box, scale, rng=None):
    w, h = image.size
    x0, y0, x1, y1 = box
    d = max(x1-x0, y1-y0)
    cx, cy = (x0+x1)/2, (y0+y1)/2
    if rng is not None:
        cx += rng.uniform(-.15, .15)*d
        cy += rng.uniform(-.15, .15)*d
        scale *= rng.uniform(.85, 1.15)
    side = max(2, int(round(scale*d)))
    left, top = int(round(cx-side/2)), int(round(cy-side/2))
    bounds = (left, top, left+side, top+side)
    fraction = max(0, min(w, left+side)-max(0, left))*max(0, min(h, top+side)-max(0, top))/side**2
    return image.crop(bounds), [v for v in (cx/w, cy/h, side/w, side/h, fraction)]


def augment_original(image, box=None):
    import random
    from PIL import ImageEnhance
    w, h = image.size
    if random.random() < .5:
        image = ImageOps.mirror(image)
        if box is not None:
            box = [w-box[2], box[1], w-box[0], box[3]]
    angle = random.uniform(-15, 15)
    image = image.rotate(angle, Image.Resampling.BILINEAR, fillcolor=(0, 0, 0))
    if box is not None:
        c, s = math.cos(math.radians(angle)), math.sin(math.radians(angle))
        corners = [(box[x]-w/2, box[y]-h/2) for x, y in ((0, 1), (2, 1), (0, 3), (2, 3))]
        rotated = [(c*x+s*y+w/2, -s*x+c*y+h/2) for x, y in corners]
        box = [min(x for x, y in rotated), min(y for x, y in rotated), max(x for x, y in rotated), max(y for x, y in rotated)]
    image = ImageEnhance.Brightness(image).enhance(random.uniform(.9, 1.1))
    image = ImageEnhance.Color(image).enhance(random.uniform(.9, 1.1))
    return image, box


class Views:
    def __init__(self, rows, config, age, roi=None, augment=False, scale_tta=1., cache=None):
        self.rows, self.config, self.roi, self.augment = rows, config, roi or {}, augment
        self.ages = age.transform([r["age"] for r in rows])
        self.scale_tta = scale_tta
        self.cache_path = str(cache) if cache else None
        self.cache_array = None
        if cache:
            metadata = read_json(Path(cache).with_suffix('.json'))
            certificate = read_json(Path(cache).with_suffix('.sha256.json'))
            if (metadata.get('names') != [r['name'] for r in rows] or metadata.get('size') != config['size'] or
                    metadata.get('recipe') != config.get('preprocess_version') or
                    digest(metadata) != certificate['identity_sha256'] or sha256(cache) != certificate['sha256']):
                raise ValueError('Global cache provenance or pixels differ')

    def __getstate__(self):
        state = self.__dict__.copy()
        state['cache_array'] = None
        return state

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        import random
        import torch
        row, cfg = self.rows[i], self.config
        if cfg.get('preprocess_version') == 'original-fov-pad-then-augment-v2':
            if cfg.get('view', 'global') != 'global' or cfg.get('ssl'):
                raise ValueError('Global v2 cache/augmentation is not a local or SSL recipe')
            from .cache import global_record, global_dtype
            if self.cache_path:
                if self.cache_array is None:
                    self.cache_array = np.load(self.cache_path, mmap_mode='r', allow_pickle=False)
                    if self.cache_array.shape != (len(self.rows),) or self.cache_array.dtype != global_dtype(cfg['size']):
                        raise ValueError('Cache row/size layout differs')
                record = self.cache_array[i]
            else:
                record = global_record(row['path'], cfg['size'])
            pixels = resize_image(Image.fromarray(record['pixels']), cfg['size'], self.augment)
            local_size = cfg.get('local_size', 512)
            views = {'global': torch.from_numpy(pixels).permute(2, 0, 1).float()/255,
                     'local': torch.zeros(0, 3, local_size, local_size),
                     'mask': torch.zeros(0, dtype=torch.bool), 'coordinates': torch.zeros(0, 5),
                     'crop': torch.tensor(np.array(record['crop'], copy=True))}
            return views, torch.tensor(self.ages[i]), torch.tensor(float(row['label'])), i
        if self.cache_path:
            raise ValueError('The original augmentation recipe cannot read a resized cache')
        image = original(row["path"])
        item = self.roi.get(row["name"], {})
        valid = bool(item.get("valid"))
        box = item.get("box") if valid else None
        if self.augment:
            image, box = augment_original(image, box)
        full, crop = fov_geometry(image)
        tensor = lambda im, size: torch.from_numpy(resize_image(im, size, False)).permute(2, 0, 1).float()/255
        global_image = tensor(full, cfg["size"])
        scales = cfg.get("roi_scales", [3., 5.]) if cfg.get("view", "global") != "global" or cfg.get("ssl") else []
        local_size = cfg.get("local_size", 512)
        local_sizes = cfg.get("local_sizes", [local_size]*len(scales))
        if len(local_sizes) != len(scales):
            raise ValueError("local_sizes must match roi_scales")
        local, coordinates = [], []
        for scale, target_size in zip(scales, local_sizes):
            if valid:
                im, coord = roi_crop(image, box, scale*self.scale_tta, random if self.augment else None)
                cx, cy, side_x, side_y, fraction = coord
                w, h = image.size
                mapped = map_box([cx*w-side_x*w/2, cy*h-side_y*h/2, cx*w+side_x*w/2, cy*h+side_y*h/2], crop, cfg["size"])
                coord = [(mapped[0]+mapped[2])/2/cfg["size"], (mapped[1]+mapped[3])/2/cfg["size"],
                         (mapped[2]-mapped[0])/cfg["size"], (mapped[3]-mapped[1])/cfg["size"], fraction]
                local.append(tensor(im, target_size))
                coordinates.append(coord)
            else:
                local.append(torch.zeros(3, target_size, target_size))
                coordinates.append([0.] * 5)
        local_value = local if cfg.get("local_sizes") else torch.stack(local) if local else torch.zeros(0, 3, local_size, local_size)
        views = {"global": global_image, "local": local_value,
                "mask": torch.full((len(scales),), valid, dtype=torch.bool),
                "coordinates": torch.tensor(coordinates).reshape(-1, 5), "crop": torch.tensor(crop)}
        if cfg.get("ssl"):
            second = resize_image(full, cfg["size"], True)
            views["global2"] = torch.from_numpy(second).permute(2, 0, 1).float()/255
        return views, torch.tensor(self.ages[i]), torch.tensor(float(row["label"])), i


def partition(rows, folds, groups, fold, inner_fold=None, seed=42):
    indices = np.arange(len(rows))
    fit = indices if fold == "full" else indices[np.asarray(folds) != int(fold)]
    hold = np.array([], dtype=int) if fold == "full" else indices[np.asarray(folds) == int(fold)]
    if inner_fold is not None:
        if fold == "full" or inner_fold not in (0, 1, 2):
            raise ValueError("Nested training needs an outer fold and inner_fold 0/1/2")
        from sklearn.model_selection import StratifiedGroupKFold
        split = StratifiedGroupKFold(3, shuffle=True, random_state=seed)
        inner_fit, inner_hold = list(split.split(fit, [rows[i]["label"] for i in fit], np.asarray(groups)[fit]))[inner_fold]
        fit, hold = fit[inner_fit], fit[inner_hold]
    if not len(fit) or (fold != "full" and not len(hold)):
        raise ValueError("Empty training or holdout partition")
    return fit, hold
