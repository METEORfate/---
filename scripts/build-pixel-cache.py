import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from glaucoma.audit import require_audit
from glaucoma.common import ARTIFACTS
from glaucoma.data import load_rows
from glaucoma.pixel_cache import build_pixel_cache

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--size", type=int, default=392)
    p.add_argument("--workers", type=int, default=6)
    args = p.parse_args()
    audit = require_audit(ROOT / "glaucoma_first")
    target = ARTIFACTS / "pixel_cache" / f"train_{args.size}_{audit['data_fingerprint'][:10]}.npy"
    build_pixel_cache(load_rows(ROOT / "glaucoma_first", "train"), args.size, target, workers=args.workers)
