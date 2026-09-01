"""
Generate 10 weakly-augmented variants per original image (Track 2).

Augmentations match the paper: RandomResizedCrop (scale 0.9-1.0), RandomRotation
(+/-10 deg), RandomAffine (translate up to 10%), ColorJitter (brightness/contrast
0.8-1.2), all with bicubic interpolation. Output layout:
    <aug-dir>/variant_<1..10>/<img_id>.jpg

Usage:
    python augmentation/generate_augmented.py --aug-dir data/image_weak_augmented
"""
import argparse
from pathlib import Path

import numpy as np
import random as _random
import torch
import torchvision.transforms as T
from torchvision.transforms import InterpolationMode as IM
from datasets import load_dataset, Image as HfImage

SEED = 42
_random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

DATASET = "SimulaMet/Kvasir-VQA-x1"
N_VARIANTS = 10


def weak(img):
    return T.Compose(
        [
            T.RandomResizedCrop(
                img.size[::-1],
                scale=(0.9, 1.0),
                ratio=(img.size[0] / img.size[1] * 0.95, img.size[0] / img.size[1] * 1.05),
                interpolation=IM.BICUBIC,
            ),
            T.RandomRotation((-10, 10), interpolation=IM.BICUBIC, fill=0),
            T.RandomAffine(0, translate=(0.1, 0.1), interpolation=IM.BICUBIC, fill=0),
            T.ColorJitter(brightness=(0.8, 1.2), contrast=(0.8, 1.2)),
        ]
    )(img)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--aug-dir", default="data/image_weak_augmented")
    args = ap.parse_args()

    aug_dir = Path(args.aug_dir)
    for v in range(1, N_VARIANTS + 1):
        (aug_dir / f"variant_{v}").mkdir(parents=True, exist_ok=True)

    ds = load_dataset(DATASET, split="train").cast_column("image", HfImage())
    uniq_idx = sorted(np.unique(ds["img_id"], return_index=True)[1])
    ds_unique = ds.select(uniq_idx)

    for row in ds_unique:
        img = row["image"].convert("RGB")
        for v in range(1, N_VARIANTS + 1):
            out = aug_dir / f"variant_{v}" / f"{row['img_id']}.jpg"
            if out.exists():
                continue
            weak(img).save(out)
    print(f"Wrote {N_VARIANTS} augmented variants per image under {aug_dir}")


if __name__ == "__main__":
    main()
