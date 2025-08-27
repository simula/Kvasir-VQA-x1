# Kvasir-VQA-x1


A Multimodal Dataset for Medical Reasoning and Robust MedVQA in Gastrointestinal Endoscopy

[Dataset on Hugging Face](https://huggingface.co/datasets/SimulaMet/Kvasir-VQA-x1)  
[Original Image Download (Simula Datasets)](https://datasets.simula.no/kvasir-vqa/) or see below.


## 🧠 About

**Kvasir-VQA-x1** is a multimodal dataset aimed at advancing medical visual question answering (MedVQA) in GI endoscopy. We build on the original [Kvasir-VQA](https://datasets.simula.no/kvasir-vqa/) by adding 159,549 new QA pairs with richer reasoning and complexity stratification.

This repo provides:

- Augmentation scripts
- Dataset generation code
- JSON validators
- Sample training/evaluation workflows
- Metric visualizations (e.g., radar plots)

## 🧾 Dataset Structure

Each sample includes:

| Field           | Description |
|----------------|-------------|
| `img_id`        | Image reference from Kvasir-VQA |
| `complexity`    | Reasoning complexity (1–3) |
| `question`      | Natural language QA |
| `answer`        | Human-validated clinical answer |
| `original`      | Source atomic QA pairs |
| `question_class`| Clinical categories (e.g., polyp type) |

See full dataset: [Hugging Face page](https://huggingface.co/datasets/SimulaMet/Kvasir-VQA-x1)

## 🧪 Evaluation Tracks

- **Standard**: QA on original images  
- **Transformed**: QA on visually perturbed images (augmented via scripts here)

## 📥 Download & Prepare Images

To ensure reproducibility, you can download the **original images** and generate **augmented (perturbed) images** locally.

---

### 1️⃣ Download Original Images

```python
from datasets import load_dataset
from pathlib import Path
from tqdm import tqdm
import os, json

# Output folder
d_path = "./Kvasir-VQA-x1/"
img_dir = Path(os.path.abspath(os.path.join(d_path, "images")))
img_dir.mkdir(exist_ok=True, parents=True)

# Download original images once from SimulaMet-HOST/Kvasir-VQA
ds_host = load_dataset("SimulaMet-HOST/Kvasir-VQA", split="raw")
seen = set()
for row in tqdm(ds_host, desc="Saving original images"):
    if row["img_id"] not in seen:
        row["image"].save(img_dir / f"{row['img_id']}.jpg")
        seen.add(row["img_id"])

# Save VLM-ready JSONLs (pointing to ORIGINAL images)
for split in ["train", "test"]:
    with open(f"{d_path}/Kvasir-VQA-x1-{split}.jsonl", "w", encoding="utf-8") as f:
        for r in load_dataset("SimulaMet/Kvasir-VQA-x1", split=split):
            f.write(json.dumps({
                "messages": [
                    {"role": "user", "content": f"<image>{r['question']}"},
                    {"role": "assistant", "content": r["answer"]}
                ],
                "images": [str(img_dir / f"{r['img_id']}.jpg")]
            }, ensure_ascii=False) + "\n")
```

---

### 2️⃣ Generate Weakly-Augmented Images

This script saves **lightly perturbed versions** of each image and creates JSONLs pointing to them.

```python
from datasets import load_dataset, Image as HfImage
from pathlib import Path
from PIL import Image
import torchvision.transforms as T
from torchvision.transforms import InterpolationMode as IM
import numpy as np, os, random, json, torch

SEED = 42
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)

# Paths
d_path = Path("./Kvasir-VQA-x1")
aug_dir = (d_path / "image_weak_augmented").resolve()
aug_dir.mkdir(parents=True, exist_ok=True)

# Define weak augmentation
weak = lambda img: T.Compose([
    T.RandomResizedCrop(img.size[::-1], scale=(0.9,1.0), ratio=(img.size[0]/img.size[1]*0.95, img.size[0]/img.size[1]*1.05), interpolation=IM.BICUBIC),
    T.RandomRotation((-10,10), interpolation=IM.BICUBIC, fill=0),
    T.RandomAffine(0, translate=(0.1,0.1), interpolation=IM.BICUBIC, fill=0),
    T.ColorJitter(0.2,0.2)
])(img)

# Work on unique images
ds_aug = {}
for split in ["train", "test"]:
    ds = load_dataset("SimulaMet/Kvasir-VQA-x1", split=split).cast_column("image", HfImage())

    # keep unique img_id
    uniq_idx = sorted(np.unique(ds["img_id"], return_index=True)[1])
    ds_unique = ds.select(uniq_idx)

    # augment and save
    def save_img_batch(batch):
        return {"weak_image":[
            (weak(img.convert("RGB")).save(p) or p) if not os.path.exists(p) else p
            for img,p in zip(batch["image"], [str(aug_dir / f"{i}.jpg") for i in batch["img_id"]])
        ]}
    ds_unique = ds_unique.map(save_img_batch, batched=True, batch_size=10, num_proc=4)

    # cast new column as HfImage
    ds_aug[split] = ds_unique.cast_column("weak_image", HfImage())

# Now you can access ad dataset object with:
ds_train_aug = ds_aug["train"]
ds_test_aug  = ds_aug["test"]
```

---

### 3️⃣ Export JSONLs with Augmented Images

```python
# Save VLM-ready JSONLs pointing to AUGMENTED images
for split in ["train", "test"]:
    out_path = f"{d_path}/Kvasir-VQA-x1-{split}-aug.jsonl"
    with open(out_path, "w", encoding="utf-8") as f:
        for r in load_dataset("SimulaMet/Kvasir-VQA-x1", split=split):
            f.write(json.dumps({
                "messages": [
                    {"role": "user", "content": f"<image>{r['question']}"},
                    {"role": "assistant", "content": r["answer"]}
                ],
                "images": [str(aug_dir / f"{r['img_id']}.jpg")]
            }, ensure_ascii=False) + "\n")
```

---

✅ With this, you’ll have both:

- `Kvasir-VQA-x1-{train,test}.jsonl` → pointing to **original** images  
- `Kvasir-VQA-x1-{train,test}-aug.jsonl` → pointing to **weakly augmented** images  

---

## 📜 License

This dataset is released under [CC BY-NC 4.0](https://creativecommons.org/licenses/by-nc/4.0/).

## 📌 Citation

Please cite the associated dataset paper if you use Kvasir-VQA-x1 in your work:
```bibtex
@article{Gautam2025Jun,
	author = {Gautam, Sushant and Riegler, Michael A. and Halvorsen, P{\aa}l},
	title = {{Kvasir-VQA-x1: A Multimodal Dataset for Medical Reasoning and Robust MedVQA in Gastrointestinal Endoscopy}},
	journal = {arXiv},
	year = {2025},
	month = jun,
	eprint = {2506.09958},
	doi = {10.48550/arXiv.2506.09958}
}
```

