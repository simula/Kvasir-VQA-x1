# Kvasir-VQA-x1


A Multimodal Dataset for Medical Reasoning and Robust MedVQA in Gastrointestinal Endoscopy

[Dataset on Hugging Face](https://huggingface.co/datasets/SimulaMet/Kvasir-VQA-x1)  
[Original Image Download (Simula Datasets)](https://datasets.simula.no/kvasir-vqa/)

---

🚧 **Work in Progress**  
This repository is under active development. The dataset generation code, augmentation scripts, and evaluation tools will be released soon.

If you urgently need access or have questions, please contact:  
📧 **sushant@simula.no**

---

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

## 📂 Download Original Images

You can download the original images directly from the official Simula site:  
➡️ [https://datasets.simula.no/kvasir-vqa/](https://datasets.simula.no/kvasir-vqa/)

## 📜 License

This dataset is released under [CC BY-NC 4.0](https://creativecommons.org/licenses/by-nc/4.0/).

## 📌 Citation

Please cite the associated dataset paper if you use Kvasir-VQA-x1 in your work (coming soon).

