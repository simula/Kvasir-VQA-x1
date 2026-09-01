"""
Compute NLG metrics (Table 6): ROUGE-1/2/L, METEOR, CHRF++, BLEU, BLEURT,
Exact Match and BERTScore, per complexity level and overall.

Installs:
    pip install evaluate datasets rouge_score bert_score sacrebleu
    pip install git+https://github.com/google-research/bleurt.git

Usage:
    python evaluation/compute_metrics.py --in results/pred_qwen25vl_ft.jsonl \
        --out-dir results/scores
    python evaluation/combine_scores.py --in-dir results/scores
"""
import os
import json
import glob
import argparse
from collections import defaultdict

import pandas as pd
from datasets import Dataset
import evaluate

from common import load_predictions

_METRICS = None


def get_metrics():
    global _METRICS
    if _METRICS is None:
        _METRICS = {
            "rouge": evaluate.load("rouge"),
            "meteor": evaluate.load("meteor"),
            "chrf": evaluate.load("chrf"),
            "bleu": evaluate.load("bleu"),
            "bleurt": evaluate.load("bleurt", config_name="bleurt-base-128"),
            "bertscore": evaluate.load("bertscore"),
        }
    return _METRICS


def score_file(input_path, output_path):
    m = get_metrics()
    merged = load_predictions(input_path)
    dataset = Dataset.from_pandas(merged)
    preds, refs, comps = dataset["response"], dataset["answer"], dataset["complexity"]

    rouge_all = m["rouge"].compute(predictions=preds, references=refs)
    meteor_all = m["meteor"].compute(predictions=preds, references=refs)
    chrf_all = m["chrf"].compute(predictions=preds, references=refs)
    bleu_all = m["bleu"].compute(predictions=preds, references=[[r] for r in refs])
    bleurt_all = m["bleurt"].compute(predictions=preds, references=refs)["scores"]
    bs = m["bertscore"].compute(predictions=preds, references=refs, lang="en", batch_size=8)
    bert_data = list(zip(bs["precision"], bs["recall"], bs["f1"]))
    exact = [p.strip().lower() == r.strip().lower() for p, r in zip(preds, refs)]

    grouped = defaultdict(list)
    for i, comp in enumerate(comps):
        grouped[comp].append(i)

    def summarize(idx):
        bert = [bert_data[i] for i in idx]
        return {
            "Exact Match": sum(exact[i] for i in idx) / len(idx),
            "BLEURT": sum(bleurt_all[i] for i in idx) / len(idx),
            "BERTScore": {
                "precision": sum(p for p, _, _ in bert) / len(bert),
                "recall": sum(r for _, r, _ in bert) / len(bert),
                "f1": sum(f for _, _, f in bert) / len(bert),
            },
        }

    results = {"per_complexity": {}, "overall": {}}
    for comp in sorted(grouped):
        idx = grouped[comp]
        gp, gr = [preds[i] for i in idx], [refs[i] for i in idx]
        rg = m["rouge"].compute(predictions=gp, references=gr)
        mg = m["meteor"].compute(predictions=gp, references=gr)
        cg = m["chrf"].compute(predictions=gp, references=gr)
        bg = m["bleu"].compute(predictions=gp, references=[[r] for r in gr])
        s = summarize(idx)
        results["per_complexity"][comp] = {
            "ROUGE-1": rg["rouge1"], "ROUGE-2": rg["rouge2"], "ROUGE-L": rg["rougeL"],
            "Exact Match": s["Exact Match"], "METEOR": mg["meteor"], "CHRF++": cg["score"],
            "BLEU": bg["bleu"], "BLEURT": s["BLEURT"], "BERTScore": s["BERTScore"],
        }

    results["overall"] = {
        "ROUGE-1": rouge_all["rouge1"], "ROUGE-2": rouge_all["rouge2"], "ROUGE-L": rouge_all["rougeL"],
        "Exact Match": sum(exact) / len(exact), "METEOR": meteor_all["meteor"], "CHRF++": chrf_all["score"],
        "BLEU": bleu_all["bleu"], "BLEURT": sum(bleurt_all) / len(bleurt_all),
        "BERTScore": {
            "precision": sum(p for p, _, _ in bert_data) / len(bert_data),
            "recall": sum(r for _, r, _ in bert_data) / len(bert_data),
            "f1": sum(f for _, _, f in bert_data) / len(bert_data),
        },
    }

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as f:
        f.write(json.dumps(results) + "\n")
    print(f"Saved {output_path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", required=True, help="prediction JSONL (or glob)")
    ap.add_argument("--out-dir", default="results/scores")
    args = ap.parse_args()

    files = glob.glob(args.inp) if any(c in args.inp for c in "*?[") else [args.inp]
    for path in files:
        out = os.path.join(args.out_dir, os.path.basename(path).replace(".jsonl", "_scores.jsonl"))
        if os.path.exists(out):
            print(f"skip (exists): {out}")
            continue
        print(f"scoring {path}")
        score_file(path, out)


if __name__ == "__main__":
    main()
