"""
Aggregate the LLM-adjudicator outputs and draw the paper figures.

Produces, under --out-dir:
  * aspect_scores.csv / _complexity{1,2,3}.csv   (Table 7 source)
  * heatmap.png        (Fig 1: rank-normalised, 1 = best)
  * radar_combined.png (Fig 2: absolute overall scores)
  * radar_complex_{1,2,3}.png (Fig 3: per-complexity radar)

Usage:
    python evaluation/make_figures.py \
        --model gemma3=results/eval/..._gemma3_eval.json \
        --model medgemma=results/eval/..._medgemma-4b_eval.json \
        --model medgemma-ft=results/eval/..._medgemma-3952_eval.json \
        --model Qwen2.5-VL-7B=results/eval/..._Qwen2.5-VL_eval.json \
        --model Qwen2.5-VL-7B-ft=results/eval/..._v1-4444_eval.json \
        --out-dir results/figures
"""
import json
import argparse
from collections import defaultdict

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def parse_models(pairs):
    out = {}
    for p in pairs:
        name, _, path = p.partition("=")
        out[name] = path
    return out


def scores_from_eval(path):
    data = json.load(open(path))
    overall = defaultdict(lambda: {"c": 0, "t": 0})
    by_comp = defaultdict(lambda: {"c": 0, "t": 0})
    for entry in data:
        comp = entry["complexity"]
        ev = entry.get("eval_json")
        if not ev:
            continue
        for qc in entry["question_class"]:
            if qc not in ev:
                continue
            hit = 1 if ev[qc]["score"] == 1 else 0
            overall[qc]["c"] += hit
            overall[qc]["t"] += 1
            by_comp[(qc, comp)]["c"] += hit
            by_comp[(qc, comp)]["t"] += 1
    ov = {qc: 100 * v["c"] / v["t"] for qc, v in overall.items() if v["t"]}
    bc = {qc: {c: 100 * v["c"] / v["t"] for (q, c), v in by_comp.items() if q == qc and v["t"]}
          for qc in overall}
    return ov, bc


def radar(df, title, fname):
    cats = list(df.index)
    N = len(cats)
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
    angles += angles[:1]
    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))
    for col in df.columns:
        vals = df[col].tolist()
        vals += vals[:1]
        ax.plot(angles, vals, linewidth=1.5, label=col)
        ax.fill(angles, vals, alpha=0.06)
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(cats, fontsize=7)
    ax.set_ylim(0, 100)
    ax.set_title(title, fontsize=11, pad=20)
    ax.legend(loc="upper right", bbox_to_anchor=(1.25, 1.1), fontsize=8)
    fig.tight_layout()
    fig.savefig(fname, dpi=150, bbox_inches="tight")
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", action="append", required=True,
                    help="name=path/to/_eval.json (repeat per model)")
    ap.add_argument("--out-dir", default="results/figures")
    args = ap.parse_args()
    import os
    os.makedirs(args.out_dir, exist_ok=True)

    models = parse_models(args.model)
    overall, by_comp = {}, {}
    for name, path in models.items():
        overall[name], by_comp[name] = scores_from_eval(path)

    df = pd.DataFrame(overall).sort_index()
    df.to_csv(f"{args.out_dir}/aspect_scores.csv")
    for c in (1, 2, 3):
        dfc = pd.DataFrame({n: by_comp[n].get(qc, {}).get(c, np.nan) for n in models for qc in df.index},
                           index=df.index)
        dfc.to_csv(f"{args.out_dir}/aspect_scores_complexity{c}.csv")

    # Fig 1: rank-normalised heatmap (1 = best per category).
    ranks = df.rank(axis=1, ascending=False)
    ranks = (ranks - 1) / (len(models) - 1) if len(models) > 1 else ranks * 0
    plt.figure(figsize=(10, 12))
    sns.heatmap(ranks, annot=df.round(1), fmt=".1f", cmap="viridis_r",
                cbar_kws={"label": "normalised rank (0=best)"})
    plt.title("Rank-normalised aspect accuracy (annotated = % correct)")
    plt.tight_layout()
    plt.savefig(f"{args.out_dir}/heatmap.png", dpi=150, bbox_inches="tight")
    plt.close()

    # Fig 2: overall radar.
    radar(df, "Overall aspect accuracy (%)", f"{args.out_dir}/radar_combined.png")

    # Fig 3: per-complexity radar.
    for c in (1, 2, 3):
        dfc = pd.DataFrame({n: by_comp[n].get(qc, {}).get(c, np.nan) for n in models for qc in df.index},
                           index=df.index).dropna(how="all")
        radar(dfc, f"Complexity level {c} (%)", f"{args.out_dir}/radar_complex_{c}.png")

    print(f"Wrote tables + figures to {args.out_dir}")


if __name__ == "__main__":
    main()
