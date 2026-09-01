"""Combine per-file *_scores.jsonl into a single total.json."""
import os
import json
import glob
import argparse


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in-dir", default="results/scores")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    out = args.out or os.path.join(args.in_dir, "total.json")
    all_data = {}
    for path in glob.glob(os.path.join(args.in_dir, "*.jsonl")):
        try:
            with open(path, "r", encoding="utf-8") as f:
                all_data[os.path.basename(path)] = json.load(f)
        except Exception as e:  # noqa: BLE001
            print(f"Skipping {path}: {e}")

    with open(out, "w", encoding="utf-8") as f:
        json.dump(all_data, f, ensure_ascii=False, indent=2)
    print(f"Combined {len(all_data)} files into {out}")


if __name__ == "__main__":
    main()
