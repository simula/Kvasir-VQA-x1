"""Shared helpers for the evaluation scripts."""
import json
import os

import pandas as pd
from datasets import load_dataset
from huggingface_hub import hf_hub_download

# Public dataset + the question-class mapping used to derive `question_class`.
DATASET = "SimulaMet/Kvasir-VQA-x1"
CLASS_MAP_REPO = "SimulaMet/Kvasir-VQA-x1"
CLASS_MAP_FILE = "kvasir_hallucination_types.json"


def load_reference_frame():
    """Reference frame: dataset rows with a derived `question_class` column."""
    dsf = load_dataset(DATASET, split="train").to_pandas()
    types = json.load(hf_hub_download(CLASS_MAP_REPO, CLASS_MAP_FILE))
    clss_map = {k: v["slug"] for k, v in types.items()}
    dsf["question_class"] = dsf["original"].apply(
        lambda x: [clss_map[item["q"]] for item in json.loads(x)]
    )
    return dsf


def load_predictions(path):
    """Parse an MS-Swift prediction JSONL into a tidy frame and merge with the reference."""
    df = pd.read_json(path, lines=True)
    df["img_id"] = df["images"].apply(lambda x: x[0]["path"].split("/")[-1].replace(".jpg", ""))
    df["images"] = df["images"].apply(lambda x: x[0]["path"])
    df["question"] = df["messages"].apply(lambda x: x[0]["content"].replace("<image>", "").strip())
    df["response"] = df["messages"].apply(lambda x: x[1]["content"].strip())
    merged = pd.merge(df, load_reference_frame(), on=["img_id", "question"], how="inner")
    return merged.drop(columns=["messages", "logprobs", "labels"])


def env(name, default=None):
    return os.environ.get(name, default)
