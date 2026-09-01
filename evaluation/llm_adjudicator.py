"""
LLM-as-a-judge adjudicator (Table 7).

For each model prediction, asks Qwen3-30B-A3B to grade the response per clinical
aspect (question_class) with a binary score + reason, and writes *_eval.json.

Start an endpoint first, e.g.:
    vllm serve Qwen/Qwen3-30B-A3B --port 8000 --host 0.0.0.0 --max-model-len 4096

Usage:
    export VLLM_BASE_URL="http://localhost:8000/v1"
    export OPENAI_API_KEY="EMPTY"
    python evaluation/llm_adjudicator.py --in "results/pred_*.jsonl" --out-dir results/eval
"""
import os
import re
import json
import glob
import asyncio
import argparse

import openai
import pandas as pd
from tqdm import tqdm

from common import load_predictions, env

client = openai.AsyncOpenAI(
    api_key=env("OPENAI_API_KEY", "EMPTY"), base_url=env("VLLM_BASE_URL", "http://localhost:8000/v1")
)
MODEL = env("ADJUDICATOR_MODEL", "Qwen/Qwen3-30B-A3B")
SYSTEM_PROMPT = (
    "You are an medical examiner evaluating a **doctor's written response** in a "
    "medical exam and always provide a **structured JSON evaluation** of the response."
)
BATCH_SIZE = 100
sem = asyncio.Semaphore(100)


def evaluate_response(question, model_response, complexity, question_class, original, answer):
    aspects_json = ",\n".join(
        f'    "{aspect}": {{\n      "score": 0 or 1,\n      "reason": "<short justification>"\n    }}'
        for aspect in question_class
    )
    return f"""
    ## CONTEXT
    The current **exam question** is derived from one or more original Q/A items (see `original`) and may vary in complexity. It has been annotated with one or more **aspect labels** (see `question_class`), where each label represents a specific area of clinical knowledge.

    ## TASK
    Grade the doctor's response against each individual aspect. For each aspect in `question_class`:
    - Compare the doctor's response to the correct answer, using the original Q/A pairs for context.
    - Assign "score": 1 if the response fully and correctly addresses that aspect, else 0.
    - Give a short reason.

    ## OUTPUT FORMAT (STRICT)
    Return a valid JSON object where each key is one aspect label and each value has "score" and "reason".
    Wrap the JSON in triple backticks with `json`. No extra text.

    ```json
    {{
    {aspects_json}
    }}
    ```

    ## Input:
    Exam Question: {question}
    Doctor's Response: {model_response}
    Correct Answer: {answer}
    Question Complexity Level: {complexity}
    Original Q/A Reference: {original}
    Evaluation Aspects: {question_class}
    """


async def _with_sem(coro):
    async with sem:
        return await coro


async def ask_batch(batch):
    tasks = []
    for _, item in batch:
        qwery = evaluate_response(
            item["question"], item["response"], item["complexity"],
            item["question_class"], item["original"], item["answer"],
        )
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": [{"type": "text", "text": qwery + " /no_think"}]},
        ]
        tasks.append(_with_sem(
            client.chat.completions.create(
                model=MODEL, messages=messages,
                extra_body={"chat_template_kwargs": {"enable_thinking": False}},
            )
        ))
    results = await asyncio.gather(*tasks, return_exceptions=True)
    output = []
    for (idx, item), response in zip(batch, results):
        entry = item.copy()
        try:
            res = response.choices[0].message.content
            txt = re.findall(r"```json(?:[^\n]*)\n(.*?)```", res, re.DOTALL)[0] if "```" in res else None
            json_res = json.loads(txt)
            assert all(k in json_res for k in item["question_class"]), f"Missing keys: {json_res}"
            entry["eval_json"] = json_res
        except Exception as e:  # noqa: BLE001
            entry["error"] = str(e)
        output.append(entry)
    return output


def batches(data, size):
    for i in range(0, len(data), size):
        yield list(enumerate(data[i:i + size], start=i))


async def runz(merged):
    all_results = []
    for batch in tqdm(batches(merged.to_dict(orient="records"), BATCH_SIZE)):
        all_results.extend(await ask_batch(batch))
        await asyncio.sleep(0.5)
    return all_results


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", required=True, help="prediction JSONL (or glob)")
    ap.add_argument("--out-dir", default="results/eval")
    args = ap.parse_args()

    files = glob.glob(args.inp) if any(c in args.inp for c in "*?[") else [args.inp]
    os.makedirs(args.out_dir, exist_ok=True)
    for path in files:
        out = os.path.join(args.out_dir, os.path.basename(path).replace(".jsonl", "_eval.json"))
        if os.path.exists(out):
            print(f"skip (exists): {out}")
            continue
        print(f"adjudicating {path}")
        merged = load_predictions(path)
        results = asyncio.run(runz(merged))
        with open(out, "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False)
        print(f"Wrote {out} ({len(results)} rows)")


if __name__ == "__main__":
    main()
