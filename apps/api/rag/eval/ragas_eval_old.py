from __future__ import annotations
import os, json, requests, pandas as pd
from datasets import Dataset
from ragas import evaluate
from ragas.metrics import faithfulness, answer_relevancy, context_precision, context_recall

API_URL = os.getenv("API_URL", "http://api:8000/v1/chat/completions")
MODEL   = os.getenv("EVAL_MODEL", "llama3.2:3b")

def call_api(question: str):
    r = requests.post(API_URL, json={"model": MODEL, "messages":[{"role":"user","content":question}]}, timeout=300)
    r.raise_for_status()
    j = r.json()
    if "choices" in j and j["choices"]:
        return j["choices"][0]["message"]["content"]
    return str(j)

def run_ragas(eval_jsonl_path: str, out_path: str = "/app/data/eval_results.json"):
    rows = [json.loads(l) for l in open(eval_jsonl_path, "r", encoding="utf-8")]
    questions = [r["question"] for r in rows]
    answers = [call_api(q) for q in questions]
    gt = [r.get("ground_truth","") for r in rows]
    ds = Dataset.from_dict({"question": questions, "answer": answers, "ground_truth": gt, "contexts": [[] for _ in questions]})
    report = evaluate(ds, metrics=[faithfulness, answer_relevancy, context_precision, context_recall])
    pd.DataFrame(report.to_pandas()).to_json(out_path, orient="records", indent=2)
    print("Saved:", out_path)

if __name__ == "__main__":
    import sys
    path = sys.argv[1] if len(sys.argv) > 1 else "/app/data/eval_set.jsonl"
    run_ragas(path)
