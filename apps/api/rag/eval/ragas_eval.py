import os
import json
import logging
from datasets import Dataset
from ragas import evaluate
from ragas.metrics import (
    faithfulness,
    answer_relevancy,
    context_precision,
)
from langchain_openai import ChatOpenAI

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("ragas_eval")

EVAL_FILE = "/app/data/eval_set.jsonl"
RESULT_FILE = "/app/data/eval_results.json"

# --- Patch ChatOpenAI to ignore 'proxies' ---
class PatchedChatOpenAI(ChatOpenAI):
    def __init__(self, **kwargs):
        kwargs.pop("proxies", None)  # remove unsupported arg
        super().__init__(
            model=kwargs.get("model", "gpt-4o-mini"),
            temperature=kwargs.get("temperature", 0),
            openai_api_key=kwargs.get("openai_api_key", os.getenv("OPENAI_API_KEY")),
            **{k: v for k, v in kwargs.items() if k not in ["model", "temperature", "openai_api_key"]}
        )

def run_ragas(eval_file: str, result_file: str):
    logger.info("🚀 Running RAGAS evaluation...")

    # --- Load dataset ---
    try:
        dataset = Dataset.from_json(eval_file)
        logger.info(f"✅ Loaded dataset with {len(dataset)} samples")
    except Exception as e:
        logger.error(f"❌ Failed to load dataset: {e}")
        return

    # --- Define metrics ---
    metrics = [
        faithfulness,
        answer_relevancy,
        context_precision,
    ]

    # --- Initialize LLM safely ---
    try:
        llm = PatchedChatOpenAI(model="gpt-4o-mini", temperature=0)
        logger.info("✅ Initialized LLM")
    except Exception as e:
        logger.error(f"❌ Failed to initialize LLM: {e}")
        return

    # --- Run evaluation ---
    try:
        results = evaluate(dataset, metrics=metrics, llm=llm)
        # Use dict(results) to get average scores as a dict (metric_name: mean_score)
        results_dict = dict(results)

        # Save JSON
        with open(result_file, "w") as f:
            json.dump(results_dict, f, indent=4)

        logger.info(f"✅ Evaluation complete! Results saved to {result_file}")
        print(json.dumps(results_dict, indent=4))

    except Exception as e:
        logger.error(f"❌ Evaluation failed: {e}")

if __name__ == "__main__":
    run_ragas(EVAL_FILE, RESULT_FILE)