"""
Evaluation on LoCoMo / LongMemEval benchmarks.

Usage:
    python scripts/eval_locomo.py --checkpoint outputs/phase3/best
"""

import argparse
import os
import json

from loguru import logger

from gmsra.config import GMSRAConfig
from gmsra.utils import set_seed, load_model_and_tokenizer, compute_f1, compute_exact_match


def main(args):
    set_seed(42)
    logger.info(f"Evaluating on LoCoMo/LongMemEval | checkpoint={args.checkpoint}")

    config = GMSRAConfig()
    model, tokenizer = load_model_and_tokenizer(config.model.model_name)

    if args.checkpoint and os.path.exists(args.checkpoint):
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, args.checkpoint)

    from gmsra.agent import GMSRAAgent
    agent = GMSRAAgent(config)
    agent.initialize(model, tokenizer, env_type="dialogue")
    if os.path.exists(os.path.join(args.checkpoint, "memory_store.json")):
        agent.load_checkpoint(args.checkpoint)

    # Load evaluation data
    eval_data = load_eval_data(args.data_dir, args.benchmark)
    logger.info(f"Loaded {len(eval_data)} eval examples from {args.benchmark}")

    # Evaluation loop
    results = []
    for idx, example in enumerate(eval_data):
        # Feed events to build memory
        for event in example.get("events", []):
            agent.step(event=event, task_context="", agent_response="")

        # Answer question
        question = example["question"]
        ground_truth = example["answer"]
        prediction = agent.answer_question(question)

        f1 = compute_f1(prediction, ground_truth)
        em = compute_exact_match(prediction, ground_truth)

        results.append({
            "idx": idx,
            "question": question,
            "ground_truth": ground_truth,
            "prediction": prediction,
            "f1": f1,
            "em": em,
            "category": example.get("category", "unknown"),
        })

        if (idx + 1) % 20 == 0:
            avg_f1 = sum(r["f1"] for r in results) / len(results)
            avg_em = sum(r["em"] for r in results) / len(results)
            logger.info(f"Progress: {idx+1}/{len(eval_data)} | F1={avg_f1:.3f} | EM={avg_em:.3f}")

    # Compute aggregate metrics
    avg_f1 = sum(r["f1"] for r in results) / len(results) if results else 0
    avg_em = sum(r["em"] for r in results) / len(results) if results else 0

    # Category breakdown
    categories = set(r["category"] for r in results)
    category_metrics = {}
    for cat in categories:
        cat_results = [r for r in results if r["category"] == cat]
        category_metrics[cat] = {
            "f1": sum(r["f1"] for r in cat_results) / len(cat_results),
            "em": sum(r["em"] for r in cat_results) / len(cat_results),
            "count": len(cat_results),
        }

    summary = {
        "benchmark": args.benchmark,
        "checkpoint": args.checkpoint,
        "num_examples": len(results),
        "avg_f1": avg_f1,
        "avg_em": avg_em,
        "memory_size_final": agent.memory_store.size(),
        "category_breakdown": category_metrics,
    }

    # Save
    os.makedirs(args.output_dir, exist_ok=True)
    with open(os.path.join(args.output_dir, f"{args.benchmark}_results.json"), "w") as f:
        json.dump({"summary": summary, "details": results}, f, indent=2, ensure_ascii=False)

    logger.info(f"\n{'='*50}")
    logger.info(f"RESULTS: {args.benchmark}")
    logger.info(f"  F1:  {avg_f1:.4f}")
    logger.info(f"  EM:  {avg_em:.4f}")
    logger.info(f"  Mem: {agent.memory_store.size()} entries")
    logger.info(f"{'='*50}")


def load_eval_data(data_dir: str, benchmark: str) -> list[dict]:
    """Load evaluation data."""
    path = os.path.join(data_dir, f"{benchmark}_test.json")
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    logger.warning(f"Eval data not found at {path}, using placeholder")
    return [{"events": ["User likes AI"], "question": "What does user like?",
             "answer": "AI", "category": "preference"}] * 10


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="G-MSRA Evaluation")
    parser.add_argument("--checkpoint", default="outputs/phase3/best")
    parser.add_argument("--output_dir", default="results")
    parser.add_argument("--data_dir", default="data")
    parser.add_argument("--benchmark", default="locomo",
                        choices=["locomo", "longmemeval"])
    parser.add_argument("--use_qlora", action="store_true")
    parser.add_argument("--load_in_4bit", action="store_true")
    args = parser.parse_args()
    main(args)
