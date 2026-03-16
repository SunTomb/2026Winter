"""
Unified baseline reproduction runner for G-MSRA.

This script runs project-local, aligned reproductions of the baselines named in
the paper. It trains each baseline on the shared dialogue format, evaluates on
LoCoMo / LongMemEval, and optionally performs online adaptation evaluation on
agent-task streams.
"""

import argparse
import json
import os

from loguru import logger

from gmsra.baselines import (
    BASELINE_SPECS,
    create_baseline,
    get_baseline_spec,
)
from gmsra.config import GMSRAConfig
from gmsra.utils import load_model_and_tokenizer, set_seed


def load_json_or_fallback(path: str, fallback: list[dict]) -> list[dict]:
    """Load JSON data if present, otherwise return fallback data."""

    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as handle:
            return json.load(handle)
    return fallback


def load_dialogue_data(data_dir: str) -> tuple[list[dict], dict[str, list[dict]]]:
    """Load shared dialogue train/test splits used by local baselines."""

    from scripts.train_phase1_rl import _generate_placeholder_data

    placeholder = _generate_placeholder_data()
    train_data = load_json_or_fallback(
        os.path.join(data_dir, "locomo_train.json"),
        placeholder,
    )

    benchmarks = {
        "locomo": load_json_or_fallback(
            os.path.join(data_dir, "locomo_test.json"),
            placeholder[:20],
        ),
        "longmemeval": load_json_or_fallback(
            os.path.join(data_dir, "longmemeval_test.json"),
            placeholder[:20],
        ),
    }
    return train_data, benchmarks


def load_agent_tasks(data_dir: str, env_name: str, num_tasks: int) -> list[dict]:
    """Load an agent-task benchmark stream."""

    from scripts.eval_agent_tasks import load_agent_tasks as load_tasks

    return load_tasks(env_name, data_dir, num_tasks)


def run_single_baseline(
    baseline_id: str,
    args,
    train_data: list[dict],
    dialogue_benchmarks: dict[str, list[dict]],
) -> dict:
    """Train and evaluate one baseline."""

    spec = get_baseline_spec(baseline_id)
    logger.info(f"\n{'=' * 70}")
    logger.info(f"BASELINE: {spec.display_name}")
    logger.info(spec.description)
    logger.info(f"{'=' * 70}")

    config = GMSRAConfig()
    config.model.model_name = args.model_name
    config.model.use_qlora = args.use_qlora
    config.model.load_in_4bit = args.load_in_4bit

    model, tokenizer = load_model_and_tokenizer(
        args.model_name,
        use_qlora=args.use_qlora,
        load_in_4bit=args.load_in_4bit,
    )
    baseline = create_baseline(
        baseline_id,
        model,
        tokenizer,
        config,
        consolidation_interval=args.consolidation_interval,
    )

    output = {
        "spec": spec.to_dict(),
        "train": baseline.train_dialogue(train_data, args.max_train_episodes),
        "benchmarks": {},
    }

    for benchmark_name, benchmark_data in dialogue_benchmarks.items():
        output["benchmarks"][benchmark_name] = baseline.evaluate_dialogue(
            benchmark_data,
            benchmark_name,
        )

    if args.include_agent_tasks and spec.supports_agent_tasks:
        tasks = load_agent_tasks(args.data_dir, args.agent_env, args.num_tasks)
        output["agent_tasks"] = baseline.evaluate_agent_tasks(tasks, args.agent_env)

    baseline_dir = os.path.join(args.output_dir, baseline_id)
    os.makedirs(baseline_dir, exist_ok=True)
    with open(os.path.join(baseline_dir, "results.json"), "w", encoding="utf-8") as handle:
        json.dump(output, handle, indent=2, ensure_ascii=False, default=str)

    return output


def main(args):
    set_seed(42)
    os.makedirs(args.output_dir, exist_ok=True)

    if args.baselines:
        selected = [item.strip() for item in args.baselines.split(",") if item.strip()]
    else:
        selected = list(BASELINE_SPECS.keys())

    train_data, dialogue_benchmarks = load_dialogue_data(args.data_dir)
    logger.info(
        "Loaded baseline data: "
        f"{len(train_data)} train episodes | "
        + ", ".join(f"{name}={len(data)}" for name, data in dialogue_benchmarks.items())
    )

    results = {}
    for baseline_id in selected:
        try:
            results[baseline_id] = run_single_baseline(
                baseline_id,
                args,
                train_data,
                dialogue_benchmarks,
            )
        except Exception as exc:
            logger.exception(f"Baseline {baseline_id} failed")
            results[baseline_id] = {
                "spec": BASELINE_SPECS.get(baseline_id, {"baseline_id": baseline_id}),
                "status": "failed",
                "error": str(exc),
            }

    summary_path = os.path.join(args.output_dir, "baseline_summary.json")
    with open(summary_path, "w", encoding="utf-8") as handle:
        json.dump(results, handle, indent=2, ensure_ascii=False, default=str)

    logger.info(f"\n{'=' * 70}")
    logger.info("BASELINE SUMMARY")
    logger.info(f"{'=' * 70}")
    logger.info(f"{'Baseline':<26} {'LoCoMo F1':>10} {'LongMem F1':>12} {'Status':>10}")
    logger.info(f"{'-' * 70}")
    for baseline_id, result in results.items():
        if result.get("status") == "failed":
            logger.info(f"{baseline_id:<26} {'--':>10} {'--':>12} {'failed':>10}")
            continue

        locomo = result["benchmarks"].get("locomo", {}).get("summary", {})
        longmem = result["benchmarks"].get("longmemeval", {}).get("summary", {})
        logger.info(
            f"{result['spec']['display_name']:<26} "
            f"{locomo.get('avg_f1', 0.0):>10.4f} "
            f"{longmem.get('avg_f1', 0.0):>12.4f} "
            f"{'ok':>10}"
        )

    logger.info(f"\nSaved aggregated baseline results to {summary_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run project-local baseline reproductions")
    parser.add_argument("--model_name", default="Qwen/Qwen2.5-7B-Instruct")
    parser.add_argument("--data_dir", default="data")
    parser.add_argument("--output_dir", default="results/baselines")
    parser.add_argument("--baselines", default=None,
                        help="Comma-separated baseline ids (default: run all)")
    parser.add_argument("--max_train_episodes", type=int, default=100,
                        help="Train episodes per baseline")
    parser.add_argument("--include_agent_tasks", action="store_true",
                        help="Also run online adaptation on task streams")
    parser.add_argument("--agent_env", default="alfworld", choices=["alfworld", "webArena"])
    parser.add_argument("--num_tasks", type=int, default=50)
    parser.add_argument("--consolidation_interval", type=int, default=25)
    parser.add_argument("--use_qlora", action="store_true")
    parser.add_argument("--load_in_4bit", action="store_true")
    args = parser.parse_args()
    main(args)
