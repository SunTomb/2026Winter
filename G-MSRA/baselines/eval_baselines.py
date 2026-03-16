"""
Unified Baseline Evaluation Harness.
Evaluates all baseline agents on LoCoMo / LongMemEval / ALFWorld
with the same metrics used in Table 1 and Table 2 of the paper.

Usage:
    # Run all baselines on LoCoMo:
    python baselines/eval_baselines.py --data_dir data --benchmark locomo

    # Run specific baseline:
    python baselines/eval_baselines.py --agent reflexion --benchmark locomo

    # Run all baselines on all benchmarks:
    python baselines/eval_baselines.py --data_dir data
"""

import argparse
import json
import os
import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = str(Path(__file__).parent.parent)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from loguru import logger
from gmsra.utils import compute_f1, compute_exact_match


# Registry of all baseline agents
AGENT_REGISTRY = {
    "reflexion": {
        "class": "baselines.reflexion_agent.ReflexionAgent",
        "description": "Verbal RL via self-reflection (Shinn 2023)",
    },
    "memory_r1": {
        "class": "baselines.memory_r1_agent.MemoryR1Agent",
        "description": "RL CRUD with QA F1 reward (Chen 2025)",
    },
    "self_consolidation": {
        "class": "baselines.self_consolidation_agent.SelfConsolidationAgent",
        "description": "Contrastive reflection + LoRA (Zhang 2026)",
    },
    "evolver": {
        "class": "baselines.evolver_agent.EvolveRAgent",
        "description": "Experience lifecycle + principle distillation (2025)",
    },
    "mem0_memory_r1": {
        "class": "baselines.mem0_memoryr1_agent.Mem0MemoryR1Agent",
        "description": "Mem0 multi-level memory + RL CRUD (2025)",
    },
}


def load_agent(agent_name: str, model_name: str = "Qwen/Qwen2.5-7B-Instruct"):
    """Dynamically load and initialize a baseline agent."""
    if agent_name not in AGENT_REGISTRY:
        raise ValueError(f"Unknown agent: {agent_name}. Available: {list(AGENT_REGISTRY.keys())}")

    class_path = AGENT_REGISTRY[agent_name]["class"]
    module_name, class_name = class_path.rsplit(".", 1)

    import importlib
    module = importlib.import_module(module_name)
    agent_class = getattr(module, class_name)

    agent = agent_class(model_name=model_name)
    return agent


def load_benchmark_data(data_dir: str, benchmark: str) -> list[dict]:
    """Load benchmark data."""
    file_map = {
        "locomo": "locomo_test.json",
        "longmemeval": "longmemeval_test.json",
        "alfworld": "alfworld_tasks.json",
        "evomemory": "evomemory_test.json",
    }

    if benchmark not in file_map:
        raise ValueError(f"Unknown benchmark: {benchmark}")

    path = os.path.join(data_dir, file_map[benchmark])
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)

    # Fallback to synthetic data
    logger.warning(f"Benchmark data not found at {path}, generating synthetic data")
    from scripts.prepare_data import _generate_synthetic_locomo
    _, test = _generate_synthetic_locomo()
    return test


def evaluate_agent_on_dialogue(agent, data: list[dict],
                                max_episodes: int = None) -> dict:
    """Evaluate agent on dialogue-based benchmarks (LoCoMo, LongMemEval).

    Returns: dict with F1, EM, per-category scores.
    """
    if max_episodes:
        data = data[:max_episodes]

    f1_scores = []
    em_scores = []
    category_scores = {}

    for ep_idx, episode in enumerate(data):
        agent.reset()

        events = episode.get("events", [])
        question = episode.get("question", "")
        answer = episode.get("answer", "")
        category = episode.get("category", "unknown")

        # Process events
        for event in events:
            try:
                agent.process_event(event, context=question)
            except Exception as e:
                logger.warning(f"[{agent.name}] process_event error: {e}")

        # Answer question
        if question and answer:
            try:
                prediction = agent.answer_question(question)
            except Exception as e:
                logger.warning(f"[{agent.name}] answer_question error: {e}")
                prediction = ""

            f1 = compute_f1(prediction, answer)
            em = compute_exact_match(prediction, answer)

            f1_scores.append(f1)
            em_scores.append(em)

            # Per-category tracking
            if category not in category_scores:
                category_scores[category] = {"f1": [], "em": []}
            category_scores[category]["f1"].append(f1)
            category_scores[category]["em"].append(em)

        if (ep_idx + 1) % 20 == 0:
            avg_f1 = sum(f1_scores) / len(f1_scores)
            logger.info(
                f"[{agent.name}] Progress: {ep_idx+1}/{len(data)} | "
                f"F1={avg_f1:.4f}"
            )

    # Compute aggregates
    result = {
        "agent": agent.name,
        "num_episodes": len(f1_scores),
        "avg_f1": sum(f1_scores) / max(len(f1_scores), 1),
        "avg_em": sum(em_scores) / max(len(em_scores), 1),
        "per_category": {},
    }

    for cat, scores in category_scores.items():
        result["per_category"][cat] = {
            "f1": sum(scores["f1"]) / max(len(scores["f1"]), 1),
            "em": sum(scores["em"]) / max(len(scores["em"]), 1),
            "count": len(scores["f1"]),
        }

    # Agent stats
    result["stats"] = agent.get_stats()

    return result


def evaluate_agent_on_tasks(agent, data: list[dict],
                             max_episodes: int = None) -> dict:
    """Evaluate agent on agent task benchmarks (ALFWorld).

    Returns: dict with Success Rate, FRR, Token Cost.
    """
    if max_episodes:
        data = data[:max_episodes]

    successes = []
    failure_recurrences = []
    token_costs = []
    failed_tasks = set()

    for ep_idx, task in enumerate(data):
        agent.reset()

        events = task.get("events", [])
        instruction = task.get("instruction", "")

        for event in events:
            try:
                agent.process_event(event, context=instruction)
            except Exception as e:
                logger.warning(f"[{agent.name}] process_event error: {e}")

        # Check task result
        env_kwargs = task.get("env_kwargs", {})
        task_result = env_kwargs.get("task_result", {})
        success = task_result.get("success", False)
        successes.append(float(success))

        # Track failure recurrence
        task_type = task.get("type", f"task_{ep_idx}")
        if not success:
            if task_type in failed_tasks:
                failure_recurrences.append(1.0)
            else:
                failure_recurrences.append(0.0)
            failed_tasks.add(task_type)
        else:
            failure_recurrences.append(0.0)
            failed_tasks.discard(task_type)

        # Track token cost
        token_costs.append(agent.total_tokens_used)

    # Compute metrics
    result = {
        "agent": agent.name,
        "num_tasks": len(successes),
        "success_rate": sum(successes) / max(len(successes), 1),
        "frr": sum(failure_recurrences) / max(len(failure_recurrences), 1),
        "avg_token_cost": (
            token_costs[-1] / max(len(token_costs), 1) if token_costs else 0
        ),
        "stats": agent.get_stats(),
    }
    return result


def run_evaluation(args):
    """Run evaluation for specified agents and benchmarks."""
    # Select agents
    if args.agent:
        agent_names = [args.agent]
    else:
        agent_names = list(AGENT_REGISTRY.keys())

    # Select benchmarks
    if args.benchmark:
        benchmarks = [args.benchmark]
    else:
        benchmarks = ["locomo", "longmemeval", "alfworld"]

    all_results = {}

    for agent_name in agent_names:
        logger.info(f"\n{'='*60}")
        logger.info(f"Evaluating: {agent_name}")
        logger.info(f"  {AGENT_REGISTRY[agent_name]['description']}")
        logger.info(f"{'='*60}")

        agent = load_agent(agent_name, model_name=args.model_name)
        agent.initialize()

        agent_results = {}

        for benchmark in benchmarks:
            logger.info(f"\n--- Benchmark: {benchmark} ---")
            data = load_benchmark_data(args.data_dir, benchmark)

            if benchmark in ("locomo", "longmemeval", "evomemory"):
                result = evaluate_agent_on_dialogue(
                    agent, data, max_episodes=args.max_episodes
                )
            else:
                result = evaluate_agent_on_tasks(
                    agent, data, max_episodes=args.max_episodes
                )

            agent_results[benchmark] = result

            # Print summary
            if "avg_f1" in result:
                logger.info(
                    f"  [{benchmark}] F1={result['avg_f1']:.4f} "
                    f"EM={result['avg_em']:.4f} "
                    f"({result['num_episodes']} episodes)"
                )
            elif "success_rate" in result:
                logger.info(
                    f"  [{benchmark}] SR={result['success_rate']:.4f} "
                    f"FRR={result['frr']:.4f} "
                    f"({result['num_tasks']} tasks)"
                )

        all_results[agent_name] = agent_results

    # Save results
    os.makedirs(args.output_dir, exist_ok=True)
    output_path = os.path.join(args.output_dir, "baseline_results.json")
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False, default=str)

    # Print summary table
    print_summary_table(all_results, benchmarks)

    logger.info(f"\nResults saved to {output_path}")
    return all_results


def print_summary_table(results: dict, benchmarks: list[str]):
    """Print results in paper table format."""
    logger.info(f"\n{'='*70}")
    logger.info("BASELINE EVALUATION SUMMARY")
    logger.info(f"{'='*70}")

    # Table 1: Dialogue benchmarks
    dialogue_benchmarks = [b for b in benchmarks if b in ("locomo", "longmemeval")]
    if dialogue_benchmarks:
        header = f"{'Method':<25}"
        for bm in dialogue_benchmarks:
            header += f" | {bm.upper()} F1  EM  "
        logger.info(header)
        logger.info("-" * 70)

        for agent_name, agent_results in results.items():
            row = f"{agent_name:<25}"
            for bm in dialogue_benchmarks:
                if bm in agent_results:
                    r = agent_results[bm]
                    row += f" | {r['avg_f1']:.3f}  {r['avg_em']:.3f}"
                else:
                    row += " |   --    --  "
            logger.info(row)

    # Table 2: Agent tasks
    task_benchmarks = [b for b in benchmarks if b == "alfworld"]
    if task_benchmarks:
        logger.info(f"\n{'Method':<25} | SR     FRR    Token Cost")
        logger.info("-" * 55)
        for agent_name, agent_results in results.items():
            if "alfworld" in agent_results:
                r = agent_results["alfworld"]
                logger.info(
                    f"{agent_name:<25} | "
                    f"{r['success_rate']:.3f}  "
                    f"{r['frr']:.3f}  "
                    f"{r['avg_token_cost']:.0f}"
                )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="G-MSRA Baseline Evaluation")
    parser.add_argument("--agent", default=None,
                        choices=list(AGENT_REGISTRY.keys()),
                        help="Specific agent to evaluate (default: all)")
    parser.add_argument("--benchmark", default=None,
                        choices=["locomo", "longmemeval", "alfworld", "evomemory"],
                        help="Specific benchmark (default: all)")
    parser.add_argument("--data_dir", default="data")
    parser.add_argument("--output_dir", default="results/baselines")
    parser.add_argument("--model_name", default="Qwen/Qwen2.5-7B-Instruct")
    parser.add_argument("--max_episodes", type=int, default=None,
                        help="Max episodes per benchmark (default: all)")
    args = parser.parse_args()

    run_evaluation(args)
