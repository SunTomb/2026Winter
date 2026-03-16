"""
Evaluation on Agent Tasks (ALFWorld / WebArena).

Usage:
    python scripts/eval_agent_tasks.py --checkpoint outputs/phase3/best --env alfworld
"""

import argparse
import os
import json

from loguru import logger

from gmsra.config import GMSRAConfig
from gmsra.utils import set_seed, load_model_and_tokenizer


def main(args):
    set_seed(42)
    logger.info(f"Evaluating on {args.env} | checkpoint={args.checkpoint}")

    config = GMSRAConfig()
    model, tokenizer = load_model_and_tokenizer(config.model.model_name)

    if args.checkpoint and os.path.exists(args.checkpoint):
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, args.checkpoint)

    from gmsra.agent import GMSRAAgent
    agent = GMSRAAgent(config)
    agent.initialize(model, tokenizer, env_type="agent_task")
    if os.path.exists(os.path.join(args.checkpoint, "memory_store.json")):
        agent.load_checkpoint(args.checkpoint)

    # Load task suite
    tasks = load_agent_tasks(args.env, args.data_dir, args.num_tasks)
    logger.info(f"Loaded {len(tasks)} tasks from {args.env}")

    # Evaluation metrics
    success_curve = []   # For long-term success rate plot
    failure_recurrence = {}  # Track failure types for FRR

    for task_idx, task in enumerate(tasks):
        # Run task
        env_kwargs = task.get("env_kwargs", {"task_result": {"success": False}})
        for event in task.get("events", []):
            result = agent.step(
                event=event,
                task_context=task.get("instruction", ""),
                agent_response="",
                env_signal_kwargs=env_kwargs,
            )

        success = env_kwargs.get("task_result", {}).get("success", False)
        task_type = task.get("type", "unknown")
        success_curve.append({"task_idx": task_idx, "success": success, "type": task_type})

        # Track failure recurrence
        if not success:
            failure_recurrence.setdefault(task_type, []).append(task_idx)

    # Compute metrics
    total = len(success_curve)
    successes = sum(1 for s in success_curve if s["success"])
    overall_sr = successes / total if total > 0 else 0

    # Failure Recurrence Rate (FRR)
    frr_by_type = {}
    for ftype, indices in failure_recurrence.items():
        if len(indices) > 1:
            # Count how many failures of same type occur after the first
            frr_by_type[ftype] = len(indices) - 1

    summary = {
        "env": args.env,
        "total_tasks": total,
        "successes": successes,
        "overall_success_rate": overall_sr,
        "failure_recurrence": frr_by_type,
        "final_memory_size": agent.memory_store.size(),
        "consolidation_count": agent.distiller.consolidation_count,
    }

    # Save
    os.makedirs(args.output_dir, exist_ok=True)
    with open(os.path.join(args.output_dir, f"{args.env}_results.json"), "w") as f:
        json.dump({"summary": summary, "success_curve": success_curve}, f, indent=2)

    logger.info(f"\nRESULTS: {args.env}")
    logger.info(f"  Success Rate: {overall_sr:.4f} ({successes}/{total})")
    logger.info(f"  Memory Size:  {agent.memory_store.size()}")
    logger.info(f"  FRR types:    {frr_by_type}")


def load_agent_tasks(env: str, data_dir: str, num_tasks: int) -> list[dict]:
    """Load agent evaluation tasks."""
    path = os.path.join(data_dir, f"{env}_tasks.json")
    if os.path.exists(path):
        with open(path, "r") as f:
            return json.load(f)[:num_tasks]

    logger.warning(f"Task data not found at {path}, using placeholder")
    return [
        {
            "instruction": f"Put the mug on the desk (task {i})",
            "events": [f"You are in room {i % 3}. You see a mug and a desk."],
            "type": "put",
            "env_kwargs": {"task_result": {"success": i % 3 == 0, "partial_score": 0.5}},
        }
        for i in range(num_tasks)
    ]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="G-MSRA Agent Task Evaluation")
    parser.add_argument("--checkpoint", default="outputs/phase3/best")
    parser.add_argument("--output_dir", default="results")
    parser.add_argument("--data_dir", default="data")
    parser.add_argument("--env", default="alfworld", choices=["alfworld", "webArena"])
    parser.add_argument("--num_tasks", type=int, default=200)
    args = parser.parse_args()
    main(args)
