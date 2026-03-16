"""
Phase 3: Full Closed-Loop with Consolidation.
No external reward — pure grounded self-reward + adaptive consolidation.

Key improvement over skeleton: actual RL policy updates during closed-loop
training, proper consolidation integration, and comprehensive diagnostics.

Usage:
    accelerate launch --num_processes 4 scripts/train_phase3_full.py \
        --checkpoint outputs/phase2/best --max_episodes 10000
"""

import argparse
import os
import json

import torch
from loguru import logger

from gmsra.config import GMSRAConfig
from gmsra.utils import set_seed, load_model_and_tokenizer


def main(args):
    set_seed(42)
    logger.info(f"Phase 3: Full Closed-Loop | checkpoint={args.checkpoint}")

    config = GMSRAConfig()

    # Load model
    model, tokenizer = load_model_and_tokenizer(config.model.model_name)
    if args.checkpoint and os.path.exists(args.checkpoint):
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, args.checkpoint)

    # Initialize agent
    from gmsra.agent import GMSRAAgent
    agent = GMSRAAgent(config)
    agent.initialize(model, tokenizer, env_type=args.env_type)

    # Load checkpoint
    if os.path.exists(os.path.join(args.checkpoint, "memory_store.json")):
        agent.load_checkpoint(args.checkpoint)

    # Setup LoRA for distillation
    if args.consolidation_enabled:
        agent.distiller.setup_dual_lora()

    # --- Setup RL optimizer for continued policy updates ---
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(
        trainable_params,
        lr=config.rl.learning_rate * 0.3,  # Lower LR for Phase 3
        weight_decay=0.01,
    )

    # Load task stream based on env_type
    task_stream = load_task_stream(args.env_type, args.data_dir, args.max_episodes)
    logger.info(f"Loaded {len(task_stream)} tasks for full-loop training")

    # Metrics tracking
    metrics_log = []
    success_window = []
    reward_baseline = 0.0
    WINDOW_SIZE = 50

    model.train()
    for ep_idx, task in enumerate(task_stream):
        # Process the task through the full G-MSRA loop
        task_events = task.get("events", [task.get("instruction", "")])
        task_context = task.get("context", task.get("question", ""))

        last_result = None
        for event in task_events:
            result = agent.step(
                event=event,
                task_context=task_context,
                agent_response="",
                env_signal_kwargs=task.get("env_kwargs", {}),
            )
            last_result = result

        if last_result is None:
            continue

        # --- RL policy update with self-reward ---
        r_total = last_result["reward"]["r_total"]
        reward_baseline = 0.95 * reward_baseline + 0.05 * r_total
        advantage = r_total - reward_baseline

        if task_events and trainable_params and abs(advantage) > 0.01:
            last_event = task_events[-1]
            _, prompt = agent.memory_manager.decide(last_event, task_context)
            operation_str = last_result["operation"].get("content", "NOOP")

            inputs = tokenizer(
                prompt + operation_str,
                return_tensors="pt", truncation=True,
                max_length=1024, padding=True,
            )
            inputs = {k: v.to(model.device) for k, v in inputs.items()}

            prompt_len = len(tokenizer.encode(prompt, truncation=True, max_length=1024))
            labels = inputs["input_ids"].clone()
            labels[0, :prompt_len] = -100

            outputs = model(**inputs, labels=labels)
            policy_loss = -advantage * outputs.loss

            policy_loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable_params, config.rl.max_grad_norm)

            if (ep_idx + 1) % config.rl.gradient_accumulation_steps == 0:
                optimizer.step()
                optimizer.zero_grad()

        # Track success rate
        task_success = last_result["reward"]["r_env"] > 0.5
        success_window.append(float(task_success))
        if len(success_window) > WINDOW_SIZE:
            success_window.pop(0)

        # Log periodically
        if (ep_idx + 1) % 50 == 0:
            avg_success = sum(success_window) / len(success_window)
            diagnostics = agent.get_full_diagnostics()

            log_entry = {
                "episode": ep_idx + 1,
                "avg_success_rate": avg_success,
                "avg_reward": r_total,
                "memory_size": diagnostics["memory_size"],
                "consolidation_count": diagnostics["consolidation_count"],
                "operation_stats": diagnostics["operation_stats"],
            }
            metrics_log.append(log_entry)

            logger.info(
                f"Episode {ep_idx+1}/{args.max_episodes} | "
                f"success_rate={avg_success:.3f} | "
                f"R_total={r_total:.3f} | "
                f"mem_size={diagnostics['memory_size']} | "
                f"consolidations={diagnostics['consolidation_count']}"
            )

        # Save checkpoint periodically
        if (ep_idx + 1) % 500 == 0:
            agent.save_checkpoint(
                os.path.join(args.output_dir, f"checkpoint_{ep_idx+1}")
            )

    # Save final results
    os.makedirs(args.output_dir, exist_ok=True)
    agent.save_checkpoint(os.path.join(args.output_dir, "best"))

    with open(os.path.join(args.output_dir, "metrics.json"), "w") as f:
        json.dump(metrics_log, f, indent=2)

    # Save full diagnostics for paper figures
    diagnostics = agent.get_full_diagnostics()
    with open(os.path.join(args.output_dir, "diagnostics.json"), "w") as f:
        json.dump(diagnostics, f, indent=2, default=str)

    logger.info("Phase 3 complete!")


def load_task_stream(env_type: str, data_dir: str,
                     max_episodes: int) -> list[dict]:
    """Load task stream based on environment type."""
    if env_type == "agent_task":
        # ALFWorld / WebArena format
        data_path = os.path.join(data_dir, "alfworld_tasks.json")
        if os.path.exists(data_path):
            with open(data_path, "r") as f:
                tasks = json.load(f)
            return tasks[:max_episodes]

        # Placeholder
        logger.warning(f"Task data not found at {data_path}, using placeholder")
        import random
        random.seed(42)
        return [
            {
                "instruction": f"Complete task {i}",
                "events": [f"Observation: you are in room {i % 5}"],
                "context": "Navigate and complete household tasks.",
                "env_kwargs": {
                    "task_result": {
                        "success": random.random() > 0.6,
                        "partial_score": random.uniform(0.2, 0.8),
                        "steps_taken": random.randint(3, 20),
                        "max_steps": 30,
                    }
                },
            }
            for i in range(max_episodes)
        ]
    else:
        # Dialogue format
        from scripts.train_phase1_rl import load_locomo_data
        data = load_locomo_data(data_dir)
        # Convert to task stream format
        tasks = []
        for episode in data[:max_episodes]:
            tasks.append({
                "events": episode.get("events", []),
                "context": episode.get("question", ""),
                "question": episode.get("question", ""),
                "answer": episode.get("answer", ""),
                "env_kwargs": {
                    "prediction": "",
                    "ground_truth": episode.get("answer", ""),
                },
            })
        return tasks


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="G-MSRA Phase 3: Full Closed-Loop")
    parser.add_argument("--checkpoint", default="outputs/phase2/best")
    parser.add_argument("--output_dir", default="outputs/phase3")
    parser.add_argument("--data_dir", default="data")
    parser.add_argument("--env_type", default="dialogue", choices=["dialogue", "agent_task"])
    parser.add_argument("--max_episodes", type=int, default=10000)
    parser.add_argument("--consolidation_enabled", action="store_true", default=True)
    args = parser.parse_args()
    main(args)
