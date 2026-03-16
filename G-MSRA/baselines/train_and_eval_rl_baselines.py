"""
Train and Evaluate RL-based Baselines (Memory-R1, Mem0+Memory-R1).

These two baselines require RL training before evaluation.
This script runs the full pipeline: train → save checkpoint → evaluate.

Designed for overnight unattended execution on the cluster.

Usage:
    # Run both RL baselines (full pipeline):
    CUDA_VISIBLE_DEVICES=6 python baselines/train_and_eval_rl_baselines.py

    # Run only Memory-R1:
    CUDA_VISIBLE_DEVICES=6 python baselines/train_and_eval_rl_baselines.py --agent memory_r1

    # Run only Mem0+Memory-R1:
    CUDA_VISIBLE_DEVICES=6 python baselines/train_and_eval_rl_baselines.py --agent mem0_memory_r1

    # Customize training:
    CUDA_VISIBLE_DEVICES=6 python baselines/train_and_eval_rl_baselines.py \\
        --train_epochs 5 --lr 5e-5 --eval_benchmark locomo
"""

import argparse
import json
import os
import sys
import time
import traceback
from pathlib import Path
from datetime import datetime

# Add project root to path
PROJECT_ROOT = str(Path(__file__).parent.parent)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from loguru import logger

# Configure logfile for overnight monitoring
LOG_DIR = os.path.join(PROJECT_ROOT, "results", "baselines")
os.makedirs(LOG_DIR, exist_ok=True)
log_file = os.path.join(LOG_DIR, f"rl_baselines_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
logger.add(log_file, level="INFO")


def load_train_data(data_dir: str) -> list[dict]:
    """Load training data (LoCoMo train split)."""
    path = os.path.join(data_dir, "locomo_train.json")
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        logger.info(f"Loaded {len(data)} training episodes from {path}")
        return data

    # Fallback: generate synthetic data
    logger.warning(f"Train data not found at {path}, generating synthetic data")
    from scripts.prepare_data import _generate_synthetic_locomo
    train, _ = _generate_synthetic_locomo()
    return train


def load_eval_data(data_dir: str, benchmark: str) -> list[dict]:
    """Load evaluation data."""
    from baselines.eval_baselines import load_benchmark_data
    return load_benchmark_data(data_dir, benchmark)


def setup_lora(model):
    """Apply LoRA to make model trainable (if not already)."""
    import torch

    # Check if model already has trainable params
    trainable = sum(1 for p in model.parameters() if p.requires_grad)
    if trainable > 0:
        logger.info(f"Model already has {trainable} trainable parameter groups")
        return model

    try:
        from peft import get_peft_model, LoraConfig, TaskType

        lora_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=8,
            lora_alpha=16,
            lora_dropout=0.05,
            target_modules=["q_proj", "v_proj"],
        )
        model = get_peft_model(model, lora_config)
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        total = sum(p.numel() for p in model.parameters())
        logger.info(
            f"LoRA applied: {trainable:,} trainable / {total:,} total params "
            f"({100*trainable/total:.2f}%)"
        )
    except ImportError:
        logger.warning("peft not installed, making last 2 layers trainable instead")
        # Fallback: unfreeze last 2 transformer layers
        all_params = list(model.named_parameters())
        for name, param in all_params:
            param.requires_grad = False
        # Unfreeze lm_head + last layer
        for name, param in all_params[-20:]:
            param.requires_grad = True
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        logger.info(f"Unfroze last layers: {trainable:,} trainable params")

    return model


def train_agent(agent, train_data: list[dict], num_epochs: int = 3,
                learning_rate: float = 5e-5, checkpoint_dir: str = None):
    """Train an RL-based baseline agent.

    Uses REINFORCE with QA F1 reward:
      1. Process events → build memory
      2. Answer question → compute F1
      3. Update policy with F1 as reward
    """
    import torch

    agent_name = agent.name
    logger.info(f"\n{'='*60}")
    logger.info(f"TRAINING: {agent_name}")
    logger.info(f"  Epochs: {num_epochs}")
    logger.info(f"  Train data: {len(train_data)} episodes")
    logger.info(f"  Learning rate: {learning_rate}")
    logger.info(f"{'='*60}")

    # Apply LoRA for training
    agent.model = setup_lora(agent.model)

    # Setup optimizer
    trainable_params = [p for p in agent.model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable_params, lr=learning_rate, weight_decay=0.01)

    # Training loop
    best_avg_reward = -1
    all_metrics = []

    for epoch in range(num_epochs):
        epoch_rewards = []
        epoch_losses = []
        epoch_start = time.time()

        # Shuffle training data each epoch
        import random
        shuffled = list(train_data)
        random.shuffle(shuffled)

        for ep_idx, episode in enumerate(shuffled):
            events = episode.get("events", [])
            question = episode.get("question", "")
            answer = episode.get("answer", "")

            if not question or not answer:
                continue

            # Reset agent for new episode
            agent.reset()

            # Step 1: Process events (build memory)
            for event in events:
                try:
                    agent.process_event(event, context=question)
                except Exception as e:
                    logger.debug(f"process_event error: {e}")

            # Step 2: Answer question
            try:
                prediction = agent.answer_question(question)
            except Exception as e:
                logger.debug(f"answer_question error: {e}")
                prediction = ""

            # Step 3: Compute reward (QA F1)
            from gmsra.utils import compute_f1
            reward = compute_f1(prediction, answer)
            epoch_rewards.append(reward)

            # Step 4: REINFORCE update
            #   Re-generate the CRUD decision to get log-probs
            optimizer.zero_grad()

            last_event = events[-1] if events else ""
            try:
                result = agent.train_step(
                    reward=reward,
                    event=last_event,
                    context=question,
                )
                if result.get("trained", False):
                    optimizer.step()
                    if "loss" in result:
                        epoch_losses.append(result["loss"])
            except Exception as e:
                logger.debug(f"train_step error: {e}")

            # Log progress
            if (ep_idx + 1) % 10 == 0:
                avg_r = sum(epoch_rewards[-10:]) / min(10, len(epoch_rewards))
                logger.info(
                    f"  [{agent_name}] Epoch {epoch+1}/{num_epochs} | "
                    f"Episode {ep_idx+1}/{len(shuffled)} | "
                    f"Reward(last10)={avg_r:.4f}"
                )

        # Epoch summary
        epoch_time = time.time() - epoch_start
        avg_reward = sum(epoch_rewards) / max(len(epoch_rewards), 1)
        avg_loss = sum(epoch_losses) / max(len(epoch_losses), 1) if epoch_losses else 0

        metrics = {
            "epoch": epoch + 1,
            "avg_reward": avg_reward,
            "avg_loss": avg_loss,
            "num_episodes": len(epoch_rewards),
            "time_seconds": epoch_time,
        }
        all_metrics.append(metrics)

        logger.info(
            f"\n  [{agent_name}] === Epoch {epoch+1}/{num_epochs} DONE ===\n"
            f"    Avg Reward (F1): {avg_reward:.4f}\n"
            f"    Avg Loss:        {avg_loss:.4f}\n"
            f"    Time:            {epoch_time:.1f}s\n"
        )

        # Save best checkpoint
        if checkpoint_dir and avg_reward > best_avg_reward:
            best_avg_reward = avg_reward
            best_path = os.path.join(checkpoint_dir, f"{agent_name}_best")
            agent.save(best_path)

            # Save LoRA weights if using peft
            try:
                agent.model.save_pretrained(os.path.join(best_path, "lora_weights"))
                logger.info(f"  Saved best checkpoint (F1={avg_reward:.4f}) → {best_path}")
            except Exception:
                logger.info(f"  Saved best checkpoint (F1={avg_reward:.4f}) → {best_path}")

    # Save training metrics
    if checkpoint_dir:
        metrics_path = os.path.join(checkpoint_dir, f"{agent_name}_train_metrics.json")
        with open(metrics_path, "w") as f:
            json.dump(all_metrics, f, indent=2)
        logger.info(f"Training metrics saved to {metrics_path}")

    return all_metrics


def evaluate_agent(agent, data_dir: str, benchmarks: list[str],
                   output_dir: str) -> dict:
    """Evaluate a trained agent on specified benchmarks."""
    from baselines.eval_baselines import evaluate_agent_on_dialogue, evaluate_agent_on_tasks

    results = {}
    for benchmark in benchmarks:
        logger.info(f"\n--- Evaluating on {benchmark} ---")
        data = load_eval_data(data_dir, benchmark)

        if benchmark in ("locomo", "longmemeval", "evomemory"):
            result = evaluate_agent_on_dialogue(agent, data)
        else:
            result = evaluate_agent_on_tasks(agent, data)

        results[benchmark] = result

        if "avg_f1" in result:
            logger.info(
                f"  [{benchmark}] F1={result['avg_f1']:.4f} "
                f"EM={result['avg_em']:.4f} "
                f"({result['num_episodes']} episodes)"
            )
        elif "success_rate" in result:
            logger.info(
                f"  [{benchmark}] SR={result['success_rate']:.4f} "
                f"FRR={result['frr']:.4f}"
            )

    # Save evaluation results
    os.makedirs(output_dir, exist_ok=True)
    eval_path = os.path.join(output_dir, f"{agent.name}_eval_results.json")
    with open(eval_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False, default=str)
    logger.info(f"Evaluation results saved to {eval_path}")

    return results


def run_pipeline(args):
    """Run the full train → evaluate pipeline for RL baselines."""
    from baselines.eval_baselines import load_agent

    agents_to_run = []
    if args.agent:
        agents_to_run = [args.agent]
    else:
        agents_to_run = ["memory_r1", "mem0_memory_r1"]

    benchmarks = [args.eval_benchmark] if args.eval_benchmark else ["locomo"]
    train_data = load_train_data(args.data_dir)

    all_results = {}

    for agent_name in agents_to_run:
        start_time = time.time()
        logger.info(f"\n\n{'#'*60}")
        logger.info(f"## Pipeline: {agent_name}")
        logger.info(f"{'#'*60}")

        try:
            # 1. Load agent
            agent = load_agent(agent_name, model_name=args.model_name)
            agent.initialize()

            # 2. Train
            checkpoint_dir = os.path.join(args.output_dir, "checkpoints")
            os.makedirs(checkpoint_dir, exist_ok=True)

            train_metrics = train_agent(
                agent, train_data,
                num_epochs=args.train_epochs,
                learning_rate=args.lr,
                checkpoint_dir=checkpoint_dir,
            )

            # 3. Evaluate
            eval_results = evaluate_agent(
                agent, args.data_dir, benchmarks, args.output_dir
            )

            elapsed = time.time() - start_time
            all_results[agent_name] = {
                "train_metrics": train_metrics,
                "eval_results": eval_results,
                "total_time_seconds": elapsed,
            }

            logger.info(
                f"\n{'='*60}\n"
                f"[{agent_name}] PIPELINE COMPLETE in {elapsed:.0f}s\n"
                f"{'='*60}"
            )

        except Exception as e:
            logger.error(f"[{agent_name}] Pipeline FAILED: {e}")
            traceback.print_exc()
            all_results[agent_name] = {"error": str(e)}

    # Save combined results
    combined_path = os.path.join(args.output_dir, "rl_baselines_combined.json")
    with open(combined_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False, default=str)

    # Print final summary
    logger.info(f"\n\n{'='*60}")
    logger.info("FINAL SUMMARY — RL BASELINE PIPELINES")
    logger.info(f"{'='*60}")
    for name, res in all_results.items():
        if "error" in res:
            logger.info(f"  {name}: FAILED — {res['error']}")
        else:
            eval_r = res.get("eval_results", {})
            for bm, bm_results in eval_r.items():
                if "avg_f1" in bm_results:
                    logger.info(
                        f"  {name} on {bm}: F1={bm_results['avg_f1']:.4f} "
                        f"EM={bm_results['avg_em']:.4f}"
                    )
            train_m = res.get("train_metrics", [])
            if train_m:
                final_r = train_m[-1].get("avg_reward", 0)
                logger.info(f"  {name} final train reward: {final_r:.4f}")
            logger.info(
                f"  {name} total time: "
                f"{res.get('total_time_seconds', 0):.0f}s"
            )

    logger.info(f"\nAll results saved to {combined_path}")
    logger.info(f"Log file: {log_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train and evaluate RL-based baselines (Memory-R1, Mem0+Memory-R1)"
    )
    parser.add_argument("--agent", default=None,
                        choices=["memory_r1", "mem0_memory_r1"],
                        help="Specific agent (default: both)")
    parser.add_argument("--data_dir", default="data",
                        help="Directory with prepared datasets")
    parser.add_argument("--output_dir", default="results/baselines",
                        help="Output directory for results and checkpoints")
    parser.add_argument("--model_name", default="Qwen/Qwen2.5-7B-Instruct")
    parser.add_argument("--train_epochs", type=int, default=3,
                        help="Number of training epochs (default: 3)")
    parser.add_argument("--lr", type=float, default=5e-5,
                        help="Learning rate (default: 5e-5)")
    parser.add_argument("--eval_benchmark", default=None,
                        choices=["locomo", "longmemeval", "alfworld", "evomemory"],
                        help="Benchmark for evaluation (default: locomo)")
    args = parser.parse_args()

    run_pipeline(args)
