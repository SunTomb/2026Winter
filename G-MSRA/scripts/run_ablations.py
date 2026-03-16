"""
Ablation Experiment Runner (A1-A7).
Runs all 7 ablation experiments defined in the G-MSRA proposal.

Each ablation modifies the GMSRAConfig and runs a shortened Phase 3
training followed by evaluation, producing results for Table 3 of the paper.

Usage:
    python scripts/run_ablations.py --base_checkpoint outputs/phase1/best
    python scripts/run_ablations.py --ablations A1_no_env_anchor,A2_no_memory_consistency
"""

import argparse
import os
import json
import sys
import random
from copy import deepcopy

from loguru import logger

from gmsra.config import GMSRAConfig
from gmsra.utils import set_seed


ABLATION_CONFIGS = {
    "A1_no_env_anchor": {
        "description": "Remove R_env, use only R_mem (pure self-reward). "
                       "THIS IS THE MOST CRITICAL ABLATION — tests Reward Hacking.",
        "priority": 5,
    },
    "A2_no_memory_consistency": {
        "description": "Remove R_mem, use only R_env. "
                       "Tests memory consistency signal's fine-grained guidance value.",
        "priority": 4,
    },
    "A3_no_confidence_filter": {
        "description": "Remove memory confidence filtering from Judge. "
                       "Tests noise pollution prevention.",
        "priority": 3,
    },
    "A4_fixed_trigger": {
        "description": "Replace adaptive trigger with fixed threshold (every 50 episodes). "
                       "Tests 3D adaptive trigger vs heuristic.",
        "priority": 3,
    },
    "A5_random_distill": {
        "description": "Random memory sampling instead of graph-based selection for distillation. "
                       "Tests graph topology's role in consolidation quality.",
        "priority": 2,
    },
    "A6_no_consolidation": {
        "description": "Remove LoRA consolidation entirely, only external memory. "
                       "Tests parametric consolidation's long-term benefit.",
        "priority": 4,
    },
    "A7_no_curriculum": {
        "description": "Skip Phase 1-2, start directly with self-reward. "
                       "Tests curriculum training's stability contribution.",
        "priority": 3,
    },
}


def apply_ablation_config(ablation_name: str, config: GMSRAConfig) -> GMSRAConfig:
    """Apply ablation-specific configuration changes.

    Returns:
        Modified config for this ablation.
    """
    ablation_config = deepcopy(config)

    if ablation_name == "A1_no_env_anchor":
        # Pure self-reward: only R_mem, no environment anchor
        ablation_config.reward.lambda_mem = 1.0
        # We'll also need to disable env_extractor in the agent

    elif ablation_name == "A2_no_memory_consistency":
        # Only environment reward, no memory consistency
        ablation_config.reward.lambda_mem = 0.0

    elif ablation_name == "A3_no_confidence_filter":
        # Use all memories (no confidence filtering)
        ablation_config.memory.confidence_topk = 999

    elif ablation_name == "A4_fixed_trigger":
        # Always trigger at fixed intervals
        ablation_config.trigger.theta = -1.0  # Score always exceeds threshold
        ablation_config.trigger.min_interval = 50  # Fixed every 50 steps
        ablation_config.trigger.alpha = 0.0
        ablation_config.trigger.beta = 0.0
        ablation_config.trigger.gamma = 0.0

    elif ablation_name == "A5_random_distill":
        # Will override distiller's subgraph extraction with random sampling
        pass  # Handled at runtime via monkey-patching

    elif ablation_name == "A6_no_consolidation":
        # Disable consolidation entirely
        # Handled by not enabling consolidation in agent
        pass

    elif ablation_name == "A7_no_curriculum":
        # No Phase 1-2: start directly with self-reward
        ablation_config.reward.anneal_start_alpha = 0.0  # Skip curriculum
        ablation_config.reward.anneal_end_alpha = 0.0

    return ablation_config


def run_ablation(ablation_name: str, config: GMSRAConfig, args) -> dict:
    """Run a single ablation experiment.

    1. Apply config modifications
    2. Run shortened Phase 3 training
    3. Evaluate on LoCoMo
    4. Return results
    """
    logger.info(f"\n{'=' * 60}")
    logger.info(f"ABLATION: {ablation_name}")
    logger.info(f"Description: {ABLATION_CONFIGS[ablation_name]['description']}")
    logger.info(f"{'=' * 60}")

    # Apply ablation-specific config changes
    ablation_config = apply_ablation_config(ablation_name, config)
    output_dir = os.path.join(args.output_dir, ablation_name)
    os.makedirs(output_dir, exist_ok=True)

    # Save ablation metadata
    with open(os.path.join(output_dir, "ablation_config.json"), "w") as f:
        json.dump({
            "name": ablation_name,
            "description": ABLATION_CONFIGS[ablation_name]["description"],
            "priority": ABLATION_CONFIGS[ablation_name]["priority"],
        }, f, indent=2)

    # --- Run training via Phase 3 with modified config ---
    from gmsra.utils import load_model_and_tokenizer
    model, tokenizer = load_model_and_tokenizer(ablation_config.model.model_name)

    if args.base_checkpoint and os.path.exists(args.base_checkpoint):
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, args.base_checkpoint)
        logger.info(f"Loaded base checkpoint: {args.base_checkpoint}")

    from gmsra.agent import GMSRAAgent
    agent = GMSRAAgent(ablation_config)

    # Determine env_type
    env_type = "dialogue"  # Default for LoCoMo evaluation
    agent.initialize(model, tokenizer, env_type=env_type)

    # A1: Disable env_extractor (pure self-reward)
    if ablation_name == "A1_no_env_anchor":
        agent.env_extractor = None
        agent.reward_generator.env_extractor = None
        logger.info("A1: Disabled environment signal extractor")

    # A5: Override distiller's subgraph extraction with random sampling
    if ablation_name == "A5_random_distill":
        _patch_random_distillation(agent)
        logger.info("A5: Patched distiller with random sampling")

    # A6: Disable consolidation
    consolidation_enabled = (ablation_name != "A6_no_consolidation")
    if consolidation_enabled and ablation_name not in ("A1_no_env_anchor", "A7_no_curriculum"):
        agent.distiller.setup_dual_lora()

    # Load task stream
    from scripts.train_phase3_full import load_task_stream
    max_episodes = args.num_episodes
    task_stream = load_task_stream("dialogue", args.data_dir, max_episodes)

    # Training loop (shortened version of Phase 3)
    import torch
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable_params, lr=ablation_config.rl.learning_rate * 0.3) if trainable_params else None

    reward_history = []
    success_window = []
    reward_baseline = 0.0

    model.train()
    for ep_idx, task in enumerate(task_stream):
        events = task.get("events", [])
        context = task.get("context", task.get("question", ""))

        last_result = None
        for event in events:
            result = agent.step(
                event=event,
                task_context=context,
                agent_response="",
                env_signal_kwargs=task.get("env_kwargs", {}),
            )
            last_result = result

        if last_result is None:
            continue

        r_total = last_result["reward"]["r_total"]
        reward_history.append(r_total)

        # RL update
        reward_baseline = 0.95 * reward_baseline + 0.05 * r_total
        advantage = r_total - reward_baseline

        if events and optimizer and trainable_params and abs(advantage) > 0.01:
            last_event = events[-1]
            _, prompt = agent.memory_manager.decide(last_event, context)

            inputs = tokenizer(
                prompt + "NOOP",
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
            torch.nn.utils.clip_grad_norm_(trainable_params, ablation_config.rl.max_grad_norm)

            if (ep_idx + 1) % ablation_config.rl.gradient_accumulation_steps == 0:
                optimizer.step()
                optimizer.zero_grad()

        task_success = last_result["reward"]["r_env"] > 0.5
        success_window.append(float(task_success))
        if len(success_window) > 50:
            success_window.pop(0)

        if (ep_idx + 1) % 100 == 0:
            avg_success = sum(success_window) / len(success_window)
            logger.info(
                f"[{ablation_name}] Ep {ep_idx+1}/{max_episodes} | "
                f"success={avg_success:.3f} | R={r_total:.3f} | "
                f"mem={agent.memory_store.size()}"
            )

    # Save checkpoint
    agent.save_checkpoint(os.path.join(output_dir, "checkpoint"))

    # --- Evaluate ---
    from gmsra.utils import compute_f1, compute_exact_match

    eval_results = _evaluate_ablation(agent, args.data_dir)

    # Save results
    result = {
        "ablation": ablation_name,
        "description": ABLATION_CONFIGS[ablation_name]["description"],
        "training_episodes": len(reward_history),
        "final_success_rate": sum(success_window) / max(len(success_window), 1),
        "avg_reward": sum(reward_history) / max(len(reward_history), 1),
        "reward_history": reward_history,
        "eval_results": eval_results,
        "memory_size": agent.memory_store.size(),
        "consolidation_count": agent.distiller.consolidation_count,
        "status": "completed",
    }

    with open(os.path.join(output_dir, "results.json"), "w") as f:
        json.dump(result, f, indent=2, default=str)

    logger.info(f"[{ablation_name}] Complete: F1={eval_results.get('avg_f1', 0):.4f}")
    return result


def _evaluate_ablation(agent, data_dir: str) -> dict:
    """Evaluate agent on LoCoMo test set."""
    from gmsra.utils import compute_f1, compute_exact_match

    data_path = os.path.join(data_dir, "locomo_test.json")
    if os.path.exists(data_path):
        with open(data_path, "r") as f:
            eval_data = json.load(f)
    else:
        # Use placeholder
        from scripts.train_phase1_rl import _generate_placeholder_data
        eval_data = _generate_placeholder_data()[:20]

    f1_scores = []
    em_scores = []
    for example in eval_data:
        for event in example.get("events", []):
            agent.step(event=event, task_context="", agent_response="")

        question = example.get("question", "")
        answer = example.get("answer", "")
        if question and answer:
            prediction = agent.answer_question(question)
            f1_scores.append(compute_f1(prediction, answer))
            em_scores.append(compute_exact_match(prediction, answer))

    return {
        "avg_f1": sum(f1_scores) / max(len(f1_scores), 1),
        "avg_em": sum(em_scores) / max(len(em_scores), 1),
        "num_examples": len(f1_scores),
    }


def _patch_random_distillation(agent):
    """A5: Patch the distiller to use random memory sampling
    instead of graph-based subgraph extraction."""
    original_consolidate = agent.distiller.consolidate

    def random_consolidate(memory_store, llm_model=None, llm_tokenizer=None):
        """Override: random sampling instead of subgraph extraction."""
        entries = list(memory_store.entries.values())
        if len(entries) < 3:
            return {"distilled": 0, "skipped": True}

        # Random sample instead of high-frequency subgraph
        k = min(len(entries), 20)
        candidates = random.sample(entries, k)

        # Continue with normal distillation pipeline
        from gmsra.utils import generate_text
        triples = agent.distiller._generate_semantic_triples(
            candidates, llm_model or agent.distiller.base_model,
            llm_tokenizer or agent.distiller.tokenizer,
        )
        if not triples:
            return {"distilled": 0, "skipped": True}

        if agent.distiller._lora_model is None:
            agent.distiller.setup_dual_lora()

        train_loss = agent.distiller._train_lora(triples)
        agent.distiller.distilled_entries.extend([e.id for e in candidates])
        agent.distiller.consolidation_count += 1

        return {
            "distilled": len(candidates),
            "triples": len(triples),
            "train_loss": train_loss,
            "consolidation_num": agent.distiller.consolidation_count,
            "method": "random_sampling",
            "skipped": False,
        }

    agent.distiller.consolidate = random_consolidate


def main(args):
    set_seed(42)
    config = GMSRAConfig()

    # Select which ablations to run
    if args.ablations:
        selected = args.ablations.split(",")
    else:
        # Run in priority order (highest first)
        selected = sorted(
            ABLATION_CONFIGS.keys(),
            key=lambda k: ABLATION_CONFIGS[k]["priority"],
            reverse=True,
        )

    logger.info(f"Running {len(selected)} ablation experiments")
    logger.info(f"Order: {selected}")

    results = {}
    for ablation_name in selected:
        if ablation_name not in ABLATION_CONFIGS:
            logger.warning(f"Unknown ablation: {ablation_name}, skipping")
            continue
        try:
            result = run_ablation(ablation_name, config, args)
            results[ablation_name] = result
        except Exception as e:
            logger.error(f"Ablation {ablation_name} FAILED: {e}")
            results[ablation_name] = {"status": "failed", "error": str(e)}

    # Save aggregated summary (for Table 3)
    summary_path = os.path.join(args.output_dir, "ablation_summary.json")
    with open(summary_path, "w") as f:
        json.dump(results, f, indent=2, default=str)

    # Print summary table
    logger.info(f"\n{'=' * 70}")
    logger.info("ABLATION SUMMARY (Table 3)")
    logger.info(f"{'=' * 70}")
    logger.info(f"{'Ablation':<30} {'F1':>6} {'EM':>6} {'Success':>8} {'Status':>10}")
    logger.info(f"{'-' * 70}")
    for name, result in results.items():
        if result.get("status") == "completed":
            eval_r = result.get("eval_results", {})
            logger.info(
                f"{name:<30} "
                f"{eval_r.get('avg_f1', 0):>6.4f} "
                f"{eval_r.get('avg_em', 0):>6.4f} "
                f"{result.get('final_success_rate', 0):>8.4f} "
                f"{'✓':>10}"
            )
        else:
            logger.info(f"{name:<30} {'—':>6} {'—':>6} {'—':>8} {'✗':>10}")

    logger.info(f"\nResults saved to {summary_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="G-MSRA Ablation Experiments")
    parser.add_argument("--base_checkpoint", default="outputs/phase1/best")
    parser.add_argument("--output_dir", default="results/ablations")
    parser.add_argument("--data_dir", default="data")
    parser.add_argument("--ablations", default=None,
                        help="Comma-separated list of ablation IDs (default: all)")
    parser.add_argument("--num_episodes", type=int, default=1000,
                        help="Training episodes per ablation (shorter than full Phase 3)")
    args = parser.parse_args()
    main(args)
