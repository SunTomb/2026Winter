"""
Phase 2: Self-Reward Transition (Curriculum Annealing).
Gradually replaces external QA F1 reward with grounded self-reward.

R_phase2(t) = α(t)·R_ext + (1-α(t))·R_total

α anneals from 1.0 → 0.0. Transition paused if Kendall τ < threshold.

Key improvement over skeleton: actual RL policy updates during annealing,
not just reward monitoring.

Usage:
    accelerate launch --num_processes 4 scripts/train_phase2_transition.py \
        --checkpoint outputs/phase1/best
"""

import argparse
import os
import json

import torch
from loguru import logger

from gmsra.config import GMSRAConfig
from gmsra.utils import set_seed, load_model_and_tokenizer, compute_f1, compute_kendall_tau


def main(args):
    set_seed(42)
    logger.info(f"Phase 2: Self-Reward Transition | checkpoint={args.checkpoint}")

    config = GMSRAConfig()
    config.reward.anneal_steps = args.anneal_steps
    config.reward.tau_threshold = args.tau_threshold

    # Load model from Phase 1
    model, tokenizer = load_model_and_tokenizer(config.model.model_name)
    if args.checkpoint and os.path.exists(args.checkpoint):
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, args.checkpoint)
        logger.info(f"Loaded Phase 1 checkpoint: {args.checkpoint}")

    # Initialize agent
    from gmsra.agent import GMSRAAgent
    agent = GMSRAAgent(config)
    agent.initialize(model, tokenizer, env_type="dialogue")

    # Load checkpoint state
    if os.path.exists(os.path.join(args.checkpoint, "memory_store.json")):
        agent.load_checkpoint(args.checkpoint)

    # Load dataset
    from scripts.train_phase1_rl import load_locomo_data
    dataset = load_locomo_data(args.data_dir)

    # --- Setup RL optimizer for policy updates during annealing ---
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable_params, lr=config.rl.learning_rate * 0.5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=config.reward.anneal_steps
    )

    # Annealing schedule
    alpha = config.reward.anneal_start_alpha  # Start at 1.0
    alpha_step = (config.reward.anneal_start_alpha - config.reward.anneal_end_alpha) / config.reward.anneal_steps
    paused = False

    # Tracking for calibration
    ext_rewards = []
    self_rewards = []
    annealed_rewards = []
    reward_baseline = 0.0

    model.train()
    for step_idx, episode in enumerate(dataset[:config.reward.anneal_steps]):
        events = episode.get("events", [])
        question = episode.get("question", "")
        answer = episode.get("answer", "")

        # Process events through Memory Manager
        for event in events:
            agent.step(
                event=event,
                task_context=question,
                agent_response="",
                env_signal_kwargs={"prediction": "", "ground_truth": answer},
            )

        # Get external reward (from QA F1)
        predicted = agent.answer_question(question)
        r_ext = compute_f1(predicted, answer)

        # Get self-reward
        r_self_result = agent.reward_generator.compute_reward(
            agent_response=predicted,
            task_context=question,
            memory_operation="NOOP",
            env_signal_kwargs={"prediction": predicted, "ground_truth": answer},
        )

        # Compute annealed reward
        r_annealed = alpha * r_ext + (1 - alpha) * r_self_result.r_total

        ext_rewards.append(r_ext)
        self_rewards.append(r_self_result.r_total)
        annealed_rewards.append(r_annealed)

        # --- RL policy update with annealed reward ---
        # Use the annealed reward to perform a REINFORCE-style update
        # on the Memory Manager's last decision
        if question and answer and trainable_params:
            # Re-generate the memory decision to get gradients
            operation_str, prompt = agent.memory_manager.decide(events[-1] if events else "", question)

            inputs = tokenizer(
                prompt + operation_str,
                return_tensors="pt", truncation=True, max_length=1024, padding=True,
            )
            inputs = {k: v.to(model.device) for k, v in inputs.items()}

            prompt_len = len(tokenizer.encode(prompt, truncation=True, max_length=1024))
            labels = inputs["input_ids"].clone()
            labels[0, :prompt_len] = -100  # Mask prompt tokens

            outputs = model(**inputs, labels=labels)
            loss = outputs.loss

            # REINFORCE: scale loss by advantage (reward - baseline)
            reward_baseline = 0.95 * reward_baseline + 0.05 * r_annealed
            advantage = r_annealed - reward_baseline
            policy_loss = -advantage * loss

            policy_loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable_params, config.rl.max_grad_norm)

            if (step_idx + 1) % config.rl.gradient_accumulation_steps == 0:
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

        # Check calibration every 100 steps
        if (step_idx + 1) % 100 == 0 and len(ext_rewards) >= 50:
            tau = compute_kendall_tau(ext_rewards[-50:], self_rewards[-50:])
            avg_annealed = sum(annealed_rewards[-50:]) / 50
            logger.info(
                f"Step {step_idx+1} | α={alpha:.3f} | τ={tau:.3f} | "
                f"R_ext={r_ext:.3f} | R_self={r_self_result.r_total:.3f} | "
                f"R_annealed={avg_annealed:.3f}"
            )

            # Pause annealing if calibration drops
            if tau < config.reward.tau_threshold:
                if not paused:
                    logger.warning(
                        f"ANNEALING PAUSED: Kendall τ={tau:.3f} < "
                        f"threshold={config.reward.tau_threshold}"
                    )
                    paused = True
            else:
                paused = False

        # Anneal alpha (unless paused)
        if not paused and alpha > config.reward.anneal_end_alpha:
            alpha = max(config.reward.anneal_end_alpha, alpha - alpha_step)

        # Periodic save
        if (step_idx + 1) % 500 == 0:
            agent.save_checkpoint(
                os.path.join(args.output_dir, f"checkpoint_{step_idx+1}")
            )

    # Save final
    os.makedirs(args.output_dir, exist_ok=True)
    agent.save_checkpoint(os.path.join(args.output_dir, "best"))

    # Save calibration data (for paper §5.1)
    calib_data = {
        "ext_rewards": ext_rewards,
        "self_rewards": self_rewards,
        "annealed_rewards": annealed_rewards,
        "final_alpha": alpha,
        "total_steps": len(ext_rewards),
    }
    with open(os.path.join(args.output_dir, "calibration.json"), "w") as f:
        json.dump(calib_data, f)

    logger.info(f"Phase 2 complete. Final α={alpha:.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="G-MSRA Phase 2: Self-Reward Transition")
    parser.add_argument("--checkpoint", default="outputs/phase1/best")
    parser.add_argument("--output_dir", default="outputs/phase2")
    parser.add_argument("--data_dir", default="data")
    parser.add_argument("--anneal_steps", type=int, default=3000)
    parser.add_argument("--tau_threshold", type=float, default=0.5)
    args = parser.parse_args()
    main(args)
