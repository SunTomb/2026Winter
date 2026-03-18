"""
Phase 1: RL Training with External Reward.
Trains Memory Manager RL policy using external QA F1 as reward signal.
Establishes performance upper bound for comparison with Phase 2-3.

Uses TRL's GRPOTrainer (Group Relative Policy Optimization) for RL training.
GRPO is simpler and more stable than PPO for language model RL,
and doesn't require a separate value head or critic model.

Usage (single A100 80GB):
    CUDA_VISIBLE_DEVICES=<GPU> python scripts/train_phase1_rl.py \
        --model_name Qwen/Qwen2.5-7B-Instruct \
        --num_episodes 5000 --no_qlora \
        --per_device_batch_size 8 --num_generations 16 \
        --max_completion_length 192 --output_dir outputs/phase1

Usage (4× A40 48GB, multi-GPU):
    CUDA_VISIBLE_DEVICES=1,5,6,7 accelerate launch \
        --config_file cluster/accelerate_a40.yaml \
        scripts/train_phase1_rl.py \
        --model_name Qwen/Qwen2.5-7B-Instruct \
        --num_episodes 5000 --gpu_preset a40 \
        --output_dir outputs/phase1
"""

import argparse
import os
import json
from dataclasses import dataclass

import torch
from loguru import logger

from gmsra.config import GMSRAConfig
from gmsra.utils import set_seed, load_model_and_tokenizer, compute_f1


# ============================================================
# TRL version detection
# ============================================================

def _get_trl_version():
    """Get installed TRL version as a tuple of ints, e.g. (0, 15, 0)."""
    try:
        from importlib.metadata import version as pkg_version
        ver_str = pkg_version("trl")
        parts = ver_str.split(".")
        return tuple(int(p) for p in parts[:3])
    except Exception:
        return (0, 0, 0)


def _check_trl_capabilities():
    """Check which TRL trainers are available and return capability dict."""
    trl_version = _get_trl_version()
    logger.info(f"TRL version detected: {'.'.join(str(v) for v in trl_version)}")

    caps = {"grpo": False, "ppo": False, "version": trl_version}

    # GRPOTrainer requires trl >= 0.14.0
    if trl_version >= (0, 14, 0):
        try:
            from trl import GRPOConfig, GRPOTrainer
            caps["grpo"] = True
            logger.info("✓ GRPOTrainer available (recommended)")
        except ImportError as e:
            logger.warning(f"GRPOTrainer import failed despite version {trl_version}: {e}")

    # PPOTrainer API differs across versions
    try:
        from trl import PPOConfig, PPOTrainer
        caps["ppo"] = True
        logger.info("✓ PPOTrainer available (fallback)")
    except ImportError as e:
        logger.warning(f"PPOTrainer import failed: {e}")

    if not caps["grpo"] and not caps["ppo"]:
        logger.warning(
            "⚠ Neither GRPOTrainer nor PPOTrainer available! "
            "Will use manual REINFORCE (slower, less stable). "
            "To fix: pip install 'trl>=0.15.0' 'accelerate>=0.34.0' 'transformers>=4.46.0'"
        )

    return caps


# ============================================================
# Dataset for RL training
# ============================================================

MAX_EVENTS_PER_EPISODE = 5  # Cap events per episode to avoid 1M+ total steps


def build_rl_prompts_from_episode(episode: dict, max_events: int = MAX_EVENTS_PER_EPISODE) -> list[dict]:
    """Convert a LoCoMo episode into RL training prompts.

    For each event in the episode (up to max_events), we create a prompt
    that asks the Memory Manager to decide the operation. The reward comes
    from the QA F1 at the end of the episode.

    Returns:
        List of prompt dicts with keys: "query", "events", "question", "answer"
    """
    events = episode.get("events", [])
    question = episode.get("question", "")
    answer = episode.get("answer", "")

    # Cap event count to avoid explosion in total training steps
    if len(events) > max_events:
        # Sample evenly-spaced events to maintain coverage
        indices = [int(i * (len(events) - 1) / (max_events - 1)) for i in range(max_events)]
        events = [events[i] for i in indices]

    prompts = []
    memory_context = "(empty memory)"
    for i, event in enumerate(events):
        prompt = (
            "You are a Memory Manager for an AI agent. "
            "Given the current memory entries and a new event, "
            "decide the best memory operation.\n\n"
            "### Available Operations\n"
            "- ADD: <content> — Store new important information\n"
            "- UPDATE <id>: <new_content> — Update existing memory\n"
            "- DELETE <id> — Remove outdated/wrong memory\n"
            "- NOOP — No action needed\n\n"
            f"### Current Memory Entries\n{memory_context}\n\n"
            f"### New Event\n{event}\n\n"
            "### Decision\n"
        )
        prompts.append({
            "query": prompt,
            "event": event,
            "question": question,
            "answer": answer,
        })
        # Simulate memory accumulation for context
        memory_context += f"\n[m{i+1}] {event[:60]}..."

    return prompts


class GMSRARLDataset(torch.utils.data.Dataset):
    """Dataset that yields RL training prompts from LoCoMo episodes."""

    def __init__(self, episodes: list[dict], tokenizer, max_length: int = 1024):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.prompts = []
        for episode in episodes:
            self.prompts.extend(build_rl_prompts_from_episode(episode))
        logger.info(f"Built {len(self.prompts)} RL training prompts from {len(episodes)} episodes")

    def __len__(self):
        return len(self.prompts)

    def __getitem__(self, idx):
        item = self.prompts[idx]
        return {
            "query": item["query"],
            "event": item["event"],
            "question": item["question"],
            "answer": item["answer"],
        }


# ============================================================
# Reward computation
# ============================================================

def compute_rl_reward(
    response: str,
    event: str,
    question: str,
    answer: str,
    agent=None,
) -> float:
    """Compute reward for a Memory Manager decision.

    In Phase 1, we use external QA F1 as the primary reward signal.
    Additionally, we add a format reward to encourage well-formatted outputs.

    Args:
        response: The Memory Manager's decision output.
        event: The event that triggered this decision.
        question: The evaluation question for this episode.
        answer: The ground truth answer.
        agent: Optional GMSRAAgent for memory-augmented QA.

    Returns:
        Reward scalar in [-1.0, 1.0].
    """
    reward = 0.0

    # --- Format reward (encourages correct CRUD format) ---
    response_upper = response.strip().upper()
    if any(response_upper.startswith(op) for op in ["ADD:", "ADD ", "UPDATE ", "DELETE ", "NOOP"]):
        reward += 0.2  # Correct format bonus
    else:
        reward -= 0.3  # Format penalty

    # --- Content quality reward ---
    # Check that ADD/UPDATE actually contains meaningful content
    if response_upper.startswith("ADD") or response_upper.startswith("UPDATE"):
        content = response.split(":", 1)[1].strip() if ":" in response else ""
        if len(content) > 10:
            reward += 0.1  # Non-trivial content
        else:
            reward -= 0.1  # Empty or trivial

    # --- QA F1 reward (if agent is available for evaluation) ---
    if agent is not None and question and answer:
        try:
            # Execute the operation on the agent's memory
            op_result = agent.memory_manager.execute_operation(
                response, event, env_reward=0.5
            )
            # Evaluate QA performance
            predicted = agent.answer_question(question)
            f1 = compute_f1(predicted, answer)
            reward += f1 * 0.7  # Scale F1 contribution
        except Exception:
            pass  # Silently handle failures during RL exploration

    return max(-1.0, min(1.0, reward))


# ============================================================
# Main training loop
# ============================================================

# ============================================================
# GPU preset configurations
# ============================================================

GPU_PRESETS = {
    "a100": {
        "per_device_batch_size": 8,
        "num_generations": 16,
        "max_completion_length": 192,
        "gradient_accumulation_steps": 2,
        "description": "Single A100 80GB — large batches, high VRAM (~65-70GB)",
    },
    "a40": {
        "per_device_batch_size": 4,
        "num_generations": 8,
        "max_completion_length": 192,
        "gradient_accumulation_steps": 4,
        "description": "Multi-GPU A40 48GB — reduced per-device sizes, use with accelerate launch",
    },
}


def _apply_gpu_preset(args):
    """Apply GPU preset defaults for any CLI args not explicitly set."""
    if args.gpu_preset and args.gpu_preset in GPU_PRESETS:
        preset = GPU_PRESETS[args.gpu_preset]
        logger.info(f"Applying GPU preset '{args.gpu_preset}': {preset['description']}")
        if args.per_device_batch_size is None:
            args.per_device_batch_size = preset["per_device_batch_size"]
        if args.num_generations is None:
            args.num_generations = preset["num_generations"]
        if args.max_completion_length is None:
            args.max_completion_length = preset["max_completion_length"]
        if args.gradient_accumulation_steps is None:
            args.gradient_accumulation_steps = preset["gradient_accumulation_steps"]
    return args


def _detect_multi_gpu():
    """Detect if we are running under accelerate multi-GPU."""
    # accelerate sets WORLD_SIZE > 1 for multi-GPU launches
    import os
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    local_rank = int(os.environ.get("LOCAL_RANK", "-1"))
    is_multi = world_size > 1 or local_rank >= 0
    if is_multi:
        logger.info(f"Multi-GPU detected: WORLD_SIZE={world_size}, LOCAL_RANK={local_rank}")
    return is_multi


def main(args):
    set_seed(42)
    logger.info(f"Phase 1: RL + External Reward | model={args.model_name}")

    # --- Apply GPU preset (fills in unset CLI args with preset defaults) ---
    args = _apply_gpu_preset(args)

    config = GMSRAConfig()
    config.rl.num_episodes = args.num_episodes
    config.rl.learning_rate = args.learning_rate
    config.rl.batch_size = args.batch_size
    if args.gradient_accumulation_steps is not None:
        config.rl.gradient_accumulation_steps = args.gradient_accumulation_steps

    # --- Detect multi-GPU (accelerate launch) ---
    is_multi_gpu = _detect_multi_gpu()

    # --- Load model ---
    use_qlora = not args.no_qlora
    model, tokenizer = load_model_and_tokenizer(
        args.model_name, use_qlora=use_qlora, use_accelerate=is_multi_gpu
    )
    logger.info(f"Model precision: {'bf16 (full)' if args.no_qlora else 'QLoRA 4-bit'}")
    if is_multi_gpu:
        logger.info("Model loaded without device_map (accelerate handles placement)")

    if args.checkpoint and os.path.exists(args.checkpoint):
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, args.checkpoint)
        logger.info(f"Loaded Phase 0 checkpoint: {args.checkpoint}")

    # --- Load dataset ---
    dataset = load_locomo_data(args.data_dir)
    logger.info(f"Loaded {len(dataset)} episodes from LoCoMo")

    # --- Initialize G-MSRA Agent for reward computation ---
    from gmsra.agent import GMSRAAgent
    agent = GMSRAAgent(config)
    agent.initialize(model, tokenizer, env_type="dialogue")

    # --- Setup RL training with TRL (version-aware) ---
    caps = _check_trl_capabilities()

    if caps["grpo"]:
        logger.info(">>> Using GRPOTrainer (preferred, most efficient)")
        _train_with_grpo(model, tokenizer, dataset, agent, config, args)
    elif caps["ppo"]:
        logger.info(">>> Using PPOTrainer (fallback)")
        _train_with_ppo(model, tokenizer, dataset, agent, config, args)
    else:
        logger.warning(
            ">>> Using manual REINFORCE (last resort). "
            "This is significantly slower. Please upgrade TRL!"
        )
        _train_with_reinforce(model, tokenizer, dataset, agent, config, args)

    logger.info("Phase 1 complete!")


def _train_with_grpo(model, tokenizer, dataset, agent, config, args):
    """Train using TRL's GRPOTrainer (preferred method)."""
    from trl import GRPOConfig, GRPOTrainer
    from datasets import Dataset as HFDataset
    import inspect

    # Prepare dataset in HF format
    all_prompts = []
    all_meta = []
    for episode in dataset[:config.rl.num_episodes]:
        for prompt_data in build_rl_prompts_from_episode(episode):
            all_prompts.append(prompt_data["query"])
            all_meta.append({
                "event": prompt_data["event"],
                "question": prompt_data["question"],
                "answer": prompt_data["answer"],
            })

    logger.info(f"Prepared {len(all_prompts)} prompts for GRPO training")

    hf_dataset = HFDataset.from_dict({
        "prompt": all_prompts,
    })

    # Define reward function for GRPO
    meta_store = all_meta  # Capture for closure

    def reward_fn(completions, prompts, **kwargs):
        """GRPO reward function: evaluates each completion."""
        rewards = []
        for i, completion in enumerate(completions):
            idx = i % len(meta_store)
            meta = meta_store[idx]
            # Extract the actual text from completion
            if isinstance(completion, list):
                text = completion[0] if completion else ""
            else:
                text = str(completion)

            r = compute_rl_reward(
                response=text,
                event=meta["event"],
                question=meta["question"],
                answer=meta["answer"],
                agent=agent,
            )
            rewards.append(r)
        return rewards

    # Detect whether wandb is configured
    try:
        import wandb
        report_to = "wandb" if wandb.api.api_key else "none"
    except Exception:
        report_to = "none"

    # GRPO config — scale up batch sizes to fill GPU memory
    num_gens = args.num_generations or config.rl.batch_size
    max_comp_len = args.max_completion_length or 256
    per_device_bs = args.per_device_batch_size or config.rl.mini_batch_size
    grad_accum = args.gradient_accumulation_steps or config.rl.gradient_accumulation_steps

    grpo_kwargs = dict(
        output_dir=args.output_dir,
        per_device_train_batch_size=per_device_bs,
        gradient_accumulation_steps=grad_accum,
        learning_rate=config.rl.learning_rate,
        max_grad_norm=config.rl.max_grad_norm,
        num_train_epochs=1,
        logging_steps=10,
        save_steps=500,
        save_total_limit=3,
        max_completion_length=max_comp_len,
        num_generations=num_gens,
        report_to=report_to,
        bf16=True,
    )

    # Optional DeepSpeed config
    if args.deepspeed and os.path.exists(args.deepspeed):
        grpo_kwargs["deepspeed"] = args.deepspeed
        logger.info(f"DeepSpeed config: {args.deepspeed}")

    grpo_config = GRPOConfig(**grpo_kwargs)
    logger.info(
        f"GRPO config: per_device_bs={per_device_bs}, "
        f"num_generations={num_gens}, max_completion_length={max_comp_len}, "
        f"gradient_accumulation_steps={grad_accum}"
    )

    # GRPOTrainer constructor — handle API differences across TRL versions
    # Some versions use `config=`, others use `args=`
    # Some versions use `reward_funcs=`, others use `reward_func=`
    trainer_sig = inspect.signature(GRPOTrainer.__init__)
    trainer_params = list(trainer_sig.parameters.keys())

    trainer_kwargs = {
        "model": model,
        "train_dataset": hf_dataset,
    }

    # config= vs args=
    if "args" in trainer_params:
        trainer_kwargs["args"] = grpo_config
    else:
        trainer_kwargs["config"] = grpo_config

    # processing_class= vs tokenizer=
    if "processing_class" in trainer_params:
        trainer_kwargs["processing_class"] = tokenizer
    elif "tokenizer" in trainer_params:
        trainer_kwargs["tokenizer"] = tokenizer

    # reward_funcs= (list) vs reward_func= (single)
    if "reward_funcs" in trainer_params:
        trainer_kwargs["reward_funcs"] = [reward_fn]
    elif "reward_func" in trainer_params:
        trainer_kwargs["reward_func"] = reward_fn

    logger.info(f"GRPOTrainer constructor args: {list(trainer_kwargs.keys())}")
    trainer = GRPOTrainer(**trainer_kwargs)

    logger.info("Starting GRPO training...")
    trainer.train()

    # Save final model
    trainer.save_model(os.path.join(args.output_dir, "best"))
    tokenizer.save_pretrained(os.path.join(args.output_dir, "best"))
    logger.info(f"GRPO training complete. Model saved to {args.output_dir}/best")


def _train_with_ppo(model, tokenizer, dataset, agent, config, args):
    """Fallback: Train using TRL's PPOTrainer."""
    from trl import PPOConfig, PPOTrainer, AutoModelForCausalLMWithValueHead

    # Wrap model with value head
    ppo_model = AutoModelForCausalLMWithValueHead.from_pretrained(
        model if isinstance(model, str) else model
    )

    # PPOConfig — `model_name` was removed in trl >= 0.12
    # Build config with only supported parameters
    ppo_config_kwargs = {
        "learning_rate": config.rl.learning_rate,
        "batch_size": config.rl.batch_size,
        "mini_batch_size": config.rl.mini_batch_size,
        "gradient_accumulation_steps": config.rl.gradient_accumulation_steps,
        "ppo_epochs": config.rl.ppo_epochs,
    }

    # Only add model_name if PPOConfig supports it (trl < 0.12)
    import inspect
    ppo_config_sig = inspect.signature(PPOConfig.__init__)
    if "model_name" in ppo_config_sig.parameters:
        ppo_config_kwargs["model_name"] = args.model_name

    # Only add log_with if supported
    if "log_with" in ppo_config_sig.parameters:
        try:
            import wandb
            ppo_config_kwargs["log_with"] = "wandb" if wandb.api.api_key else None
        except Exception:
            pass

    ppo_config = PPOConfig(**ppo_config_kwargs)

    # PPOTrainer constructor — handle API differences
    ppo_trainer_sig = inspect.signature(PPOTrainer.__init__)
    ppo_trainer_kwargs = {
        "model": ppo_model,
        "config": ppo_config,
    }
    if "tokenizer" in ppo_trainer_sig.parameters:
        ppo_trainer_kwargs["tokenizer"] = tokenizer
    elif "processing_class" in ppo_trainer_sig.parameters:
        ppo_trainer_kwargs["processing_class"] = tokenizer

    ppo_trainer = PPOTrainer(**ppo_trainer_kwargs)

    # Training loop
    best_reward = -float("inf")
    all_prompts = []
    for episode in dataset[:config.rl.num_episodes]:
        all_prompts.extend(build_rl_prompts_from_episode(episode))

    total_batches = (len(all_prompts) + config.rl.batch_size - 1) // config.rl.batch_size
    logger.info(f"PPO training: {len(all_prompts)} prompts, {total_batches} batches")

    for batch_start in range(0, len(all_prompts), config.rl.batch_size):
        batch = all_prompts[batch_start:batch_start + config.rl.batch_size]
        if not batch:
            continue

        # Tokenize queries
        query_tensors = [
            tokenizer.encode(item["query"], return_tensors="pt",
                             truncation=True, max_length=1024).squeeze(0)
            for item in batch
        ]

        # Generate responses
        response_tensors = []
        for qt in query_tensors:
            response = ppo_trainer.generate(
                qt.unsqueeze(0).to(ppo_model.pretrained_model.device),
                max_new_tokens=256,
                do_sample=True,
                temperature=0.7,
            )
            response_tensors.append(response.squeeze(0)[qt.shape[0]:])

        # Compute rewards
        rewards = []
        for i, (item, rt) in enumerate(zip(batch, response_tensors)):
            response_text = tokenizer.decode(rt, skip_special_tokens=True)
            r = compute_rl_reward(
                response=response_text,
                event=item["event"],
                question=item["question"],
                answer=item["answer"],
                agent=agent,
            )
            rewards.append(torch.tensor(r, dtype=torch.float32))

        # PPO step
        stats = ppo_trainer.step(query_tensors, response_tensors, rewards)

        avg_reward = sum(r.item() for r in rewards) / len(rewards)
        step_num = batch_start // config.rl.batch_size + 1

        if step_num % 10 == 0:
            logger.info(
                f"PPO Step {step_num}/{total_batches} | "
                f"avg_reward={avg_reward:.3f} | "
                f"kl={stats.get('objective/kl', 0):.4f}"
            )

        if avg_reward > best_reward:
            best_reward = avg_reward
            ppo_trainer.save_pretrained(os.path.join(args.output_dir, "best"))

    logger.info(f"PPO training complete. Best reward={best_reward:.4f}")


def _train_with_reinforce(model, tokenizer, dataset, agent, config, args):
    """Manual REINFORCE fallback when TRL trainers are not available.

    This is a last-resort fallback. It uses mini-batched REINFORCE with
    an exponential moving average baseline to avoid zero gradients.

    For best performance, please upgrade TRL: pip install 'trl>=0.15.0'
    """
    from peft import LoraConfig, get_peft_model, TaskType

    # Setup LoRA if not already
    if not hasattr(model, "peft_config"):
        lora_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=16, lora_alpha=32,
            target_modules=["q_proj", "v_proj"],
            lora_dropout=0.05,
        )
        model = get_peft_model(model, lora_config)

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config.rl.learning_rate
    )

    best_avg_reward = -float("inf")
    reward_history = []

    all_prompts = []
    for episode in dataset[:config.rl.num_episodes]:
        all_prompts.extend(build_rl_prompts_from_episode(episode))

    total_steps = len(all_prompts)
    batch_size = config.rl.mini_batch_size  # Use mini-batch instead of serial
    total_batches = (total_steps + batch_size - 1) // batch_size

    logger.info(
        f"REINFORCE training: {total_steps} prompts, "
        f"batch_size={batch_size}, {total_batches} batches"
    )

    # Exponential moving average baseline (avoids zero-gradient trap)
    ema_baseline = 0.0
    ema_alpha = 0.05  # Smoothing factor
    baseline_warmup = 20  # Don't subtract baseline during warmup

    model.train()
    optimizer.zero_grad()

    global_step = 0
    for batch_idx in range(0, total_steps, batch_size):
        batch_items = all_prompts[batch_idx:batch_idx + batch_size]
        batch_loss = 0.0
        batch_rewards = []

        for item in batch_items:
            query = item["query"]

            # Forward pass: generate response
            inputs = tokenizer(
                query, return_tensors="pt", truncation=True, max_length=1024
            )
            inputs = {k: v.to(model.device) for k, v in inputs.items()}

            with torch.no_grad():
                outputs = model.generate(
                    **inputs, max_new_tokens=128,  # Shorter for speed
                    do_sample=True, temperature=0.7,
                    pad_token_id=tokenizer.pad_token_id,
                )

            response_ids = outputs[0][inputs["input_ids"].shape[1]:]
            response_text = tokenizer.decode(response_ids, skip_special_tokens=True)

            # Compute reward
            reward = compute_rl_reward(
                response=response_text,
                event=item["event"],
                question=item["question"],
                answer=item["answer"],
                agent=agent,
            )
            reward_history.append(reward)
            batch_rewards.append(reward)

            # REINFORCE loss: -advantage * log p(response | query)
            full_ids = outputs[0].unsqueeze(0)
            labels = full_ids.clone()
            labels[0, :inputs["input_ids"].shape[1]] = -100  # Mask prompt

            outputs_with_loss = model(full_ids, labels=labels)

            # Advantage = reward - baseline
            advantage = reward - ema_baseline
            # During warmup, use raw reward as advantage
            if global_step < baseline_warmup:
                advantage = reward

            # Prevent zero gradients: clamp minimum absolute advantage
            if abs(advantage) < 0.01:
                advantage = 0.01 if advantage >= 0 else -0.01

            # Accumulate loss (negative because we want to maximize reward)
            sample_loss = -advantage * outputs_with_loss.loss
            batch_loss += sample_loss / len(batch_items)  # Average over batch

            global_step += 1

        # Backward pass on accumulated batch loss
        if isinstance(batch_loss, torch.Tensor):
            batch_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.rl.max_grad_norm)

        batch_num = batch_idx // batch_size + 1

        # Gradient accumulation
        if batch_num % config.rl.gradient_accumulation_steps == 0:
            optimizer.step()
            optimizer.zero_grad()

        # Update EMA baseline with batch average
        batch_avg_reward = sum(batch_rewards) / len(batch_rewards)
        ema_baseline = ema_alpha * batch_avg_reward + (1 - ema_alpha) * ema_baseline

        if batch_num % 20 == 0:
            recent_avg = sum(reward_history[-100:]) / len(reward_history[-100:])
            logger.info(
                f"REINFORCE Batch {batch_num}/{total_batches} | "
                f"batch_reward={batch_avg_reward:.3f} | "
                f"avg_100={recent_avg:.3f} | "
                f"baseline={ema_baseline:.3f}"
            )
            if recent_avg > best_avg_reward:
                best_avg_reward = recent_avg
                os.makedirs(args.output_dir, exist_ok=True)
                model.save_pretrained(os.path.join(args.output_dir, "best"))
                tokenizer.save_pretrained(os.path.join(args.output_dir, "best"))

    # Final optimizer step for any remaining gradients
    optimizer.step()
    optimizer.zero_grad()

    # Save final
    os.makedirs(args.output_dir, exist_ok=True)
    agent.save_checkpoint(os.path.join(args.output_dir, "final"))

    # Save reward history
    with open(os.path.join(args.output_dir, "reward_history.json"), "w") as f:
        json.dump(reward_history, f)

    logger.info(f"REINFORCE complete. Best avg reward={best_avg_reward:.4f}")


# ============================================================
# Data loading
# ============================================================

def load_locomo_data(data_dir: str) -> list[dict]:
    """Load LoCoMo dataset.

    Expected format per episode:
    {
        "events": ["event1", "event2", ...],
        "question": "What is ...?",
        "answer": "The answer is ..."
    }
    """
    # Try to load from local file
    data_path = os.path.join(data_dir, "locomo_train.json")
    if os.path.exists(data_path):
        with open(data_path, "r", encoding="utf-8") as f:
            return json.load(f)

    # Fallback: generate placeholder data for testing
    logger.warning(f"LoCoMo data not found at {data_path}, using placeholder")
    return _generate_placeholder_data()


def _generate_placeholder_data() -> list[dict]:
    """Generate diverse placeholder LoCoMo data for testing."""
    placeholders = [
        {
            "events": [
                "User says: My name is Alice.",
                "User says: I work at a tech company in Silicon Valley.",
                "User says: I'm working on a machine learning project.",
            ],
            "question": "What is the user's name?",
            "answer": "Alice",
        },
        {
            "events": [
                "User says: I moved to Shanghai last month.",
                "User says: I work at Alibaba as a data scientist.",
                "User says: I prefer working from home on Fridays.",
            ],
            "question": "Where does the user live?",
            "answer": "Shanghai",
        },
        {
            "events": [
                "User says: My favorite programming language is Python.",
                "User says: I've been learning Rust recently.",
                "User says: I use VSCode for development.",
            ],
            "question": "What programming language does the user prefer?",
            "answer": "Python",
        },
        {
            "events": [
                "User says: I have two cats named Luna and Star.",
                "User says: I live in a small apartment in downtown.",
                "User says: I enjoy reading sci-fi novels before bed.",
            ],
            "question": "What are the names of the user's cats?",
            "answer": "Luna and Star",
        },
        {
            "events": [
                "User says: I used to live in Beijing.",
                "User says: I recently moved to Shenzhen for a new job.",
                "User says: I work at Tencent now.",
            ],
            "question": "Where does the user currently work?",
            "answer": "Tencent",
        },
        {
            "events": [
                "User says: I'm allergic to peanuts.",
                "User says: I also can't eat shellfish.",
                "User says: I love Italian cuisine though.",
            ],
            "question": "What food allergies does the user have?",
            "answer": "Peanuts and shellfish",
        },
        {
            "events": [
                "User says: My daughter started kindergarten this year.",
                "User says: She's 5 years old.",
                "User says: Her name is Sophie.",
            ],
            "question": "How old is the user's daughter?",
            "answer": "5 years old",
        },
        {
            "events": [
                "User says: I exercise every morning at 6am.",
                "User says: I run 5km and then do yoga.",
                "User says: I've been doing this routine for 2 years.",
            ],
            "question": "What is the user's morning exercise routine?",
            "answer": "Running 5km and yoga at 6am",
        },
    ]
    # Extend to ~50 episodes for testing
    return placeholders * 7


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="G-MSRA Phase 1: RL + External Reward")
    parser.add_argument("--model_name", default="Qwen/Qwen2.5-7B-Instruct")
    parser.add_argument("--checkpoint", default="outputs/phase0/best")
    parser.add_argument("--output_dir", default="outputs/phase1")
    parser.add_argument("--data_dir", default="data")
    parser.add_argument("--num_episodes", type=int, default=5000)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--learning_rate", type=float, default=1.41e-5)
    parser.add_argument("--num_gpus", type=int, default=4)

    # --- GPU preset ---
    parser.add_argument("--gpu_preset", type=str, default=None,
                        choices=list(GPU_PRESETS.keys()),
                        help="GPU preset for quick configuration. "
                             "'a100': single A100 80GB (per_device_bs=8, num_gens=16). "
                             "'a40': multi-GPU A40 48GB (per_device_bs=4, num_gens=8, grad_accum=4). "
                             "Preset values are overridden by explicit CLI args.")

    # --- Performance tuning flags ---
    parser.add_argument("--no_qlora", action="store_true",
                        help="Use bf16 full-precision + LoRA instead of QLoRA 4-bit. "
                             "Uses ~14GB for 7B model (vs ~5GB) but eliminates "
                             "dequantize overhead → faster training.")
    parser.add_argument("--per_device_batch_size", type=int, default=None,
                        help="Override per-device train batch size (default: mini_batch_size from config)")
    parser.add_argument("--num_generations", type=int, default=None,
                        help="Number of generations per prompt for GRPO (default: batch_size). "
                             "Increasing this uses more VRAM but gives better advantage estimates.")
    parser.add_argument("--max_completion_length", type=int, default=None,
                        help="Max tokens per generated completion (default: 256). "
                             "Memory operations are short, so 128-192 is usually enough.")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=None,
                        help="Gradient accumulation steps. Increase to compensate for "
                             "smaller per-device batch sizes on limited-VRAM GPUs.")
    parser.add_argument("--deepspeed", type=str, default=None,
                        help="Path to DeepSpeed config JSON (e.g. cluster/ds_zero2_a40.json). "
                             "Enables ZeRO optimizer partitioning for additional memory savings.")

    args = parser.parse_args()
    main(args)
