"""
Phase 1: RL Training with External Reward.
Trains Memory Manager RL policy using external QA F1 as reward signal.
Establishes performance upper bound for comparison with Phase 2-3.

Uses TRL's GRPOTrainer (Group Relative Policy Optimization) for RL training.
GRPO is simpler and more stable than PPO for language model RL,
and doesn't require a separate value head or critic model.

Usage:
    accelerate launch --num_processes 4 scripts/train_phase1_rl.py \
        --model_name Qwen/Qwen2.5-7B-Instruct --num_episodes 5000
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
# Dataset for RL training
# ============================================================

def build_rl_prompts_from_episode(episode: dict) -> list[dict]:
    """Convert a LoCoMo episode into RL training prompts.

    For each event in the episode, we create a prompt that asks the
    Memory Manager to decide the operation. The reward comes from
    the QA F1 at the end of the episode.

    Returns:
        List of prompt dicts with keys: "query", "events", "question", "answer"
    """
    events = episode.get("events", [])
    question = episode.get("question", "")
    answer = episode.get("answer", "")

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

def main(args):
    set_seed(42)
    logger.info(f"Phase 1: RL + External Reward | model={args.model_name}")

    config = GMSRAConfig()
    config.rl.num_episodes = args.num_episodes
    config.rl.learning_rate = args.learning_rate
    config.rl.batch_size = args.batch_size

    # --- Load model ---
    model, tokenizer = load_model_and_tokenizer(args.model_name)

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

    # --- Setup RL training with TRL ---
    try:
        from trl import GRPOConfig, GRPOTrainer
        use_grpo = True
        logger.info("Using TRL GRPOTrainer for RL training")
    except ImportError:
        try:
            from trl import PPOConfig, PPOTrainer, AutoModelForCausalLMWithValueHead
            use_grpo = False
            logger.info("GRPOTrainer not available, falling back to PPOTrainer")
        except ImportError:
            logger.warning("TRL RL trainers not available, using manual REINFORCE")
            use_grpo = None

    if use_grpo is True:
        _train_with_grpo(model, tokenizer, dataset, agent, config, args)
    elif use_grpo is False:
        _train_with_ppo(model, tokenizer, dataset, agent, config, args)
    else:
        _train_with_reinforce(model, tokenizer, dataset, agent, config, args)

    logger.info("Phase 1 complete!")


def _train_with_grpo(model, tokenizer, dataset, agent, config, args):
    """Train using TRL's GRPOTrainer (preferred method)."""
    from trl import GRPOConfig, GRPOTrainer
    from datasets import Dataset as HFDataset

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

    # GRPO config
    grpo_config = GRPOConfig(
        output_dir=args.output_dir,
        per_device_train_batch_size=config.rl.mini_batch_size,
        gradient_accumulation_steps=config.rl.gradient_accumulation_steps,
        learning_rate=config.rl.learning_rate,
        max_grad_norm=config.rl.max_grad_norm,
        num_train_epochs=1,
        logging_steps=10,
        save_steps=500,
        save_total_limit=3,
        max_completion_length=256,
        num_generations=config.rl.batch_size,
        report_to="wandb",
        bf16=True,
    )

    trainer = GRPOTrainer(
        model=model,
        config=grpo_config,
        train_dataset=hf_dataset,
        processing_class=tokenizer,
        reward_funcs=reward_fn,
    )

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

    ppo_config = PPOConfig(
        model_name=args.model_name,
        learning_rate=config.rl.learning_rate,
        batch_size=config.rl.batch_size,
        mini_batch_size=config.rl.mini_batch_size,
        gradient_accumulation_steps=config.rl.gradient_accumulation_steps,
        ppo_epochs=config.rl.ppo_epochs,
        log_with="wandb",
    )

    ppo_trainer = PPOTrainer(
        model=ppo_model,
        config=ppo_config,
        tokenizer=tokenizer,
    )

    # Training loop
    best_reward = -float("inf")
    all_prompts = []
    for episode in dataset[:config.rl.num_episodes]:
        all_prompts.extend(build_rl_prompts_from_episode(episode))

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
                f"PPO Step {step_num} | "
                f"avg_reward={avg_reward:.3f} | "
                f"kl={stats.get('objective/kl', 0):.4f}"
            )

        if avg_reward > best_reward:
            best_reward = avg_reward
            ppo_trainer.save_pretrained(os.path.join(args.output_dir, "best"))

    logger.info(f"PPO training complete. Best reward={best_reward:.4f}")


def _train_with_reinforce(model, tokenizer, dataset, agent, config, args):
    """Manual REINFORCE fallback when TRL is not available."""
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

    model.train()
    for step_idx, item in enumerate(all_prompts):
        query = item["query"]

        # Forward pass: generate response
        inputs = tokenizer(
            query, return_tensors="pt", truncation=True, max_length=1024
        )
        inputs = {k: v.to(model.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = model.generate(
                **inputs, max_new_tokens=256,
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

        # REINFORCE: -R * log p(response | query)
        full_ids = outputs[0].unsqueeze(0)
        labels = full_ids.clone()
        labels[0, :inputs["input_ids"].shape[1]] = -100  # Mask prompt

        outputs_with_loss = model(full_ids, labels=labels)
        loss = -reward * outputs_with_loss.loss  # REINFORCE gradient

        # Baseline subtraction (running mean)
        baseline = sum(reward_history[-50:]) / max(len(reward_history[-50:]), 1)
        loss = -(reward - baseline) * outputs_with_loss.loss

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), config.rl.max_grad_norm)

        if (step_idx + 1) % config.rl.gradient_accumulation_steps == 0:
            optimizer.step()
            optimizer.zero_grad()

        if (step_idx + 1) % 50 == 0:
            recent_avg = sum(reward_history[-50:]) / len(reward_history[-50:])
            logger.info(
                f"REINFORCE Step {step_idx+1}/{len(all_prompts)} | "
                f"reward={reward:.3f} | avg_50={recent_avg:.3f}"
            )
            if recent_avg > best_avg_reward:
                best_avg_reward = recent_avg
                os.makedirs(args.output_dir, exist_ok=True)
                model.save_pretrained(os.path.join(args.output_dir, "best"))
                tokenizer.save_pretrained(os.path.join(args.output_dir, "best"))

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
    args = parser.parse_args()
    main(args)
