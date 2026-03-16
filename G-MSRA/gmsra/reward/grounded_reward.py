"""
Grounded Self-Reward Generator.
Core innovation of G-MSRA: dual-layer composite reward function.

R_total = R_env (anchor) + λ · R_mem (refiner)

- R_env: Environment grounding signal (prevents reward hacking)
- R_mem: Memory-aware consistency signal (fine-grained guidance)
- Memory confidence filtering prevents noise pollution of Judge
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Optional

from loguru import logger

from gmsra.config import RewardConfig
from gmsra.memory.store import MemoryStore
from gmsra.reward.env_signals import EnvironmentSignalExtractor


@dataclass
class RewardResult:
    """Container for reward computation results."""
    r_env: float           # Environment anchor reward
    r_mem: float           # Memory consistency reward
    r_total: float         # Composite reward
    judge_rationale: str   # Judge's textual reasoning (for logging)
    memories_used: int     # Number of memories consulted by Judge


class GroundedRewardGenerator:
    """Dual-layer composite reward generator.

    Architecture:
        1. R_env from EnvironmentSignalExtractor (external world feedback)
        2. R_mem from LLM-as-Judge (memory-aware self-assessment)
        3. R_total = R_env + λ · R_mem

    Key design: R_mem ONLY refines R_env, never overrides it (λ < 1).
    This prevents the "self-consistent but wrong" reward hacking failure mode.
    """

    def __init__(
        self,
        config: RewardConfig,
        memory_store: MemoryStore,
        judge_model=None,
        judge_tokenizer=None,
        env_extractor: Optional[EnvironmentSignalExtractor] = None,
    ):
        self.config = config
        self.memory_store = memory_store
        self.judge_model = judge_model
        self.judge_tokenizer = judge_tokenizer
        self.env_extractor = env_extractor

        # Tracking for reward drift monitoring
        self.reward_history: list[RewardResult] = []

    def compute_reward(
        self,
        agent_response: str,
        task_context: str,
        memory_operation: str,       # "ADD xxx" / "UPDATE idx xxx" / etc.
        env_signal_kwargs: dict,     # Passed to env_extractor.extract()
    ) -> RewardResult:
        """Compute the composite grounded reward.

        Args:
            agent_response: The agent's response/action output.
            task_context: The task/question/current context.
            memory_operation: The memory operation performed (for judging).
            env_signal_kwargs: Keyword args for environment signal extraction.

        Returns:
            RewardResult with r_env, r_mem, r_total, and judge rationale.
        """
        # --- Layer 1: Environment Anchor ---
        r_env = 0.5  # Default neutral
        if self.env_extractor is not None:
            r_env = self.env_extractor.extract(**env_signal_kwargs)

        # --- Layer 2: Memory-Aware Self-Assessment ---
        r_mem = 0.5  # Default neutral
        judge_rationale = ""
        memories_used = 0

        if self.judge_model is not None and self.judge_tokenizer is not None:
            r_mem, judge_rationale, memories_used = self._compute_memory_reward(
                agent_response, task_context, memory_operation, r_env
            )

        # --- Composite Reward ---
        r_total = r_env + self.config.lambda_mem * r_mem
        r_total *= self.config.reward_scale

        result = RewardResult(
            r_env=r_env,
            r_mem=r_mem,
            r_total=r_total,
            judge_rationale=judge_rationale,
            memories_used=memories_used,
        )
        self.reward_history.append(result)

        logger.debug(
            f"Reward: R_env={r_env:.3f}, R_mem={r_mem:.3f}, "
            f"R_total={r_total:.3f} (memories={memories_used})"
        )
        return result

    def _compute_memory_reward(
        self,
        agent_response: str,
        task_context: str,
        memory_operation: str,
        r_env: float,
    ) -> tuple[float, str, int]:
        """Compute R_mem using LLM-as-Judge with confidence-filtered memories.

        The Judge sees:
        (a) Current response
        (b) Confident historical memories
        (c) Environment feedback result
        (d) Memory operation performed

        Returns: (r_mem, rationale, num_memories_used)
        """
        from gmsra.utils import generate_text

        # Retrieve confidence-filtered memories
        relevant_memories = self.memory_store.retrieve_confident(
            query=task_context,
            topk=self.memory_store.config.confidence_topk,
        )
        memories_text = "\n".join([
            f"  [{entry.id}] (conf={entry.confidence:.2f}) {entry.content}"
            for entry, score in relevant_memories
        ])
        num_memories = len(relevant_memories)

        # Build Judge prompt
        prompt = self._build_judge_prompt(
            agent_response=agent_response,
            task_context=task_context,
            memory_operation=memory_operation,
            memories_text=memories_text,
            r_env=r_env,
        )

        # Generate Judge assessment
        try:
            output = generate_text(
                self.judge_model, self.judge_tokenizer, prompt,
                max_new_tokens=self.config.judge_max_tokens,
                temperature=0.3, do_sample=True,
            )
            r_mem, rationale = self._parse_judge_output(output)
        except Exception as e:
            logger.warning(f"Judge failed: {e}, returning neutral R_mem=0.5")
            r_mem, rationale = 0.5, "Judge failed"

        return r_mem, rationale, num_memories

    def _build_judge_prompt(
        self,
        agent_response: str,
        task_context: str,
        memory_operation: str,
        memories_text: str,
        r_env: float,
    ) -> str:
        """Build the Judge prompt with full context."""
        return (
            "You are an impartial judge evaluating the quality of a memory "
            "management decision and the resulting response.\n\n"
            f"### Task Context\n{task_context[:500]}\n\n"
            f"### Agent Response\n{agent_response[:500]}\n\n"
            f"### Memory Operation Performed\n{memory_operation}\n\n"
            f"### Relevant Historical Memories (confidence-filtered)\n"
            f"{memories_text if memories_text else '(no relevant memories)'}\n\n"
            f"### Environment Feedback Score\n{r_env:.2f}\n\n"
            "### Your Task\n"
            "Evaluate whether the memory operation was appropriate given the "
            "historical context. Consider:\n"
            "1. Is the memory operation consistent with what we know from history?\n"
            "2. Does the response align with accumulated knowledge?\n"
            "3. Was the decision to ADD/UPDATE/DELETE/NOOP well-justified?\n\n"
            "Output format:\n"
            "Score: [0.0-1.0]\n"
            "Rationale: [brief explanation]\n"
        )

    def _parse_judge_output(self, output: str) -> tuple[float, str]:
        """Parse the Judge's output to extract score and rationale."""
        lines = output.strip().split("\n")
        score = 0.5
        rationale = output.strip()

        for line in lines:
            line = line.strip()
            if line.lower().startswith("score:"):
                try:
                    score_str = line.split(":", 1)[1].strip()
                    # Handle "0.8/1.0" or "0.8" formats
                    score_str = score_str.split("/")[0].strip()
                    score = float(score_str)
                    score = max(0.0, min(1.0, score))
                except ValueError:
                    pass
            elif line.lower().startswith("rationale:"):
                rationale = line.split(":", 1)[1].strip()

        return score, rationale

    # ---- Reward Annealing (Phase 2) ----

    def compute_annealed_reward(
        self,
        r_external: float,
        agent_response: str,
        task_context: str,
        memory_operation: str,
        env_signal_kwargs: dict,
        alpha: float,
    ) -> RewardResult:
        """Compute annealed reward during Phase 2 transition.

        R_phase2 = α · R_ext + (1 - α) · R_total

        Args:
            r_external: External QA F1 reward (from Phase 1).
            alpha: Annealing coefficient (1.0 → 0.0 over transition).
            ...: Same as compute_reward.

        Returns:
            RewardResult with annealed r_total.
        """
        self_result = self.compute_reward(
            agent_response, task_context, memory_operation, env_signal_kwargs
        )
        annealed_total = alpha * r_external + (1 - alpha) * self_result.r_total
        return RewardResult(
            r_env=self_result.r_env,
            r_mem=self_result.r_mem,
            r_total=annealed_total,
            judge_rationale=self_result.judge_rationale,
            memories_used=self_result.memories_used,
        )

    # ---- Monitoring ----

    def get_reward_variance(self, window: int = 20) -> float:
        """Compute reward variance over recent window (for trigger)."""
        if len(self.reward_history) < 2:
            return 0.0
        recent = [r.r_total for r in self.reward_history[-window:]]
        import numpy as np
        return float(np.var(recent))

    def get_reward_drift(self) -> list[float]:
        """Return R_total history for drift visualization."""
        return [r.r_total for r in self.reward_history]

    def get_calibration_data(self) -> tuple[list[float], list[float]]:
        """Return (r_env, r_mem) pairs for Kendall τ calibration analysis."""
        envs = [r.r_env for r in self.reward_history]
        mems = [r.r_mem for r in self.reward_history]
        return envs, mems
