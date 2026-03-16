"""
Adaptive Consolidation Trigger.
3D signal function deciding WHEN to trigger parametric consolidation.

Trigger(t) = 𝟙[α·Conflict(t) + β·Var(R) + γ·Growth(t) > θ]

Three dimensions:
1. Memory Conflict Index — semantic contradictions between memories
2. Reward Variance — instability in self-reward scores
3. Memory Growth Rate — external memory size pressure
"""

from __future__ import annotations
from loguru import logger

from gmsra.config import TriggerConfig
from gmsra.memory.store import MemoryStore
from gmsra.reward.grounded_reward import GroundedRewardGenerator


class ConsolidationTrigger:
    """Adaptive 3D consolidation trigger.

    Replaces heuristic "every N episodes" thresholds (as in Self-Consolidation)
    with a principled multi-signal trigger function.
    """

    def __init__(
        self,
        config: TriggerConfig,
        memory_store: MemoryStore,
        reward_generator: GroundedRewardGenerator,
    ):
        self.config = config
        self.store = memory_store
        self.reward_gen = reward_generator
        self.last_trigger_step: int = 0
        self.trigger_count: int = 0
        self.trigger_history: list[dict] = []

    def should_trigger(self, current_step: int) -> bool:
        """Check if consolidation should be triggered at current step.

        Returns:
            True if all conditions met for consolidation.
        """
        # Enforce minimum interval between consolidations
        if current_step - self.last_trigger_step < self.config.min_interval:
            return False

        # Need enough memory to consolidate
        if self.store.size() < 10:
            return False

        # Compute 3D signal
        conflict = self._compute_conflict_index()
        variance = self._compute_reward_variance()
        growth = self._compute_growth_rate()

        score = (
            self.config.alpha * conflict +
            self.config.beta * variance +
            self.config.gamma * growth
        )

        triggered = score > self.config.theta

        self.trigger_history.append({
            "step": current_step,
            "conflict": conflict,
            "variance": variance,
            "growth": growth,
            "score": score,
            "triggered": triggered,
        })

        if triggered:
            self.last_trigger_step = current_step
            self.trigger_count += 1
            logger.info(
                f"CONSOLIDATION TRIGGERED at step {current_step} "
                f"(score={score:.3f} > θ={self.config.theta})"
                f" | conflict={conflict:.3f}, var={variance:.3f}, growth={growth:.3f}"
            )

        return triggered

    def _compute_conflict_index(self) -> float:
        """Compute Memory Conflict Index.

        Measures semantic contradictions between related memories.
        High conflict → knowledge is inconsistent → needs consolidation.
        """
        if self.store.size() < 2:
            return 0.0

        conflicts = 0
        total_pairs = 0

        for entry in list(self.store.entries.values())[:50]:  # Sample up to 50
            for linked_id in entry.links:
                if linked_id in self.store.entries:
                    linked = self.store.entries[linked_id]
                    total_pairs += 1

                    # Simple heuristic: if linked memories have very different
                    # confidence scores, they may be contradictory
                    conf_diff = abs(entry.confidence - linked.confidence)
                    if conf_diff > 0.3:
                        conflicts += 1

                    # Semantic contradiction detection via embedding similarity
                    if entry.embedding and linked.embedding:
                        import numpy as np
                        sim = np.dot(entry.embedding, linked.embedding)
                        # Linked but highly similar → consistent
                        # Linked but dissimilar → potentially contradictory
                        if sim < 0.3:  # Low similarity despite being linked
                            conflicts += 1

        if total_pairs == 0:
            return 0.0
        return min(1.0, conflicts / max(total_pairs, 1))

    def _compute_reward_variance(self) -> float:
        """Compute reward variance over recent window.
        High variance → model's judgment is unstable → needs consolidation.
        Normalized to [0, 1].
        """
        variance = self.reward_gen.get_reward_variance(
            window=self.config.window_size
        )
        # Normalize: typical variance range is [0, 0.25] (for rewards in [0,1])
        return min(1.0, variance / 0.25)

    def _compute_growth_rate(self) -> float:
        """Compute memory growth rate pressure.
        High growth → external memory expanding too fast → needs parametric offloading.
        Normalized to [0, 1].
        """
        rate = self.store.get_growth_rate(window_entries=50)
        # Normalize: >5 entries per hour is considered high pressure
        return min(1.0, rate / 5.0)

    def get_diagnostics(self) -> dict:
        """Get trigger diagnostics for visualization."""
        return {
            "total_triggers": self.trigger_count,
            "history": self.trigger_history,
            "current_memory_size": self.store.size(),
        }
