"""
Abstract Base Agent for Baseline Comparisons.
All baseline agents implement this interface so they can be evaluated
by the unified evaluation harness (`eval_baselines.py`).
"""

import os
import json
from abc import ABC, abstractmethod
from typing import Optional

from loguru import logger


class BaseAgent(ABC):
    """Abstract base class for all baseline agents.

    Every baseline must implement:
      - process_event(): handle a new event (dialogue turn / environment obs)
      - answer_question(): produce a prediction for a QA query
      - reset(): clear state between episodes

    Optionally:
      - train_step(): perform one training step (for RL-based baselines)
      - save / load: checkpoint management
    """

    name: str = "base"

    def __init__(self, model_name: str = "Qwen/Qwen2.5-7B-Instruct",
                 max_memories: int = 200, **kwargs):
        self.model_name = model_name
        self.max_memories = max_memories
        self.model = None
        self.tokenizer = None
        self._initialized = False

        # Counters
        self.total_events_processed = 0
        self.total_tokens_used = 0

    def initialize(self, model=None, tokenizer=None):
        """Initialize model and tokenizer.

        If model/tokenizer are provided, use them directly (allows sharing
        across baselines for fair comparison). Otherwise, load from model_name.
        """
        if model is not None and tokenizer is not None:
            self.model = model
            self.tokenizer = tokenizer
        else:
            from gmsra.utils import load_model_and_tokenizer
            self.model, self.tokenizer = load_model_and_tokenizer(self.model_name)

        self._initialized = True
        logger.info(f"[{self.name}] Initialized with model: {self.model_name}")

    @abstractmethod
    def process_event(self, event: str, context: str = "") -> dict:
        """Process an incoming event (dialogue turn, observation, etc.).

        Args:
            event: The new event text.
            context: Optional task context or question.

        Returns:
            dict with at least:
              - "operation": str (what memory operation was performed)
              - "details": str (operation details)
        """
        ...

    @abstractmethod
    def answer_question(self, question: str) -> str:
        """Answer a question using the agent's current memory state.

        Args:
            question: The question to answer.

        Returns:
            Predicted answer string.
        """
        ...

    @abstractmethod
    def reset(self):
        """Reset agent state for a new episode.

        Subclasses should clear memory stores, episodic buffers, etc.
        but keep model weights intact.
        """
        ...

    def train_step(self, reward: float, **kwargs) -> dict:
        """Perform one training step (for RL-based baselines).

        Default: no-op. Override in RL-based baselines.

        Returns:
            dict with training stats (loss, grad_norm, etc.)
        """
        return {"trained": False}

    def get_memory_contents(self) -> list[str]:
        """Return current memory contents as list of strings.

        Used for diagnostics and Judge evaluation.
        """
        return []

    def get_stats(self) -> dict:
        """Return agent statistics for logging."""
        return {
            "name": self.name,
            "total_events": self.total_events_processed,
            "total_tokens": self.total_tokens_used,
            "memory_size": len(self.get_memory_contents()),
        }

    def save(self, path: str):
        """Save agent state to disk."""
        os.makedirs(path, exist_ok=True)
        stats = self.get_stats()
        with open(os.path.join(path, "agent_stats.json"), "w") as f:
            json.dump(stats, f, indent=2)

    def load(self, path: str):
        """Load agent state from disk."""
        pass

    def _generate(self, prompt: str, max_new_tokens: int = 256,
                  temperature: float = 0.7) -> str:
        """Generate text using the loaded model.

        Shared utility for all baselines.
        """
        if not self._initialized:
            raise RuntimeError(f"[{self.name}] Must call initialize() first")

        from gmsra.utils import generate_text
        result = generate_text(
            self.model, self.tokenizer, prompt,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
        )
        # Track token usage (approximate)
        self.total_tokens_used += len(self.tokenizer.encode(prompt)) + max_new_tokens
        return result

    def _compute_f1(self, prediction: str, ground_truth: str) -> float:
        """Compute token-level F1 between prediction and ground truth."""
        from gmsra.utils import compute_f1
        return compute_f1(prediction, ground_truth)
