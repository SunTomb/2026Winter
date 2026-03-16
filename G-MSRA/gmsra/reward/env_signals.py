"""
Environment Signal Extractors.
Provides R_env — the environment grounding anchor that prevents Reward Hacking.

Supports multiple benchmark types:
- Agent tasks (ALFWorld/WebArena): direct success/failure from environment
- Dialogue tasks (LoCoMo/LongMemEval): user reaction proxy signals
"""

from __future__ import annotations
from abc import ABC, abstractmethod
from loguru import logger


class EnvironmentSignalExtractor(ABC):
    """Base class for environment signal extraction."""

    @abstractmethod
    def extract(self, **kwargs) -> float:
        """Extract R_env from environment feedback.

        Returns:
            float: Environment reward signal in [0, 1].
        """
        raise NotImplementedError


class AgentTaskSignalExtractor(EnvironmentSignalExtractor):
    """Environment signal for Agent tasks (ALFWorld, WebArena).
    Uses direct environment feedback: task success/failure/partial completion.
    """

    def extract(self, task_result: dict) -> float:
        """Extract R_env from Agent task result.

        Args:
            task_result: Dict with keys:
                - 'success': bool — whether the task was completed
                - 'partial_score': float (optional) — partial completion score
                - 'steps_taken': int — number of steps agent took
                - 'max_steps': int — maximum allowed steps

        Returns:
            R_env in [0, 1].
        """
        if task_result.get("success", False):
            # Full success, bonus for efficiency (fewer steps = higher reward)
            max_steps = task_result.get("max_steps", 30)
            steps = task_result.get("steps_taken", max_steps)
            efficiency_bonus = max(0, 1.0 - steps / max_steps) * 0.2
            return min(1.0, 0.8 + efficiency_bonus)

        # Partial completion
        partial = task_result.get("partial_score", 0.0)
        return partial * 0.6  # Scale partial to [0, 0.6]


class DialogueSignalExtractor(EnvironmentSignalExtractor):
    """Environment signal for Dialogue tasks (LoCoMo, LongMemEval).
    Uses user reaction proxy: corrections, follow-ups, satisfaction cues.

    In the absence of explicit labels, we use the NEXT user turn as
    an implicit environment signal.
    """

    def __init__(self, llm_model=None, tokenizer=None):
        self.model = llm_model
        self.tokenizer = tokenizer

    def extract(self, agent_response: str, next_user_turn: str = None,
                qa_ground_truth: str = None) -> float:
        """Extract R_env from dialogue context.

        Priority:
        1. If qa_ground_truth is available (Phase 1): use F1 score
        2. If next_user_turn is available: use user reaction analysis
        3. Fallback: return neutral 0.5

        Args:
            agent_response: The agent's response in this turn.
            next_user_turn: The user's next message (if available).
            qa_ground_truth: Ground truth answer (if available, Phase 1 only).

        Returns:
            R_env in [0, 1].
        """
        # Priority 1: External QA ground truth (Phase 1 training)
        if qa_ground_truth is not None:
            from gmsra.utils import compute_f1
            f1 = compute_f1(agent_response, qa_ground_truth)
            return f1

        # Priority 2: User reaction proxy
        if next_user_turn is not None:
            return self._analyze_user_reaction(agent_response, next_user_turn)

        # Fallback: neutral
        return 0.5

    def _analyze_user_reaction(self, agent_response: str,
                               next_user_turn: str) -> float:
        """Analyze user's next turn to estimate satisfaction.

        Uses simple heuristics + optional LLM analysis:
        - Correction signals: "no", "wrong", "actually", "I meant" → negative
        - Confirmation signals: "thanks", "great", "yes", "exactly" → positive
        - Follow-up question: neutral to slightly positive
        """
        turn_lower = next_user_turn.lower().strip()

        # Rule-based fast path
        negative_cues = ["no,", "wrong", "incorrect", "actually,",
                         "not what i", "i meant", "that's not", "please fix",
                         "你说错了", "不对", "不是"]
        positive_cues = ["thanks", "great", "perfect", "exactly", "correct",
                         "yes", "good", "谢谢", "没错", "对的", "很好"]

        neg_score = sum(1 for cue in negative_cues if cue in turn_lower)
        pos_score = sum(1 for cue in positive_cues if cue in turn_lower)

        if neg_score > pos_score:
            return max(0.0, 0.3 - neg_score * 0.1)
        elif pos_score > neg_score:
            return min(1.0, 0.7 + pos_score * 0.1)

        # If LLM is available, use it for ambiguous cases
        if self.model is not None and self.tokenizer is not None:
            return self._llm_analyze_reaction(agent_response, next_user_turn)

        return 0.5  # Neutral

    def _llm_analyze_reaction(self, agent_response: str,
                              next_user_turn: str) -> float:
        """Use LLM to analyze user reaction (for ambiguous cases)."""
        from gmsra.utils import generate_text

        prompt = (
            "Given the assistant's response and the user's next message, "
            "rate the user's satisfaction from 0.0 (very unsatisfied) to 1.0 "
            "(very satisfied). Output only a number.\n\n"
            f"Assistant: {agent_response[:200]}\n"
            f"User next: {next_user_turn[:200]}\n"
            "Satisfaction score:"
        )
        try:
            result = generate_text(self.model, self.tokenizer, prompt,
                                   max_new_tokens=8, temperature=0.1)
            score = float(result.strip())
            return max(0.0, min(1.0, score))
        except (ValueError, RuntimeError):
            return 0.5


class ExternalQASignalExtractor(EnvironmentSignalExtractor):
    """Direct QA-based signal (for Phase 1 with external labels)."""

    def extract(self, prediction: str, ground_truth: str) -> float:
        from gmsra.utils import compute_f1
        return compute_f1(prediction, ground_truth)
