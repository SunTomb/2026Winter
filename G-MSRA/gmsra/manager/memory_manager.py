"""
RL-Based Memory Manager.
Based on Memory-R1 architecture (Chen et al., 2025) with key modification:
reward signal replaced from external QA F1 to G-MSRA's grounded composite reward.

Action Space: ADD / UPDATE <id> / DELETE <id> / NOOP
Training: PPO or GRPO via TRL library.
"""

from __future__ import annotations
from typing import Optional, Literal

from loguru import logger

from gmsra.config import RLConfig, MemoryConfig
from gmsra.memory.store import MemoryStore
from gmsra.memory.entry import MemoryEntry


MemoryOp = Literal["ADD", "UPDATE", "DELETE", "NOOP"]


class MemoryManager:
    """RL-based Memory Manager Agent.

    Decides how to manage memories in response to new events.
    The RL policy is trained to optimize the grounded composite reward.

    Architecture mirrors Memory-R1's Memory Manager but replaces
    the reward source from QA F1 to G-MSRA's R_total.
    """

    def __init__(
        self,
        model=None,
        tokenizer=None,
        memory_store: Optional[MemoryStore] = None,
        rl_config: Optional[RLConfig] = None,
        memory_config: Optional[MemoryConfig] = None,
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.store = memory_store or MemoryStore(memory_config)
        self.rl_config = rl_config or RLConfig()
        self.operation_history: list[dict] = []

    # ---- Core Decision Making ----

    def decide(self, new_event: str, task_context: str = "") -> tuple[str, str]:
        """Given a new event, decide the memory operation.

        Args:
            new_event: New information/experience to process.
            task_context: Current task or conversation context.

        Returns:
            (operation_str, formatted_prompt) — e.g., ("ADD new fact...", prompt)
            operation_str is the raw model output describing the operation.
        """
        # Retrieve relevant existing memories for context
        relevant = self.store.retrieve(new_event, topk=self.store.config.retrieval_topk)
        relevant_entries = [entry for entry, score in relevant]

        prompt = self._build_manager_prompt(new_event, relevant_entries, task_context)

        from gmsra.utils import generate_text
        output = generate_text(
            self.model, self.tokenizer, prompt,
            max_new_tokens=256, temperature=0.7
        )

        return output.strip(), prompt

    def execute_operation(self, operation_str: str,
                          new_event: str,
                          env_reward: float = 0.0) -> dict:
        """Parse and execute the model's memory operation decision.

        Args:
            operation_str: Model output, e.g., "ADD: User prefers tea over coffee"
            new_event: The original event that triggered this operation.
            env_reward: Environment reward at this timestep.

        Returns:
            Dict with operation details for logging.
        """
        op, target_id, content = self._parse_operation(operation_str, new_event)

        result = {"op": op, "target_id": target_id, "content": content,
                  "success": False}

        if op == "ADD":
            entry = self.store.add(
                content=content, env_reward=env_reward,
                source=new_event[:100]
            )
            result["success"] = True
            result["entry_id"] = entry.id

        elif op == "UPDATE" and target_id:
            entry = self.store.update(target_id, content, env_reward)
            result["success"] = entry is not None
            result["entry_id"] = target_id

        elif op == "DELETE" and target_id:
            result["success"] = self.store.delete(target_id)
            result["entry_id"] = target_id

        elif op == "NOOP":
            result["success"] = True

        self.operation_history.append(result)
        logger.debug(f"Executed: {op} → success={result['success']}")
        return result

    # ---- Prompt Construction ----

    def _build_manager_prompt(self, new_event: str,
                              history: list[MemoryEntry],
                              task_context: str = "") -> str:
        """Build the Memory Manager's decision prompt."""
        memory_str = "\n".join([
            f"  [{e.id}] (conf={e.confidence:.2f}) {e.content}"
            for e in history
        ]) if history else "(empty memory)"

        prompt = (
            "You are a Memory Manager for an AI agent. "
            "Given the current memory entries and a new event, "
            "decide the best memory operation.\n\n"
            "### Available Operations\n"
            "- ADD: <content> — Store new important information\n"
            "- UPDATE <id>: <new_content> — Update existing memory\n"
            "- DELETE <id> — Remove outdated/wrong memory\n"
            "- NOOP — No action needed\n\n"
        )

        if task_context:
            prompt += f"### Current Task Context\n{task_context[:300]}\n\n"

        prompt += (
            f"### Current Memory Entries\n{memory_str}\n\n"
            f"### New Event\n{new_event}\n\n"
            "### Decision\n"
            "Think about whether this event contains new information worth "
            "storing, updates existing knowledge, or contradicts stored facts. "
            "Output your operation:\n"
        )
        return prompt

    def _parse_operation(self, output: str,
                         fallback_content: str) -> tuple[MemoryOp, str, str]:
        """Parse model output into (op, target_id, content).

        Handles formats:
        - "ADD: some content"
        - "UPDATE abc123: new content"
        - "DELETE abc123"
        - "NOOP" or anything else
        """
        output = output.strip()
        upper = output.upper()

        if upper.startswith("ADD"):
            content = output.split(":", 1)[1].strip() if ":" in output else fallback_content
            return "ADD", "", content

        elif upper.startswith("UPDATE"):
            parts = output.split(":", 1)
            if len(parts) == 2:
                # "UPDATE abc123: new content"
                id_part = parts[0].replace("UPDATE", "").strip()
                content = parts[1].strip()
                return "UPDATE", id_part, content
            return "NOOP", "", ""

        elif upper.startswith("DELETE"):
            target_id = output.replace("DELETE", "").strip().rstrip(".")
            return "DELETE", target_id, ""

        else:
            return "NOOP", "", ""

    # ---- SFT Data Generation ----

    @staticmethod
    def generate_sft_examples(
        events: list[str],
        operations: list[str],
    ) -> list[dict]:
        """Generate SFT training examples for Phase 0 warmup.

        Args:
            events: List of event texts.
            operations: Corresponding correct operations.

        Returns:
            List of {"prompt": ..., "completion": ...} dicts.
        """
        examples = []
        for event, op in zip(events, operations):
            prompt = (
                "You are a Memory Manager. Given the new event, "
                "decide the operation.\n\n"
                f"New Event: {event}\n\n"
                "Decision:\n"
            )
            examples.append({"prompt": prompt, "completion": op})
        return examples

    # ---- Statistics ----

    def get_operation_stats(self) -> dict:
        """Get counts of each operation type."""
        stats = {"ADD": 0, "UPDATE": 0, "DELETE": 0, "NOOP": 0, "total": 0}
        for record in self.operation_history:
            op = record.get("op", "NOOP")
            stats[op] = stats.get(op, 0) + 1
            stats["total"] += 1
        return stats
