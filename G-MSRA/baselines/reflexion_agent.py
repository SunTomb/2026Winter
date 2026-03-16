"""
Baseline 1: Reflexion Agent.
Verbal reinforcement learning via self-reflection.

Reference: Shinn et al., "Reflexion: Language Agents with Verbal
Reinforcement Learning", NeurIPS 2023.
Source: https://github.com/noahshinn/reflexion

Core idea:
  - On failure, the agent generates a textual "reflection"
  - Reflections are stored in an episodic buffer (up to N entries)
  - Reflections are prepended to subsequent prompts as guidance
  - NO parameter updates — purely prompt-based learning

This reproduction adapts Reflexion to the memory management task:
  - Actor generates CRUD memory operations
  - Evaluator uses environment feedback (R_env)
  - Self-Reflector generates verbal reflection on failures
  - Episodic buffer stores reflections (NOT structured memories)
"""

from baselines.base_agent import BaseAgent
from loguru import logger


class ReflexionAgent(BaseAgent):
    """Reflexion: verbal RL via self-reflection prompts."""

    name = "reflexion"

    def __init__(self, max_reflections: int = 10, reflection_threshold: float = 0.3,
                 **kwargs):
        super().__init__(**kwargs)
        self.max_reflections = max_reflections
        self.reflection_threshold = reflection_threshold
        self.reflections: list[str] = []  # Episodic reflection buffer
        self.memories: list[str] = []      # Simple text memory store
        self.last_event = ""
        self.last_operation = ""
        self.trial_count = 0

    def reset(self):
        """Reset per-episode state. Keep reflections across episodes
        (that's the whole point of Reflexion)."""
        self.memories = []
        self.last_event = ""
        self.last_operation = ""
        self.trial_count += 1

    def process_event(self, event: str, context: str = "") -> dict:
        """Process event: decide memory operation, then possibly reflect."""
        self.total_events_processed += 1
        self.last_event = event

        # Build prompt with reflections
        prompt = self._build_actor_prompt(event, context)

        # Actor: generate memory operation
        response = self._generate(prompt, max_new_tokens=200, temperature=0.3)
        operation = self._parse_operation(response)

        # Execute operation on simple memory store
        self._execute_operation(operation, event)
        self.last_operation = operation["type"]

        return {
            "operation": operation["type"],
            "details": operation.get("content", ""),
        }

    def reflect_on_failure(self, event: str, context: str, reward: float):
        """Generate reflection after receiving low reward.

        This is the core Reflexion mechanism: convert scalar failure
        signals into verbal learning.
        """
        if reward >= self.reflection_threshold:
            return  # No need to reflect on success

        reflection_prompt = self._build_reflection_prompt(event, context, reward)
        reflection = self._generate(reflection_prompt, max_new_tokens=200, temperature=0.5)
        reflection = reflection.strip()

        if reflection:
            self.reflections.append(reflection)
            # Trim to max
            if len(self.reflections) > self.max_reflections:
                self.reflections = self.reflections[-self.max_reflections:]
            logger.debug(f"[Reflexion] New reflection: {reflection[:80]}...")

    def answer_question(self, question: str) -> str:
        """Answer question using memories + reflections."""
        memory_context = "\n".join(f"- {m}" for m in self.memories[-20:])
        reflection_context = "\n".join(f"- {r}" for r in self.reflections[-5:])

        prompt = (
            "You are a helpful assistant with memory.\n\n"
        )
        if memory_context:
            prompt += f"### Your Memories:\n{memory_context}\n\n"
        if reflection_context:
            prompt += f"### Lessons Learned:\n{reflection_context}\n\n"
        prompt += f"### Question:\n{question}\n\n### Answer (be concise):\n"

        return self._generate(prompt, max_new_tokens=100, temperature=0.1).strip()

    def get_memory_contents(self) -> list[str]:
        return list(self.memories)

    def _build_actor_prompt(self, event: str, context: str) -> str:
        """Build the Actor prompt with reflections prepended."""
        parts = ["You are a memory management agent. Decide what to do with the following event.\n"]

        # Prepend reflections (key Reflexion mechanism)
        if self.reflections:
            parts.append("### Lessons from Past Mistakes:")
            for r in self.reflections[-5:]:
                parts.append(f"- {r}")
            parts.append("")

        # Current memories
        if self.memories:
            parts.append("### Current Memories:")
            for m in self.memories[-15:]:
                parts.append(f"- {m}")
            parts.append("")

        parts.append(f"### New Event:\n{event}\n")
        if context:
            parts.append(f"### Task Context:\n{context}\n")

        parts.append(
            "### Available Operations:\n"
            "- ADD: <content> — store new information\n"
            "- UPDATE: <old> -> <new> — update existing memory\n"
            "- DELETE: <content> — remove outdated memory\n"
            "- NOOP — no action needed\n\n"
            "### Your Decision:\n"
        )
        return "\n".join(parts)

    def _build_reflection_prompt(self, event: str, context: str, reward: float) -> str:
        """Build the Self-Reflection prompt."""
        memory_list = "\n".join(f"- {m}" for m in self.memories[-10:])
        return (
            "You are reflecting on a past memory management decision.\n\n"
            f"Event: {event}\n"
            f"Context: {context}\n"
            f"Your action: {self.last_operation}\n"
            f"Reward received: {reward:.2f} (low = bad)\n"
            f"Current memories:\n{memory_list}\n\n"
            "What went wrong? What should you do differently next time? "
            "Write a concise lesson (1-2 sentences):\n"
        )

    def _parse_operation(self, response: str) -> dict:
        """Parse LLM response into memory operation."""
        response = response.strip().upper()
        if response.startswith("ADD"):
            content = response[4:].strip().strip(":").strip()
            return {"type": "ADD", "content": content}
        elif response.startswith("UPDATE"):
            content = response[7:].strip().strip(":").strip()
            return {"type": "UPDATE", "content": content}
        elif response.startswith("DELETE"):
            content = response[7:].strip().strip(":").strip()
            return {"type": "DELETE", "content": content}
        else:
            return {"type": "NOOP", "content": ""}

    def _execute_operation(self, operation: dict, event: str):
        """Execute memory operation on simple list-based store."""
        op_type = operation["type"]
        content = operation.get("content", "")

        if op_type == "ADD" and content:
            self.memories.append(content if content else event)
            if len(self.memories) > self.max_memories:
                self.memories.pop(0)

        elif op_type == "UPDATE" and content:
            # Find most similar memory and update
            if "->" in content:
                parts = content.split("->")
                old_part = parts[0].strip().lower()
                new_part = parts[1].strip()
                for i, m in enumerate(self.memories):
                    if old_part in m.lower():
                        self.memories[i] = new_part
                        break
                else:
                    self.memories.append(new_part)

        elif op_type == "DELETE" and content:
            target = content.lower()
            self.memories = [m for m in self.memories if target not in m.lower()]

    def save(self, path: str):
        import os, json
        super().save(path)
        with open(os.path.join(path, "reflections.json"), "w") as f:
            json.dump({"reflections": self.reflections, "memories": self.memories}, f, indent=2)

    def load(self, path: str):
        import os, json
        fp = os.path.join(path, "reflections.json")
        if os.path.exists(fp):
            with open(fp) as f:
                data = json.load(f)
            self.reflections = data.get("reflections", [])
            self.memories = data.get("memories", [])
