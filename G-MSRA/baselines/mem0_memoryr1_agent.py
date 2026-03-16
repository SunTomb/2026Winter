"""
Baseline 5: Mem0 + Memory-R1 Agent.
Combines Mem0's structured multi-level memory with Memory-R1's RL CRUD.

Reference:
  - Chhikara et al., "Mem0: Building Production-Ready AI Agents with
    Scalable Long-Term Memory", 2025. (github.com/mem0ai/mem0)
  - Chen et al., "Memory-R1", 2025.

Core idea:
  - Mem0 contributes: entity extraction, deduplication, structured
    multi-level memory (User / Session / Agent level)
  - Memory-R1 contributes: RL-trained CRUD policy
  - NO self-reward, NO consolidation — just better memory organization + RL

This is the "engineering combination" baseline that tests whether
simply composing existing tools is sufficient vs. our principled
reward design.
"""

import os
import json

from baselines.base_agent import BaseAgent
from loguru import logger


class Mem0MemoryR1Agent(BaseAgent):
    """Mem0 + Memory-R1: structured memory + RL CRUD."""

    name = "mem0_memory_r1"

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # Mem0-style multi-level memory
        self.user_memories: list[dict] = []       # Long-term user facts
        self.session_memories: list[dict] = []    # Session-level context
        self.agent_memories: list[dict] = []      # Agent's own observations

        # Entity tracking (Mem0 feature)
        self.entities: dict[str, dict] = {}       # entity_name → {facts, last_updated}

    def initialize(self, model=None, tokenizer=None):
        super().initialize(model, tokenizer)

    def process_event(self, event: str, context: str = "") -> dict:
        """Process event: Mem0 entity extraction + RL CRUD decision."""
        self.total_events_processed += 1

        # Step 1: Mem0-style entity extraction
        entities = self._extract_entities(event)

        # Step 2: Mem0-style deduplication
        deduplicated_event = self._deduplicate(event, entities)

        # Step 3: RL-trained CRUD decision (Memory-R1 component)
        operation = self._rl_crud_decision(deduplicated_event, context)

        # Step 4: Execute on multi-level memory
        self._execute_on_multilevel(operation, event, entities)

        return {
            "operation": operation["type"],
            "details": operation.get("content", ""),
        }

    def answer_question(self, question: str) -> str:
        """Answer using Mem0's multi-level memory retrieval."""
        # Retrieve from all levels
        user_context = self._retrieve_from_level(self.user_memories, question, 5)
        session_context = self._retrieve_from_level(self.session_memories, question, 3)
        agent_context = self._retrieve_from_level(self.agent_memories, question, 2)

        # Build prompt with structured memory
        prompt = "You are a helpful assistant with structured memory.\n\n"

        if user_context:
            prompt += f"### User Profile (long-term):\n{user_context}\n\n"
        if session_context:
            prompt += f"### Session Context:\n{session_context}\n\n"
        if agent_context:
            prompt += f"### Agent Notes:\n{agent_context}\n\n"

        # Entity knowledge
        if self.entities:
            entity_info = "\n".join(
                f"- {name}: {info.get('summary', 'N/A')}"
                for name, info in list(self.entities.items())[:10]
            )
            prompt += f"### Known Entities:\n{entity_info}\n\n"

        prompt += f"### Question:\n{question}\n\n### Answer (be concise):\n"

        return self._generate(prompt, max_new_tokens=100, temperature=0.1).strip()

    def train_step(self, reward: float, event: str = "",
                   context: str = "", **kwargs) -> dict:
        """Train RL CRUD policy with QA F1 reward (same as Memory-R1)."""
        import torch

        if not event:
            return {"trained": False}

        trainable_params = [p for p in self.model.parameters() if p.requires_grad]
        if not trainable_params:
            return {"trained": False}

        # Build prompt for the CRUD decision
        prompt = self._build_crud_prompt(event, context)
        operation = self._rl_crud_decision(event, context)

        inputs = self.tokenizer(
            prompt + operation["type"],
            return_tensors="pt", truncation=True,
            max_length=1024, padding=True,
        )
        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}

        prompt_len = len(self.tokenizer.encode(prompt, truncation=True, max_length=1024))
        labels = inputs["input_ids"].clone()
        labels[0, :prompt_len] = -100

        outputs = self.model(**inputs, labels=labels)
        loss = -reward * outputs.loss

        loss.backward()
        torch.nn.utils.clip_grad_norm_(trainable_params, 1.0)

        return {"trained": True, "loss": loss.item(), "reward": reward}

    def reset(self):
        """Reset session memories. User-level memories persist."""
        self.session_memories = []

    def get_memory_contents(self) -> list[str]:
        all_memories = (
            [m["content"] for m in self.user_memories] +
            [m["content"] for m in self.session_memories] +
            [m["content"] for m in self.agent_memories]
        )
        return all_memories

    def _extract_entities(self, event: str) -> list[str]:
        """Mem0-style entity extraction using heuristics.

        In production Mem0, this uses NER or LLM extraction.
        We use a lightweight approach for reproduction.
        """
        entities = []
        words = event.split()

        # Title-case words are likely entities
        for word in words:
            clean = word.strip(".,!?:;\"'()[]")
            if clean and clean[0].isupper() and len(clean) > 1:
                if clean.lower() not in {"i", "the", "a", "an", "my", "user", "says"}:
                    entities.append(clean)

        return list(set(entities))

    def _deduplicate(self, event: str, entities: list[str]) -> str:
        """Mem0-style deduplication: check if information already exists."""
        event_lower = event.lower()

        # Check if this fact is already known
        for mem in self.user_memories:
            if self._similarity(event_lower, mem["content"].lower()) > 0.7:
                return ""  # Skip duplicate

        return event

    def _rl_crud_decision(self, event: str, context: str) -> dict:
        """RL-trained CRUD decision (Memory-R1 component).

        Uses LLM to generate a CRUD operation, similar to Memory-R1.
        """
        if not event:
            return {"type": "NOOP", "content": ""}

        prompt = self._build_crud_prompt(event, context)
        response = self._generate(prompt, max_new_tokens=150, temperature=0.3)

        return self._parse_operation(response)

    def _build_crud_prompt(self, event: str, context: str) -> str:
        """Build CRUD prompt with multi-level memory context."""
        parts = ["You are a memory management agent with structured memory.\n"]

        if self.user_memories:
            parts.append("### User Profile:")
            for m in self.user_memories[-10:]:
                parts.append(f"- {m['content']}")
            parts.append("")

        if self.session_memories:
            parts.append("### Current Session:")
            for m in self.session_memories[-5:]:
                parts.append(f"- {m['content']}")
            parts.append("")

        parts.append(f"### New Event:\n{event}\n")
        if context:
            parts.append(f"### Context:\n{context}\n")

        parts.append(
            "### Operations:\n"
            "- ADD: <content> (add to user profile)\n"
            "- UPDATE: <content> (update existing fact)\n"
            "- DELETE: <content> (remove outdated fact)\n"
            "- NOOP (no action)\n\n"
            "### Decision:\n"
        )
        return "\n".join(parts)

    def _execute_on_multilevel(self, operation: dict, event: str,
                                entities: list[str]):
        """Execute operation on Mem0-style multi-level memory."""
        op_type = operation["type"]
        content = operation.get("content", event)

        if op_type == "ADD" and content:
            # Classify into memory level
            level = self._classify_memory_level(content)

            memory_entry = {
                "content": content,
                "entities": entities,
                "level": level,
            }

            if level == "user":
                self.user_memories.append(memory_entry)
                if len(self.user_memories) > self.max_memories:
                    self.user_memories.pop(0)
            elif level == "session":
                self.session_memories.append(memory_entry)
                if len(self.session_memories) > 50:
                    self.session_memories.pop(0)
            else:
                self.agent_memories.append(memory_entry)
                if len(self.agent_memories) > 30:
                    self.agent_memories.pop(0)

            # Update entity tracking
            for entity in entities:
                if entity not in self.entities:
                    self.entities[entity] = {"facts": [], "summary": ""}
                self.entities[entity]["facts"].append(content[:100])
                self.entities[entity]["summary"] = content[:100]

        elif op_type == "UPDATE" and content:
            # Update in all levels
            for mem_list in [self.user_memories, self.session_memories, self.agent_memories]:
                for i, m in enumerate(mem_list):
                    if self._similarity(content.lower(), m["content"].lower()) > 0.3:
                        mem_list[i]["content"] = content
                        break

        elif op_type == "DELETE" and content:
            target = content.lower()
            for mem_list in [self.user_memories, self.session_memories, self.agent_memories]:
                mem_list[:] = [m for m in mem_list if target not in m["content"].lower()]

    def _classify_memory_level(self, content: str) -> str:
        """Classify memory into Mem0 levels."""
        content_lower = content.lower()

        # User-level: persistent personal facts
        user_indicators = [
            "name is", "live", "work", "born", "prefer", "like",
            "married", "family", "age", "phone", "email", "hobby",
        ]
        if any(ind in content_lower for ind in user_indicators):
            return "user"

        # Session-level: current conversation context
        session_indicators = [
            "today", "right now", "currently", "this meeting",
            "just", "recently", "earlier",
        ]
        if any(ind in content_lower for ind in session_indicators):
            return "session"

        # Default: agent level
        return "agent"

    def _retrieve_from_level(self, memories: list[dict], query: str,
                              top_k: int) -> str:
        """Retrieve relevant memories from a level using keyword matching."""
        if not memories:
            return ""

        query_words = set(query.lower().split())
        scored = []
        for m in memories:
            m_words = set(m["content"].lower().split())
            overlap = len(query_words & m_words)
            scored.append((overlap, m["content"]))

        scored.sort(reverse=True)
        relevant = [content for _, content in scored[:top_k] if _ > 0]
        return "\n".join(f"- {c}" for c in relevant)

    def _similarity(self, a: str, b: str) -> float:
        """Simple Jaccard similarity for deduplication."""
        words_a = set(a.split())
        words_b = set(b.split())
        if not words_a or not words_b:
            return 0.0
        intersection = len(words_a & words_b)
        union = len(words_a | words_b)
        return intersection / union if union > 0 else 0.0

    def _parse_operation(self, response: str) -> dict:
        """Parse LLM response into memory operation."""
        response = response.strip().upper()
        if response.startswith("ADD"):
            return {"type": "ADD", "content": response[4:].strip().strip(":").strip()}
        elif response.startswith("UPDATE"):
            return {"type": "UPDATE", "content": response[7:].strip().strip(":").strip()}
        elif response.startswith("DELETE"):
            return {"type": "DELETE", "content": response[7:].strip().strip(":").strip()}
        return {"type": "NOOP", "content": ""}

    def save(self, path: str):
        super().save(path)
        with open(os.path.join(path, "mem0_state.json"), "w") as f:
            json.dump({
                "user_memories": self.user_memories,
                "session_memories": self.session_memories,
                "agent_memories": self.agent_memories,
                "entities": self.entities,
            }, f, indent=2, ensure_ascii=False)

    def load(self, path: str):
        fp = os.path.join(path, "mem0_state.json")
        if os.path.exists(fp):
            with open(fp) as f:
                data = json.load(f)
            self.user_memories = data.get("user_memories", [])
            self.session_memories = data.get("session_memories", [])
            self.agent_memories = data.get("agent_memories", [])
            self.entities = data.get("entities", {})
