"""
Baseline 3: Self-Consolidation Agent.
Contrastive reflection + LoRA parameter distillation.

Reference: Zhang et al., "Self-Consolidation for Self-Evolving Agents",
arXiv:2602.01966, 2026.

Core idea:
  - Compare model output with/without memory context
  - Extract positive (correct memory use) and negative (contradictions) pairs
  - Fine-tune LoRA on positive triples
  - Fixed trigger: every N episodes
  - NO RL, NO self-reward, NO environment grounding

This is the parametric consolidation baseline — it tests whether
consolidation alone (without RL or grounded reward) is sufficient.
"""

import os
import json

from baselines.base_agent import BaseAgent
from loguru import logger


class SelfConsolidationAgent(BaseAgent):
    """Self-Consolidation: contrastive reflection + LoRA distillation."""

    name = "self_consolidation"

    def __init__(self, consolidation_interval: int = 50,
                 consolidation_lr: float = 1e-4,
                 **kwargs):
        super().__init__(**kwargs)
        self.consolidation_interval = consolidation_interval
        self.consolidation_lr = consolidation_lr
        self.memories: list[str] = []
        self.episode_count = 0
        self.consolidation_count = 0
        self.positive_pairs: list[tuple[str, str]] = []  # (input, correct_output)
        self._lora_model = None

    def initialize(self, model=None, tokenizer=None):
        super().initialize(model, tokenizer)

    def process_event(self, event: str, context: str = "") -> dict:
        """Process event: heuristic memory management + contrastive reflection."""
        self.total_events_processed += 1

        # Simple heuristic CRUD (no RL)
        operation = self._heuristic_crud(event, context)
        self._execute_operation(operation, event)

        # Contrastive reflection: compare response with/without memories
        if context and self.memories:
            self._contrastive_reflect(event, context)

        # Fixed-interval consolidation (not adaptive)
        self.episode_count += 1
        if self.episode_count % self.consolidation_interval == 0:
            self._consolidate()

        return {
            "operation": operation["type"],
            "details": operation.get("content", ""),
        }

    def answer_question(self, question: str) -> str:
        """Answer question using memories."""
        memory_context = "\n".join(f"- {m}" for m in self.memories[-20:])

        prompt = "You are a helpful assistant with memory.\n\n"
        if memory_context:
            prompt += f"### Your Memories:\n{memory_context}\n\n"
        prompt += f"### Question:\n{question}\n\n### Answer (be concise):\n"

        return self._generate(prompt, max_new_tokens=100, temperature=0.1).strip()

    def reset(self):
        """Reset per-episode state, keep LoRA weights and consolidation history."""
        self.memories = []

    def get_memory_contents(self) -> list[str]:
        return list(self.memories)

    def _heuristic_crud(self, event: str, context: str) -> dict:
        """Heuristic memory management (no RL).

        Self-Consolidation uses rule-based memory decisions:
          - If event contains factual content → ADD
          - If event contradicts existing memory → UPDATE
          - Otherwise → NOOP
        """
        event_lower = event.lower()

        # Check for contradictions with existing memories
        for i, mem in enumerate(self.memories):
            mem_lower = mem.lower()
            # Simple contradiction detection: shared subjects with different predicates
            event_words = set(event_lower.split())
            mem_words = set(mem_lower.split())
            overlap = event_words & mem_words
            if len(overlap) > 3 and event_lower != mem_lower:
                return {"type": "UPDATE", "content": event, "target_idx": i}

        # Check if event has factual content worth storing
        factual_indicators = [
            "is", "was", "are", "were", "has", "have", "had",
            "lives", "works", "moved", "started", "stopped",
            "prefers", "likes", "dislikes", "wants", "needs",
        ]
        if any(ind in event_lower for ind in factual_indicators):
            return {"type": "ADD", "content": event}

        return {"type": "NOOP", "content": ""}

    def _execute_operation(self, operation: dict, event: str):
        """Execute heuristic memory operation."""
        op_type = operation["type"]
        content = operation.get("content", event)

        if op_type == "ADD":
            self.memories.append(content)
            if len(self.memories) > self.max_memories:
                self.memories.pop(0)
        elif op_type == "UPDATE":
            idx = operation.get("target_idx", -1)
            if 0 <= idx < len(self.memories):
                self.memories[idx] = content

    def _contrastive_reflect(self, event: str, context: str):
        """Contrastive reflection: compare output quality with/without memory.

        If memory-augmented response is better → extract as positive triple.
        If memory-augmented response is worse → evidence of noise, skip.
        """
        # Response WITHOUT memory
        prompt_no_mem = (
            f"Answer the following based only on the given event.\n"
            f"Event: {event}\nQuestion: {context}\nAnswer:\n"
        )
        response_no_mem = self._generate(prompt_no_mem, max_new_tokens=80, temperature=0.1)

        # Response WITH memory
        memory_text = "\n".join(f"- {m}" for m in self.memories[-10:])
        prompt_with_mem = (
            f"Answer using the memories and event.\n"
            f"Memories:\n{memory_text}\n"
            f"Event: {event}\nQuestion: {context}\nAnswer:\n"
        )
        response_with_mem = self._generate(prompt_with_mem, max_new_tokens=80, temperature=0.1)

        # If memory-augmented response is longer/more detailed → positive pair
        if len(response_with_mem.split()) > len(response_no_mem.split()):
            triple = f"Given: {event[:100]}. Memory says: {self.memories[-1][:100] if self.memories else 'N/A'}. Correct response: {response_with_mem[:150]}"
            self.positive_pairs.append((context, triple))

            # Keep manageable size
            if len(self.positive_pairs) > 200:
                self.positive_pairs = self.positive_pairs[-200:]

    def _consolidate(self):
        """LoRA consolidation at fixed intervals.

        Unlike G-MSRA's adaptive trigger, Self-Consolidation fires
        every N episodes regardless of memory state.
        """
        if len(self.positive_pairs) < 5:
            logger.debug("[SelfConsolidation] Not enough positive pairs, skipping")
            return

        logger.info(
            f"[SelfConsolidation] Consolidation #{self.consolidation_count + 1}: "
            f"{len(self.positive_pairs)} positive pairs"
        )

        try:
            import torch

            # Build training data from positive pairs
            train_texts = []
            for context, triple in self.positive_pairs[-50:]:
                text = f"Q: {context}\nA: {triple}"
                train_texts.append(text)

            # LoRA SFT on positive triples
            trainable_params = [p for p in self.model.parameters() if p.requires_grad]
            if not trainable_params:
                logger.debug("[SelfConsolidation] No trainable params, skipping LoRA")
                return

            optimizer = torch.optim.AdamW(trainable_params, lr=self.consolidation_lr)
            total_loss = 0.0

            self.model.train()
            for text in train_texts[:20]:  # Limit per consolidation
                inputs = self.tokenizer(
                    text, return_tensors="pt", truncation=True,
                    max_length=512, padding=True,
                )
                inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
                labels = inputs["input_ids"].clone()

                outputs = self.model(**inputs, labels=labels)
                loss = outputs.loss
                loss.backward()
                total_loss += loss.item()

            torch.nn.utils.clip_grad_norm_(trainable_params, 1.0)
            optimizer.step()
            optimizer.zero_grad()

            self.consolidation_count += 1
            avg_loss = total_loss / max(len(train_texts[:20]), 1)
            logger.info(f"[SelfConsolidation] Consolidation done, avg_loss={avg_loss:.4f}")

            # Clear used pairs
            self.positive_pairs = []

        except Exception as e:
            logger.warning(f"[SelfConsolidation] Consolidation failed: {e}")

    def save(self, path: str):
        super().save(path)
        with open(os.path.join(path, "consolidation_state.json"), "w") as f:
            json.dump({
                "memories": self.memories,
                "consolidation_count": self.consolidation_count,
                "episode_count": self.episode_count,
                "num_positive_pairs": len(self.positive_pairs),
            }, f, indent=2)
