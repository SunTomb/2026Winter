"""
Semantic Distiller: Parametric Consolidation via LoRA.
Converts high-value memories (selected by graph subgraph extraction)
into semantic triples, then distills them into LoRA parameters.

Key design: Graph topology determines WHAT to distill (selection),
but the distillation itself uses text-based SFT (not graph serialization).
This avoids the information loss problem of graph→sequence conversion.

Dual-LoRA architecture:
- Long-term LoRA (high-rank): preserved knowledge, EWC-protected
- Short-term LoRA (low-rank): new adaptation, freely updated
"""

from __future__ import annotations
from typing import Optional
import os

import torch
from loguru import logger

from gmsra.config import LoRAConfig
from gmsra.memory.store import MemoryStore
from gmsra.memory.entry import MemoryEntry


class SemanticDistiller:
    """Semantic distillation module for parametric consolidation.

    Flow:
    1. Extract high-frequency subgraph from memory store
    2. Convert memories to semantic triple declarations
    3. SFT-train LoRA layers on these declarations
    4. Clear distilled memories from external store
    """

    def __init__(
        self,
        base_model=None,
        tokenizer=None,
        config: Optional[LoRAConfig] = None,
    ):
        self.base_model = base_model
        self.tokenizer = tokenizer
        self.config = config or LoRAConfig()
        self.consolidation_count = 0
        self.distilled_entries: list[str] = []  # IDs of distilled memories
        self._lora_model = None
        self._fisher_information = None  # For EWC regularization

    def setup_dual_lora(self):
        """Initialize dual-LoRA architecture.

        - Long-term LoRA: high-rank, EWC-protected
        - Short-term LoRA: low-rank, freely updated
        """
        from peft import LoraConfig as PeftLoraConfig, get_peft_model, TaskType

        # Start with short-term LoRA for initial adaptation
        lora_config = PeftLoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=self.config.shortterm_rank,
            lora_alpha=self.config.shortterm_alpha,
            target_modules=self.config.target_modules,
            lora_dropout=self.config.lora_dropout,
            bias="none",
        )
        self._lora_model = get_peft_model(self.base_model, lora_config)
        logger.info(
            f"Initialized LoRA: rank={self.config.shortterm_rank}, "
            f"alpha={self.config.shortterm_alpha}, "
            f"targets={self.config.target_modules}"
        )
        return self._lora_model

    def consolidate(self, memory_store: MemoryStore,
                    llm_model=None, llm_tokenizer=None) -> dict:
        """Execute parametric consolidation.

        Steps:
        1. Extract high-value memories via graph subgraph
        2. Generate semantic triples
        3. Train LoRA on triples
        4. Clear distilled memories from store

        Args:
            memory_store: The external memory store.
            llm_model: Model for generating semantic triples (can be same as base).
            llm_tokenizer: Tokenizer for triple generation.

        Returns:
            Dict with consolidation statistics.
        """
        llm_model = llm_model or self.base_model
        llm_tokenizer = llm_tokenizer or self.tokenizer

        # Step 1: Extract high-value subgraph
        candidates = memory_store.extract_high_frequency_subgraph(
            min_links=2, min_confidence=0.5
        )
        if len(candidates) < 3:
            logger.info("Not enough high-value memories for consolidation")
            return {"distilled": 0, "skipped": True}

        logger.info(f"Consolidation: {len(candidates)} candidate memories")

        # Step 2: Generate semantic triples
        triples = self._generate_semantic_triples(
            candidates, llm_model, llm_tokenizer
        )
        if not triples:
            logger.warning("No semantic triples generated, skipping consolidation")
            return {"distilled": 0, "skipped": True}

        logger.info(f"Generated {len(triples)} semantic triples")

        # Step 3: Train LoRA
        if self._lora_model is None:
            self.setup_dual_lora()

        train_loss = self._train_lora(triples)

        # Step 4: Record and optionally clear distilled memories
        distilled_ids = [e.id for e in candidates]
        self.distilled_entries.extend(distilled_ids)
        self.consolidation_count += 1

        # Compute EWC Fisher information for protection
        self._compute_fisher_information(triples)

        stats = {
            "distilled": len(candidates),
            "triples": len(triples),
            "train_loss": train_loss,
            "consolidation_num": self.consolidation_count,
            "skipped": False,
        }
        logger.info(f"Consolidation #{self.consolidation_count} complete: {stats}")
        return stats

    def _generate_semantic_triples(
        self,
        memories: list[MemoryEntry],
        model, tokenizer,
    ) -> list[str]:
        """Convert high-value memories into semantic triple declarations.

        Examples of generated triples:
        - "The user prefers Python over Java for data analysis."
        - "When encountering a FileNotFoundError, check the working directory first."
        - "The user's meeting schedule changed from Tuesdays to Wednesdays."
        """
        from gmsra.utils import generate_text

        triples = []
        # Process memories in related groups (use links)
        for memory in memories:
            # Get linked memories for context
            context_parts = [memory.content]
            for link_id in memory.links[:3]:
                if link_id in {m.id for m in memories}:
                    linked = next((m for m in memories if m.id == link_id), None)
                    if linked:
                        context_parts.append(linked.content)

            context = "\n".join(f"- {c}" for c in context_parts)
            prompt = (
                "Convert the following related memory entries into clear, "
                "concise knowledge statements. Each statement should be a "
                "self-contained fact, preference, or rule.\n\n"
                f"Memory entries:\n{context}\n\n"
                "Output 1-3 knowledge statements, one per line:\n"
            )

            try:
                output = generate_text(
                    model, tokenizer, prompt,
                    max_new_tokens=150, temperature=0.5
                )
                for line in output.strip().split("\n"):
                    line = line.strip().lstrip("- •0123456789.)")
                    if len(line) > 10:  # Filter out empty/trivial lines
                        triples.append(line.strip())
            except Exception as e:
                logger.warning(f"Triple generation failed for [{memory.id}]: {e}")

        return triples

    def _train_lora(self, triples: list[str]) -> float:
        """Train LoRA layers on semantic triple declarations via SFT.

        Uses standard cross-entropy language modeling loss
        with EWC regularization to prevent catastrophic forgetting.
        """
        self._lora_model.train()
        optimizer = torch.optim.AdamW(
            self._lora_model.parameters(),
            lr=self.config.consolidation_lr,
            weight_decay=0.01,
        )

        total_loss = 0.0
        num_steps = 0

        for epoch in range(self.config.consolidation_epochs):
            for text in triples:
                # Prepend instruction for SFT
                full_text = f"Knowledge: {text}"
                inputs = self.tokenizer(
                    full_text, return_tensors="pt", truncation=True,
                    max_length=256, padding=True,
                )
                inputs = {k: v.to(self._lora_model.device) for k, v in inputs.items()}

                outputs = self._lora_model(**inputs, labels=inputs["input_ids"])
                loss = outputs.loss

                # EWC regularization
                if self._fisher_information is not None:
                    ewc_loss = self._compute_ewc_penalty()
                    loss = loss + self.config.ewc_lambda * ewc_loss

                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    self._lora_model.parameters(), max_norm=1.0
                )
                optimizer.step()
                optimizer.zero_grad()

                total_loss += loss.item()
                num_steps += 1

        avg_loss = total_loss / max(num_steps, 1)
        self._lora_model.eval()
        logger.info(f"LoRA training done: avg_loss={avg_loss:.4f}, steps={num_steps}")
        return avg_loss

    def _compute_fisher_information(self, triples: list[str]):
        """Compute Fisher Information Matrix diagonal for EWC.
        Protects previously consolidated knowledge from being overwritten.
        """
        self._fisher_information = {}
        self._param_snapshot = {}

        self._lora_model.eval()
        for name, param in self._lora_model.named_parameters():
            if param.requires_grad:
                self._fisher_information[name] = torch.zeros_like(param)
                self._param_snapshot[name] = param.data.clone()

        # Estimate Fisher via gradients on consolidation data
        self._lora_model.train()
        for text in triples[:20]:  # Sample subset for efficiency
            full_text = f"Knowledge: {text}"
            inputs = self.tokenizer(
                full_text, return_tensors="pt", truncation=True,
                max_length=256,
            )
            inputs = {k: v.to(self._lora_model.device) for k, v in inputs.items()}

            outputs = self._lora_model(**inputs, labels=inputs["input_ids"])
            loss = outputs.loss
            loss.backward()

            for name, param in self._lora_model.named_parameters():
                if param.requires_grad and param.grad is not None:
                    self._fisher_information[name] += param.grad.data ** 2

            self._lora_model.zero_grad()

        # Normalize
        n = len(triples[:20])
        for name in self._fisher_information:
            self._fisher_information[name] /= n

        self._lora_model.eval()
        logger.debug("Computed Fisher Information for EWC protection")

    def _compute_ewc_penalty(self) -> torch.Tensor:
        """Compute EWC penalty: Σ F_i · (θ_i - θ*_i)²"""
        penalty = torch.tensor(0.0, device=self._lora_model.device)
        for name, param in self._lora_model.named_parameters():
            if name in self._fisher_information and param.requires_grad:
                fisher = self._fisher_information[name]
                old_params = self._param_snapshot[name]
                penalty += (fisher * (param - old_params) ** 2).sum()
        return penalty

    def save_lora(self, path: str):
        """Save LoRA weights."""
        if self._lora_model is not None:
            os.makedirs(path, exist_ok=True)
            self._lora_model.save_pretrained(path)
            logger.info(f"Saved LoRA weights to {path}")

    def load_lora(self, path: str):
        """Load LoRA weights."""
        if self._lora_model is not None:
            from peft import PeftModel
            self._lora_model = PeftModel.from_pretrained(self.base_model, path)
            logger.info(f"Loaded LoRA weights from {path}")
