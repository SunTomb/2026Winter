"""
G-MSRA Agent: Main orchestrator connecting all modules.

Lifecycle:
  Online Phase:
    1. Receive event → Memory Manager decides operation → Execute
    2. Agent responds to task → Grounded Reward computed
    3. RL policy updated with R_total

  Offline Phase (triggered by ConsolidationTrigger):
    4. Extract high-value subgraph → Semantic triples → LoRA distillation
    5. Clear distilled memories → Continue

Training Phases:
  Phase 0: SFT warmup for Memory Manager format
  Phase 1: RL + external reward (QA F1)
  Phase 2: Curriculum annealing (external → self-reward)
  Phase 3: Full closed-loop with consolidation
"""

from __future__ import annotations
from typing import Optional

from loguru import logger

from gmsra.config import GMSRAConfig
from gmsra.memory.store import MemoryStore
from gmsra.manager.memory_manager import MemoryManager
from gmsra.reward.grounded_reward import GroundedRewardGenerator
from gmsra.reward.env_signals import (
    EnvironmentSignalExtractor,
    AgentTaskSignalExtractor,
    DialogueSignalExtractor,
)
from gmsra.consolidation.trigger import ConsolidationTrigger
from gmsra.consolidation.distiller import SemanticDistiller


class GMSRAAgent:
    """Grounded Memory-Guided Self-Rewarding Agent.

    Integrates:
    - MemoryStore: Zettelkasten-style external memory with graph links
    - MemoryManager: RL policy for CRUD operations
    - GroundedRewardGenerator: Dual-layer composite reward
    - ConsolidationTrigger: 3D adaptive trigger
    - SemanticDistiller: LoRA parametric distillation

    Usage:
        agent = GMSRAAgent(config)
        agent.initialize(model, tokenizer)

        for event in event_stream:
            result = agent.step(event, task_context, env_kwargs)
    """

    def __init__(self, config: Optional[GMSRAConfig] = None):
        self.config = config or GMSRAConfig()
        self.step_count = 0
        self.episode_count = 0

        # --- Modules (initialized in setup) ---
        self.memory_store: Optional[MemoryStore] = None
        self.memory_manager: Optional[MemoryManager] = None
        self.reward_generator: Optional[GroundedRewardGenerator] = None
        self.trigger: Optional[ConsolidationTrigger] = None
        self.distiller: Optional[SemanticDistiller] = None
        self.env_extractor: Optional[EnvironmentSignalExtractor] = None

    def initialize(self, model, tokenizer,
                   env_type: str = "agent_task",
                   judge_model=None, judge_tokenizer=None):
        """Initialize all modules with the loaded model.

        Args:
            model: Base LLM (will be used for Manager, Judge, Distiller).
            tokenizer: Tokenizer for the base model.
            env_type: "agent_task" or "dialogue" — determines signal extractor.
            judge_model: Separate judge model (optional, defaults to base model).
            judge_tokenizer: Tokenizer for judge (optional).
        """
        judge_model = judge_model or model
        judge_tokenizer = judge_tokenizer or tokenizer

        # 1. Memory Store
        self.memory_store = MemoryStore(self.config.memory)

        # 2. Environment Signal Extractor
        if env_type == "agent_task":
            self.env_extractor = AgentTaskSignalExtractor()
        elif env_type == "dialogue":
            self.env_extractor = DialogueSignalExtractor(judge_model, judge_tokenizer)
        else:
            raise ValueError(f"Unknown env_type: {env_type}")

        # 3. Grounded Reward Generator
        self.reward_generator = GroundedRewardGenerator(
            config=self.config.reward,
            memory_store=self.memory_store,
            judge_model=judge_model,
            judge_tokenizer=judge_tokenizer,
            env_extractor=self.env_extractor,
        )

        # 4. Memory Manager (RL policy)
        self.memory_manager = MemoryManager(
            model=model,
            tokenizer=tokenizer,
            memory_store=self.memory_store,
            rl_config=self.config.rl,
            memory_config=self.config.memory,
        )

        # 5. Consolidation Trigger
        self.trigger = ConsolidationTrigger(
            config=self.config.trigger,
            memory_store=self.memory_store,
            reward_generator=self.reward_generator,
        )

        # 6. Semantic Distiller
        self.distiller = SemanticDistiller(
            base_model=model,
            tokenizer=tokenizer,
            config=self.config.lora,
        )

        logger.info(
            f"G-MSRA Agent initialized: env_type={env_type}, "
            f"model={self.config.model.model_name}"
        )

    # ---- Online Phase ----

    def step(
        self,
        event: str,
        task_context: str = "",
        agent_response: str = "",
        env_signal_kwargs: dict = None,
    ) -> dict:
        """Execute one step of the G-MSRA loop.

        Args:
            event: New event/observation to process.
            task_context: Current task context.
            agent_response: Agent's response to the task (for reward).
            env_signal_kwargs: Environment signal kwargs (e.g., task_result).

        Returns:
            Step result dict with operation, reward, and trigger info.
        """
        self.step_count += 1
        env_signal_kwargs = env_signal_kwargs or {}

        # Step 1: Memory Manager decides operation
        operation_str, prompt = self.memory_manager.decide(event, task_context)

        # Step 2: Execute the memory operation
        op_result = self.memory_manager.execute_operation(
            operation_str, event,
            env_reward=env_signal_kwargs.get("env_reward_hint", 0.5)
        )

        # Step 3: Compute grounded composite reward
        reward_result = self.reward_generator.compute_reward(
            agent_response=agent_response or event,
            task_context=task_context,
            memory_operation=operation_str,
            env_signal_kwargs=env_signal_kwargs,
        )

        # Step 4: Record hit for confidence tracking
        if op_result["success"] and op_result["op"] != "NOOP":
            task_success = reward_result.r_env > 0.5
            entry_id = op_result.get("entry_id")
            if entry_id and entry_id in self.memory_store.entries:
                self.memory_store.entries[entry_id].record_hit(task_success)

        # Step 5: Check consolidation trigger
        trigger_fired = False
        consolidation_stats = None
        if self.trigger.should_trigger(self.step_count):
            trigger_fired = True
            consolidation_stats = self._run_consolidation()

        result = {
            "step": self.step_count,
            "operation": op_result,
            "reward": {
                "r_env": reward_result.r_env,
                "r_mem": reward_result.r_mem,
                "r_total": reward_result.r_total,
            },
            "memory_size": self.memory_store.size(),
            "trigger_fired": trigger_fired,
            "consolidation": consolidation_stats,
        }

        if self.step_count % 100 == 0:
            logger.info(
                f"Step {self.step_count}: mem_size={self.memory_store.size()}, "
                f"R_total={reward_result.r_total:.3f}, ops={self.memory_manager.get_operation_stats()}"
            )

        return result

    # ---- Offline Phase ----

    def _run_consolidation(self) -> dict:
        """Execute offline parametric consolidation (sleep-time)."""
        logger.info("=" * 50)
        logger.info("OFFLINE CONSOLIDATION PHASE")
        logger.info("=" * 50)

        # Recalibrate memory confidence before selection
        self.memory_store.recalibrate_confidence()

        # Run semantic distillation
        stats = self.distiller.consolidate(
            memory_store=self.memory_store,
            llm_model=self.memory_manager.model,
            llm_tokenizer=self.memory_manager.tokenizer,
        )

        # Optionally clear distilled memories to control store growth
        if not stats.get("skipped", True):
            cleared = 0
            for entry_id in self.distiller.distilled_entries[-stats.get("distilled", 0):]:
                if entry_id in self.memory_store.entries:
                    self.memory_store.delete(entry_id)
                    cleared += 1
            stats["cleared_from_store"] = cleared
            logger.info(f"Cleared {cleared} distilled memories from store")

        return stats

    # ---- Evaluation Helpers ----

    def answer_question(self, question: str) -> str:
        """Answer a question using the memory-augmented agent.

        Combines parametric knowledge (from LoRA) with retrieved memories.
        """
        from gmsra.utils import generate_text

        # Retrieve relevant memories
        relevant = self.memory_store.retrieve(question, topk=5)
        memory_context = "\n".join([
            f"- {entry.content}" for entry, score in relevant
        ]) if relevant else "(no relevant memories)"

        prompt = (
            "Answer the following question using the provided context.\n\n"
            f"### Relevant Memories\n{memory_context}\n\n"
            f"### Question\n{question}\n\n"
            "### Answer\n"
        )

        model = self.distiller._lora_model or self.memory_manager.model
        tokenizer = self.memory_manager.tokenizer

        return generate_text(model, tokenizer, prompt, max_new_tokens=256)

    # ---- Persistence ----

    def save_checkpoint(self, path: str):
        """Save full agent state."""
        import os
        os.makedirs(path, exist_ok=True)

        self.memory_store.save(os.path.join(path, "memory_store.json"))
        self.distiller.save_lora(os.path.join(path, "lora"))

        # Save reward history and trigger diagnostics
        import json
        meta = {
            "step_count": self.step_count,
            "episode_count": self.episode_count,
            "trigger_diagnostics": self.trigger.get_diagnostics() if self.trigger else {},
            "operation_stats": self.memory_manager.get_operation_stats() if self.memory_manager else {},
        }
        with open(os.path.join(path, "agent_meta.json"), "w") as f:
            json.dump(meta, f, indent=2, default=str)

        logger.info(f"Agent checkpoint saved to {path}")

    def load_checkpoint(self, path: str):
        """Load agent state from checkpoint."""
        import os

        mem_path = os.path.join(path, "memory_store.json")
        if os.path.exists(mem_path):
            self.memory_store.load(mem_path)

        lora_path = os.path.join(path, "lora")
        if os.path.exists(lora_path):
            self.distiller.load_lora(lora_path)

        meta_path = os.path.join(path, "agent_meta.json")
        if os.path.exists(meta_path):
            import json
            with open(meta_path, "r") as f:
                meta = json.load(f)
            self.step_count = meta.get("step_count", 0)
            self.episode_count = meta.get("episode_count", 0)

        logger.info(f"Agent checkpoint loaded from {path}")

    # ---- Diagnostics ----

    def get_full_diagnostics(self) -> dict:
        """Get comprehensive diagnostics for analysis and paper figures."""
        return {
            "step_count": self.step_count,
            "memory_size": self.memory_store.size() if self.memory_store else 0,
            "operation_stats": self.memory_manager.get_operation_stats() if self.memory_manager else {},
            "reward_drift": self.reward_generator.get_reward_drift() if self.reward_generator else [],
            "trigger_diagnostics": self.trigger.get_diagnostics() if self.trigger else {},
            "consolidation_count": self.distiller.consolidation_count if self.distiller else 0,
        }
