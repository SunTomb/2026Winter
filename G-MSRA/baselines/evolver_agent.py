"""
Baseline 4: EvolveR Agent.
Self-evolving LLM agent with experience-driven lifecycle.

Reference: "EvolveR: Self-Evolving LLM Agents through an
Experience-Driven Lifecycle", arXiv:2510.16079, 2025.

Core idea:
  - Two-stage loop:
    1. Online: Agent interacts, stores full trajectories
    2. Offline: LLM analyzes trajectories → extracts abstract "strategic principles"
  - Principles are stored in a vector-indexed repository
  - At inference, relevant principles are retrieved and prepended to prompts
  - NO RL weight updates, NO LoRA, NO self-reward score

Difference from Reflexion:
  - Reflexion stores per-failure reflections (specific)
  - EvolveR extracts abstract, reusable strategic principles (general)
"""

import os
import json
import random

from baselines.base_agent import BaseAgent
from loguru import logger


class EvolveRAgent(BaseAgent):
    """EvolveR: experience lifecycle with principle distillation."""

    name = "evolver"

    def __init__(self, max_principles: int = 50,
                 distillation_interval: int = 20,
                 max_trajectory_buffer: int = 100,
                 **kwargs):
        super().__init__(**kwargs)
        self.max_principles = max_principles
        self.distillation_interval = distillation_interval
        self.max_trajectory_buffer = max_trajectory_buffer

        # Memory store
        self.memories: list[str] = []

        # Trajectory buffer: stores (events, actions, outcomes) per episode
        self.trajectory_buffer: list[dict] = []
        self.current_trajectory: list[dict] = []

        # Strategic principle repository
        self.principles: list[str] = []

        # Counters
        self.episodes_since_distillation = 0

    def process_event(self, event: str, context: str = "") -> dict:
        """Online phase: interact and record trajectory."""
        self.total_events_processed += 1

        # Build prompt with principles
        prompt = self._build_action_prompt(event, context)

        # Generate action (memory operation)
        response = self._generate(prompt, max_new_tokens=200, temperature=0.3)
        operation = self._parse_operation(response)
        self._execute_operation(operation, event)

        # Record in current trajectory
        self.current_trajectory.append({
            "event": event,
            "context": context,
            "action": operation["type"],
            "content": operation.get("content", ""),
        })

        return {
            "operation": operation["type"],
            "details": operation.get("content", ""),
        }

    def end_episode(self, episode_reward: float):
        """Called at end of episode to finalize trajectory and possibly distill."""
        if self.current_trajectory:
            self.trajectory_buffer.append({
                "trajectory": self.current_trajectory,
                "reward": episode_reward,
            })
            # Trim buffer
            if len(self.trajectory_buffer) > self.max_trajectory_buffer:
                self.trajectory_buffer = self.trajectory_buffer[-self.max_trajectory_buffer:]

        self.current_trajectory = []
        self.episodes_since_distillation += 1

        # Offline distillation at fixed intervals
        if self.episodes_since_distillation >= self.distillation_interval:
            self._distill_principles()
            self.episodes_since_distillation = 0

    def answer_question(self, question: str) -> str:
        """Answer question using memories + principles."""
        memory_context = "\n".join(f"- {m}" for m in self.memories[-20:])

        # Retrieve relevant principles
        relevant_principles = self._retrieve_principles(question)
        principle_context = "\n".join(f"- {p}" for p in relevant_principles)

        prompt = "You are a helpful assistant with memory and strategic knowledge.\n\n"
        if principle_context:
            prompt += f"### Strategic Principles:\n{principle_context}\n\n"
        if memory_context:
            prompt += f"### Your Memories:\n{memory_context}\n\n"
        prompt += f"### Question:\n{question}\n\n### Answer (be concise):\n"

        return self._generate(prompt, max_new_tokens=100, temperature=0.1).strip()

    def reset(self):
        """Reset per-episode state, keep principles and trajectory buffer."""
        self.memories = []
        self.current_trajectory = []

    def get_memory_contents(self) -> list[str]:
        return list(self.memories)

    def _build_action_prompt(self, event: str, context: str) -> str:
        """Build prompt with principles prepended (key EvolveR mechanism)."""
        parts = ["You are a memory management agent.\n"]

        # Prepend relevant principles (core EvolveR feature)
        relevant_principles = self._retrieve_principles(event)
        if relevant_principles:
            parts.append("### Strategic Principles (learned from experience):")
            for p in relevant_principles[:5]:
                parts.append(f"- {p}")
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
            "- ADD: <content>\n- UPDATE: <old> -> <new>\n"
            "- DELETE: <content>\n- NOOP\n\n"
            "### Your Decision:\n"
        )
        return "\n".join(parts)

    def _distill_principles(self):
        """Offline phase: analyze trajectories and extract strategic principles.

        This is the core EvolveR contribution — converting specific
        trajectory experiences into abstract, reusable principles.
        """
        if len(self.trajectory_buffer) < 3:
            return

        logger.info(
            f"[EvolveR] Distilling principles from "
            f"{len(self.trajectory_buffer)} trajectories"
        )

        # Select high-reward and low-reward trajectories for comparison
        sorted_trajs = sorted(
            self.trajectory_buffer, key=lambda t: t["reward"], reverse=True
        )
        good_trajs = sorted_trajs[:5]
        bad_trajs = sorted_trajs[-5:]

        # Format trajectories for LLM analysis
        good_summary = self._format_trajectories(good_trajs, "successful")
        bad_summary = self._format_trajectories(bad_trajs, "unsuccessful")

        # LLM generates abstract principles
        distill_prompt = (
            "You are analyzing agent trajectories to extract general strategic principles.\n\n"
            f"### Successful Trajectories:\n{good_summary}\n\n"
            f"### Unsuccessful Trajectories:\n{bad_summary}\n\n"
            "### Task:\n"
            "Extract 3-5 abstract, reusable strategic principles that explain "
            "what distinguishes successful from unsuccessful behavior. "
            "Each principle should be general (not specific to one case) and actionable.\n\n"
            "### Principles (one per line, starting with -):\n"
        )

        result = self._generate(distill_prompt, max_new_tokens=300, temperature=0.5)

        # Parse principles
        new_principles = []
        for line in result.strip().split("\n"):
            line = line.strip()
            if line.startswith("-") or line.startswith("*"):
                principle = line.lstrip("-*").strip()
                if len(principle) > 10:
                    new_principles.append(principle)

        if new_principles:
            self.principles.extend(new_principles)
            # Deduplicate and trim
            self.principles = list(dict.fromkeys(self.principles))
            if len(self.principles) > self.max_principles:
                self.principles = self.principles[-self.max_principles:]

            logger.info(
                f"[EvolveR] Extracted {len(new_principles)} new principles, "
                f"total: {len(self.principles)}"
            )

        # Clear processed trajectories
        self.trajectory_buffer = []

    def _format_trajectories(self, trajectories: list[dict], label: str) -> str:
        """Format trajectories for LLM analysis."""
        parts = []
        for i, traj in enumerate(trajectories[:3]):
            parts.append(f"Trajectory {i+1} ({label}, reward={traj['reward']:.2f}):")
            for step in traj["trajectory"][:5]:
                parts.append(
                    f"  Event: {step['event'][:80]} → "
                    f"Action: {step['action']} {step.get('content', '')[:50]}"
                )
        return "\n".join(parts)

    def _retrieve_principles(self, query: str, top_k: int = 5) -> list[str]:
        """Retrieve relevant principles for a query.

        Simple keyword-based retrieval. In full EvolveR, this would
        use embedding-based retrieval.
        """
        if not self.principles:
            return []

        # Score by keyword overlap
        query_words = set(query.lower().split())
        scored = []
        for p in self.principles:
            p_words = set(p.lower().split())
            overlap = len(query_words & p_words)
            scored.append((overlap, p))

        scored.sort(reverse=True)
        return [p for _, p in scored[:top_k] if _ > 0] or self.principles[:top_k]

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

    def _execute_operation(self, operation: dict, event: str):
        """Execute memory operation on simple store."""
        op_type = operation["type"]
        content = operation.get("content", event)

        if op_type == "ADD" and content:
            self.memories.append(content)
            if len(self.memories) > self.max_memories:
                self.memories.pop(0)
        elif op_type == "UPDATE" and content and "->" in content:
            parts = content.split("->")
            old_part = parts[0].strip().lower()
            new_part = parts[1].strip()
            for i, m in enumerate(self.memories):
                if old_part in m.lower():
                    self.memories[i] = new_part
                    break
        elif op_type == "DELETE" and content:
            target = content.lower()
            self.memories = [m for m in self.memories if target not in m.lower()]

    def save(self, path: str):
        super().save(path)
        with open(os.path.join(path, "evolver_state.json"), "w") as f:
            json.dump({
                "principles": self.principles,
                "memories": self.memories,
                "num_trajectories": len(self.trajectory_buffer),
            }, f, indent=2, ensure_ascii=False)

    def load(self, path: str):
        fp = os.path.join(path, "evolver_state.json")
        if os.path.exists(fp):
            with open(fp) as f:
                data = json.load(f)
            self.principles = data.get("principles", [])
            self.memories = data.get("memories", [])
