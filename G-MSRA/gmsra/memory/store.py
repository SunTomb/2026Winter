"""
MemoryStore: Zettelkasten-style memory database with FAISS-backed retrieval.
Supports CRUD operations, graph-based linking, confidence filtering,
and subgraph extraction for consolidation.
"""

from __future__ import annotations
import json
import os
import time
from typing import Optional

import numpy as np
from loguru import logger

from gmsra.memory.entry import MemoryEntry
from gmsra.config import MemoryConfig


class MemoryStore:
    """External memory store with FAISS indexing and graph structure.

    Key Design Choices (per G-MSRA proposal):
    - Zettelkasten-style cards with explicit links (inspired by A-MEM)
    - Confidence-weighted retrieval for Judge (prevents noise pollution)
    - Subgraph extraction for consolidation (graph filters WHAT to distill)
    """

    def __init__(self, config: Optional[MemoryConfig] = None):
        self.config = config or MemoryConfig()
        self.entries: dict[str, MemoryEntry] = {}  # id -> MemoryEntry
        self._index = None  # FAISS index (lazy init)
        self._id_list: list[str] = []  # Ordered list of IDs in FAISS index
        self._encoder = None  # Sentence encoder (lazy init)

    # ---- Lazy initialization ----

    def _init_encoder(self):
        """Lazy-load sentence-transformers encoder."""
        if self._encoder is None:
            from sentence_transformers import SentenceTransformer
            self._encoder = SentenceTransformer(self.config.embedding_model)
            logger.info(f"Loaded encoder: {self.config.embedding_model}")

    def _init_faiss_index(self):
        """Lazy-init FAISS index."""
        if self._index is None:
            import faiss
            self._index = faiss.IndexFlatIP(self.config.embedding_dim)  # Inner product
            logger.info(f"Initialized FAISS index (dim={self.config.embedding_dim})")

    def _encode(self, text: str) -> np.ndarray:
        """Encode text to embedding vector."""
        self._init_encoder()
        emb = self._encoder.encode([text], normalize_embeddings=True)
        return emb[0].astype(np.float32)

    # ---- CRUD Operations ----

    def add(self, content: str, env_reward: float = 0.0,
            keywords: list[str] = None, tags: list[str] = None,
            source: str = "") -> MemoryEntry:
        """ADD operation: Create a new memory entry."""
        entry = MemoryEntry(
            content=content,
            keywords=keywords or [],
            tags=tags or [],
            env_reward_at_write=env_reward,
            source=source,
        )
        entry.embedding = self._encode(content).tolist()

        # Add to FAISS index
        self._init_faiss_index()
        self._index.add(np.array([entry.embedding], dtype=np.float32))
        self._id_list.append(entry.id)

        # Auto-link: find similar existing memories and link
        if len(self.entries) > 0:
            similar = self.retrieve(content, topk=3)
            for sim_entry, score in similar:
                if score > 0.7:  # High similarity threshold for auto-linking
                    entry.add_link(sim_entry.id)
                    sim_entry.add_link(entry.id)

        self.entries[entry.id] = entry

        # Check capacity
        if len(self.entries) > self.config.max_entries:
            self._evict_lowest_confidence()

        logger.debug(f"ADD memory [{entry.id}]: {content[:50]}...")
        return entry

    def update(self, entry_id: str, new_content: str,
               env_reward: float = 0.0) -> Optional[MemoryEntry]:
        """UPDATE operation: Modify an existing memory entry."""
        if entry_id not in self.entries:
            logger.warning(f"UPDATE failed: entry {entry_id} not found")
            return None

        entry = self.entries[entry_id]
        entry.content = new_content
        entry.updated_at = time.time()
        entry.env_reward_at_write = max(entry.env_reward_at_write, env_reward)

        # Update embedding in FAISS
        new_emb = self._encode(new_content)
        entry.embedding = new_emb.tolist()
        idx = self._id_list.index(entry_id)
        self._rebuild_index()  # FAISS doesn't support in-place update

        logger.debug(f"UPDATE memory [{entry_id}]: {new_content[:50]}...")
        return entry

    def delete(self, entry_id: str) -> bool:
        """DELETE operation: Remove a memory entry."""
        if entry_id not in self.entries:
            logger.warning(f"DELETE failed: entry {entry_id} not found")
            return False

        entry = self.entries.pop(entry_id)

        # Remove links from other entries
        for linked_id in entry.links:
            if linked_id in self.entries:
                other = self.entries[linked_id]
                other.links = [lid for lid in other.links if lid != entry_id]

        # Rebuild FAISS index without this entry
        self._id_list = [eid for eid in self._id_list if eid != entry_id]
        self._rebuild_index()

        logger.debug(f"DELETE memory [{entry_id}]")
        return True

    # ---- Retrieval ----

    def retrieve(self, query: str, topk: int = None) -> list[tuple[MemoryEntry, float]]:
        """Retrieve top-K relevant memories by semantic similarity."""
        topk = topk or self.config.retrieval_topk
        if len(self.entries) == 0:
            return []

        self._init_faiss_index()
        query_emb = self._encode(query).reshape(1, -1)
        k = min(topk, len(self._id_list))
        scores, indices = self._index.search(query_emb, k)

        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < 0 or idx >= len(self._id_list):
                continue
            entry_id = self._id_list[idx]
            if entry_id in self.entries:
                results.append((self.entries[entry_id], float(score)))

        return results

    def retrieve_confident(self, query: str,
                           topk: int = None) -> list[tuple[MemoryEntry, float]]:
        """Retrieve top-K memories, filtered by confidence score.
        Used by the Judge to prevent memory noise pollution.
        """
        topk = topk or self.config.confidence_topk
        # Retrieve more candidates, then filter by confidence
        candidates = self.retrieve(query, topk=topk * 3)

        # Re-rank by confidence * similarity
        scored = []
        for entry, sim_score in candidates:
            combined = entry.confidence * 0.5 + sim_score * 0.5
            scored.append((entry, combined))

        scored.sort(key=lambda x: x[1], reverse=True)
        return scored[:topk]

    # ---- Graph Operations ----

    def get_linked_entries(self, entry_id: str,
                          depth: int = 1) -> list[MemoryEntry]:
        """Get all entries linked to the given entry up to a certain depth."""
        if entry_id not in self.entries:
            return []

        visited = set()
        queue = [(entry_id, 0)]
        result = []

        while queue:
            current_id, d = queue.pop(0)
            if current_id in visited or d > depth:
                continue
            visited.add(current_id)
            if current_id != entry_id and current_id in self.entries:
                result.append(self.entries[current_id])
            if d < depth:
                entry = self.entries.get(current_id)
                if entry:
                    for linked_id in entry.links:
                        queue.append((linked_id, d + 1))

        return result

    def extract_high_frequency_subgraph(self,
                                        min_links: int = 2,
                                        min_confidence: float = 0.5
                                        ) -> list[MemoryEntry]:
        """Extract high-frequency subgraph for consolidation.
        Returns memories with many links AND high confidence.
        These are the "crystallized knowledge" worth distilling to LoRA.
        """
        candidates = [
            entry for entry in self.entries.values()
            if len(entry.links) >= min_links and entry.confidence >= min_confidence
        ]
        # Sort by (links × confidence) — most connected & trustworthy first
        candidates.sort(key=lambda e: len(e.links) * e.confidence, reverse=True)
        return candidates

    # ---- Bulk Operations ----

    def recalibrate_confidence(self, weights: dict[str, float] = None):
        """Recalibrate all memory confidence scores."""
        weights = weights or self.config.confidence_weights
        for entry in self.entries.values():
            entry.update_confidence(weights)
        logger.info(f"Recalibrated confidence for {len(self.entries)} memories")

    def get_growth_rate(self, window_entries: int = 50) -> float:
        """Compute memory growth rate (entries added in recent window / total)."""
        if len(self.entries) == 0:
            return 0.0
        sorted_entries = sorted(self.entries.values(),
                                key=lambda e: e.created_at, reverse=True)
        recent = sorted_entries[:window_entries]
        if len(recent) < 2:
            return 0.0
        time_span = recent[0].created_at - recent[-1].created_at
        if time_span <= 0:
            return 0.0
        return len(recent) / (time_span / 3600)  # Entries per hour

    def size(self) -> int:
        """Number of memories in the store."""
        return len(self.entries)

    # ---- Persistence ----

    def save(self, path: str):
        """Save memory store to JSON."""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        data = [entry.to_dict() for entry in self.entries.values()]
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        logger.info(f"Saved {len(data)} memories to {path}")

    def load(self, path: str):
        """Load memory store from JSON."""
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        self.entries = {}
        for item in data:
            entry = MemoryEntry.from_dict(item)
            entry.embedding = self._encode(entry.content).tolist()
            self.entries[entry.id] = entry
        self._rebuild_index()
        logger.info(f"Loaded {len(self.entries)} memories from {path}")

    # ---- Internal ----

    def _evict_lowest_confidence(self):
        """Remove the memory with the lowest confidence score."""
        if not self.entries:
            return
        worst = min(self.entries.values(), key=lambda e: e.confidence)
        self.delete(worst.id)
        logger.debug(f"Evicted lowest-confidence memory [{worst.id}]")

    def _rebuild_index(self):
        """Rebuild the FAISS index from current entries."""
        import faiss
        self._index = faiss.IndexFlatIP(self.config.embedding_dim)
        new_id_list = []
        embeddings = []
        for entry_id in self._id_list:
            if entry_id in self.entries:
                entry = self.entries[entry_id]
                if entry.embedding:
                    embeddings.append(entry.embedding)
                    new_id_list.append(entry_id)
        self._id_list = new_id_list
        if embeddings:
            self._index.add(np.array(embeddings, dtype=np.float32))

    def get_all_as_text(self, max_entries: int = 20) -> str:
        """Get all memories as formatted text (for prompts)."""
        sorted_entries = sorted(self.entries.values(),
                                key=lambda e: e.confidence, reverse=True)
        texts = [e.to_text() for e in sorted_entries[:max_entries]]
        return "\n\n".join(texts)
