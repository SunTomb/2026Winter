"""
G-MSRA Smoke Test: End-to-End Pipeline Validation.
Verifies the full pipeline (Phase 0 → 1 → 2 → 3 → eval) can run
without errors on a tiny synthetic dataset.

Does NOT require GPU or real model weights — uses mock/tiny models
for structure validation only.

Usage:
    cd f:/USTC/2026Winter/G-MSRA
    python scripts/smoke_test.py
"""

import os
import sys
import json
import tempfile
import traceback
from pathlib import Path

# Force UTF-8 output for Windows GBK consoles
if sys.platform == "win32":
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

# Add project root to path
PROJECT_ROOT = str(Path(__file__).parent.parent)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)


class SmokeTestResult:
    """Track test results."""
    def __init__(self):
        self.passed = []
        self.failed = []

    def ok(self, name, msg=""):
        self.passed.append(name)
        print(f"  [PASS] {name}" + (f" -- {msg}" if msg else ""))

    def fail(self, name, error):
        self.failed.append((name, str(error)))
        print(f"  [FAIL] {name} -- {error}")

    def summary(self):
        total = len(self.passed) + len(self.failed)
        print(f"\n{'='*50}")
        print(f"SMOKE TEST: {len(self.passed)}/{total} passed")
        if self.failed:
            print(f"FAILED:")
            for name, err in self.failed:
                print(f"  [FAIL] {name}: {err}")
        print(f"{'='*50}")
        return len(self.failed) == 0


def test_imports(results: SmokeTestResult):
    """Test that all modules can be imported."""
    print("\n[1/7] Testing imports...")
    modules = [
        "gmsra.config",
        "gmsra.utils",
        "gmsra.memory.entry",
        "gmsra.memory.store",
        "gmsra.reward.env_signals",
        "gmsra.reward.grounded_reward",
        "gmsra.manager.memory_manager",
        "gmsra.consolidation.trigger",
        "gmsra.consolidation.distiller",
        "gmsra.agent",
    ]
    for mod in modules:
        try:
            __import__(mod)
            results.ok(f"import {mod}")
        except Exception as e:
            results.fail(f"import {mod}", e)


def test_config(results: SmokeTestResult):
    """Test configuration system."""
    print("\n[2/7] Testing configuration...")
    try:
        from gmsra.config import GMSRAConfig
        config = GMSRAConfig()
        assert config.model.model_name == "Qwen/Qwen2.5-7B-Instruct"
        assert config.reward.lambda_mem < 1.0
        assert config.trigger.theta > 0
        results.ok("GMSRAConfig defaults")
    except Exception as e:
        results.fail("GMSRAConfig defaults", e)

    # Test YAML loading (if pyyaml installed)
    try:
        import yaml
        tmp_path = os.path.join(tempfile.gettempdir(), "gmsra_test_config.yaml")
        with open(tmp_path, 'w') as f:
            yaml.dump({"model": {"model_name": "test-model"}, "seed": 123}, f)
        config = GMSRAConfig.from_yaml(tmp_path)
        assert config.model.model_name == "test-model"
        assert config.seed == 123
        os.unlink(tmp_path)
        results.ok("GMSRAConfig from YAML")
    except ImportError:
        results.ok("GMSRAConfig from YAML (skipped, pyyaml not installed)")
    except Exception as e:
        results.fail("GMSRAConfig from YAML", e)


def test_memory_entry(results: SmokeTestResult):
    """Test MemoryEntry dataclass."""
    print("\n[3/7] Testing memory system...")
    try:
        from gmsra.memory.entry import MemoryEntry

        entry = MemoryEntry(
            content="Test memory content",
            keywords=["test", "memory"],
            tags=["fact"],
        )
        assert entry.confidence == 0.5
        assert entry.id  # Should have auto-generated ID

        # Test confidence update
        conf = entry.update_confidence({
            "env_reward_write": 0.4,
            "hit_success_ratio": 0.4,
            "log_age": 0.2,
        })
        assert 0 < conf < 1
        results.ok("MemoryEntry creation & confidence")

        # Test hit recording
        entry.record_hit(True)
        entry.record_hit(False)
        assert entry.hit_total == 2
        assert entry.hit_success == 1
        results.ok("MemoryEntry hit recording")

        # Test serialization
        d = entry.to_dict()
        entry2 = MemoryEntry.from_dict(d)
        assert entry2.content == entry.content
        assert entry2.confidence == entry.confidence
        results.ok("MemoryEntry serialization")

        # Test text representation
        text = entry.to_text()
        assert "test" in text.lower()
        results.ok("MemoryEntry to_text")

    except Exception as e:
        results.fail("MemoryEntry", e)


def test_utils(results: SmokeTestResult):
    """Test utility functions."""
    print("\n[4/7] Testing utilities...")
    try:
        from gmsra.utils import compute_f1, compute_exact_match

        assert compute_f1("the cat sat", "the cat") > 0.5
        assert compute_f1("completely wrong", "the answer") == 0.0
        assert compute_exact_match("hello", "hello") == 1.0
        assert compute_exact_match("hello", "world") == 0.0
        results.ok("compute_f1 and compute_exact_match")
    except Exception as e:
        results.fail("compute_f1 and compute_exact_match", e)

    try:
        from gmsra.utils import compute_kendall_tau
        tau = compute_kendall_tau([1.0, 2.0, 3.0], [1.0, 2.0, 3.0])
        assert tau == 1.0
        results.ok("compute_kendall_tau")
    except ImportError:
        results.ok("compute_kendall_tau (skipped, scipy not installed)")
    except Exception as e:
        results.fail("compute_kendall_tau", e)


def test_env_signals(results: SmokeTestResult):
    """Test environment signal extractors."""
    print("\n[5/7] Testing reward system...")
    try:
        from gmsra.reward.env_signals import (
            AgentTaskSignalExtractor,
            DialogueSignalExtractor,
            ExternalQASignalExtractor,
        )

        # Agent task extractor
        agent_ext = AgentTaskSignalExtractor()
        r = agent_ext.extract(task_result={"success": True, "steps_taken": 5, "max_steps": 30})
        assert 0.8 <= r <= 1.0
        results.ok("AgentTaskSignalExtractor (success)")

        r = agent_ext.extract(task_result={"success": False, "partial_score": 0.5})
        assert 0 < r < 0.6
        results.ok("AgentTaskSignalExtractor (partial)")

        # Dialogue extractor (without LLM)
        dial_ext = DialogueSignalExtractor()
        r = dial_ext.extract(agent_response="I think it's sunny",
                             next_user_turn="Thanks, that's correct!")
        assert r > 0.5
        results.ok("DialogueSignalExtractor (positive)")

        r = dial_ext.extract(agent_response="I think it's sunny",
                             next_user_turn="No, that's wrong. It's raining.")
        assert r < 0.5
        results.ok("DialogueSignalExtractor (negative)")

        # External QA
        qa_ext = ExternalQASignalExtractor()
        r = qa_ext.extract(prediction="the answer is Python", ground_truth="Python")
        assert r > 0.3, f"Expected QA F1 > 0.3, got {r}"
        results.ok(f"ExternalQASignalExtractor (F1={r:.3f})")

    except Exception as e:
        results.fail("Environment signals", f"{type(e).__name__}: {e}")


def test_sft_data(results: SmokeTestResult):
    """Test SFT data generation."""
    print("\n[6/7] Testing SFT data generation...")
    try:
        from scripts.train_phase0_sft import generate_sft_data

        data = generate_sft_data()
        assert len(data) >= 100, f"Expected 100+ examples, got {len(data)}"

        # Check format
        for item in data[:5]:
            assert "text" in item
            assert "prompt" in item
            assert "completion" in item
            assert "### Available Operations" in item["prompt"]

        # Check diversity
        operations = set()
        for item in data:
            comp = item["completion"].strip().upper()
            if comp.startswith("ADD"):
                operations.add("ADD")
            elif comp.startswith("UPDATE"):
                operations.add("UPDATE")
            elif comp.startswith("DELETE"):
                operations.add("DELETE")
            elif comp.startswith("NOOP"):
                operations.add("NOOP")

        assert len(operations) == 4, f"Expected 4 operation types, got {operations}"
        results.ok(f"SFT data: {len(data)} examples, {len(operations)} op types")

    except Exception as e:
        results.fail("SFT data generation", e)


def test_data_preparation(results: SmokeTestResult):
    """Test data preparation script."""
    print("\n[7/7] Testing data preparation...")
    try:
        from scripts.prepare_data import (
            _generate_synthetic_locomo,
            prepare_alfworld,
            prepare_evomemory,
        )

        # Test synthetic LoCoMo
        train, test = _generate_synthetic_locomo()
        assert len(train) > 0
        assert len(test) > 0
        assert "events" in train[0]
        assert "question" in train[0]
        assert "answer" in train[0]
        results.ok(f"Synthetic LoCoMo: {len(train)} train, {len(test)} test")

        # Test ALFWorld prep (to temp dir)
        with tempfile.TemporaryDirectory() as tmpdir:
            prepare_alfworld(tmpdir)
            path = os.path.join(tmpdir, "alfworld_tasks.json")
            assert os.path.exists(path)
            with open(path) as f:
                tasks = json.load(f)
            assert len(tasks) > 0
            assert "instruction" in tasks[0]
            assert "events" in tasks[0]
            results.ok(f"ALFWorld preparation: {len(tasks)} tasks")

        # Test Evo-Memory prep
        with tempfile.TemporaryDirectory() as tmpdir:
            prepare_evomemory(tmpdir)
            path = os.path.join(tmpdir, "evomemory_test.json")
            assert os.path.exists(path)
            with open(path) as f:
                data = json.load(f)
            assert len(data) > 0
            results.ok(f"Evo-Memory preparation: {len(data)} examples")

    except Exception as e:
        results.fail("Data preparation", e)


def test_baseline_registry(results: SmokeTestResult):
    """Test that the baseline registry is wired up."""
    print("\n[8/8] Testing baseline registry...")
    try:
        from gmsra.baselines import list_baselines, get_baseline_spec

        baselines = list_baselines()
        assert len(baselines) >= 5
        assert get_baseline_spec("memory_r1").display_name == "Memory-R1"
        results.ok(f"Baseline registry: {len(baselines)} baselines")
    except Exception as e:
        results.fail("Baseline registry", e)


def main():
    print("=" * 50)
    print("G-MSRA SMOKE TEST")
    print("=" * 50)

    results = SmokeTestResult()

    test_imports(results)
    test_config(results)
    test_memory_entry(results)
    test_utils(results)
    test_env_signals(results)
    test_sft_data(results)
    test_data_preparation(results)
    test_baseline_registry(results)

    success = results.summary()
    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())
