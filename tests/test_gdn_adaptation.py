"""Tests for adapted official GDN assets, stratified workblocks, GDN ABI validation, and cli run."""

from __future__ import annotations

from contextlib import redirect_stderr, redirect_stdout
from io import StringIO
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from kernel_agent_v1.backends import SimulatedExecutionBackend
from kernel_agent_v1.cli import main as cli_main
from kernel_agent_v1.clock import SimClock
from kernel_agent_v1.coder import Coder
from kernel_agent_v1.evaluator import TrustedEvaluator
from kernel_agent_v1.gdn_task import (
    load_official_definition,
    load_official_workloads,
    seed_gdn_candidate_source,
    stratified_split_workblocks,
    validate_gdn_candidate_sources,
)
from kernel_agent_v1.leader import Leader
from kernel_agent_v1.llm import ScriptedDeepSeekClient
from kernel_agent_v1.protocol import CommandKind, EventType, TargetRef
from kernel_agent_v1.store import RunStore
from kernel_agent_v1.wiki import HardwareKnowledgePlugin, KernelWiki


class TestGdnAdaptation(unittest.TestCase):
    def test_official_100_workloads_and_stratified_blocks(self) -> None:
        defn = load_official_definition()
        self.assertEqual(defn["name"], "gdn_prefill_qk4_v8_d128_k_last")

        rows = load_official_workloads()
        self.assertEqual(len(rows), 100)
        blocks = stratified_split_workblocks(rows)
        self.assertEqual(set(blocks.keys()), {"W1", "W2", "W3", "W4"})
        for b, uids in blocks.items():
            self.assertEqual(len(uids), 25, b)
        all_uids = [u for uids in blocks.values() for u in uids]
        self.assertEqual(len(set(all_uids)), 100)

        ev = TrustedEvaluator.from_official_gdn_dataset(device_id="NVIDIA-L4:0")
        self.assertEqual(len(ev.uuids), 100)
        self.assertEqual(ev.blocks["W1"], blocks["W1"])

    def test_gdn_abi_validator_and_coder_enforcement(self) -> None:
        # Seed source passes strict GDN ABI check
        seed_src = seed_gdn_candidate_source()
        self.assertEqual(validate_gdn_candidate_sources({"kernel.py": seed_src}), [])

        # Wrong signature fails
        bad_sig = "import torch\ndef run(q, k, v):\n    return q, k\n"
        errs = validate_gdn_candidate_sources({"kernel.py": bad_sig})
        self.assertTrue(any("ABI" in e for e in errs))

        # Forbidden import fails
        bad_imp = "import os\ndef run(q,k,v,state,A_log,a,dt_bias,b,cu_seqlens,scale):\n    return q, state\n"
        errs2 = validate_gdn_candidate_sources({"kernel.py": bad_imp})
        self.assertTrue(any("disallowed import" in e for e in errs2))

        # Coder with enforce_gdn_abi=True blocks invalid kernel with code_blocked
        with tempfile.TemporaryDirectory() as tmp:
            clock = SimClock()
            wiki = KernelWiki(HardwareKnowledgePlugin("NVIDIA-L4", "cuda-13.0/triton-3.8"))
            leader = Leader("r-gdn", wiki, "NVIDIA-L4", clock)
            coder = Coder(Path(tmp), "r-gdn", clock, wiki=wiki, enforce_gdn_abi=True)
            lookup = wiki.query("NVIDIA-L4", "gdn", ("l4", "ada", "gdn"), clock.now_iso())
            self.assertTrue(any(e.gpu_arch == "NVIDIA-L4" for e in lookup.hardware_entries))

            cmd = leader.build_command(
                kind=CommandKind.IMPLEMENT_KERNEL.value,
                research_id="r-001",
                direction_id="gdn",
                target=TargetRef("V0", "fla-baseline"),
                hypothesis="test",
                evidence_refs=(lookup.lookup_id,),
                instruction="test",
                deliverable="kernel.py",
                allowed_paths=("kernel.py",),
                preserve_target=False,
            )
            evt_bad, pkg_bad = coder.execute_command(
                cmd, {"entrypoint": "kernel.py", "files": {"kernel.py": bad_sig}}
            )
            self.assertEqual(evt_bad.event, EventType.CODE_BLOCKED.value)
            self.assertIsNone(pkg_bad)

            evt_ok, pkg_ok = coder.execute_command(
                cmd, {"entrypoint": "kernel.py", "files": {"kernel.py": seed_src}}
            )
            self.assertEqual(evt_ok.event, EventType.CODE_READY.value)
            self.assertIsNotNone(pkg_ok)

    def test_cli_run_sim_dry_run_with_official_gdn_dataset(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            rc = cli_main(
                [
                    "run",
                    "--backend",
                    "sim",
                    "--gpu",
                    "NVIDIA-L4",
                    "--run-dir",
                    tmp,
                ]
            )
            self.assertEqual(rc, 0)

    def test_real_gpu_cli_reserves_time_for_three_long_final_blocks(self) -> None:
        class TimedBackend(SimulatedExecutionBackend):
            async def async_wait(self, handle, clock, poll_interval_s=0.01):
                clock.advance_to(min(handle.expected_finish_at, handle.timeout_at))
                if handle.expected_finish_at > handle.timeout_at:
                    handle.metadata["behavior"]["timeout"] = True
                return self.wait(handle, clock)

        for timeout_override, final_timeout in ((None, 3600), (3000, 3000), (2000, 2000)):
            with self.subTest(final_timeout=final_timeout), tempfile.TemporaryDirectory() as tmp:
                evaluator = TrustedEvaluator.from_official_gdn_dataset()
                backend = TimedBackend(evaluator)
                backend.set_behavior("benchmark_block", {"duration_s": 115.0, "speedup": 1.2})
                backend.set_behavior("final_eval", {"duration_s": 2500.0, "speedup": 1.2})
                args = ["run", "--backend", "real-gpu", "--run-dir", tmp]
                if timeout_override is not None:
                    args += ["--final-eval-timeout-s", str(timeout_override)]
                stdout = StringIO()
                with patch("kernel_agent_v1.cli.RealClock", return_value=SimClock()), \
                     patch("kernel_agent_v1.cli.RealSingleGpuBackend", return_value=backend), \
                     redirect_stdout(stdout), redirect_stderr(StringIO()):
                    rc = cli_main(args)

                self.assertEqual(rc, 0)
                metrics = json.loads(stdout.getvalue())
                if final_timeout >= 2500:
                    self.assertEqual(metrics["verification_level"], "final_verified")
                    self.assertEqual(metrics["stop_reason"], "completed")
                else:
                    self.assertEqual(metrics["verification_level"], "block_verified")
                    self.assertEqual(metrics["stop_reason"], "final_eval_incomplete")
                self.assertEqual(metrics["total_wall_s"], 4 * 115 + 3 * min(final_timeout, 2500))
                tasks = RunStore(tmp).load_snapshot()["tasks"].values()
                blocks = {t["block_id"]: t["timeout_s"] for t in tasks if t["kind"] == "final_eval"}
                self.assertEqual(blocks, {"F1": final_timeout, "F2": final_timeout, "F3": final_timeout})
                self.assertEqual({t["timeout_s"] for t in tasks if t["kind"] == "benchmark_block"}, {180.0})

    def test_invalid_model_replies_allow_seed_evaluation_to_finish(self) -> None:
        plan = {"allowed_paths": ["kernel.py"], "preserve_target": False}
        for replies, message in (
            (["   ", "not JSON"], "[model-error] stage=researcher"),
            ([plan, "null", "{"], "[coder-blocked]"),
        ):
            with self.subTest(message=message), tempfile.TemporaryDirectory() as tmp:
                pending_replies = iter(replies)
                client = ScriptedDeepSeekClient(lambda messages: next(pending_replies))
                stdout, stderr = StringIO(), StringIO()
                with patch("kernel_agent_v1.cli.DeepSeekClient", return_value=client), \
                     redirect_stdout(stdout), redirect_stderr(stderr):
                    rc = cli_main(["run", "--deepseek-live", "--run-dir", tmp])
                self.assertEqual(rc, 0)
                self.assertIn(message, stderr.getvalue())
                self.assertEqual(len(client.calls), len(replies))
                metrics = json.loads(stdout.getvalue())
                self.assertEqual(metrics["versions_closed"], 1)
                self.assertEqual(metrics["stop_reason"], "completed")
                ledger = RunStore(tmp).load_snapshot()["ledger"]
                self.assertAlmostEqual(ledger["model_cost_used"], 0.0005 * len(replies))


if __name__ == "__main__":
    unittest.main()
