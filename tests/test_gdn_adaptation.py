"""Tests for adapted official GDN assets, stratified workblocks, GDN ABI validation, and cli run."""

from __future__ import annotations

from pathlib import Path
import tempfile
import unittest

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
from kernel_agent_v1.protocol import CommandKind, EventType, TargetRef
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


if __name__ == "__main__":
    unittest.main()
