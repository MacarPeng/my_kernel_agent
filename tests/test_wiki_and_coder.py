"""Tests for KernelWiki, 3-round stagnation rules, Coder constraints & immutability."""

from __future__ import annotations

from pathlib import Path
import tempfile
import unittest

from kernel_agent_v1.clock import SimClock
from kernel_agent_v1.coder import Coder, verify_package_integrity
from kernel_agent_v1.leader import Leader
from kernel_agent_v1.protocol import CommandKind, EventType, RouteStatus, TargetRef
from kernel_agent_v1.wiki import ExperimentRouteTree, HardwareKnowledgePlugin, KernelWiki


class TestWikiAndCoder(unittest.TestCase):
    def test_stagnation_sequence_1_2_0_1_and_pause_only_that_route(self) -> None:
        tree = ExperimentRouteTree(min_gain_threshold=0.01, stagnation_limit=3)
        tree.ensure_route("route-a", "reduce register spill", priority=2.0, reference_score=1.0)
        tree.ensure_route("route-b", "swizzle smem layout", priority=1.0, reference_score=1.0)

        # Partial evidence & compile failures must NOT increment stagnation
        tree.record_reusable_evidence("route-a", "rec-p1", "V1", "ncu_parsed", "case-07", {})
        tree.record_failure_repair("route-a", "compile_failure", "V1", "art-cf", is_repair=True)
        self.assertEqual(tree.routes["route-a"].stagnation_count, 0)
        self.assertEqual(tree.routes["route-a"].compile_failure_count, 1)
        self.assertEqual(tree.routes["route-a"].repair_count, 1)

        # Sequence: no gain (1), no gain (2), significant gain > 0.01 (0), no gain (1)
        tree.record_trusted_verdict("route-a", "V1", "art-1", 1.005, "rejected_no_gain", "rec-1")
        self.assertEqual(tree.routes["route-a"].stagnation_count, 1)

        # Same-artifact retry must NOT increment stagnation
        tree.record_trusted_verdict("route-a", "V1", "art-1", 1.005, "rejected_no_gain", "rec-1-dup")
        self.assertEqual(tree.routes["route-a"].stagnation_count, 1)

        tree.record_trusted_verdict("route-a", "V2", "art-2", 1.008, "rejected_no_gain", "rec-2")
        self.assertEqual(tree.routes["route-a"].stagnation_count, 2)

        # Significant gain resets stagnation_count to 0 and updates reference
        tree.record_trusted_verdict("route-a", "V3", "art-3", 1.050, "promoted", "rec-3")
        self.assertEqual(tree.routes["route-a"].stagnation_count, 0)
        self.assertEqual(tree.routes["route-a"].reference_version_id, "V3")

        tree.record_trusted_verdict("route-a", "V4", "art-4", 1.055, "rejected_no_gain", "rec-4")
        self.assertEqual(tree.routes["route-a"].stagnation_count, 1)

        # Renaming hypothesis does NOT reset stagnation_count
        tree.rename_hypothesis("route-a", "reworded register spill hypothesis")
        self.assertEqual(tree.routes["route-a"].stagnation_count, 1)

        # 2 more no-gain rounds -> reaches 3 -> route-a paused, sibling route-b still active!
        tree.record_trusted_verdict("route-a", "V5", "art-5", 1.052, "rejected_no_gain", "rec-5")
        tree.record_trusted_verdict("route-a", "V6", "art-6", 1.058, "rejected_no_gain", "rec-6")
        self.assertEqual(tree.routes["route-a"].stagnation_count, 3)
        self.assertEqual(tree.routes["route-a"].status, RouteStatus.PAUSED.value)
        self.assertEqual(tree.routes["route-b"].status, RouteStatus.ACTIVE.value)

    def test_wiki_lookup_credential_and_coder_constraints(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            clock = SimClock()
            wiki = KernelWiki(HardwareKnowledgePlugin("NVIDIA-H100-SXM", "cuda-12.6/triton-3.1"))
            leader = Leader("r1", wiki, "NVIDIA-H100-SXM", clock)
            coder = Coder(Path(tmp) / "pkgs", "r1", clock, wiki=wiki)

            # Leader cannot build command without wiki-lookup credential
            with self.assertRaises(ValueError):
                leader.build_command(
                    kind=CommandKind.WRITE_PROBE.value,
                    research_id="r-001",
                    direction_id="reg",
                    target=TargetRef("V1", "art-1", digest="abc123"),
                    hypothesis="spill",
                    evidence_refs=("eval-V1-W1",),
                    instruction="probe",
                    deliverable="probe.py",
                )

            lookup = wiki.query("NVIDIA-H100-SXM", "reg", ("register", "spill"), clock.now_iso())
            self.assertTrue(lookup.lookup_id.startswith("wiki-lookup-"))
            self.assertGreater(len(lookup.hardware_entries), 0)

            cmd = leader.build_command(
                kind=CommandKind.WRITE_PROBE.value,
                research_id="r-001",
                direction_id="reg",
                target=TargetRef("V1", "art-1", digest="abc123"),
                hypothesis="spill",
                evidence_refs=("eval-V1-W1", lookup.lookup_id),
                instruction="probe case-07",
                deliverable="probe.py",
                workload_ids=("case-07",),
                allowed_paths=("probe/",),
                preserve_target=True,
            )

            # Out-of-bounds allowed_paths must be rejected with code_blocked
            evt_bad, pkg_bad = coder.execute_command(
                cmd,
                provided_spec={
                    "entrypoint": "kernel/hack.py",
                    "files": {"kernel/hack.py": "print('illegal')"},
                },
            )
            self.assertEqual(evt_bad.event, EventType.CODE_BLOCKED.value)
            self.assertIsNone(pkg_bad)

            # Target preservation violation must be rejected
            evt_pres, pkg_pres = coder.execute_command(
                cmd,
                provided_spec={
                    "entrypoint": "probe/p.py",
                    "files": {"probe/p.py": "print('ok')"},
                    "target_digest_after": "tampered-digest",
                },
            )
            self.assertEqual(evt_pres.event, EventType.CODE_BLOCKED.value)
            self.assertIsNone(pkg_pres)

            # Compliant package seals immutably with code_ready
            evt_ok, pkg_ok = coder.execute_command(
                cmd,
                provided_spec={
                    "artifact_id": "probe-art-1",
                    "entrypoint": "probe/p.py",
                    "files": {"probe/p.py": "print('ok')"},
                },
            )
            self.assertEqual(evt_ok.event, EventType.CODE_READY.value)
            self.assertIsNotNone(pkg_ok)
            ok, msg = verify_package_integrity(pkg_ok)
            self.assertTrue(ok, msg)


if __name__ == "__main__":
    unittest.main()
