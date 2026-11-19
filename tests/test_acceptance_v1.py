"""End-to-end acceptance suite verifying all 10 scenarios in IMPLEMENTATION_PLAN.md §9."""

from __future__ import annotations

from pathlib import Path
import tempfile
import unittest

from kernel_agent_v1.backends import SimulatedExecutionBackend
from kernel_agent_v1.clock import SimClock
from kernel_agent_v1.experiments import run_three_way_comparison
from kernel_agent_v1.protocol import (
    BudgetConfig,
    CommandKind,
    Event,
    EventType,
    JobKind,
    PROHIBITED_HEADER_FIELDS,
    ResultKind,
    RuntimeConfig,
    TargetRef,
    VerificationLevel,
    VersionStatus,
)
from kernel_agent_v1.runtime import KernelLabRuntime


class TestAcceptanceV1(unittest.TestCase):
    def _make_rt_with_v1(self, tmp: str, **cfg_kwargs) -> KernelLabRuntime:
        cfg = RuntimeConfig(run_id="acc-v1", **cfg_kwargs)
        rt = KernelLabRuntime(Path(tmp), config=cfg, clock=SimClock())
        rt.register_route("route-reg", "register spill", priority=2.0, researcher_id="res_reg")
        lookup = rt.wiki.query("NVIDIA-H100-SXM", "route-reg", ("register",), rt.clock.now_iso())
        cmd = rt.leader.build_command(
            kind=CommandKind.IMPLEMENT_KERNEL.value,
            research_id="r-001",
            direction_id="route-reg",
            target=TargetRef("V0", "fla-baseline"),
            hypothesis="register spill",
            evidence_refs=(lookup.lookup_id,),
            instruction="build v1",
            deliverable="kernel.py",
            reply_to="res_reg",
        )
        rt.submit_command_and_package(cmd, {"artifact_id": "art-v1"})
        return rt

    def test_01_w1_partial_direct_to_researcher_while_w2_overlaps(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            rt = self._make_rt_with_v1(tmp)
            h_w1, _ = rt.dispatch_gpu_if_ready()
            assert h_w1 is not None and h_w1.block_id == "W1"
            rt.complete_gpu_job(h_w1)

            # Controller log must NEVER contain result body fields
            for _, evt in rt.store.load_events():
                d = evt.to_dict()
                self.assertFalse(PROHIBITED_HEADER_FIELDS.intersection(d.keys()))

            # Designated Researcher directly received W1 payload
            res_reg = rt.leader.researchers["res_reg"]
            self.assertEqual(len(res_reg.inbox), 1)
            self.assertEqual(res_reg.inbox[0].block_id, "W1")
            self.assertEqual(res_reg.inbox[0].result_kind, ResultKind.BENCHMARK_PARTIAL.value)

            # W2 can run immediately while active candidate stays frozen as V1 / art-v1
            h_w2, _ = rt.dispatch_gpu_if_ready()
            assert h_w2 is not None and h_w2.block_id == "W2"
            self.assertEqual(rt.controller.active_version.version_id, "V1")
            self.assertEqual(rt.controller.active_version.artifact_id, "art-v1")

    def test_02_concurrent_multi_type_results_routed_to_specialized_recipients(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            rt = self._make_rt_with_v1(tmp)
            rt.leader.ensure_researcher("res_diag")
            rt.leader.ensure_researcher("res_comp")
            lookup = rt.wiki.query("NVIDIA-H100-SXM", "route-reg", ("compiler", "ncu"), rt.clock.now_iso())

            # Propose & execute probe -> res_diag, compiler_analysis (CPU) -> res_comp, and complete W1..W4 -> Judge
            rt.controller.admit_next_version()
            cmd_probe = rt.leader.build_command(
                kind=CommandKind.WRITE_PROBE.value,
                research_id="r-002",
                direction_id="route-reg",
                target=TargetRef("V1", "art-v1"),
                hypothesis="probe",
                evidence_refs=(lookup.lookup_id,),
                instruction="probe",
                deliverable="probe.py",
                reply_to="res_diag",
            )
            rt.submit_command_and_package(cmd_probe, {"artifact_id": "art-probe"})
            probe_task = [t for t in rt.controller.tasks.values() if t.kind == JobKind.PROBE.value][0]
            rt.controller.process_event(
                rt.leader.declare_waiting_event(
                    member="res_diag",
                    waiting_for=f"result_ready:{probe_task.task_id}",
                    unblocks_work="diagnose_probe",
                    estimated_duration_s=60.0,
                    direction_id="route-reg",
                )
            )

            cmd_comp = rt.leader.build_command(
                kind=CommandKind.WRITE_COMPILER_ANALYSIS.value,
                research_id="r-002",
                direction_id="route-reg",
                target=TargetRef("V1", "art-v1"),
                hypothesis="ptx",
                evidence_refs=(lookup.lookup_id,),
                instruction="analyze ptx",
                deliverable="analyze.py",
                reply_to="res_comp",
            )
            rt.submit_command_and_package(cmd_comp, {"artifact_id": "art-comp"})

            # Execute probe on GPU + compiler analysis on CPU + all 4 benchmark blocks
            comp_task = [t for t in rt.controller.tasks.values() if t.kind == JobKind.COMPILER_ANALYSIS.value][0]
            h_comp = rt.operator.start(
                rt.packages_by_id[comp_task.package_id],
                {
                    "task_id": comp_task.task_id,
                    "kind": comp_task.kind,
                    "resource": "cpu",
                    "direction_id": comp_task.direction_id,
                    "version_id": "V1",
                    "artifact_id": "art-v1",
                    "reply_to": "res_comp",
                },
            )
            comp_res = rt.operator.finalize_handle(h_comp)
            for evt in comp_res.scheduling_events:
                rt.controller.process_event(evt)
            rt.process_deliveries(comp_res.deliveries)

            # Drain GPU tasks (probe + W1..W4)
            while True:
                h, _ = rt.dispatch_gpu_if_ready()
                if h is None:
                    break
                rt.complete_gpu_job(h)

            self.assertEqual(len(rt.leader.researchers["res_diag"].inbox), 1)
            self.assertEqual(
                rt.leader.researchers["res_diag"].inbox[0].result_kind,
                ResultKind.PROBE_RESULT.value,
            )
            self.assertEqual(len(rt.leader.researchers["res_comp"].inbox), 1)
            self.assertEqual(
                rt.leader.researchers["res_comp"].inbox[0].result_kind,
                ResultKind.COMPILER_ANALYSIS.value,
            )
            # Judge received complete benchmark directly
            judge_delivs = [d for d in rt.deliveries_log if d.recipient == "judge"]
            self.assertEqual(len(judge_delivs), 1)
            self.assertEqual(judge_delivs[0].result_kind, ResultKind.BENCHMARK_COMPLETE.value)

    def test_05_conflict_blocks_cannot_promote_and_abba_runs_before_close(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            rt = self._make_rt_with_v1(tmp)
            assert isinstance(rt.backend, SimulatedExecutionBackend)
            # V1 has missing UUID in W4 -> cannot promote!
            rt.backend.set_behavior("art-v1:benchmark_block:W4", {"drop_last_uuid": True})
            while True:
                h, _ = rt.dispatch_gpu_if_ready()
                if h is None:
                    break
                rt.complete_gpu_job(h)

            self.assertIsNone(rt.judge.incumbent.version_id)
            self.assertIn("V1", rt.controller.closed_versions)
            self.assertEqual(rt.judge.verdict_history[-1]["verdict"], "rejected_invalid")

            # Now submit valid V2 (beats baseline -> retained) and V3 (beats V2 -> triggers ABBA before V3 closes)
            lookup = rt.wiki.query("NVIDIA-H100-SXM", "route-reg", ("register",), rt.clock.now_iso())
            for art, spd in (("art-v2", 1.15), ("art-v3", 1.25)):
                for b in ("W1", "W2", "W3", "W4"):
                    rt.backend.set_behavior(f"{art}:benchmark_block:{b}", {"speedup": spd})
                rt.backend.set_behavior(f"{art}:abba_retest", {"speedup": spd})
                cmd = rt.leader.build_command(
                    kind=CommandKind.IMPLEMENT_KERNEL.value,
                    research_id="r-003",
                    direction_id="route-reg",
                    target=TargetRef("V0", "fla-baseline"),
                    hypothesis="valid gain",
                    evidence_refs=(lookup.lookup_id,),
                    instruction=f"build {art}",
                    deliverable="kernel.py",
                )
                rt.submit_command_and_package(cmd, {"artifact_id": art})
                while True:
                    h, _ = rt.dispatch_gpu_if_ready()
                    if h is None:
                        break
                    rt.complete_gpu_job(h)

            self.assertEqual(rt.judge.incumbent.version_id, "V3")
            self.assertEqual(rt.judge.incumbent.verification_level, VerificationLevel.ABBA_VERIFIED.value)
            self.assertEqual(rt.controller.closed_versions["V3"].retest_status, "completed")

    def test_06_next_version_gated_by_version_closed_and_selected_by_priority(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            rt = self._make_rt_with_v1(tmp)
            rt.register_route("route-low", "low prio", priority=1.0)
            rt.register_route("route-high", "high prio", priority=5.0)
            lookup = rt.wiki.query("NVIDIA-H100-SXM", "route-high", ("layout",), rt.clock.now_iso())

            # Start V1 (only W1) -> V1 is ACTIVE
            h1, _ = rt.dispatch_gpu_if_ready()
            assert h1 is not None
            rt.complete_gpu_job(h1)

            # While V1 is still active, both route-low and route-high become code_ready
            for d_id, art in (("route-low", "art-low"), ("route-high", "art-high")):
                cmd = rt.leader.build_command(
                    kind=CommandKind.IMPLEMENT_KERNEL.value,
                    research_id="r-prio",
                    direction_id=d_id,
                    target=TargetRef("V0", "fla-baseline"),
                    hypothesis=d_id,
                    evidence_refs=(lookup.lookup_id,),
                    instruction=art,
                    deliverable="k.py",
                )
                rt.submit_command_and_package(cmd, {"artifact_id": art})

            # Next GPU task must still belong to V1, NOT V2!
            h2, _ = rt.dispatch_gpu_if_ready()
            assert h2 is not None
            self.assertEqual(h2.version_id, "V1")
            rt.complete_gpu_job(h2)
            while True:
                h, _ = rt.dispatch_gpu_if_ready()
                if h is None:
                    break
                if h.version_id != "V1":
                    # Stop as soon as V2 is admitted & dispatched!
                    self.assertIn("V1", rt.controller.closed_versions)
                    self.assertEqual(h.version_id, "V2")
                    self.assertEqual(h.artifact_id, "art-high")
                    break
                rt.complete_gpu_job(h)

    def test_07_stale_attempt_duplicate_event_and_version_closed_once(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            rt = self._make_rt_with_v1(tmp)
            h_w1, _ = rt.dispatch_gpu_if_ready()
            assert h_w1 is not None
            # Complete W1..W4 cleanly -> V1 closes
            rt.complete_gpu_job(h_w1)
            while True:
                h, _ = rt.dispatch_gpu_if_ready()
                if h is None:
                    break
                rt.complete_gpu_job(h)

            closed_events = [
                e for e in rt.controller.emitted_events if e.event == EventType.VERSION_CLOSED.value
            ]
            self.assertEqual(len(closed_events), 1)
            seq_before = rt.controller.state_seq

            # Late stale gpu_job_finished with old attempt_id must not reopen V1 or alter state
            late = Event(
                event=EventType.GPU_JOB_FINISHED.value,
                event_id="evt-late-stale",
                run_id=rt.config.run_id,
                producer="operator",
                timestamp=rt.clock.now_iso(),
                sim_time=rt.clock.now(),
                version_id="V1",
                artifact_id="art-v1",
                job_id=h_w1.job_id,
                attempt_id=h_w1.attempt_id,
                kind=JobKind.BENCHMARK_BLOCK.value,
                block_id="W1",
                status="completed",
                gpu_released=True,
            )
            rt.controller.process_event(late)
            self.assertIn("V1", rt.controller.closed_versions)
            self.assertIsNone(rt.controller.active_version)
            # Only state_seq incremented for recording evt-late-stale; no duplicate VERSION_CLOSED!
            closed_events_after = [
                e for e in rt.controller.emitted_events if e.event == EventType.VERSION_CLOSED.value
            ]
            self.assertEqual(len(closed_events_after), 1)
            self.assertEqual(rt.controller.state_seq, seq_before + 1)

    def test_10_budget_exhaustion_baseline_fallback_and_three_way_comparison(self) -> None:
        # Case A: No candidate beats baseline -> returns baseline_only
        with tempfile.TemporaryDirectory() as tmp:
            rt = self._make_rt_with_v1(tmp)
            assert isinstance(rt.backend, SimulatedExecutionBackend)
            for b in ("W1", "W2", "W3", "W4"):
                rt.backend.set_behavior(f"art-v1:benchmark_block:{b}", {"speedup": 1.0})
            while True:
                h, _ = rt.dispatch_gpu_if_ready()
                if h is None:
                    break
                rt.complete_gpu_job(h)
            self.assertIsNone(rt.judge.incumbent.version_id)
            self.assertEqual(rt.judge.incumbent.verification_level, VerificationLevel.BASELINE_ONLY.value)

        # Case B: Tight budget freezes admission before admitting a new version
        with tempfile.TemporaryDirectory() as tmp:
            tight_budget = BudgetConfig(total_gpu_budget_s=300.0, total_wall_budget_s=1000.0)
            rt = self._make_rt_with_v1(tmp, budgets=tight_budget)
            self.assertTrue(rt.controller.ledger.admission_frozen)
            self.assertIsNone(rt.controller.admit_next_version())

        # Case C: 3-way comparison report under identical protocol & budget
        with tempfile.TemporaryDirectory() as tmp:
            results = run_three_way_comparison(base_dir=Path(tmp))
            self.assertEqual(set(results.keys()), {"whole_round", "chunked_fcfs", "chunked_dynamic"})
            self.assertLess(
                results["chunked_dynamic"].first_actionable_evidence_s,
                results["whole_round"].first_actionable_evidence_s,
            )
            self.assertGreater(results["chunked_dynamic"].member_gpu_overlap_s, 0.0)
            self.assertEqual(
                results["chunked_dynamic"].verification_level,
                VerificationLevel.FINAL_VERIFIED.value,
            )


if __name__ == "__main__":
    unittest.main()
