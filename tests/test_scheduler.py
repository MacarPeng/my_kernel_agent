"""Tests for Controller dynamic horizon-H scoring, sensitivity check, tie-break & DurationModel."""

from __future__ import annotations

from pathlib import Path
import tempfile
import unittest

from kernel_agent_v1.clock import SimClock
from kernel_agent_v1.controller import Controller
from kernel_agent_v1.duration import DurationModel
from kernel_agent_v1.protocol import Event, EventType, JobKind, RuntimeConfig, TaskStatus
from kernel_agent_v1.store import RunStore


class TestScheduler(unittest.TestCase):
    def _setup_w4_vs_probe(self, tmp: str) -> tuple[Controller, SimClock]:
        clock = SimClock()
        cfg = RuntimeConfig(run_id="sched-test", horizon_h_s=100.0)
        ctrl = Controller(cfg, RunStore(Path(tmp)), clock)

        # Register candidate & admit V1
        ctrl.process_event(
            Event(
                event=EventType.CODE_READY.value,
                event_id="e-code-v1",
                run_id="sched-test",
                producer="coder",
                timestamp=clock.now_iso(),
                sim_time=0.0,
                command_id="cmd-1",
                direction_id="reg",
                artifact_id="art-1",
                parent_version_id="V0",
                kind="implement_kernel",
                extra={"package_id": "pkg-1", "reply_to": "researcher"},
            )
        )
        v1 = ctrl.admit_next_version()
        assert v1 is not None

        # Mark W1, W2, W3 complete; leave W4 = 60s
        v1.completed_blocks = ["W1", "W2", "W3"]
        for t in ctrl.tasks.values():
            if t.kind == JobKind.BENCHMARK_BLOCK.value:
                if t.block_id in ("W1", "W2", "W3"):
                    t.status = TaskStatus.COMPLETED.value
                else:
                    t.estimated_duration_s = 60.0

        # Propose probe (20s GPU + 10s CPU parse) on V1
        ctrl.process_event(
            Event(
                event=EventType.TASK_PROPOSED.value,
                event_id="e-probe",
                run_id="sched-test",
                producer="leader",
                timestamp=clock.now_iso(),
                sim_time=0.0,
                direction_id="reg",
                version_id="V1",
                artifact_id="art-1",
                kind=JobKind.PROBE.value,
                extra={
                    "task_id": "probe-1",
                    "resource": "gpu",
                    "estimated_duration_s": 20.0,
                    "post_cpu_kind": "cpu_probe",
                    "post_cpu_duration_s": 10.0,
                    "is_required": False,
                },
            )
        )
        # Leader waits on probe-1 to do 50s analysis
        ctrl.process_event(
            Event(
                event=EventType.TASK_WAITING.value,
                event_id="e-wait-leader",
                run_id="sched-test",
                producer="leader",
                timestamp=clock.now_iso(),
                sim_time=0.0,
                direction_id="reg",
                extra={
                    "wait_id": "wait-leader-probe",
                    "member": "leader",
                    "waiting_for": "result_ready:probe-1",
                    "unblocks_work": "analysis",
                    "estimated_duration_s": 50.0,
                },
            )
        )
        return ctrl, clock

    def test_exact_section6_example_probe_wins_and_withdraw_reverts(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            ctrl, clock = self._setup_w4_vs_probe(tmp)
            chosen, dec = ctrl.select_next_gpu_task()
            self.assertIsNotNone(chosen)
            self.assertEqual(chosen.task_id, "probe-1")

            by_id = {c["task_id"]: c for c in dec.candidates}
            self.assertAlmostEqual(by_id["probe-1"]["typical_score"], 70.0)
            w4_id = [tid for tid in by_id if tid != "probe-1"][0]
            self.assertAlmostEqual(by_id[w4_id]["typical_score"], 40.0)

            # Withdraw waiting relation -> choice flips back to W4 (40 > 20)
            ctrl.process_event(
                Event(
                    event=EventType.TASK_WITHDRAWN.value,
                    event_id="e-withdraw",
                    run_id="sched-test",
                    producer="leader",
                    timestamp=clock.now_iso(),
                    sim_time=0.0,
                    extra={"wait_id": "wait-leader-probe"},
                )
            )
            chosen2, dec2 = ctrl.select_next_gpu_task()
            self.assertIsNotNone(chosen2)
            self.assertEqual(chosen2.block_id, "W4")
            by_id2 = {c["task_id"]: c for c in dec2.candidates}
            self.assertAlmostEqual(by_id2["probe-1"]["typical_score"], 20.0)
            self.assertAlmostEqual(by_id2[w4_id]["typical_score"], 40.0)

    def test_p90_sensitivity_fallback_reverts_to_benchmark(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            ctrl, clock = self._setup_w4_vs_probe(tmp)
            # Record historical diagnostic samples where median is fast (20s)
            # but P90 is slow (75s), pushing W4 + Judge out of horizon H=100s!
            k, fb = DurationModel.diagnostic_keys("probe", 1, ctrl.config.gpu_name)
            for d in (18.0, 20.0, 20.0, 22.0, 75.0, 80.0):
                ctrl.duration_model.record_sample(k, d, fallback_keys=fb)

            # Set Leader wait duration to 35s so typical probe = 35+20=55 > W4=40,
            # while slow probe (P90 ~ 77.5s GPU + 10s parse -> Leader gets only 12.5s, Judge 0s)
            # drops below W4's slow_score (40s)!
            ctrl.waiting_relations["wait-leader-probe"]["estimated_duration_s"] = 35.0

            chosen, dec = ctrl.select_next_gpu_task()
            self.assertIsNotNone(chosen)
            self.assertEqual(chosen.block_id, "W4")
            self.assertEqual(dec.reason, "sensitivity_reverted_to_required_benchmark")

    def test_duration_model_censored_timeouts_not_ignored(self) -> None:
        dm = DurationModel(min_samples=3)
        k, fb = DurationModel.diagnostic_keys("ncu", 1, "H100")
        dm.record_sample(k, 10.0, timed_out=False, fallback_keys=fb)
        dm.record_sample(k, 45.0, timed_out=True, fallback_keys=fb)
        dm.record_sample(k, 45.0, timed_out=True, fallback_keys=fb)
        est = dm.estimate(k, fb)
        # Because >=50% timed out, typical must be pulled up to slow bound (45.0), not fast 10.0
        self.assertGreaterEqual(est.typical_s, 45.0)
        self.assertGreaterEqual(est.slow_s, 45.0)


if __name__ == "__main__":
    unittest.main()
