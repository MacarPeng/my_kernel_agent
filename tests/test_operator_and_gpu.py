"""Tests for Operator CPU/GPU split, GPU cleanup halt, retry once, and subprocess stop."""

from __future__ import annotations

from pathlib import Path
import sys
import tempfile
import unittest

from kernel_agent_v1.backends import LocalSubprocessBackend, SimulatedExecutionBackend
from kernel_agent_v1.clock import RealClock, SimClock
from kernel_agent_v1.controller import Controller
from kernel_agent_v1.evaluator import TrustedEvaluator
from kernel_agent_v1.operator import Operator
from kernel_agent_v1.protocol import Event, EventType, JobKind, RuntimeConfig, TaskStatus
from kernel_agent_v1.runtime import KernelLabRuntime
from kernel_agent_v1.store import RunStore


class TestOperatorAndGPU(unittest.TestCase):
    def test_ncu_capture_releases_gpu_before_cpu_parse_unblocks_research(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            rt = KernelLabRuntime(Path(tmp))
            rt.register_route("reg", "register pressure", priority=1.0)
            rt.controller.process_event(
                Event(
                    event=EventType.CODE_READY.value,
                    event_id="e-c1",
                    run_id=rt.config.run_id,
                    producer="coder",
                    timestamp=rt.clock.now_iso(),
                    sim_time=0.0,
                    command_id="cmd-1",
                    direction_id="reg",
                    artifact_id="art-1",
                    parent_version_id="V0",
                    kind="implement_kernel",
                    extra={"package_id": "pkg-1", "reply_to": "researcher"},
                )
            )
            rt.controller.admit_next_version()
            # Propose NCU capture (GPU 20s) + CPU parse (15s) and a research task waiting on it
            rt.controller.process_event(
                rt.leader.propose_task_event(
                    task_id="ncu-1",
                    kind=JobKind.NCU_CAPTURE.value,
                    resource="gpu",
                    direction_id="reg",
                    version_id="V1",
                    artifact_id="art-1",
                    estimated_duration_s=20.0,
                    post_cpu_kind=JobKind.NCU_PARSE.value,
                    post_cpu_duration_s=15.0,
                )
            )
            rt.controller.process_event(
                rt.leader.propose_task_event(
                    task_id="res-after-ncu",
                    kind=JobKind.RESEARCH.value,
                    resource="model",
                    direction_id="reg",
                    assigned_member="researcher",
                    depends_on=["result_ready:ncu-1"],
                    estimated_duration_s=40.0,
                )
            )
            rt.controller.process_event(
                rt.leader.declare_waiting_event(
                    member="researcher",
                    waiting_for="result_ready:ncu-1",
                    unblocks_work="analysis",
                    estimated_duration_s=40.0,
                    direction_id="reg",
                )
            )

            # Dispatch NCU capture
            h_ncu, _ = rt.dispatch_gpu_if_ready()
            assert h_ncu is not None
            self.assertEqual(h_ncu.kind, JobKind.NCU_CAPTURE.value)

            # Finish GPU NCU capture -> GPU released immediately, CPU parse pending, research STILL blocked!
            op_res = rt.complete_gpu_job(h_ncu)
            self.assertFalse(rt.controller.gpu_busy)
            self.assertTrue(rt.controller.gpu_cleanup_confirmed)
            self.assertNotIn("result_ready:ncu-1", rt.controller.satisfied_tokens)
            self.assertEqual(len(op_res.pending_cpu_jobs), 1)

            # Next GPU job (e.g. W1) can start immediately while CPU parse runs!
            h_next, _ = rt.dispatch_gpu_if_ready()
            self.assertIsNotNone(h_next)
            self.assertEqual(h_next.kind, JobKind.BENCHMARK_BLOCK.value)

            # Now run CPU parse off-GPU -> emits result_ready -> unblocks research task!
            cpu_handle = rt.operator.start(None, op_res.pending_cpu_jobs[0])
            cpu_res = rt.operator.finalize_handle(cpu_handle)
            for evt in cpu_res.scheduling_events:
                rt.controller.process_event(evt)
            rt.process_deliveries(cpu_res.deliveries)
            self.assertIn("result_ready:ncu-1", rt.controller.satisfied_tokens)
            self.assertFalse(
                any(r["active"] for r in rt.controller.waiting_relations.values())
            )

    def test_transient_infra_retry_once_and_unconfirmed_cleanup_halts_gpu(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            rt = KernelLabRuntime(Path(tmp))
            rt.register_route("reg", "hyp", priority=1.0)
            rt.controller.process_event(
                Event(
                    event=EventType.CODE_READY.value,
                    event_id="e-c1",
                    run_id=rt.config.run_id,
                    producer="coder",
                    timestamp=rt.clock.now_iso(),
                    sim_time=0.0,
                    command_id="cmd-1",
                    direction_id="reg",
                    artifact_id="art-1",
                    parent_version_id="V0",
                    kind="implement_kernel",
                    extra={"package_id": "pkg-1", "reply_to": "researcher"},
                )
            )
            rt.controller.admit_next_version()
            assert isinstance(rt.backend, SimulatedExecutionBackend)

            # First attempt of W1 fails with transient infra error
            rt.backend.set_behavior("job-0001/1", {"infra_fail": True, "duration_s": 10.0})
            h1, _ = rt.dispatch_gpu_if_ready()
            assert h1 is not None
            self.assertEqual(h1.attempt_id, "job-0001/1")
            rt.complete_gpu_job(h1)

            # Same task is reset to PENDING for retry with retry_count=1 and new attempt_id!
            t_w1 = rt.controller.tasks[h1.task_id]
            self.assertEqual(t_w1.status, TaskStatus.PENDING.value)
            self.assertEqual(t_w1.retry_count, 1)

            h2, _ = rt.dispatch_gpu_if_ready()
            assert h2 is not None
            self.assertEqual(h2.task_id, h1.task_id)
            self.assertEqual(h2.attempt_id, "job-0002/2")
            rt.complete_gpu_job(h2)
            self.assertEqual(t_w1.status, TaskStatus.COMPLETED.value)

            # Now inject unconfirmed GPU cleanup on W2 cancellation -> must HALT normal GPU dispatch!
            h_w2, _ = rt.dispatch_gpu_if_ready()
            assert h_w2 is not None
            rt.backend.unconfirmed_cleanup_attempts.add(h_w2.attempt_id)
            cleanup = rt.operator.stop(h_w2, "timeout")
            self.assertFalse(cleanup.gpu_released)
            op_res = rt.operator.finalize_handle(h_w2)
            for evt in op_res.scheduling_events:
                rt.controller.process_event(evt)

            self.assertFalse(rt.controller.gpu_cleanup_confirmed)
            self.assertEqual(rt.controller.gpu_halt_reason, "gpu_cleanup_unconfirmed")
            h_blocked, dec = rt.dispatch_gpu_if_ready()
            self.assertIsNone(h_blocked)
            self.assertIn("halted:gpu_cleanup_unconfirmed", dec.reason)

    def test_local_subprocess_backend_idempotent_stop(self) -> None:
        clock = RealClock()
        backend = LocalSubprocessBackend()
        h = backend.start(
            job_id="sub-1",
            attempt_id="sub-1/1",
            task_id="t-sub",
            kind="probe",
            resource="gpu",
            package=None,
            inputs={
                "command": [sys.executable, "-c", "import time; time.sleep(30)"],
                "timeout_s": 10.0,
            },
            clock=clock,
        )
        r1 = backend.stop(h, "user_cancel", clock)
        r2 = backend.stop(h, "user_cancel", clock)
        self.assertTrue(r1.gpu_released)
        self.assertFalse(r1.already_stopped)
        self.assertTrue(r2.gpu_released)
        self.assertTrue(r2.already_stopped)


if __name__ == "__main__":
    unittest.main()
