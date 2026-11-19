"""Tests for AsyncKernelLabRunner, DeepSeek LLM interfaces, RealSingleGpuBackend, and snapshot batching."""

from __future__ import annotations

import asyncio
from pathlib import Path
import sys
import tempfile
import unittest

from kernel_agent_v1.async_runtime import AsyncKernelLabRunner, AsyncRunnerHooks
from kernel_agent_v1.backends import (
    RealSingleGpuBackend,
    parse_ncu_csv,
)
from kernel_agent_v1.clock import RealClock, SimClock
from kernel_agent_v1.closed_loop_tester import _subprocess_worker_script
from kernel_agent_v1.controller import Controller
from kernel_agent_v1.evaluator import TrustedEvaluator
from kernel_agent_v1.llm import (
    DeepSeekClient,
    DeepSeekCoderAdapter,
    DeepSeekConfig,
    DeepSeekResearcherAdapter,
    LLMMessage,
    LLMNotConfiguredError,
    ScriptedDeepSeekClient,
)
from kernel_agent_v1.protocol import (
    CommandKind,
    Event,
    EventType,
    ResultKind,
    RuntimeConfig,
    TargetRef,
    VerificationLevel,
)
from kernel_agent_v1.runtime import KernelLabRuntime
from kernel_agent_v1.store import RunStore
from kernel_agent_v1.wiki import HardwareKnowledgePlugin


class TestAsyncAndDeepSeek(unittest.TestCase):
    def test_snapshot_batching_still_recovers_exact_state(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            store = RunStore(Path(tmp))
            cfg = RuntimeConfig(run_id="snap-batch", snapshot_interval=50)
            clock = SimClock()
            ctrl = Controller(cfg, store, clock)

            # Emit 12 priority events (below snapshot_interval=50) -> no intermediate snapshot saved
            for i in range(12):
                ctrl.process_event(
                    Event(
                        event=EventType.PRIORITY_CHANGED.value,
                        event_id=f"e-batch-{i}",
                        run_id="snap-batch",
                        producer="leader",
                        timestamp=clock.now_iso(),
                        sim_time=float(i),
                        direction_id=f"dir-{i}",
                        extra={"priority": float(i + 1)},
                    )
                )
            self.assertIsNone(store.load_snapshot())
            self.assertEqual(len(store.load_events()), 12)

            recovered = Controller.recover(cfg, store, clock)
            self.assertEqual(recovered.state_seq, 12)
            self.assertEqual(recovered.route_priorities["dir-11"], 12.0)

    def test_deepseek_client_interface_disabled_by_default(self) -> None:
        client = DeepSeekClient(DeepSeekConfig())
        self.assertFalse(client.is_ready())
        with self.assertRaises(LLMNotConfiguredError):
            asyncio.run(client.complete([LLMMessage("user", "hi")]))

    def test_deepseek_adapters_enforce_wiki_and_path_constraints(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            rt = KernelLabRuntime(Path(tmp))
            rt.register_route("reg", "register spill", priority=2.0)

            res_llm = ScriptedDeepSeekClient(
                lambda msgs: {
                    "observation": "W1 shows short-seq latency",
                    "explanation": "Register spill in inner loop",
                    "prediction": "Tile split improves 1.2x",
                    "hypothesis": "Reduce per-thread registers <= 128",
                    "command_kind": CommandKind.IMPLEMENT_KERNEL.value,
                    "instruction": "Split inner accumulator tile",
                    "deliverable": "kernel/run_kernel.py",
                    "allowed_paths": ["kernel/"],
                    "workload_ids": [],
                }
            )
            res_adapter = DeepSeekResearcherAdapter(res_llm, rt.leader)
            cmd, _, usage = asyncio.run(
                res_adapter.plan_from_evidence(
                    researcher_id="res_reg",
                    direction_id="reg",
                    deliveries=[],
                    default_target=TargetRef("V0", "fla-baseline"),
                )
            )
            self.assertGreater(usage.cost_usd, 0.0)
            self.assertTrue(
                any(r.startswith("wiki-lookup-") for r in cmd.body.evidence_refs)
            )
            # Must query Wiki once and append ResearchNote once (no duplicate lookup/note)
            self.assertEqual(len(rt.wiki.valid_lookups), 1)
            res_obj = rt.leader.researchers["res_reg"]
            self.assertEqual(len(res_obj.notes), 1)
            self.assertEqual(res_obj.notes[0].wiki_lookup_id, cmd.body.evidence_refs[0])

            # Out-of-bounds path from Coder LLM must be blocked by Coder constraint check
            bad_coder_llm = ScriptedDeepSeekClient(
                lambda msgs: {
                    "artifact_id": "art-bad",
                    "entrypoint": "outside/hack.py",
                    "files": {"outside/hack.py": "print('nope')"},
                }
            )
            evt_bad, pkg_bad, _ = asyncio.run(
                DeepSeekCoderAdapter(bad_coder_llm, rt.coder).implement(cmd)
            )
            self.assertEqual(evt_bad.event, EventType.CODE_BLOCKED.value)
            self.assertIsNone(pkg_bad)

            # Valid path seals package cleanly
            ok_coder_llm = ScriptedDeepSeekClient(
                lambda msgs: {
                    "artifact_id": "art-ok",
                    "entrypoint": "kernel/run_kernel.py",
                    "files": {
                        "kernel/run_kernel.py": _subprocess_worker_script(
                            rt.evaluator, speedup=1.18
                        )
                    },
                }
            )
            evt_ok, pkg_ok, _ = asyncio.run(
                DeepSeekCoderAdapter(ok_coder_llm, rt.coder).implement(cmd)
            )
            self.assertEqual(evt_ok.event, EventType.CODE_READY.value)
            self.assertIsNotNone(pkg_ok)

    def test_async_concurrent_runner_with_real_subprocesses_and_deepseek_adapters(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            cfg = RuntimeConfig(run_id="async-e2e", scheduler_mode="chunked_dynamic")
            clock = RealClock()
            evaluator = TrustedEvaluator(
                protocol=cfg.protocol_name,
                device_id=f"{cfg.gpu_name}:0",
                total_cases=100,
                num_blocks=4,
            )
            backend = RealSingleGpuBackend(
                enable_gpu_probe=False,
                eval_cmd_template=[
                    sys.executable, "{package_dir}/{entrypoint}",
                    "--inputs", "inputs.json", "--output", "output.json",
                ],
            )
            rt = KernelLabRuntime(Path(tmp), config=cfg, clock=clock, backend=backend)
            rt.register_route("reg", "register spill", priority=2.0, researcher_id="res_reg")

            # Add brief sleep in worker script so GPU subprocess & async research genuinely overlap
            slow_worker = (
                "import time\n"
                + _subprocess_worker_script(evaluator, speedup=1.18).replace(
                    "def main():", "def main():\n    time.sleep(0.03)"
                )
            )
            fast_worker_v2 = _subprocess_worker_script(evaluator, speedup=1.26)

            research_replies = iter([
                "   ",
                {
                    "observation": "W1 shows gain, propose v2",
                    "explanation": "Further unroll cut",
                    "prediction": "1.26x",
                    "hypothesis": "Tune register cap for v2",
                    "command_kind": CommandKind.IMPLEMENT_KERNEL.value,
                    "instruction": "Build art-v2",
                    "deliverable": "kernel/run_kernel.py",
                    "allowed_paths": ["kernel/"],
                },
            ])
            res_llm = ScriptedDeepSeekClient(lambda msgs: next(research_replies), latency_s=0.02)
            coder_llm = ScriptedDeepSeekClient(
                lambda msgs: {
                    "artifact_id": "art-v2",
                    "entrypoint": "kernel/run_kernel.py",
                    "files": {"kernel/run_kernel.py": fast_worker_v2},
                },
                latency_s=0.02,
            )
            res_adapter = DeepSeekResearcherAdapter(res_llm, rt.leader)
            coder_adapter = DeepSeekCoderAdapter(coder_llm, rt.coder)

            # Seed V1 synchronously before starting async loop
            lookup = rt.wiki.query("NVIDIA-H100-SXM", "reg", ("register",), clock.now_iso())
            cmd1 = rt.leader.build_command(
                kind=CommandKind.IMPLEMENT_KERNEL.value,
                research_id="r-001",
                direction_id="reg",
                target=TargetRef("V0", "fla-baseline"),
                hypothesis="v1",
                evidence_refs=(lookup.lookup_id,),
                instruction="build v1",
                deliverable="kernel/run_kernel.py",
                reply_to="res_reg",
                allowed_paths=("kernel/",),
                preserve_target=False,
            )
            rt.submit_command_and_package(
                cmd1,
                {
                    "artifact_id": "art-v1",
                    "entrypoint": "kernel/run_kernel.py",
                    "files": {"kernel/run_kernel.py": slow_worker},
                },
            )

            v2_submitted = False

            async def _on_evidence(runtime: KernelLabRuntime, delivery) -> None:
                nonlocal v2_submitted
                if (
                    delivery.result_kind == ResultKind.BENCHMARK_PARTIAL.value
                    and delivery.block_id == "W1"
                    and not v2_submitted
                ):
                    v2_submitted = True
                    cmd2, _, u1 = await res_adapter.plan_from_evidence(
                        researcher_id="res_reg",
                        direction_id="reg",
                        deliveries=[delivery],
                        default_target=TargetRef("V1", "art-v1"),
                    )
                    evt2, pkg2, u2 = await coder_adapter.implement(cmd2)
                    runtime.controller.ledger.model_cost_used += u1.cost_usd + u2.cost_usd
                    if pkg2 is not None:
                        runtime.packages_by_id[pkg2.manifest.package_id] = pkg2
                    await runner.submit_external_event(evt2)

            runner = AsyncKernelLabRunner(
                rt,
                hooks=AsyncRunnerHooks(on_research_evidence=_on_evidence),
            )
            metrics = asyncio.run(runner.run(max_wall_s=30.0))

            self.assertTrue(v2_submitted)
            self.assertEqual(metrics.best_artifact_id, "art-v2")
            self.assertEqual(metrics.verification_level, VerificationLevel.FINAL_VERIFIED.value)
            self.assertGreater(metrics.member_gpu_overlap_s, 0.0)
            self.assertEqual(len(res_llm.calls), 2)
            self.assertAlmostEqual(metrics.model_cost_used, 0.0015)

            packages = {p.manifest.artifact_id: p.manifest.package_id for p in rt.packages_by_id.values()}
            abba = next(t for t in rt.controller.tasks.values() if t.kind == "abba_retest")
            self.assertEqual(abba.package_id, packages["art-v2"])
            self.assertEqual(abba.metadata["incumbent_package_id"], packages["art-v1"])
            for _, event in rt.store.load_events():
                if event.kind == "benchmark_block" and event.event == EventType.GPU_JOB_FINISHED.value:
                    evidence = rt.store.read_evidence(event.receipt_id)
                    self.assertTrue(evidence["valid"], evidence["issues"])
                    self.assertEqual({row["attempt_id"] for row in evidence["cases"]}, {event.attempt_id})

    def test_ncu_csv_parser_and_real_single_gpu_backend_cleanup_probe(self) -> None:
        csv_sample = (
            '"ID","Metric Name","Metric Value"\n'
            '"0","lmem_spill_bytes","256"\n'
            '"0","sm__warps_active.avg.pct_of_peak_sustained_active","41.5"\n'
            '"0","launch__registers_per_thread","168"\n'
        )
        parsed = parse_ncu_csv(csv_sample)
        self.assertTrue(parsed["register_spill_detected"])
        self.assertEqual(parsed["counters"]["lmem_spill_bytes"], 256.0)
        self.assertAlmostEqual(parsed["counters"]["achieved_occupancy"], 0.415)

        # Verify RealSingleGpuBackend detects lingering PID if mock nvidia-smi reports it
        clock = RealClock()
        backend = RealSingleGpuBackend(
            gpu_index=0,
            enable_gpu_probe=True,
            nvidia_smi_cmd=("echo", "999999"),
        )
        h = backend.start(
            job_id="g1",
            attempt_id="g1/1",
            task_id="t1",
            kind="probe",
            resource="gpu",
            package=None,
            inputs={"command": [sys.executable, "-c", "import time; time.sleep(10)"]},
            clock=clock,
        )
        # Mock nvidia-smi returning handle.pid -> stop must flag gpu_released=False!
        backend.nvidia_smi_cmd = [sys.executable, "-c", f"print({h.pid})"]
        receipt = backend.stop(h, "test_linger", clock)
        self.assertFalse(receipt.gpu_released)
        self.assertIn("gpu_pid_still_active", receipt.reason)

        # Expanded Wiki covers A100/L40S/H100 and learned lessons
        wiki_plugin = HardwareKnowledgePlugin("NVIDIA-A100-SXM", "cuda-12.6")
        hits = wiki_plugin.query("NVIDIA-A100-SXM", ("cp.async", "ampere"))
        self.assertTrue(any(e.gpu_arch == "NVIDIA-A100-SXM" for e in hits))
        learned = wiki_plugin.add_verified_lesson(
            topic="gdn_tile_64",
            scope="SM80",
            guidance="64-token chunk tile avoids bank conflict",
            source="run-001",
            keywords=("gdn", "tile"),
        )
        self.assertTrue(learned.entry_id.startswith("hw-learned-"))

    def test_rejected_abba_verdict_counts_as_route_stagnation_round(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            rt = KernelLabRuntime(Path(tmp))
            rt.register_route("reg", "hyp", priority=1.0, reference_score=1.10)
            from kernel_agent_v1.protocol import EvidenceDelivery

            deliv = EvidenceDelivery(
                delivery_id="d-abba-rej",
                receipt_id="r-abba-rej",
                recipient="leader",
                result_kind="judge_abba_retest",
                producer="judge",
                timestamp=rt.clock.now_iso(),
                version_id="V2",
                artifact_id="art-v2-abba-fail",
                direction_id="reg",
                command_id=None,
                block_id=None,
                evidence_path="/tmp/r.json",
                payload={
                    "verdict": "rejected_abba",
                    "geomean_speedup": 1.105,
                    "version_id": "V2",
                    "artifact_id": "art-v2-abba-fail",
                    "direction_id": "reg",
                },
            )
            rt.leader.deliver_evidence(deliv)
            self.assertEqual(rt.wiki.route_tree.routes["reg"].stagnation_count, 1)


if __name__ == "__main__":
    unittest.main()
