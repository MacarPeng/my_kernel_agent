"""Tests for adapted official GDN assets, stratified workblocks, GDN ABI validation, and cli run."""

from __future__ import annotations

from contextlib import redirect_stderr, redirect_stdout
from io import StringIO
import asyncio
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from kernel_agent_v1.backends import SimulatedExecutionBackend
from kernel_agent_v1.cli import main as cli_main
from kernel_agent_v1.clock import SimClock
from kernel_agent_v1.coder import Coder
from kernel_agent_v1.evaluator import TrustedEvaluator
from kernel_agent_v1.gdn_gpu_runner import _run_benchmark_block
from kernel_agent_v1.gdn_task import (
    load_official_definition,
    load_official_workloads,
    seed_gdn_candidate_source,
    stratified_split_workblocks,
    validate_gdn_candidate_sources,
)
from kernel_agent_v1.leader import Leader
from kernel_agent_v1.llm import LLMResponse, ScriptedDeepSeekClient
from kernel_agent_v1.protocol import CommandKind, EventType, TargetRef
from kernel_agent_v1.store import RunStore
from kernel_agent_v1.wiki import HardwareKnowledgePlugin, KernelWiki


class TimedBackend(SimulatedExecutionBackend):
    async def async_wait(self, handle, clock, poll_interval_s=0.01):
        clock.advance_to(min(handle.expected_finish_at, handle.timeout_at))
        if handle.expected_finish_at > handle.timeout_at:
            handle.metadata["behavior"]["timeout"] = True
        return self.wait(handle, clock)


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

    def test_real_gpu_cli_allows_long_abba_before_final_evaluation(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            backend = TimedBackend(TrustedEvaluator.from_official_gdn_dataset())
            backend.set_behavior("benchmark_block", {"duration_s": 40.0, "speedup": 1.2})
            backend.set_behavior("improved:benchmark_block", {"duration_s": 40.0, "speedup": 2.0})
            backend.set_behavior("abba_retest", {"duration_s": 1050.0, "speedup": 2.0})
            replies = iter([
                {"allowed_paths": ["kernel.py"], "preserve_target": False},
                {"artifact_id": "improved", "entrypoint": "kernel.py",
                 "files": {"kernel.py": seed_gdn_candidate_source() + "\n# optimized\n"}},
            ])
            client = ScriptedDeepSeekClient(lambda messages: next(replies))
            stdout = StringIO()
            with patch("kernel_agent_v1.cli.RealClock", return_value=SimClock()), \
                 patch("kernel_agent_v1.cli.RealSingleGpuBackend", return_value=backend), \
                 patch("kernel_agent_v1.cli.DeepSeekClient", return_value=client), \
                 redirect_stdout(stdout):
                self.assertEqual(cli_main([
                    "run", "--backend", "real-gpu", "--deepseek-live",
                    "--max-rounds", "1", "--run-dir", tmp,
                ]), 0)
            result = json.loads(stdout.getvalue())
            self.assertEqual(result["best_artifact_id"], "improved")
            self.assertEqual(result["verification_level"], "final_verified")
            tasks = RunStore(tmp).load_snapshot()["tasks"].values()
            self.assertEqual(
                [t["timeout_s"] for t in tasks if t["kind"] == "abba_retest"], [1800.0]
            )

    def test_dual_researcher_uses_diagnosis_available_before_command(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            store = RunStore(tmp)

            class CoordinatedClient(ScriptedDeepSeekClient):
                def __init__(self):
                    super().__init__(lambda _messages: {})
                    self.design_started = None
                    self.diagnosis_returning = None
                    self.coder_input = None

                async def complete(self, messages, *, json_mode=True):
                    if self.design_started is None:
                        self.design_started = asyncio.Event()
                        self.diagnosis_returning = asyncio.Event()
                    system, user = messages
                    request = json.loads(user.content)
                    self.calls.append(messages)
                    if "Diagnosis Researcher" in system.content:
                        measurement = request["measurement"]
                        receipt = measurement["receipt_id"]
                        assert store.read_evidence(f"wiki-{receipt}") == measurement
                        if receipt == "receipt-0001":
                            await self.design_started.wait()
                            self.diagnosis_returning.set()
                            return LLMResponse(json.dumps({
                                "observation": "Short input latency is high.",
                                "design_impact": "Use a short-input tile.",
                            }), "scripted")
                        return LLMResponse("{}", "scripted")
                    if "Design Researcher" in system.content:
                        assert store.read_evidence("wiki-receipt-0001")
                        assert not request["available_diagnoses"]
                        self.design_started.set()
                        await self.diagnosis_returning.wait()
                        return LLMResponse(json.dumps({
                            "instruction": "Use a large tile.",
                            "allowed_paths": ["kernel.py"], "preserve_target": False,
                        }), "scripted")
                    if "Review the proposed kernel design" in system.content:
                        return LLMResponse(json.dumps({
                            "instruction": "Use a short-input tile.",
                            "diagnosis_refs": ["diagnosis-receipt-0001"],
                        }), "scripted")
                    self.coder_input = request
                    return LLMResponse(json.dumps({
                        "artifact_id": "short-tile", "entrypoint": "kernel.py",
                        "files": {"kernel.py": seed_gdn_candidate_source() + "\n# short tile\n"},
                    }), "scripted")

            client = CoordinatedClient()
            stdout = StringIO()
            with patch("kernel_agent_v1.cli.DeepSeekClient", return_value=client), redirect_stdout(stdout):
                self.assertEqual(cli_main([
                    "run", "--deepseek-live", "--dual-researcher",
                    "--max-rounds", "1", "--run-dir", tmp,
                ]), 0)
            body = client.coder_input["body"]
            self.assertEqual(body["instruction"], "Use a short-input tile.")
            self.assertIn("diagnosis-receipt-0001", body["evidence_refs"])
            trace = store.read_evidence("design-" + client.coder_input["head"]["command_id"])
            self.assertTrue(trace["late_diagnosis_changed_instruction"])
            self.assertEqual(trace["ready_before_command"], ["diagnosis-receipt-0001"])
            self.assertEqual(json.loads(stdout.getvalue())["diagnosis"]["conclusions"], 1)
            inputs = [json.loads(call[1].content) for call in client.calls
                      if "Diagnosis Researcher" in call[0].content]
            self.assertEqual(sum(p["measurement"]["receipt_id"] == "receipt-0001" for p in inputs), 1)
            self.assertFalse(any(event.event == EventType.TASK_WAITING.value for _, event in store.load_events()))

    def test_benchmark_runner_reports_triton_compilation_error(self) -> None:
        triton_error = type(
            "UnsupportedLanguageConstruct", (Exception,), {"__module__": "triton.compiler.errors"}
        )

        def candidate():
            raise triton_error("unsupported AST node type: Break")

        torch = SimpleNamespace(cuda=SimpleNamespace(is_available=lambda: False))
        with patch("kernel_agent_v1.gdn_gpu_runner._build_reference_fn", return_value=lambda: ()), \
             patch("kernel_agent_v1.gdn_gpu_runner._make_deterministic_gdn_inputs", return_value=()), \
             patch("kernel_agent_v1.gdn_gpu_runner._load_python_entrypoint",
                   return_value=SimpleNamespace(run=candidate)):
            result = _run_benchmark_block({
                "attempt_id": "job-1/1", "candidate_path": "kernel.py", "block_id": "W1",
            }, torch)

        self.assertEqual(result["candidate_failure"], "compile_failed")
        self.assertEqual(result["failure_stage"], "compile")
        self.assertTrue(result["workload_id"])
        self.assertIn("unsupported AST node type: Break", result["error"])

    def test_candidate_compile_failure_reaches_leader_and_coder(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            backend = SimulatedExecutionBackend(TrustedEvaluator.from_official_gdn_dataset())
            backend.set_behavior("art-seed-v0:benchmark_block:W1", {
                "compile_fail": True, "error": "unsupported AST node type: Break",
            })
            source = seed_gdn_candidate_source()
            replies = iter([
                {"explanation": "Triton rejects break", "instruction": "Use a bounded loop"},
                {"artifact_id": "art-seed-v0", "package_id": "pkg-0001",
                 "entrypoint": "kernel.py", "files": {"kernel.py": source + "\n# bounded loop\n"}},
            ])
            client = ScriptedDeepSeekClient(lambda messages: next(replies))
            stdout = StringIO()
            with patch("kernel_agent_v1.cli.SimulatedExecutionBackend", return_value=backend), \
                 patch("kernel_agent_v1.cli.DeepSeekClient", return_value=client), \
                 redirect_stdout(stdout), redirect_stderr(StringIO()):
                self.assertEqual(cli_main([
                    "run", "--deepseek-live", "--max-rounds", "1", "--run-dir", tmp,
                ]), 0)

            result = json.loads(stdout.getvalue())
            self.assertEqual(set(result["role_active_s"]), {
                "Controller", "Leader", "Coder", "Operator", "Judge",
            })
            timeline = Path(result["pipeline_timeline"]).read_text(encoding="utf-8")
            for label in ("Plan from receipt-", "Implement cmd-", "GPU benchmark_block V1 W1", "compile_failed"):
                self.assertIn(label, timeline)

            leader_call, coder_call = (json.loads(call[-1].content) for call in client.calls)
            context = leader_call["repair_context"]
            self.assertEqual(context["files"]["kernel.py"], source)
            self.assertIn("unsupported AST node type: Break", context["diagnostic"]["compiler_stderr"])
            self.assertEqual(coder_call["repair_context"], context)
            self.assertEqual(coder_call["body"]["instruction"], "Use a bounded loop")

            events = [event for _, event in RunStore(tmp).load_events()]
            failed = [event for event in events if event.status == "compile_failed"]
            self.assertEqual(len(failed), 1)
            self.assertTrue(failed[0].attempt_id.endswith("/1"))
            repaired = [event for event in events if event.event == EventType.CODE_READY.value][-1]
            self.assertEqual(repaired.parent_version_id, failed[0].version_id)
            self.assertNotEqual(repaired.artifact_id, failed[0].artifact_id)
            self.assertNotEqual(repaired.extra["package_id"], context["diagnostic"]["package_id"])
            self.assertTrue(any(
                event.event == EventType.GPU_JOB_FINISHED.value
                and event.artifact_id == repaired.artifact_id
                and event.status == "completed" for event in events
            ))

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
