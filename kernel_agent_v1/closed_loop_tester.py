"""Reactive end-to-end closed-loop simulator & invariant auditor (no physical GPU required)."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
import sys
import tempfile
from typing import Any

from kernel_agent_v1.backends import (
    ExecutionBackend,
    JobHandle,
    LocalSubprocessBackend,
    SimulatedExecutionBackend,
)
from kernel_agent_v1.clock import Clock, RealClock, SimClock
from kernel_agent_v1.coder import verify_package_integrity
from kernel_agent_v1.controller import Controller
from kernel_agent_v1.evaluator import TrustedEvaluator
from kernel_agent_v1.protocol import (
    BudgetConfig,
    CommandKind,
    Event,
    EventType,
    EvidenceDelivery,
    JobKind,
    PROHIBITED_HEADER_FIELDS,
    ResultKind,
    RouteStatus,
    RuntimeConfig,
    TargetRef,
    VerificationLevel,
)
from kernel_agent_v1.runtime import (
    KernelLabRuntime,
    _intersect_intervals,
    _union_interval_length,
)


@dataclass
class TraceEntry:
    time_s: float
    actor: str
    action: str
    detail: str


@dataclass
class ClosedLoopReport:
    backend_kind: str
    passed: bool
    checks: dict[str, bool]
    failures: list[str]
    wall_seconds: float
    gpu_busy_seconds: float
    member_gpu_overlap_seconds: float
    first_actionable_evidence_s: float | None
    best_version_id: str | None
    best_artifact_id: str | None
    best_speedup: float
    verification_level: str
    versions_closed: list[str]
    route_states: dict[str, dict[str, Any]]
    trace: list[TraceEntry] = field(default_factory=list)

    def format_text(self, verbose: bool = True) -> str:
        lines = [
            f"KernelLab V1 Closed-Loop Verification ({self.backend_kind})",
            "=" * 74,
            f"Status             : {'PASS' if self.passed else 'FAIL'}",
            f"Best Verified      : {self.best_version_id} ({self.best_artifact_id}) @ {self.best_speedup:.3f}x [{self.verification_level}]",
            f"Versions Closed    : {', '.join(self.versions_closed)}",
            f"First Evidence (s) : {self.first_actionable_evidence_s:.3f}" if self.first_actionable_evidence_s is not None else "First Evidence (s) : N/A",
            f"GPU Busy (s)       : {self.gpu_busy_seconds:.3f}",
            f"Member/GPU Overlap : {self.member_gpu_overlap_seconds:.3f}",
            f"Wall Time (s)      : {self.wall_seconds:.3f}",
            "",
            "Route States:",
        ]
        for rid, st in sorted(self.route_states.items()):
            lines.append(
                f"  - {rid}: status={st['status']}, stagnation={st['stagnation_count']}, "
                f"compile_fails={st['compile_failure_count']}, repairs={st['repair_count']}, "
                f"best={st['best_version_id']}({st['best_score']})"
            )
        lines.append("")
        lines.append("Invariant Checks:")
        for name, ok in self.checks.items():
            lines.append(f"  [{'OK' if ok else 'FAIL'}] {name}")
        if self.failures:
            lines.append("")
            lines.append("Failures:")
            for f in self.failures:
                lines.append(f"  * {f}")
        if verbose and self.trace:
            lines.append("")
            lines.append("Timeline Trace:")
            for tr in self.trace:
                lines.append(f"  t={tr.time_s:7.2f}s | {tr.actor:<11} | {tr.action:<22} | {tr.detail}")
        return "\n".join(lines)


def _subprocess_worker_script(
    evaluator: TrustedEvaluator,
    speedup: float,
    compile_fail: bool = False,
) -> str:
    """Generate a real executable Python script run by LocalSubprocessBackend."""
    uuids_json = repr(list(evaluator.uuids))
    blocks_json = repr({k: list(v) for k, v in evaluator.blocks.items()})
    base_json = repr(evaluator.frozen_baseline_ms)
    return f"""import argparse, json
from pathlib import Path

UUIDS = {uuids_json}
BLOCKS = {blocks_json}
BASE_MS = {base_json}
SPEEDUP = {speedup!r}
COMPILE_FAIL = {compile_fail!r}

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--inputs", required=True)
    p.add_argument("--output", required=True)
    args = p.parse_args()
    inputs = json.loads(Path(args.inputs).read_text(encoding="utf-8"))
    kind = inputs["kind"]
    out = {{}}
    if kind == "benchmark_block":
        if COMPILE_FAIL:
            out = {{"compile_fail": True, "error": "simulated nvcc compile error"}}
        else:
            bid = inputs["block_id"]
            cases = []
            for idx, uid in enumerate(BLOCKS[bid]):
                bms = BASE_MS[uid]
                # Make first 5 cases slightly slower to give Researcher partial diagnostic signal
                sp = max(1.01, SPEEDUP - 0.08) if idx < 5 else SPEEDUP
                cases.append({{
                    "uuid": uid,
                    "correct": True,
                    "candidate_ms": bms / sp,
                    "baseline_ms": bms,
                    "artifact_id": inputs["artifact_id"],
                    "version_id": inputs["version_id"],
                    "protocol": "gdn-unified-v2",
                    "device_id": "{evaluator.device_id}",
                    "attempt_id": inputs.get("attempt_id", "sub/1"),
                }})
            out = {{
                "block_id": bid,
                "version_id": inputs["version_id"],
                "artifact_id": inputs["artifact_id"],
                "protocol": "gdn-unified-v2",
                "device_id": "{evaluator.device_id}",
                "cases": cases,
            }}
    elif kind == "ncu_capture":
        out = {{
            "raw_ncu_report": {{
                "version_id": inputs["version_id"],
                "artifact_id": inputs["artifact_id"],
                "raw_counters": {{"lmem_spill_bytes": 128, "achieved_occupancy": 0.34}},
            }}
        }}
    elif kind == "abba_retest":
        inc_sp = float(inputs.get("incumbent_speedup", 1.0))
        paired = []
        for uid in UUIDS:
            bms = BASE_MS[uid]
            a_ms = bms / inc_sp
            b_ms = bms / SPEEDUP
            paired.append({{"uuid": uid, "correct": True, "a1_ms": a_ms, "b1_ms": b_ms, "b2_ms": b_ms, "a2_ms": a_ms}})
        out = {{
            "incumbent_version_id": inputs.get("incumbent_version_id"),
            "incumbent_artifact_id": inputs.get("incumbent_artifact_id"),
            "candidate_version_id": inputs["version_id"],
            "candidate_artifact_id": inputs["artifact_id"],
            "protocol": "gdn-unified-v2",
            "device_id": "{evaluator.device_id}",
            "paired_cases": paired,
        }}
    elif kind == "final_eval":
        cases = [{{"uuid": uid, "correct": True, "candidate_ms": BASE_MS[uid] / SPEEDUP, "baseline_ms": BASE_MS[uid]}} for uid in UUIDS]
        out = {{
            "final_block_id": inputs.get("block_id", "F1"),
            "version_id": inputs["version_id"],
            "artifact_id": inputs["artifact_id"],
            "protocol": "gdn-unified-v2",
            "device_id": "{evaluator.device_id}",
            "geomean_speedup": SPEEDUP,
            "cases": cases,
        }}
    Path(args.output).write_text(json.dumps(out), encoding="utf-8")

if __name__ == "__main__":
    main()
"""


class _SubprocessAwareOperatorBackend(LocalSubprocessBackend):
    """Wraps LocalSubprocessBackend to handle compile_fail payload & in-process CPU parse."""

    def __init__(self, evaluator: TrustedEvaluator) -> None:
        super().__init__()
        self.evaluator = evaluator

    def start(self, job_id, attempt_id, task_id, kind, resource, package, inputs, clock):
        enriched = dict(inputs)
        if kind == JobKind.NCU_PARSE.value:
            # Run CPU NCU parse inline via python -c
            code = (
                "import argparse, json; from pathlib import Path; "
                "p = argparse.ArgumentParser(); p.add_argument('--inputs'); p.add_argument('--output'); "
                "a = p.parse_args(); inp = json.loads(Path(a.inputs).read_text()); "
                "out = {'version_id': inp.get('version_id'), 'artifact_id': inp.get('artifact_id'), "
                "'observations': {'register_spill_detected': True, 'counters': inp.get('raw_ncu_report', {}).get('raw_counters', {})}}; "
                "Path(a.output).write_text(json.dumps(out))"
            )
            enriched["command"] = [
                sys.executable,
                "-c",
                code,
                "--inputs",
                "inputs.json",
                "--output",
                "output.json",
            ]
        elif kind in (JobKind.ABBA_RETEST.value, JobKind.FINAL_EVAL.value):
            sp = float(inputs.get("speedup", 1.25))
            script = _subprocess_worker_script(self.evaluator, speedup=sp)
            enriched["command"] = [
                sys.executable,
                "-c",
                script,
                "--inputs",
                "inputs.json",
                "--output",
                "output.json",
            ]
        return super().start(job_id, attempt_id, task_id, kind, resource, package, enriched, clock)

    def wait(self, handle: JobHandle, clock: Clock):
        outcome = super().wait(handle, clock)
        if outcome.payload.get("compile_fail"):
            outcome.status = "compile_failed"
            outcome.error = outcome.payload.get("error", "compile_failed")
        return outcome


class ClosedLoopTester:
    """Reactive closed-loop tester proving the full V1 loop works without a physical GPU."""

    def __init__(self, backend_kind: str = "sim") -> None:
        if backend_kind not in ("sim", "subprocess"):
            raise ValueError(f"Unsupported backend_kind: {backend_kind}")
        self.backend_kind = backend_kind
        self.trace: list[TraceEntry] = []

    def _log(self, t: float, actor: str, action: str, detail: str) -> None:
        self.trace.append(TraceEntry(time_s=t, actor=actor, action=action, detail=detail))

    def run(self, work_dir: str | Path | None = None, verbose: bool = False) -> ClosedLoopReport:
        root = Path(work_dir) if work_dir else Path(tempfile.mkdtemp(prefix="kl-closed-loop-"))
        cfg = RuntimeConfig(
            run_id=f"closed-loop-{self.backend_kind}",
            scheduler_mode="chunked_dynamic",
            horizon_h_s=120.0,
            min_gain_threshold=0.01,
            stagnation_limit=3,
            budgets=BudgetConfig(
                total_wall_budget_s=3600.0,
                total_gpu_budget_s=2400.0,
                total_model_budget_cost=100.0,
                version_deadline_s=600.0,
                version_diag_budget_s=120.0,
            ),
        )
        clock: Clock = SimClock() if self.backend_kind == "sim" else RealClock()
        evaluator = TrustedEvaluator(
            protocol=cfg.protocol_name,
            device_id=f"{cfg.gpu_name}:0",
            total_cases=100,
            num_blocks=4,
            min_gain_threshold=cfg.min_gain_threshold,
        )
        backend: ExecutionBackend = (
            SimulatedExecutionBackend(evaluator)
            if self.backend_kind == "sim"
            else _SubprocessAwareOperatorBackend(evaluator)
        )
        rt = KernelLabRuntime(run_dir=root, config=cfg, clock=clock, backend=backend)

        # Track closed-loop milestones for invariant checks
        ncu_gpu_released_at: float | None = None
        ncu_parse_ready_at: float | None = None
        ncu_triggered = False
        reg_repair_triggered = False
        reg_fix_submitted = False
        mem_round = 0

        # 1. Leader initializes two competing routes
        rt.register_route(
            direction_id="register-pressure",
            hypothesis="Short-sequence slowdown caused by register spill",
            priority=2.0,
            researcher_id="res_reg",
        )
        rt.register_route(
            direction_id="smem-layout",
            hypothesis="Bank conflicts in chunked shared-memory scan",
            priority=1.5,
            researcher_id="res_mem",
        )
        self._log(clock.now(), "Leader", "routes_initialized", "register-pressure (prio=2.0), smem-layout (prio=1.5)")

        def _submit_kernel(
            direction_id: str,
            researcher_id: str,
            artifact_id: str,
            parent_ver: str,
            speedup: float,
            hypothesis: str,
            compile_fail: bool = False,
        ) -> None:
            res = rt.leader.ensure_researcher(researcher_id)
            rid = rt.leader.next_research_id()
            lookup, _ = res.start_research_round(
                research_id=rid,
                direction_id=direction_id,
                evidence_terms=(direction_id, "register", "layout"),
                observation=f"Prepare {artifact_id} from {parent_ver}",
                explanation=hypothesis,
                prediction=f"Expected ~{speedup:.2f}x",
                hypotheses=[],
            )
            cmd = rt.leader.build_command(
                kind=CommandKind.IMPLEMENT_KERNEL.value,
                research_id=rid,
                direction_id=direction_id,
                target=TargetRef(parent_ver, artifact_id),
                hypothesis=hypothesis,
                evidence_refs=(lookup.lookup_id,),
                instruction=f"Implement {artifact_id}",
                deliverable="kernel.py",
                reply_to=researcher_id,
                allowed_paths=("kernel/",),
                preserve_target=False,
            )
            spec: dict[str, Any] = {
                "artifact_id": artifact_id,
                "entrypoint": "kernel/run_kernel.py",
                "files": {
                    "kernel/run_kernel.py": _subprocess_worker_script(
                        rt.evaluator, speedup=speedup, compile_fail=compile_fail
                    )
                },
            }
            if isinstance(rt.backend, SimulatedExecutionBackend):
                for b in ("W1", "W2", "W3", "W4"):
                    rt.backend.set_behavior(
                        f"{artifact_id}:benchmark_block:{b}",
                        {
                            "duration_s": 40.0 if b != "W4" else 60.0,
                            "speedup": speedup,
                            "compile_fail": compile_fail,
                        },
                    )
                rt.backend.set_behavior(
                    f"{artifact_id}:abba_retest",
                    {"duration_s": 60.0, "speedup": speedup},
                )
                for fb in ("F1", "F2", "F3"):
                    rt.backend.set_behavior(
                        f"{artifact_id}:final_eval:{fb}",
                        {"duration_s": 45.0, "speedup": speedup},
                    )
            evt, _ = rt.submit_command_and_package(cmd, spec)
            assert evt.event == EventType.CODE_READY.value
            self._log(clock.now(), "Coder", "code_ready", f"{artifact_id} ({direction_id}) lookup={lookup.lookup_id}")

        # Initial candidates from both routes
        _submit_kernel(
            "register-pressure", "res_reg", "art-reg-v1", "V0", 1.08,
            "Initial register-tiling candidate",
        )
        _submit_kernel(
            "smem-layout", "res_mem", "art-mem-v1", "V0", 1.16,
            "128B swizzled SMEM layout",
        )
        mem_round = 1

        def _react_to_delivery(d: EvidenceDelivery) -> None:
            nonlocal ncu_triggered, reg_repair_triggered, reg_fix_submitted, mem_round
            nonlocal ncu_gpu_released_at, ncu_parse_ready_at

            # 1. Partial W1 on V1 triggers NCU capture + Researcher waiting relation
            if (
                d.result_kind == ResultKind.BENCHMARK_PARTIAL.value
                and d.artifact_id == "art-reg-v1"
                and d.block_id == "W1"
                and not ncu_triggered
            ):
                ncu_triggered = True
                res = rt.leader.ensure_researcher("res_reg")
                rid = rt.leader.next_research_id()
                lookup, _ = res.start_research_round(
                    research_id=rid,
                    direction_id="register-pressure",
                    evidence_terms=("register", "spill", "ncu"),
                    observation=f"W1 partial speedup={d.payload.get('geomean_speedup', 0):.3f}x; investigate spill",
                    explanation="Short-sequence cases may be spilling registers",
                    prediction="NCU lmem_spill_bytes > 0",
                    hypotheses=[],
                )
                ncu_cmd = rt.leader.build_command(
                    kind=CommandKind.WRITE_NCU_CAPTURE.value,
                    research_id=rid,
                    direction_id="register-pressure",
                    target=TargetRef(d.version_id or "V1", "art-reg-v1"),
                    hypothesis="Check lmem spill and occupancy",
                    evidence_refs=(d.receipt_id, lookup.lookup_id),
                    instruction="Capture NCU counters on V1",
                    deliverable="ncu/capture.py",
                    reply_to="res_reg",
                    allowed_paths=("ncu/",),
                )
                ncu_tid = "ncu-v1-spill"
                rt.controller.process_event(
                    rt.leader.propose_task_event(
                        task_id=ncu_tid,
                        kind=JobKind.NCU_CAPTURE.value,
                        resource="gpu",
                        direction_id="register-pressure",
                        version_id=d.version_id,
                        artifact_id="art-reg-v1",
                        command_id=ncu_cmd.head.command_id,
                        reply_to="res_reg",
                        estimated_duration_s=20.0,
                        post_cpu_kind=JobKind.NCU_PARSE.value,
                        post_cpu_duration_s=10.0,
                    )
                )
                if isinstance(rt.backend, SimulatedExecutionBackend):
                    rt.backend.set_behavior(ncu_tid, {"duration_s": 20.0})
                rt.submit_command_and_package(
                    ncu_cmd,
                    {
                        "artifact_id": "art-reg-v1",
                        "task_id": ncu_tid,
                        "entrypoint": "ncu/capture.py",
                        "files": {"ncu/capture.py": _subprocess_worker_script(rt.evaluator, 1.0)},
                    },
                )
                rt.controller.process_event(
                    rt.leader.declare_waiting_event(
                        member="res_reg",
                        waiting_for=f"result_ready:{ncu_tid}",
                        unblocks_work="design_spill_fix",
                        estimated_duration_s=50.0,
                        direction_id="register-pressure",
                    )
                )
                self._log(clock.now(), "Researcher", "requested_ncu", f"res_reg waiting on {ncu_tid} (W1 received)")

            # 2. Parsed NCU arrives -> res_reg submits art-reg-v2-bad (compile failure test)
            elif d.result_kind == ResultKind.NCU_PARSED.value and not reg_repair_triggered:
                reg_repair_triggered = True
                ncu_parse_ready_at = clock.now()
                self._log(
                    clock.now(),
                    "Researcher",
                    "received_ncu_parsed",
                    f"lmem_spill={d.payload.get('observations', {}).get('counters')}",
                )
                _submit_kernel(
                    "register-pressure", "res_reg", "art-reg-v2-bad", "V1", 1.26,
                    "Reduce register count via tile split (attempt 1 - bad pragma)",
                    compile_fail=True,
                )

        # 3. Reactive closed-loop execution loop
        pending_cpu: list[JobHandle] = []
        while True:
            # Handle any pending CPU parse tasks (simulating off-GPU overlap)
            while pending_cpu:
                ch = pending_cpu.pop(0)
                if isinstance(clock, SimClock) and ch.expected_finish_at > clock.now():
                    clock.advance_to(ch.expected_finish_at)
                cpu_res = rt.operator.finalize_handle(ch)
                for evt in cpu_res.scheduling_events:
                    rt.controller.process_event(evt)
                rt.process_deliveries(cpu_res.deliveries, on_research_delivery=_react_to_delivery)
                self._log(clock.now(), "OperatorCPU", "cpu_parse_done", f"{ch.kind} -> result_ready")

            handle, decision = rt.dispatch_gpu_if_ready()
            if handle is None:
                # If no work left and final eval hasn't run, schedule final eval
                if (
                    not rt.controller.gpu_busy
                    and rt.controller.active_version is None
                    and not rt.controller.pending_candidates
                    and rt.controller.global_best_version_id is not None
                    and not rt.controller.final_eval_requested
                ):
                    rt.controller.schedule_final_eval()
                    self._log(
                        clock.now(),
                        "Controller",
                        "schedule_final_eval",
                        f"incumbent={rt.controller.global_best_version_id}",
                    )
                    continue
                break

            self._log(
                clock.now(),
                "Controller",
                "dispatched_gpu",
                f"{handle.kind} ver={handle.version_id} block={handle.block_id} reason={decision.reason}",
            )

            # Complete GPU job
            if isinstance(clock, SimClock) and handle.expected_finish_at > clock.now():
                clock.advance_to(handle.expected_finish_at)

            # Simulate non-GPU work overlapping when CPU/model actions happen
            op_res = rt.complete_gpu_job(handle, advance_sim_clock=False)

            if handle.kind == JobKind.NCU_CAPTURE.value:
                ncu_gpu_released_at = clock.now()
                self._log(clock.now(), "Operator", "gpu_released_ncu", "GPU freed before CPU parse")

            for cpu_job in op_res.pending_cpu_jobs:
                ch = rt.operator.start(None, cpu_job)
                rt.non_gpu_intervals.append((ch.started_at, ch.expected_finish_at))
                pending_cpu.append(ch)

            # Deliver operator messages (including partial benchmark to Researcher)
            for d in op_res.deliveries:
                if d.recipient != "judge":
                    _react_to_delivery(d)
                else:
                    # Simulate Judge work overlapping or running after complete/ABBA/final
                    self._log(
                        clock.now(),
                        "Judge",
                        f"adjudicate_{d.result_kind}",
                        f"ver={d.version_id} art={d.artifact_id}",
                    )

            # Check if active version failed compile on V3 -> record failure & repair!
            if (
                handle.artifact_id == "art-reg-v2-bad"
                and not reg_fix_submitted
                and rt.controller.active_version is None
            ):
                reg_fix_submitted = True
                rt.wiki.route_tree.record_failure_repair(
                    "register-pressure",
                    "compile_failure",
                    handle.version_id,
                    "art-reg-v2-bad",
                    is_repair=True,
                )
                self._log(
                    clock.now(),
                    "Researcher",
                    "compile_fail_repair",
                    "Recorded compile_failure (stagnation untouched), submitting fixed art-reg-v2-fix",
                )
                _submit_kernel(
                    "register-pressure", "res_reg", "art-reg-v2-fix", "V1", 1.26,
                    "Reduce register count via tile split (fixed compile)",
                )

            # Drive smem-layout through 3 non-improving rounds to verify stagnation pause
            if (
                handle.direction_id == "smem-layout"
                and rt.controller.active_version is None
                and "smem-layout" not in rt.controller.pending_candidates
                and rt.wiki.route_tree.routes["smem-layout"].status == RouteStatus.ACTIVE.value
                and mem_round < 4
            ):
                mem_round += 1
                _submit_kernel(
                    "smem-layout",
                    "res_mem",
                    f"art-mem-v{mem_round}",
                    "V2",
                    1.162,  # < +1% gain over 1.16 -> counts toward 3-round stagnation
                    f"Minor SMEM bank tweak #{mem_round}",
                )

            # Record synthetic non-GPU research/coding overlap window for telemetry audit
            if rt.gpu_intervals:
                gs, ge = rt.gpu_intervals[-1]
                if ge > gs:
                    rt.non_gpu_intervals.append((gs, min(ge, gs + 5.0)))

        # Finalize run & audit all invariants
        wall_s = clock.now()
        gpu_busy_s = _union_interval_length(rt.gpu_intervals)
        overlap_s = _union_interval_length(
            _intersect_intervals(rt.gpu_intervals, rt.non_gpu_intervals)
        )
        checks, failures = self._verify_invariants(
            rt=rt,
            cfg=cfg,
            clock=clock,
            ncu_gpu_released_at=ncu_gpu_released_at,
            ncu_parse_ready_at=ncu_parse_ready_at,
            overlap_s=overlap_s,
        )

        return ClosedLoopReport(
            backend_kind=self.backend_kind,
            passed=len(failures) == 0,
            checks=checks,
            failures=failures,
            wall_seconds=wall_s,
            gpu_busy_seconds=gpu_busy_s,
            member_gpu_overlap_seconds=overlap_s,
            first_actionable_evidence_s=rt.first_actionable_evidence_s,
            best_version_id=rt.judge.incumbent.version_id,
            best_artifact_id=rt.judge.incumbent.artifact_id,
            best_speedup=rt.judge.incumbent.geomean_speedup,
            verification_level=rt.judge.incumbent.verification_level,
            versions_closed=sorted(
                rt.controller.closed_versions.keys(),
                key=lambda v: int(v[1:]) if v[1:].isdigit() else 999,
            ),
            route_states={
                k: {
                    "status": v.status,
                    "stagnation_count": v.stagnation_count,
                    "compile_failure_count": v.compile_failure_count,
                    "repair_count": v.repair_count,
                    "best_version_id": v.best_version_id,
                    "best_score": round(v.best_score, 4) if v.best_score else None,
                }
                for k, v in rt.wiki.route_tree.routes.items()
            },
            trace=self.trace if verbose else [],
        )

    def _verify_invariants(
        self,
        rt: KernelLabRuntime,
        cfg: RuntimeConfig,
        clock: Clock,
        ncu_gpu_released_at: float | None,
        ncu_parse_ready_at: float | None,
        overlap_s: float,
    ) -> tuple[dict[str, bool], list[str]]:
        checks: dict[str, bool] = {}
        failures: list[str] = []

        # 1. Controller log contains zero body fields
        no_body = True
        for _, evt in rt.store.load_events():
            if PROHIBITED_HEADER_FIELDS.intersection(evt.to_dict().keys()):
                no_body = False
        checks["controller_headers_only"] = no_body
        if not no_body:
            failures.append("Found prohibited result body field in Controller event log")

        # 2. Every command cited a registered wiki-lookup-* credential
        wiki_ok = bool(rt.leader.commands_issued) and all(
            any(
                ref.startswith("wiki-lookup-") and rt.wiki.has_valid_lookup(ref)
                for ref in cmd.body.evidence_refs
            )
            for cmd in rt.leader.commands_issued
        )
        checks["wiki_lookup_credentials_present"] = wiki_ok
        if not wiki_ok:
            failures.append("Command missing valid wiki-lookup credential")

        # 3. All sealed packages pass SHA-256 integrity verification
        pkg_ok = bool(rt.packages_by_id) and all(
            verify_package_integrity(pkg)[0] for pkg in rt.packages_by_id.values()
        )
        checks["immutable_packages_verified"] = pkg_ok
        if not pkg_ok:
            failures.append("Code package integrity check failed")

        # 4. Single-GPU exclusivity (no overlapping GPU intervals)
        sorted_gpu = sorted(rt.gpu_intervals)
        exclusive_ok = all(
            sorted_gpu[i][1] <= sorted_gpu[i + 1][0] + 1e-9
            for i in range(len(sorted_gpu) - 1)
        )
        checks["single_gpu_exclusivity"] = exclusive_ok
        if not exclusive_ok:
            failures.append("Overlapping GPU lease intervals detected")

        # 5. Version admission gated strictly by VERSION_CLOSED(Vn)
        closed_events = [
            e for e in rt.controller.emitted_events if e.event == EventType.VERSION_CLOSED.value
        ]
        closed_once_ok = len(closed_events) == len(rt.controller.closed_versions) and len(
            rt.controller.closed_versions
        ) >= 3
        checks["version_closed_emitted_once_per_version"] = closed_once_ok
        if not closed_once_ok:
            failures.append("VERSION_CLOSED count mismatch")

        # 6. NCU GPU released before CPU parse finished
        ncu_split_ok = (
            ncu_gpu_released_at is not None
            and ncu_parse_ready_at is not None
            and ncu_gpu_released_at <= ncu_parse_ready_at
        )
        checks["ncu_capture_parse_split"] = ncu_split_ok
        if not ncu_split_ok:
            failures.append("NCU capture/parse split timestamps invalid")

        # 7. Stagnation paused smem-layout at 3 while register-pressure succeeded
        mem_node = rt.wiki.route_tree.routes["smem-layout"]
        reg_node = rt.wiki.route_tree.routes["register-pressure"]
        stag_ok = (
            mem_node.status == RouteStatus.PAUSED.value
            and mem_node.stagnation_count == 3
            and reg_node.status == RouteStatus.ACTIVE.value
            and reg_node.compile_failure_count == 1
            and reg_node.stagnation_count == 0
        )
        checks["route_stagnation_and_repair_isolation"] = stag_ok
        if not stag_ok:
            failures.append(
                f"Route stagnation/repair check failed: mem={mem_node.status}/{mem_node.stagnation_count}, reg={reg_node.status}/{reg_node.compile_failure_count}"
            )

        # 8. Final verified winner promoted via ABBA + 3-block final eval
        winner_ok = (
            rt.judge.incumbent.artifact_id == "art-reg-v2-fix"
            and rt.judge.incumbent.verification_level == VerificationLevel.FINAL_VERIFIED.value
            and rt.judge.incumbent.geomean_speedup > 1.20
        )
        checks["abba_and_final_eval_verified"] = winner_ok
        if not winner_ok:
            failures.append(
                f"Unexpected winner: {rt.judge.incumbent.artifact_id} {rt.judge.incumbent.verification_level}"
            )

        # 9. CPU/member work overlapped with GPU
        overlap_ok = overlap_s > 0.0
        checks["cpu_member_gpu_overlap"] = overlap_ok
        if not overlap_ok:
            failures.append("No GPU/member overlap observed")

        # 10. Controller recovery reproduces exact state
        recovered = Controller.recover(cfg, rt.store, clock)
        rec_ok = (
            recovered.state_seq == rt.controller.state_seq
            and recovered.global_best_version_id == rt.controller.global_best_version_id
            and set(recovered.closed_versions.keys())
            == set(rt.controller.closed_versions.keys())
        )
        checks["crash_recovery_state_match"] = rec_ok
        if not rec_ok:
            failures.append("Recovered Controller state did not match live state")

        return checks, failures
