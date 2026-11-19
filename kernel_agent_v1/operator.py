"""Operator: fixed execution runner, GPU lease enforcement, CPU/GPU split & dual-channel routing."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from kernel_agent_v1.backends import (
    CleanupReceipt,
    ExecutionBackend,
    JobHandle,
    RawOutcome,
)
from kernel_agent_v1.clock import Clock
from kernel_agent_v1.coder import verify_package_integrity
from kernel_agent_v1.evaluator import TrustedEvaluator
from kernel_agent_v1.protocol import (
    CodePackage,
    Event,
    EventType,
    EvidenceDelivery,
    JobKind,
    ResultKind,
)
from kernel_agent_v1.store import RunStore


@dataclass
class OperatorDispatchResult:
    """Outputs produced by Operator when a job finishes or is stopped."""

    scheduling_events: list[Event] = field(default_factory=list)
    deliveries: list[EvidenceDelivery] = field(default_factory=list)
    pending_cpu_jobs: list[dict[str, Any]] = field(default_factory=list)


class Operator:
    """Fixed executor owning start/wait/stop, trusted aggregation, and dual-channel output (§2, §4, §5)."""

    def __init__(
        self,
        run_id: str,
        backend: ExecutionBackend,
        evaluator: TrustedEvaluator,
        store: RunStore,
        clock: Clock,
    ) -> None:
        self.run_id = run_id
        self.backend = backend
        self.evaluator = evaluator
        self.store = store
        self.clock = clock
        self._job_seq = 0
        self._evt_seq = 0
        self._receipt_seq = 0
        self._delivery_seq = 0

        self.active_gpu_handle: JobHandle | None = None
        self.active_cpu_handles: dict[str, JobHandle] = {}
        self.route_researchers: dict[str, str] = {}
        # Cache verified block payloads keyed by (version_id, artifact_id) -> {block_id: payload}
        self._version_blocks: dict[tuple[str, str], dict[str, dict[str, Any]]] = {}
        self._complete_sent_versions: set[tuple[str, str]] = set()
        self.gpu_cleanup_confirmed: bool = True

    def register_route_researcher(self, direction_id: str, researcher_id: str) -> None:
        self.route_researchers[direction_id] = researcher_id

    def _next_job_id(self) -> str:
        self._job_seq += 1
        return f"job-{self._job_seq:04d}"

    def _next_event_id(self) -> str:
        self._evt_seq += 1
        return f"evt-op-{self._evt_seq:05d}"

    def _next_receipt_id(self) -> str:
        self._receipt_seq += 1
        return f"receipt-{self._receipt_seq:04d}"

    def _next_delivery_id(self) -> str:
        self._delivery_seq += 1
        return f"deliv-{self._delivery_seq:05d}"

    def start(
        self,
        package: CodePackage | None,
        inputs: dict[str, Any],
        attempt_num: int = 1,
        job_id: str | None = None,
    ) -> JobHandle:
        """Unified start(package, inputs) interface (§4)."""
        resource = str(inputs["resource"])
        if resource == "gpu":
            if not self.gpu_cleanup_confirmed:
                raise RuntimeError("Cannot start GPU job while previous GPU cleanup is unconfirmed")
            if self.active_gpu_handle is not None:
                raise RuntimeError(
                    f"Single GPU already occupied by {self.active_gpu_handle.attempt_id}"
                )

        if package is not None:
            ok, msg = verify_package_integrity(package)
            if not ok:
                raise ValueError(f"Immutable package integrity check failed: {msg}")

        jid = job_id or self._next_job_id()
        attempt_id = f"{jid}/{attempt_num}"
        handle = self.backend.start(
            job_id=jid,
            attempt_id=attempt_id,
            task_id=str(inputs["task_id"]),
            kind=str(inputs["kind"]),
            resource=resource,
            package=package,
            inputs=inputs,
            clock=self.clock,
        )
        if resource == "gpu":
            self.active_gpu_handle = handle
        else:
            self.active_cpu_handles[attempt_id] = handle
        return handle

    def wait(self, handle: JobHandle) -> RawOutcome:
        return self.backend.wait(handle, self.clock)

    def stop(self, handle: JobHandle, reason: str) -> CleanupReceipt:
        """Idempotent stop(handle, reason) confirming GPU cleanup before lease release."""
        receipt = self.backend.stop(handle, reason, self.clock)
        if handle.resource == "gpu":
            if receipt.gpu_released:
                if self.active_gpu_handle and self.active_gpu_handle.attempt_id == handle.attempt_id:
                    self.active_gpu_handle = None
                self.gpu_cleanup_confirmed = True
            else:
                self.gpu_cleanup_confirmed = False
        else:
            self.active_cpu_handles.pop(handle.attempt_id, None)
        return receipt

    def finalize_handle(
        self,
        handle: JobHandle,
        outcome: RawOutcome | None = None,
    ) -> OperatorDispatchResult:
        """Collect outcome, persist evidence, and emit dual-channel headers + direct deliveries."""
        if outcome is None:
            outcome = self.wait(handle)

        if handle.resource == "gpu":
            if outcome.gpu_released:
                if self.active_gpu_handle and self.active_gpu_handle.attempt_id == handle.attempt_id:
                    self.active_gpu_handle = None
                self.gpu_cleanup_confirmed = True
            else:
                self.gpu_cleanup_confirmed = False
        else:
            self.active_cpu_handles.pop(handle.attempt_id, None)

        res = OperatorDispatchResult()
        receipt_id = self._next_receipt_id()

        if handle.kind == JobKind.BENCHMARK_BLOCK.value:
            return self._handle_benchmark_block(handle, outcome, receipt_id, res)
        if handle.kind == JobKind.NCU_CAPTURE.value:
            return self._handle_ncu_capture(handle, outcome, receipt_id, res)
        if handle.kind == JobKind.NCU_PARSE.value:
            return self._handle_ncu_parse(handle, outcome, receipt_id, res)
        if handle.kind in (JobKind.PROBE.value, JobKind.CPU_PROBE.value):
            return self._handle_probe(handle, outcome, receipt_id, res)
        if handle.kind in (
            JobKind.COMPILER_ANALYSIS.value,
            JobKind.GPU_COMPILER_ANALYSIS.value,
            JobKind.CPU_COMPILE.value,
        ):
            return self._handle_compiler_or_compile(handle, outcome, receipt_id, res)
        if handle.kind == JobKind.ABBA_RETEST.value:
            return self._handle_abba_or_final(
                handle, outcome, receipt_id, ResultKind.ABBA_RETEST.value, res
            )
        if handle.kind == JobKind.FINAL_EVAL.value:
            return self._handle_abba_or_final(
                handle, outcome, receipt_id, ResultKind.FINAL_EVAL.value, res
            )
        return res

    def _researcher_recipient(self, handle: JobHandle) -> str:
        if handle.kind == JobKind.BENCHMARK_BLOCK.value:
            return self.route_researchers.get(handle.direction_id, handle.reply_to)
        return handle.reply_to

    def _emit_gpu_or_cpu_header(
        self,
        handle: JobHandle,
        outcome: RawOutcome,
        receipt_id: str,
        result_kind: str | None,
        status_override: str | None = None,
    ) -> Event:
        status = status_override or outcome.status
        if handle.resource == "gpu":
            return Event(
                event=EventType.GPU_JOB_FINISHED.value,
                event_id=self._next_event_id(),
                run_id=self.run_id,
                producer="operator",
                timestamp=self.clock.now_iso(),
                sim_time=self.clock.now(),
                direction_id=handle.direction_id,
                version_id=handle.version_id,
                artifact_id=handle.artifact_id,
                job_id=handle.job_id,
                attempt_id=handle.attempt_id,
                kind=handle.kind,
                block_id=handle.block_id,
                status=status,
                gpu_released=outcome.gpu_released,
                receipt_id=receipt_id,
                result_kind=result_kind,
                extra={
                    "task_id": handle.task_id,
                    "duration_s": outcome.duration_s,
                    "error": outcome.error,
                },
            )
        return Event(
            event=EventType.RESULT_READY.value,
            event_id=self._next_event_id(),
            run_id=self.run_id,
            producer="operator",
            timestamp=self.clock.now_iso(),
            sim_time=self.clock.now(),
            direction_id=handle.direction_id,
            version_id=handle.version_id,
            artifact_id=handle.artifact_id,
            job_id=handle.job_id,
            attempt_id=handle.attempt_id,
            kind=handle.kind,
            block_id=handle.block_id,
            status=status,
            receipt_id=receipt_id,
            result_kind=result_kind,
            extra={
                "task_id": handle.task_id,
                "duration_s": outcome.duration_s,
                "error": outcome.error,
                "parent_gpu_task_id": handle.metadata.get("inputs", {}).get(
                    "parent_gpu_task_id"
                ),
            },
        )

    def _deliver(
        self,
        receipt_id: str,
        recipient: str,
        result_kind: str,
        handle: JobHandle,
        payload: dict[str, Any],
    ) -> EvidenceDelivery:
        ev_path = self.store.write_evidence(receipt_id, payload)
        return EvidenceDelivery(
            delivery_id=self._next_delivery_id(),
            receipt_id=receipt_id,
            recipient=recipient,
            result_kind=result_kind,
            producer="operator",
            timestamp=self.clock.now_iso(),
            version_id=handle.version_id,
            artifact_id=handle.artifact_id,
            direction_id=handle.direction_id,
            command_id=handle.metadata.get("inputs", {}).get("command_id"),
            block_id=handle.block_id,
            evidence_path=str(ev_path),
            payload=payload,
        )

    def _handle_benchmark_block(
        self,
        handle: JobHandle,
        outcome: RawOutcome,
        receipt_id: str,
        res: OperatorDispatchResult,
    ) -> OperatorDispatchResult:
        if outcome.status in ("infra_failed", "timed_out", "cancelled", "compile_failed"):
            res.scheduling_events.append(
                self._emit_gpu_or_cpu_header(
                    handle, outcome, receipt_id, ResultKind.BENCHMARK_PARTIAL.value
                )
            )
            if outcome.payload:
                res.deliveries.append(
                    self._deliver(
                        receipt_id=receipt_id,
                        recipient=self._researcher_recipient(handle),
                        result_kind=ResultKind.BENCHMARK_PARTIAL.value,
                        handle=handle,
                        payload={
                            "receipt_id": receipt_id,
                            "status": outcome.status,
                            "error": outcome.error,
                            **outcome.payload,
                        },
                    )
                )
            return res

        block_id = handle.block_id or "W1"
        raw_cases = outcome.payload.get("cases", [])
        block_eval = self.evaluator.verify_block(
            block_id=block_id,
            raw_cases=raw_cases,
            expected_version_id=handle.version_id or "",
            expected_artifact_id=handle.artifact_id or "",
            expected_attempt_id=handle.attempt_id,
        )

        # Save block payload to disk and route partial result directly to Researcher
        block_payload = {
            "receipt_id": receipt_id,
            "result_kind": ResultKind.BENCHMARK_PARTIAL.value,
            "version_id": handle.version_id,
            "artifact_id": handle.artifact_id,
            "block_id": block_id,
            "protocol": outcome.payload.get("protocol", self.evaluator.protocol),
            "device_id": outcome.payload.get("device_id", self.evaluator.device_id),
            "attempt_id": handle.attempt_id,
            "all_correct": block_eval.all_correct,
            "valid": block_eval.valid,
            "issues": block_eval.issues,
            "geomean_speedup": block_eval.geomean_speedup,
            "cases": raw_cases,
        }
        res.deliveries.append(
            self._deliver(
                receipt_id=receipt_id,
                recipient=self._researcher_recipient(handle),
                result_kind=ResultKind.BENCHMARK_PARTIAL.value,
                handle=handle,
                payload=block_payload,
            )
        )

        # Header-only notification to Controller (never includes geomean or cases!)
        status_for_ctrl = "completed" if block_eval.all_correct else "numerical_failed"
        res.scheduling_events.append(
            self._emit_gpu_or_cpu_header(
                handle,
                outcome,
                receipt_id,
                ResultKind.BENCHMARK_PARTIAL.value,
                status_override=status_for_ctrl,
            )
        )

        # Record block in aggregation cache (even if block has issues, Judge checks complete coverage)
        if handle.version_id and handle.artifact_id and block_eval.all_correct:
            vkey = (handle.version_id, handle.artifact_id)
            self._version_blocks.setdefault(vkey, {})[block_id] = block_payload

            expected_blocks = set(
                handle.metadata.get("inputs", {}).get(
                    "required_blocks", list(self.evaluator.blocks.keys())
                )
            )
            if (
                vkey not in self._complete_sent_versions
                and set(self._version_blocks[vkey].keys()) == expected_blocks
            ):
                self._complete_sent_versions.add(vkey)
                agg = self.evaluator.aggregate_complete(
                    block_payloads=self._version_blocks[vkey],
                    expected_version_id=handle.version_id,
                    expected_artifact_id=handle.artifact_id,
                )
                comp_receipt_id = self._next_receipt_id()
                comp_payload = {
                    "receipt_id": comp_receipt_id,
                    "result_kind": ResultKind.BENCHMARK_COMPLETE.value,
                    "version_id": agg.version_id,
                    "artifact_id": agg.artifact_id,
                    "protocol": agg.protocol,
                    "device_id": agg.device_id,
                    "total_cases": agg.total_cases,
                    "unique_uuids": agg.unique_uuids,
                    "all_correct": agg.all_correct,
                    "geomean_speedup": agg.geomean_speedup,
                    "valid": agg.valid,
                    "issues": agg.issues,
                    "block_receipts": agg.block_receipts,
                    "blocks": self._version_blocks[vkey],
                }
                # Direct evidence delivery to Judge
                res.deliveries.append(
                    self._deliver(
                        receipt_id=comp_receipt_id,
                        recipient="judge",
                        result_kind=ResultKind.BENCHMARK_COMPLETE.value,
                        handle=handle,
                        payload=comp_payload,
                    )
                )
                # Header-only BENCHMARK_COMPLETE to Controller
                res.scheduling_events.append(
                    Event(
                        event=EventType.BENCHMARK_COMPLETE.value,
                        event_id=self._next_event_id(),
                        run_id=self.run_id,
                        producer="operator",
                        timestamp=self.clock.now_iso(),
                        sim_time=self.clock.now(),
                        direction_id=handle.direction_id,
                        version_id=handle.version_id,
                        artifact_id=handle.artifact_id,
                        receipt_id=comp_receipt_id,
                        result_kind=ResultKind.BENCHMARK_COMPLETE.value,
                        status="valid" if agg.valid else "invalid",
                        extra={"valid": agg.valid},
                    )
                )

        return res

    def _handle_ncu_capture(
        self,
        handle: JobHandle,
        outcome: RawOutcome,
        receipt_id: str,
        res: OperatorDispatchResult,
    ) -> OperatorDispatchResult:
        # Release GPU first; do NOT mark parsed result ready yet!
        res.scheduling_events.append(
            self._emit_gpu_or_cpu_header(
                handle, outcome, receipt_id, ResultKind.NCU_RAW.value
            )
        )
        raw_payload = {
            "receipt_id": receipt_id,
            "result_kind": ResultKind.NCU_RAW.value,
            "version_id": handle.version_id,
            "artifact_id": handle.artifact_id,
            "status": outcome.status,
            "error": outcome.error,
            **outcome.payload,
        }
        self.store.write_evidence(receipt_id, raw_payload)

        if outcome.status == "completed":
            # Schedule follow-up CPU parse job off-GPU
            inputs = handle.metadata.get("inputs", {})
            res.pending_cpu_jobs.append(
                {
                    "task_id": f"{handle.task_id}__parse",
                    "kind": JobKind.NCU_PARSE.value,
                    "resource": "cpu",
                    "version_id": handle.version_id,
                    "artifact_id": handle.artifact_id,
                    "direction_id": handle.direction_id,
                    "reply_to": handle.reply_to,
                    "command_id": inputs.get("command_id"),
                    "parent_gpu_task_id": handle.task_id,
                    "raw_ncu_report": outcome.payload.get("raw_ncu_report", {}),
                    "duration_s": inputs.get("post_cpu_duration_s", 10.0),
                    "sim_behavior": inputs.get("parse_sim_behavior", {}),
                }
            )
        return res

    def _handle_ncu_parse(
        self,
        handle: JobHandle,
        outcome: RawOutcome,
        receipt_id: str,
        res: OperatorDispatchResult,
    ) -> OperatorDispatchResult:
        res.scheduling_events.append(
            self._emit_gpu_or_cpu_header(
                handle, outcome, receipt_id, ResultKind.NCU_PARSED.value
            )
        )
        res.deliveries.append(
            self._deliver(
                receipt_id=receipt_id,
                recipient=self._researcher_recipient(handle),
                result_kind=ResultKind.NCU_PARSED.value,
                handle=handle,
                payload={
                    "receipt_id": receipt_id,
                    "result_kind": ResultKind.NCU_PARSED.value,
                    "status": outcome.status,
                    **outcome.payload,
                },
            )
        )
        return res

    def _handle_probe(
        self,
        handle: JobHandle,
        outcome: RawOutcome,
        receipt_id: str,
        res: OperatorDispatchResult,
    ) -> OperatorDispatchResult:
        res.scheduling_events.append(
            self._emit_gpu_or_cpu_header(
                handle, outcome, receipt_id, ResultKind.PROBE_RESULT.value
            )
        )
        if handle.resource == "gpu" and outcome.status == "completed":
            # Also emit result_ready for tasks waiting on result_ready:<task_id>
            res.scheduling_events.append(
                Event(
                    event=EventType.RESULT_READY.value,
                    event_id=self._next_event_id(),
                    run_id=self.run_id,
                    producer="operator",
                    timestamp=self.clock.now_iso(),
                    sim_time=self.clock.now(),
                    direction_id=handle.direction_id,
                    version_id=handle.version_id,
                    artifact_id=handle.artifact_id,
                    job_id=handle.job_id,
                    attempt_id=handle.attempt_id,
                    kind=handle.kind,
                    status=outcome.status,
                    receipt_id=receipt_id,
                    result_kind=ResultKind.PROBE_RESULT.value,
                    extra={"task_id": handle.task_id},
                )
            )
        res.deliveries.append(
            self._deliver(
                receipt_id=receipt_id,
                recipient=self._researcher_recipient(handle),
                result_kind=ResultKind.PROBE_RESULT.value,
                handle=handle,
                payload={
                    "receipt_id": receipt_id,
                    "result_kind": ResultKind.PROBE_RESULT.value,
                    "status": outcome.status,
                    "error": outcome.error,
                    **outcome.payload,
                },
            )
        )
        return res

    def _handle_compiler_or_compile(
        self,
        handle: JobHandle,
        outcome: RawOutcome,
        receipt_id: str,
        res: OperatorDispatchResult,
    ) -> OperatorDispatchResult:
        res.scheduling_events.append(
            self._emit_gpu_or_cpu_header(
                handle, outcome, receipt_id, ResultKind.COMPILER_ANALYSIS.value
            )
        )
        res.deliveries.append(
            self._deliver(
                receipt_id=receipt_id,
                recipient=self._researcher_recipient(handle),
                result_kind=ResultKind.COMPILER_ANALYSIS.value,
                handle=handle,
                payload={
                    "receipt_id": receipt_id,
                    "result_kind": ResultKind.COMPILER_ANALYSIS.value,
                    "status": outcome.status,
                    "error": outcome.error,
                    **outcome.payload,
                },
            )
        )
        return res

    def _handle_abba_or_final(
        self,
        handle: JobHandle,
        outcome: RawOutcome,
        receipt_id: str,
        result_kind: str,
        res: OperatorDispatchResult,
    ) -> OperatorDispatchResult:
        res.scheduling_events.append(
            self._emit_gpu_or_cpu_header(handle, outcome, receipt_id, result_kind)
        )
        res.deliveries.append(
            self._deliver(
                receipt_id=receipt_id,
                recipient="judge",
                result_kind=result_kind,
                handle=handle,
                payload={
                    "receipt_id": receipt_id,
                    "result_kind": result_kind,
                    "status": outcome.status,
                    "error": outcome.error,
                    **outcome.payload,
                },
            )
        )
        return res
