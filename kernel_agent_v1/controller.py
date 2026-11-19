"""Controller: exclusive state/lease/version writer and dynamic lookahead scheduler (§2, §5, §6, §7)."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from kernel_agent_v1.clock import Clock
from kernel_agent_v1.duration import DurationModel
from kernel_agent_v1.evaluator import BLOCK_IDS, FINAL_BLOCK_IDS
from kernel_agent_v1.protocol import (
    BudgetConfig,
    BudgetLedger,
    Event,
    EventType,
    JobKind,
    OPTIONAL_DIAGNOSTIC_KINDS,
    RuntimeConfig,
    TaskSpec,
    TaskStatus,
    VerificationLevel,
    VersionRecord,
    VersionStatus,
    assert_header_only,
)
from kernel_agent_v1.scheduler import (
    CandidateScoreRecord,
    DynamicScheduler,
    SchedulingDecision,
)
from kernel_agent_v1.store import RunStore


class Controller:
    """Single writer of scheduling state, GPU lease, version numbers, and VERSION_CLOSED."""

    def __init__(
        self,
        config: RuntimeConfig,
        store: RunStore,
        clock: Clock,
        duration_model: DurationModel | None = None,
    ) -> None:
        self.config = config
        self.store = store
        self.clock = clock
        self.duration_model = duration_model or DurationModel()
        self.scheduler = DynamicScheduler(config=self.config, duration_model=self.duration_model)

        self.state_seq: int = 0
        self.processed_event_ids: set[str] = set()
        self._events_since_snapshot: int = 0

        # GPU lease & safety gate
        self.gpu_busy: bool = False
        self.gpu_cleanup_confirmed: bool = True
        self.gpu_halt_reason: str | None = None
        self.active_gpu_attempt_id: str | None = None
        self.active_gpu_task_id: str | None = None
        self.active_gpu_job_id: str | None = None
        self.gpu_lease_started_at: float | None = None

        # Versions
        self.version_seq: int = 0
        self.active_version: VersionRecord | None = None
        self.closed_versions: dict[str, VersionRecord] = {}
        self._closed_version_ids_emitted: set[str] = set()

        # Tasks, packages, waiting relations
        self.tasks: dict[str, TaskSpec] = {}
        self.packages_ready: dict[str, dict[str, Any]] = {}  # package_id -> info
        # Pending candidate kernel per route (max_pending_per_route = 1)
        self.pending_candidates: dict[str, dict[str, Any]] = {}  # direction_id -> info
        self.candidate_ready_seq: int = 0
        self.route_priorities: dict[str, float] = {}
        self.paused_routes: set[str] = set()
        self.route_stats: dict[str, dict[str, Any]] = {}

        # Dependency tokens & member/worker tracking
        self.satisfied_tokens: set[str] = set()
        self.failed_tokens: set[str] = set()
        self.waiting_relations: dict[str, dict[str, Any]] = {}
        self.worker_states: dict[str, dict[str, Any]] = {}

        # Global incumbent & final eval
        self.global_best_version_id: str | None = None
        self.global_best_artifact_id: str | None = None
        self.verification_level: str = VerificationLevel.BASELINE_ONLY.value
        self.final_eval_requested: bool = False
        self.final_eval_completed: bool = False
        self.final_eval_version_id: str | None = None

        # Budgets & termination
        self.ledger = BudgetLedger(wall_start_s=clock.now())
        self.run_finished: bool = False
        self.stop_reason: str | None = None

        # Counters & explainability
        self._task_seq = 0
        self._evt_seq = 0
        self.latest_tentative_decision: SchedulingDecision | None = None
        self.decision_history: list[SchedulingDecision] = []
        self.emitted_events: list[Event] = []

    def _next_event_id(self) -> str:
        self._evt_seq += 1
        return f"evt-ctrl-{self._evt_seq:05d}"

    def _next_task_id(self, prefix: str) -> str:
        self._task_seq += 1
        return f"{prefix}-{self._task_seq:04d}"

    # ------------------------------------------------------------------
    # Event ingestion & atomic state commit (§5)
    # ------------------------------------------------------------------
    def process_event(self, event: Event, persist: bool = True) -> list[Event]:
        """Process a scheduling header event atomically; returns any Controller-emitted events."""
        assert_header_only(event.to_dict())
        if event.event_id in self.processed_event_ids:
            return []

        self.processed_event_ids.add(event.event_id)
        self.state_seq += 1
        emitted: list[Event] = []

        if persist:
            self.store.append_event(event, self.state_seq)

        etype = event.event
        if etype == EventType.TASK_PROPOSED.value:
            self._on_task_proposed(event)
        elif etype == EventType.TASK_WAITING.value:
            self._on_task_waiting(event)
        elif etype == EventType.PRIORITY_CHANGED.value:
            self._on_priority_changed(event)
        elif etype == EventType.TASK_WITHDRAWN.value:
            self._on_task_withdrawn(event)
        elif etype == EventType.CODE_READY.value:
            self._on_code_ready(event)
        elif etype == EventType.CODE_BLOCKED.value:
            self._on_code_blocked(event)
        elif etype == EventType.WORKER_STATE_CHANGED.value:
            self._on_worker_state_changed(event)
        elif etype == EventType.GPU_JOB_FINISHED.value:
            self._on_gpu_job_finished(event)
        elif etype == EventType.RESULT_READY.value:
            self._on_result_ready(event)
        elif etype == EventType.BENCHMARK_COMPLETE.value:
            self._on_benchmark_complete(event)
        elif etype == EventType.JUDGE_RESULT.value:
            self._on_judge_result(event)
        elif etype == EventType.VERSION_CLOSED.value:
            # Replay path on restart
            if event.version_id:
                self.satisfied_tokens.add(f"VERSION_CLOSED:{event.version_id}")
                self._closed_version_ids_emitted.add(event.version_id)

        self._check_budget_limits()
        closed_evt = self._try_close_active_version(persist=persist)
        if closed_evt is not None:
            emitted.append(closed_evt)

        self.latest_tentative_decision = self.recompute_plan()

        if persist:
            self._events_since_snapshot += 1
            force_snap = (
                closed_evt is not None
                or self.gpu_halt_reason is not None
                or self.run_finished
                or self._events_since_snapshot >= max(1, self.config.snapshot_interval)
            )
            if force_snap:
                self.flush_snapshot()

        return emitted

    def flush_snapshot(self) -> None:
        """Persist full Controller snapshot to disk and reset dirty event counter."""
        self.store.save_snapshot(self.to_dict())
        self._events_since_snapshot = 0

    def _on_task_proposed(self, event: Event) -> None:
        x = event.extra
        task_id = str(x.get("task_id") or self._next_task_id("task"))
        replaces = x.get("replaces_task_id")
        if replaces and replaces in self.tasks:
            old = self.tasks[replaces]
            if old.status == TaskStatus.PENDING.value:
                old.status = TaskStatus.CANCELLED.value

        direction_id = event.direction_id or ""
        priority = float(x.get("priority", self.route_priorities.get(direction_id, 1.0)))
        if direction_id:
            self.route_priorities.setdefault(direction_id, priority)

        task = TaskSpec(
            task_id=task_id,
            kind=event.kind or str(x.get("kind", "research")),
            resource=str(x.get("resource", "model")),
            direction_id=direction_id,
            version_id=event.version_id,
            artifact_id=event.artifact_id,
            parent_version_id=event.parent_version_id,
            command_id=event.command_id,
            package_id=x.get("package_id"),
            block_id=event.block_id,
            is_required=bool(x.get("is_required", False)),
            priority=priority,
            assigned_member=x.get("assigned_member") or x.get("member"),
            reply_to=str(x.get("reply_to", "researcher")),
            depends_on=list(x.get("depends_on", [])),
            produces=list(x.get("produces", [])),
            estimated_duration_s=x.get("estimated_duration_s"),
            duration_category=str(x.get("duration_category", "normal")),
            timeout_s=x.get("timeout_s"),
            template_id=x.get("template_id"),
            workload_count=int(x.get("workload_count", 1)),
            post_cpu_kind=x.get("post_cpu_kind"),
            post_cpu_duration_s=x.get("post_cpu_duration_s"),
            ready_at=event.sim_time,
            unblocks_work=x.get("unblocks_work"),
            replaces_task_id=replaces,
            metadata=dict(x.get("metadata", {})),
        )
        # If targeting a version that is already closing/closed, optional tasks skip immediately.
        if (
            task.resource == "gpu"
            and not task.is_required
            and task.version_id
            and (
                task.version_id in self.closed_versions
                or (
                    self.active_version is not None
                    and self.active_version.version_id == task.version_id
                    and self.active_version.status != VersionStatus.ACTIVE.value
                )
            )
        ):
            task.status = TaskStatus.SKIPPED.value
        self.tasks[task_id] = task

    def _on_task_waiting(self, event: Event) -> None:
        x = event.extra
        wait_id = str(x.get("wait_id") or f"wait:{x.get('member')}:{x.get('waiting_for')}")
        self.waiting_relations[wait_id] = {
            "wait_id": wait_id,
            "member": str(x.get("member") or event.producer),
            "waiting_for": str(x.get("waiting_for", "")),
            "unblocks_work": str(x.get("unblocks_work", "research")),
            "estimated_duration_s": float(x.get("estimated_duration_s", 30.0)),
            "direction_id": event.direction_id,
            "active": True,
            "registered_at": event.sim_time,
            "deadline_s": x.get("deadline_s"),
        }

    def _on_priority_changed(self, event: Event) -> None:
        x = event.extra
        direction_id = event.direction_id or ""
        if "priority" in x and direction_id:
            new_prio = float(x["priority"])
            self.route_priorities[direction_id] = new_prio
            if direction_id in self.pending_candidates:
                self.pending_candidates[direction_id]["priority"] = new_prio
            for task in self.tasks.values():
                if task.direction_id == direction_id and task.status == TaskStatus.PENDING.value:
                    task.priority = new_prio
        if x.get("route_status") == "paused" and direction_id:
            self.paused_routes.add(direction_id)
            self.pending_candidates.pop(direction_id, None)
        elif x.get("route_status") == "active" and direction_id:
            self.paused_routes.discard(direction_id)

    def _on_task_withdrawn(self, event: Event) -> None:
        x = event.extra
        task_id = x.get("task_id")
        if task_id and task_id in self.tasks:
            t = self.tasks[task_id]
            if t.status == TaskStatus.PENDING.value:
                t.status = TaskStatus.CANCELLED.value
        wait_id = x.get("wait_id")
        if wait_id and wait_id in self.waiting_relations:
            self.waiting_relations[wait_id]["active"] = False
        waiting_for = x.get("waiting_for")
        member = x.get("member")
        if waiting_for or member:
            for rel in self.waiting_relations.values():
                if (waiting_for is None or rel["waiting_for"] == waiting_for) and (
                    member is None or rel["member"] == member
                ):
                    rel["active"] = False

    def _on_code_ready(self, event: Event) -> None:
        x = event.extra
        package_id = str(x.get("package_id") or "")
        direction_id = event.direction_id or ""
        info = {
            "package_id": package_id,
            "package_digest": x.get("package_digest"),
            "artifact_id": event.artifact_id,
            "parent_version_id": event.parent_version_id or "V0",
            "command_id": event.command_id,
            "direction_id": direction_id,
            "kind": event.kind,
            "executable_stages": list(x.get("executable_stages", [])),
            "reply_to": str(x.get("reply_to", "researcher")),
            "ready_at": event.sim_time,
            "depends_on": list(x.get("depends_on", [])),
            "priority": self.route_priorities.get(direction_id, 1.0),
            "task_id": x.get("task_id"),
        }
        if package_id:
            self.packages_ready[package_id] = info
        self.satisfied_tokens.add(f"code_ready:{package_id}")
        if event.command_id:
            self.satisfied_tokens.add(f"code_ready:{event.command_id}")

        if event.kind == "implement_kernel":
            if direction_id in self.paused_routes or self.version_cap_reached:
                return
            self.candidate_ready_seq += 1
            info["ready_seq"] = self.candidate_ready_seq
            self.route_stats.setdefault(
                direction_id, {"ready_count": 0, "skip_count": 0, "wait_time_s": 0.0}
            )["ready_count"] += 1
            # Max 1 pending candidate per route (§7): newer replaces older pending candidate on same route
            self.pending_candidates[direction_id] = info
        elif event.kind in ("write_ncu_capture", "write_probe"):
            target_vid = x.get("target_version_id") or (
                self.active_version.version_id if self.active_version else None
            )
            linked_tid = x.get("task_id")
            if linked_tid and linked_tid in self.tasks:
                t = self.tasks[linked_tid]
                t.package_id = package_id
                t.artifact_id = t.artifact_id or event.artifact_id
                t.ready_at = event.sim_time
                if target_vid:
                    t.version_id = target_vid
                if t.kind == JobKind.NCU_CAPTURE.value and not t.post_cpu_kind:
                    t.post_cpu_kind = JobKind.NCU_PARSE.value
                    t.post_cpu_duration_s = t.post_cpu_duration_s or 10.0
                # Check if target version is already closing/closed
                if (
                    target_vid
                    and (
                        target_vid in self.closed_versions
                        or (
                            self.active_version
                            and self.active_version.version_id == target_vid
                            and self.active_version.status != VersionStatus.ACTIVE.value
                        )
                    )
                ):
                    t.status = TaskStatus.SKIPPED.value
            else:
                gpu_kind = (
                    JobKind.NCU_CAPTURE.value
                    if event.kind == "write_ncu_capture"
                    else JobKind.PROBE.value
                )
                tid = self._next_task_id("diag")
                is_closed_or_closing = bool(
                    target_vid
                    and (
                        target_vid in self.closed_versions
                        or (
                            self.active_version
                            and self.active_version.version_id == target_vid
                            and self.active_version.status != VersionStatus.ACTIVE.value
                        )
                    )
                )
                self.tasks[tid] = TaskSpec(
                    task_id=tid,
                    kind=gpu_kind,
                    resource="gpu",
                    direction_id=direction_id,
                    version_id=target_vid,
                    artifact_id=event.artifact_id,
                    command_id=event.command_id,
                    package_id=package_id,
                    is_required=False,
                    priority=self.route_priorities.get(direction_id, 1.0),
                    reply_to=info["reply_to"],
                    ready_at=event.sim_time,
                    post_cpu_kind=(
                        JobKind.NCU_PARSE.value
                        if gpu_kind == JobKind.NCU_CAPTURE.value
                        else None
                    ),
                    post_cpu_duration_s=(
                        10.0 if gpu_kind == JobKind.NCU_CAPTURE.value else None
                    ),
                    status=(
                        TaskStatus.SKIPPED.value
                        if is_closed_or_closing
                        else TaskStatus.PENDING.value
                    ),
                )
        elif event.kind == "write_compiler_analysis":
            tid = x.get("task_id") or self._next_task_id("cpuanalysis")
            if tid in self.tasks:
                self.tasks[tid].package_id = package_id
                self.tasks[tid].ready_at = event.sim_time
            else:
                self.tasks[tid] = TaskSpec(
                    task_id=tid,
                    kind=JobKind.COMPILER_ANALYSIS.value,
                    resource="cpu",
                    direction_id=direction_id,
                    version_id=x.get("target_version_id"),
                    artifact_id=event.artifact_id,
                    command_id=event.command_id,
                    package_id=package_id,
                    is_required=False,
                    priority=self.route_priorities.get(direction_id, 1.0),
                    reply_to=info["reply_to"],
                    ready_at=event.sim_time,
                )

    def _on_code_blocked(self, event: Event) -> None:
        if event.command_id:
            self.failed_tokens.add(f"code_ready:{event.command_id}")
        tid = event.extra.get("task_id")
        if tid and tid in self.tasks:
            self.tasks[tid].status = TaskStatus.FAILED.value
            self.failed_tokens.add(f"result_ready:{tid}")

    def _on_worker_state_changed(self, event: Event) -> None:
        x = event.extra
        worker_id = str(x.get("worker_id") or event.producer)
        state = str(x.get("worker_state") or event.status or "idle")
        est_dur = float(x.get("estimated_duration_s", 0.0))
        self.worker_states[worker_id] = {
            "state": state,
            "task_id": x.get("task_id"),
            "started_at": event.sim_time if state == "busy" else None,
            "busy_until": (event.sim_time + est_dur) if state == "busy" else event.sim_time,
        }
        # If worker started from another report, deactivate matching wait if indicated
        cleared_wait_for = x.get("clears_wait_for")
        cleared_wait_id = x.get("wait_id")
        if cleared_wait_id and cleared_wait_id in self.waiting_relations:
            self.waiting_relations[cleared_wait_id]["active"] = False
        if cleared_wait_for:
            for rel in self.waiting_relations.values():
                if rel["member"] == worker_id and rel["waiting_for"] == cleared_wait_for:
                    rel["active"] = False

    def _on_gpu_job_finished(self, event: Event) -> None:
        # Stale attempt check (§5, §7): ignore receipts from non-active attempts
        if event.attempt_id != self.active_gpu_attempt_id:
            return

        x = event.extra
        task_id = str(x.get("task_id") or self.active_gpu_task_id or "")
        task = self.tasks.get(task_id)
        duration_s = float(
            x.get(
                "duration_s",
                max(0.0, event.sim_time - (self.gpu_lease_started_at or event.sim_time)),
            )
        )
        self.ledger.gpu_seconds_used += duration_s

        if event.gpu_released:
            self.gpu_busy = False
            self.gpu_cleanup_confirmed = True
            self.active_gpu_attempt_id = None
            self.active_gpu_task_id = None
            self.active_gpu_job_id = None
        else:
            # Unconfirmed cleanup halts normal GPU dispatch (§7, §9)
            self.gpu_busy = True
            self.gpu_cleanup_confirmed = False
            self.gpu_halt_reason = "gpu_cleanup_unconfirmed"

        if task is None:
            return

        if (
            self.active_version is not None
            and task.version_id == self.active_version.version_id
            and task.kind in OPTIONAL_DIAGNOSTIC_KINDS
        ):
            self.active_version.diag_spent_s += duration_s

        timed_out = event.status == "timed_out"
        self._record_task_duration(task, duration_s, event.sim_time, timed_out=timed_out)

        # Transient infra retry (1x on same frozen artifact & inputs)
        if (
            event.status == "infra_failed"
            and task.retry_count < task.max_retries
            and self.active_version is not None
            and task.version_id == self.active_version.version_id
            and self.active_version.status != VersionStatus.CLOSED.value
        ):
            task.retry_count += 1
            task.status = TaskStatus.PENDING.value
            return

        if event.status in (
            "compile_failed",
            "candidate_failed",
            "numerical_failed",
            "infra_failed",
            "timed_out",
            "cancelled",
        ):
            task.status = (
                TaskStatus.TIMED_OUT.value
                if event.status == "timed_out"
                else (
                    TaskStatus.CANCELLED.value
                    if event.status == "cancelled"
                    else TaskStatus.FAILED.value
                )
            )
            task.finished_at = event.sim_time
            self.failed_tokens.add(f"result_ready:{task.task_id}")
            self._deactivate_waits_for(f"result_ready:{task.task_id}")

            if (
                self.active_version is not None
                and task.version_id == self.active_version.version_id
                and task.is_required
            ):
                if task.kind == JobKind.BENCHMARK_BLOCK.value and task.block_id:
                    self.active_version.failed_blocks.append(task.block_id)
                    self.active_version.retest_required = False
                    self.active_version.retest_status = "not_needed"
                    self.active_version.close_reason = f"{event.status}:{task.block_id}"
                elif task.kind == JobKind.ABBA_RETEST.value:
                    self.active_version.retest_status = "failed"
                    self.active_version.close_reason = f"abba_{event.status}"
                self._enter_closing(self.active_version, event.sim_time)
            return

        # Completed normally
        task.status = TaskStatus.COMPLETED.value
        task.finished_at = event.sim_time

        if task.kind != JobKind.NCU_CAPTURE.value:
            self.satisfied_tokens.add(f"result_ready:{task.task_id}")
            self._deactivate_waits_for(f"result_ready:{task.task_id}")

        if (
            self.active_version is not None
            and task.version_id == self.active_version.version_id
        ):
            v = self.active_version
            if task.kind == JobKind.BENCHMARK_BLOCK.value and task.block_id:
                if task.block_id not in v.completed_blocks:
                    v.completed_blocks.append(task.block_id)
                if event.receipt_id:
                    v.block_receipts[task.block_id] = event.receipt_id
                if set(v.completed_blocks) == set(v.required_blocks):
                    self._enter_closing(v, event.sim_time)
            elif task.kind == JobKind.ABBA_RETEST.value:
                v.retest_status = "completed"
                if event.receipt_id:
                    v.close_receipt_ids.append(event.receipt_id)

    def _on_result_ready(self, event: Event) -> None:
        x = event.extra
        task_id = str(x.get("task_id") or "")
        parent_gpu_tid = x.get("parent_gpu_task_id")
        dur = float(x.get("duration_s", 0.0))

        if task_id and task_id in self.tasks:
            t = self.tasks[task_id]
            t.status = (
                TaskStatus.COMPLETED.value
                if event.status == "completed"
                else TaskStatus.FAILED.value
            )
            t.finished_at = event.sim_time
            self._record_task_duration(
                t, dur, event.sim_time, timed_out=(event.status == "timed_out")
            )

        if event.status == "completed":
            if task_id:
                self.satisfied_tokens.add(f"result_ready:{task_id}")
                self._deactivate_waits_for(f"result_ready:{task_id}")
            if parent_gpu_tid:
                self.satisfied_tokens.add(f"result_ready:{parent_gpu_tid}")
                self._deactivate_waits_for(f"result_ready:{parent_gpu_tid}")
        else:
            if task_id:
                self.failed_tokens.add(f"result_ready:{task_id}")
                self._deactivate_waits_for(f"result_ready:{task_id}")
            if parent_gpu_tid:
                self.failed_tokens.add(f"result_ready:{parent_gpu_tid}")
                self._deactivate_waits_for(f"result_ready:{parent_gpu_tid}")

    def _on_benchmark_complete(self, event: Event) -> None:
        v = self.active_version
        if v is None or v.version_id != event.version_id or v.status == VersionStatus.CLOSED.value:
            return
        v.benchmark_complete_sent = True
        v.benchmark_complete_receipt_id = event.receipt_id
        if event.receipt_id:
            v.close_receipt_ids.append(event.receipt_id)
        self.satisfied_tokens.add(f"BENCHMARK_COMPLETE:{v.version_id}")
        self._enter_closing(v, event.sim_time)

    def _on_judge_result(self, event: Event) -> None:
        x = event.extra
        verdict = str(x.get("verdict") or event.status or "")
        ver_id = event.version_id

        if x.get("incumbent_id"):
            self.global_best_version_id = str(x["incumbent_id"])
            if x.get("incumbent_artifact_id"):
                self.global_best_artifact_id = str(x["incumbent_artifact_id"])
        if x.get("verification_level"):
            self.verification_level = str(x["verification_level"])

        if ver_id:
            self.satisfied_tokens.add(f"judge_result:{ver_id}")

        if x.get("final_eval_done"):
            self.final_eval_completed = True

        # Never reopen a closed version (§5, §7)
        if ver_id and ver_id in self.closed_versions:
            return

        v = self.active_version
        if v is None or (ver_id and v.version_id != ver_id):
            return

        if event.receipt_id and event.receipt_id not in v.close_receipt_ids:
            v.close_receipt_ids.append(event.receipt_id)

        if verdict == "needs_retest":
            v.retest_required = True
            v.retest_status = "scheduled"
            incumbent_version_id = x.get("expected_incumbent_id", "V0")
            incumbent_artifact_id = x.get("expected_incumbent_artifact_id", "fla-baseline")
            abba_tid = self._next_task_id(f"abba-{v.version_id}")
            self.tasks[abba_tid] = TaskSpec(
                task_id=abba_tid,
                kind=JobKind.ABBA_RETEST.value,
                resource="gpu",
                direction_id=v.direction_id,
                version_id=v.version_id,
                artifact_id=v.artifact_id,
                package_id=self._benchmark_package_id(v.version_id, v.artifact_id),
                is_required=True,
                priority=v.priority + 100.0,
                timeout_s=self.config.budgets.abba_retest_job_timeout_s,
                reply_to="judge",
                ready_at=event.sim_time,
                metadata={
                    "incumbent_version_id": incumbent_version_id,
                    "incumbent_artifact_id": incumbent_artifact_id,
                    "incumbent_package_id": self._benchmark_package_id(
                        incumbent_version_id, incumbent_artifact_id
                    ),
                    "incumbent_speedup": x.get("incumbent_speedup", 1.0),
                    "speedup": x.get("candidate_speedup", 1.05),
                },
            )
        elif verdict in (
            "promoted",
            "rejected_no_gain",
            "rejected_invalid",
            "rejected_abba",
            "rejected_abba_failed",
        ):
            if v.retest_required is None:
                v.retest_required = False
                v.retest_status = "not_needed"

    def _enter_closing(self, v: VersionRecord, now: float) -> None:
        if v.status == VersionStatus.ACTIVE.value:
            v.status = VersionStatus.CLOSING.value
            v.judge_decision_deadline_at = now + self.config.budgets.judge_decision_timeout_s
            # Skip unstarted optional diagnostics for this version (§7)
            for t in self.tasks.values():
                if (
                    t.version_id == v.version_id
                    and not t.is_required
                    and t.resource == "gpu"
                    and t.status == TaskStatus.PENDING.value
                ):
                    t.status = TaskStatus.SKIPPED.value
                    self.failed_tokens.add(f"result_ready:{t.task_id}")
                    self._deactivate_waits_for(f"result_ready:{t.task_id}")

    def _try_close_active_version(self, persist: bool = True) -> Event | None:
        v = self.active_version
        if v is None or v.status == VersionStatus.CLOSED.value:
            return None
        if v.version_id in self._closed_version_ids_emitted:
            return None
        if self.gpu_busy or not self.gpu_cleanup_confirmed:
            return None

        now = self.clock.now()
        # Did version fail or time out?
        if v.failed_blocks or (now >= v.deadline_at and set(v.completed_blocks) != set(v.required_blocks)):
            if v.status == VersionStatus.ACTIVE.value:
                self._enter_closing(v, now)
            v.close_reason = v.close_reason or "failed_or_timeout"
            # Cancel any remaining unstarted required tasks of this version
            for t in self.tasks.values():
                if t.version_id == v.version_id and t.status == TaskStatus.PENDING.value:
                    t.status = TaskStatus.SKIPPED.value
            return self._finalize_version_close(v, now, persist=persist)

        # Normal path: all required blocks complete
        if set(v.completed_blocks) != set(v.required_blocks):
            return None
        if not v.benchmark_complete_sent:
            return None

        # Wait for Judge's time-bounded decision on whether ABBA retest is needed (§7)
        if v.retest_required is None:
            if v.judge_decision_deadline_at is not None and now >= v.judge_decision_deadline_at:
                v.retest_required = False
                v.retest_status = "decision_timed_out"
                v.close_reason = "judge_decision_timeout"
            else:
                return None

        if v.retest_required and v.retest_status not in ("completed", "failed"):
            return None

        v.close_reason = v.close_reason or "completed"
        return self._finalize_version_close(v, now, persist=persist)

    def _finalize_version_close(
        self, v: VersionRecord, now: float, persist: bool
    ) -> Event:
        v.status = VersionStatus.CLOSED.value
        v.closed_at = now
        self._closed_version_ids_emitted.add(v.version_id)
        self.closed_versions[v.version_id] = v
        self.active_version = None
        self.satisfied_tokens.add(f"VERSION_CLOSED:{v.version_id}")

        evt = Event(
            event=EventType.VERSION_CLOSED.value,
            event_id=self._next_event_id(),
            run_id=self.config.run_id,
            producer="controller",
            timestamp=self.clock.now_iso(),
            sim_time=now,
            direction_id=v.direction_id,
            version_id=v.version_id,
            artifact_id=v.artifact_id,
            status="closed",
            extra={
                "close_reason": v.close_reason,
                "receipt_ids": list(v.close_receipt_ids),
            },
        )
        self.processed_event_ids.add(evt.event_id)
        self.state_seq += 1
        if persist:
            self.store.append_event(evt, self.state_seq)
        self.emitted_events.append(evt)
        return evt

    def _deactivate_waits_for(self, token: str) -> None:
        for rel in self.waiting_relations.values():
            if rel["waiting_for"] == token:
                rel["active"] = False

    # ------------------------------------------------------------------
    # Version admission, dispatch, and timeout handling
    # ------------------------------------------------------------------
    def _deps_met(self, deps: list[str]) -> tuple[bool, str | None]:
        for d in deps:
            if d in self.failed_tokens or f"failed:{d}" in self.failed_tokens:
                return False, f"dep_failed:{d}"
            if d not in self.satisfied_tokens:
                return False, f"deps_unmet:{d}"
        return True, None

    def _check_budget_limits(self) -> None:
        if self.final_eval_requested:
            return  # Admissions are closed while final eval consumes its reserve.
        b = self.config.budgets
        now = self.clock.now()
        wall_elapsed = now - self.ledger.wall_start_s
        gpu_rem = b.total_gpu_budget_s - self.ledger.gpu_seconds_used
        wall_rem = b.total_wall_budget_s - wall_elapsed
        model_rem = b.total_model_budget_cost - self.ledger.model_cost_used

        final_reserve = (
            b.reserve_final_eval_s
            if (self.global_best_version_id is not None and not self.final_eval_completed)
            else 0.0
        )
        min_version_need = (
            len(BLOCK_IDS) * b.reserve_required_block_s + b.reserve_retest_s + final_reserve
        )
        if gpu_rem < min_version_need or wall_rem < min_version_need or model_rem <= 0:
            self.freeze_admissions("budget_near_exhaustion")

    def freeze_admissions(self, reason: str) -> None:
        if not self.ledger.admission_frozen:
            self.ledger.admission_frozen = True
            self.ledger.freeze_reason = reason

    @property
    def version_cap_reached(self) -> bool:
        cap = self.config.max_versions
        return cap is not None and self.version_seq >= cap

    @property
    def version_search_finished(self) -> bool:
        return (
            self.config.max_versions is None
            or self.version_cap_reached
            or (self.config.finalize_when_admission_frozen and self.ledger.admission_frozen)
        )

    def admit_next_version(self) -> VersionRecord | None:
        """Admit Vn+1 from pending candidates ONLY when no version is active/closing and GPU is clean."""
        if self.active_version is not None:
            return None
        if self.gpu_busy or not self.gpu_cleanup_confirmed or self.gpu_halt_reason:
            return None
        if self.ledger.admission_frozen or self.final_eval_requested or self.version_cap_reached:
            return None

        now = self.clock.now()
        eligible: list[tuple[float, int, str, dict[str, Any]]] = []
        for direction_id, cand in list(self.pending_candidates.items()):
            if direction_id in self.paused_routes:
                continue
            ok, _ = self._deps_met(cand.get("depends_on", []))
            if not ok:
                continue
            prio = float(self.route_priorities.get(direction_id, cand.get("priority", 1.0)))
            eligible.append((-prio, int(cand.get("ready_seq", 0)), direction_id, cand))

        if not eligible:
            return None

        eligible.sort(key=lambda item: (item[0], item[1]))
        _, _, chosen_dir, chosen = eligible[0]

        # Record skip stats for other ready candidates
        for _, _, d_id, cand in eligible[1:]:
            st = self.route_stats.setdefault(
                d_id, {"ready_count": 0, "skip_count": 0, "wait_time_s": 0.0}
            )
            st["skip_count"] += 1
            st["wait_time_s"] = now - float(cand["ready_at"])

        self.pending_candidates.pop(chosen_dir, None)
        self.version_seq += 1
        vid = f"V{self.version_seq}"
        self.store.append_lifecycle({
            "event": "version_admitted", "timestamp": self.clock.now_iso(),
            "version_id": vid, "artifact_id": chosen["artifact_id"],
            "package_id": chosen.get("package_id"), "direction_id": chosen_dir,
            "source_parent_version": chosen.get("parent_version_id") or "V0",
        })
        art_id = str(chosen["artifact_id"])
        parent_vid = str(chosen.get("parent_version_id") or "V0")
        b = self.config.budgets

        v = VersionRecord(
            version_id=vid,
            artifact_id=art_id,
            source_parent_version=parent_vid,
            direction_id=chosen_dir,
            priority=float(self.route_priorities.get(chosen_dir, 1.0)),
            reply_to=str(chosen.get("reply_to", "researcher")),
            admitted_at=now,
            deadline_at=now + b.version_deadline_s,
            diag_budget_s=b.version_diag_budget_s,
            required_blocks=list(BLOCK_IDS),
        )
        self.active_version = v

        # Materialize the 4 required benchmark block GPU tasks for Vn
        for block_id in BLOCK_IDS:
            tid = self._next_task_id(f"bench-{vid}-{block_id}")
            self.tasks[tid] = TaskSpec(
                task_id=tid,
                kind=JobKind.BENCHMARK_BLOCK.value,
                resource="gpu",
                direction_id=chosen_dir,
                version_id=vid,
                artifact_id=art_id,
                parent_version_id=parent_vid,
                command_id=chosen.get("command_id"),
                package_id=chosen.get("package_id"),
                block_id=block_id,
                is_required=True,
                priority=v.priority,
                reply_to=v.reply_to,
                ready_at=now,
                timeout_s=b.default_job_timeout_s,
            )

        # Register Judge's expected adjudication task for horizon-H lookahead scoring (§6)
        judge_tid = self._next_task_id(f"judge-{vid}")
        self.tasks[judge_tid] = TaskSpec(
            task_id=judge_tid,
            kind=JobKind.JUDGE_EVAL.value,
            resource="model",
            direction_id=chosen_dir,
            version_id=vid,
            artifact_id=art_id,
            is_required=False,
            priority=v.priority,
            assigned_member="judge",
            depends_on=[f"BENCHMARK_COMPLETE:{vid}"],
            estimated_duration_s=50.0,
            ready_at=now,
        )

        # Re-bind any optional diagnostic tasks proposed for this artifact/command before admission
        for t in self.tasks.values():
            if (
                t.status == TaskStatus.PENDING.value
                and t.version_id is None
                and (
                    (t.artifact_id and t.artifact_id == art_id)
                    or (t.command_id and t.command_id == chosen.get("command_id"))
                )
            ):
                t.version_id = vid

        self.flush_snapshot()
        return v

    def _benchmark_package_id(self, version_id: str, artifact_id: str) -> str | None:
        return next(
            (
                task.package_id
                for task in self.tasks.values()
                if task.kind == JobKind.BENCHMARK_BLOCK.value
                and task.version_id == version_id
                and task.artifact_id == artifact_id
                and task.package_id
            ),
            None,
        )

    def schedule_final_eval(self) -> list[TaskSpec]:
        """Register the 3 independent final eval GPU blocks for the retained incumbent (§7)."""
        if (
            self.final_eval_requested
            or self.active_version is not None
            or self.gpu_busy
            or not self.version_search_finished
            or self.global_best_version_id is None
            or self.global_best_artifact_id is None
        ):
            return []
        self.final_eval_requested = True
        self.final_eval_version_id = self.global_best_version_id
        self.store.append_lifecycle({
            "event": "final_eval_scheduled", "timestamp": self.clock.now_iso(),
            "version_id": self.global_best_version_id, "artifact_id": self.global_best_artifact_id,
            "last_admitted_version": self.version_seq, "blocks": FINAL_BLOCK_IDS,
        })
        package_id = self._benchmark_package_id(
            self.global_best_version_id, self.global_best_artifact_id
        )
        budgets = self.config.budgets
        timeout_s = budgets.final_eval_job_timeout_s or budgets.default_job_timeout_s
        now = self.clock.now()
        created: list[TaskSpec] = []
        for fb in FINAL_BLOCK_IDS:
            tid = self._next_task_id(f"final-{self.global_best_version_id}-{fb}")
            spec = TaskSpec(
                task_id=tid,
                kind=JobKind.FINAL_EVAL.value,
                resource="gpu",
                direction_id="final_eval",
                version_id=self.global_best_version_id,
                artifact_id=self.global_best_artifact_id,
                package_id=package_id,
                block_id=fb,
                is_required=True,
                priority=1000.0,
                reply_to="judge",
                ready_at=now,
                timeout_s=timeout_s,
            )
            self.tasks[tid] = spec
            created.append(spec)
        return created

    def select_next_gpu_task(self) -> tuple[TaskSpec | None, SchedulingDecision]:
        """Select the next GPU task to dispatch when GPU is idle and clean."""
        if self.gpu_halt_reason:
            dec = SchedulingDecision(
                state_seq=self.state_seq,
                sim_time=self.clock.now(),
                scheduler_mode=self.config.scheduler_mode,
                selected_task_id=None,
                reason=f"halted:{self.gpu_halt_reason}",
            )
            return None, dec

        if self.gpu_busy or not self.gpu_cleanup_confirmed:
            dec = SchedulingDecision(
                state_seq=self.state_seq,
                sim_time=self.clock.now(),
                scheduler_mode=self.config.scheduler_mode,
                selected_task_id=None,
                reason="gpu_busy_or_unconfirmed",
            )
            return None, dec

        # Try closing active version first if ready
        self._try_close_active_version(persist=True)

        # If no active version and not running final eval, try admitting next version
        if self.active_version is None and not self.final_eval_requested:
            self.admit_next_version()

        decision = self.recompute_plan()
        self.decision_history.append(decision)
        self.store.append_decision(decision.to_dict())
        if decision.selected_task_id is None:
            return None, decision
        return self.tasks[decision.selected_task_id], decision

    def mark_gpu_dispatched(
        self, task: TaskSpec, job_id: str, attempt_id: str
    ) -> None:
        now = self.clock.now()
        task.status = TaskStatus.RUNNING.value
        task.started_at = now
        self.gpu_busy = True
        self.active_gpu_task_id = task.task_id
        self.active_gpu_job_id = job_id
        self.active_gpu_attempt_id = attempt_id
        self.gpu_lease_started_at = now
        self.flush_snapshot()

    # ------------------------------------------------------------------
    # Delegation to DynamicScheduler & DurationModel (§6)
    # ------------------------------------------------------------------
    def _record_task_duration(
        self,
        task: TaskSpec,
        duration_s: float,
        finished_at: float,
        timed_out: bool = False,
    ) -> None:
        if task.kind == JobKind.BENCHMARK_BLOCK.value:
            k, fb = DurationModel.benchmark_keys(
                self.config.gpu_name,
                self.config.protocol_name,
                task.block_id or "W1",
            )
        elif task.kind in (JobKind.NCU_CAPTURE.value, JobKind.PROBE.value, JobKind.GPU_COMPILER_ANALYSIS.value):
            k, fb = DurationModel.diagnostic_keys(
                task.template_id or task.kind,
                task.workload_count,
                self.config.gpu_name,
            )
        elif task.resource == "cpu":
            k, fb = DurationModel.cpu_keys(self.config.toolchain, "warm", task.kind)
        else:
            k, fb = DurationModel.model_keys(
                task.assigned_member or "leader", task.kind, task.duration_category
            )
        self.duration_model.record_sample(
            group_key=k,
            duration_s=duration_s,
            timed_out=timed_out,
            fallback_keys=fb,
            task_id=task.task_id,
            ready_at=task.ready_at,
            started_at=task.started_at,
            gpu_released_at=finished_at if task.resource == "gpu" else None,
            result_ready_at=finished_at,
        )

    def recompute_plan(self) -> SchedulingDecision:
        """Recompute tentative plan on every event via DynamicScheduler (§1, §6)."""
        return self.scheduler.recompute_plan(
            state_seq=self.state_seq,
            now=self.clock.now(),
            tasks=self.tasks,
            active_version=self.active_version,
            final_eval_requested=self.final_eval_requested,
            ledger=self.ledger,
            satisfied_tokens=self.satisfied_tokens,
            waiting_relations=self.waiting_relations,
            worker_states=self.worker_states,
            deps_met_fn=self._deps_met,
        )

    # ------------------------------------------------------------------
    # Serialization & Restart Recovery (§5)
    # ------------------------------------------------------------------
    def to_dict(self) -> dict[str, Any]:
        return {
            "state_seq": self.state_seq,
            "processed_event_ids": sorted(self.processed_event_ids),
            "gpu_busy": self.gpu_busy,
            "gpu_cleanup_confirmed": self.gpu_cleanup_confirmed,
            "gpu_halt_reason": self.gpu_halt_reason,
            "active_gpu_attempt_id": self.active_gpu_attempt_id,
            "active_gpu_task_id": self.active_gpu_task_id,
            "active_gpu_job_id": self.active_gpu_job_id,
            "gpu_lease_started_at": self.gpu_lease_started_at,
            "version_seq": self.version_seq,
            "active_version": self.active_version.to_dict() if self.active_version else None,
            "closed_versions": {k: v.to_dict() for k, v in self.closed_versions.items()},
            "closed_version_ids_emitted": sorted(self._closed_version_ids_emitted),
            "tasks": {k: v.to_dict() for k, v in self.tasks.items()},
            "packages_ready": self.packages_ready,
            "pending_candidates": self.pending_candidates,
            "candidate_ready_seq": self.candidate_ready_seq,
            "route_priorities": self.route_priorities,
            "paused_routes": sorted(self.paused_routes),
            "satisfied_tokens": sorted(self.satisfied_tokens),
            "failed_tokens": sorted(self.failed_tokens),
            "waiting_relations": self.waiting_relations,
            "worker_states": self.worker_states,
            "global_best_version_id": self.global_best_version_id,
            "global_best_artifact_id": self.global_best_artifact_id,
            "verification_level": self.verification_level,
            "final_eval_requested": self.final_eval_requested,
            "final_eval_completed": self.final_eval_completed,
            "final_eval_version_id": self.final_eval_version_id,
            "ledger": self.ledger.to_dict(),
            "run_finished": self.run_finished,
            "stop_reason": self.stop_reason,
            "task_seq": self._task_seq,
            "evt_seq": self._evt_seq,
            "duration_model": self.duration_model.to_dict(),
        }

    @classmethod
    def recover(
        cls,
        config: RuntimeConfig,
        store: RunStore,
        clock: Clock,
    ) -> "Controller":
        """Recover committed snapshot and replay any unapplied tail events from log."""
        ctrl = cls(config=config, store=store, clock=clock)
        snap = store.load_snapshot()
        if snap is not None:
            ctrl._load_snapshot(snap)

        events = store.load_events()
        for seq, evt in events:
            if evt.event_id in ctrl.processed_event_ids:
                continue
            ctrl.process_event(evt, persist=False)
        return ctrl

    def _load_snapshot(self, snap: dict[str, Any]) -> None:
        self.state_seq = int(snap["state_seq"])
        self.processed_event_ids = set(snap["processed_event_ids"])
        self.gpu_busy = bool(snap["gpu_busy"])
        self.gpu_cleanup_confirmed = bool(snap["gpu_cleanup_confirmed"])
        self.gpu_halt_reason = snap.get("gpu_halt_reason")
        self.active_gpu_attempt_id = snap.get("active_gpu_attempt_id")
        self.active_gpu_task_id = snap.get("active_gpu_task_id")
        self.active_gpu_job_id = snap.get("active_gpu_job_id")
        self.gpu_lease_started_at = snap.get("gpu_lease_started_at")
        self.version_seq = int(snap["version_seq"])
        av = snap.get("active_version")
        self.active_version = VersionRecord(**av) if av else None
        self.closed_versions = {
            k: VersionRecord(**v) for k, v in snap.get("closed_versions", {}).items()
        }
        self._closed_version_ids_emitted = set(snap.get("closed_version_ids_emitted", []))
        self.tasks = {k: TaskSpec(**v) for k, v in snap.get("tasks", {}).items()}
        self.packages_ready = dict(snap.get("packages_ready", {}))
        self.pending_candidates = dict(snap.get("pending_candidates", {}))
        self.candidate_ready_seq = int(snap.get("candidate_ready_seq", 0))
        self.route_priorities = dict(snap.get("route_priorities", {}))
        self.paused_routes = set(snap.get("paused_routes", []))
        self.satisfied_tokens = set(snap.get("satisfied_tokens", []))
        self.failed_tokens = set(snap.get("failed_tokens", []))
        self.waiting_relations = dict(snap.get("waiting_relations", {}))
        self.worker_states = dict(snap.get("worker_states", {}))
        self.global_best_version_id = snap.get("global_best_version_id")
        self.global_best_artifact_id = snap.get("global_best_artifact_id")
        self.verification_level = str(
            snap.get("verification_level", VerificationLevel.BASELINE_ONLY.value)
        )
        self.final_eval_requested = bool(snap.get("final_eval_requested", False))
        self.final_eval_completed = bool(snap.get("final_eval_completed", False))
        self.final_eval_version_id = snap.get("final_eval_version_id")
        self.ledger = BudgetLedger(**snap["ledger"])
        self.run_finished = bool(snap.get("run_finished", False))
        self.stop_reason = snap.get("stop_reason")
        self._task_seq = int(snap.get("task_seq", 0))
        self._evt_seq = int(snap.get("evt_seq", 0))
        if "duration_model" in snap:
            self.duration_model = DurationModel.from_dict(snap["duration_model"])
