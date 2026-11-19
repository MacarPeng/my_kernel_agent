"""KernelLabRuntime: integrated single-GPU async runtime, discrete-event driver, and telemetry."""

from __future__ import annotations

from dataclasses import dataclass, field
import heapq
from pathlib import Path
from typing import Any, Callable

from kernel_agent_v1.backends import ExecutionBackend, JobHandle, SimulatedExecutionBackend
from kernel_agent_v1.clock import Clock, SimClock
from kernel_agent_v1.coder import Coder
from kernel_agent_v1.controller import Controller, SchedulingDecision
from kernel_agent_v1.evaluator import TrustedEvaluator
from kernel_agent_v1.judge import Judge
from kernel_agent_v1.leader import Leader
from kernel_agent_v1.operator import Operator, OperatorDispatchResult
from kernel_agent_v1.protocol import (
    CodePackage,
    Command,
    Event,
    EventType,
    EvidenceDelivery,
    JobKind,
    ResultKind,
    RuntimeConfig,
    TargetRef,
    TaskSpec,
    TaskStatus,
    VerificationLevel,
)
from kernel_agent_v1.store import RunStore
from kernel_agent_v1.timeline import PipelineTimeline
from kernel_agent_v1.wiki import HardwareKnowledgePlugin, KernelWiki


@dataclass
class RunMetrics:
    scheduler_mode: str
    protocol: str
    total_wall_s: float
    gpu_busy_s: float
    gpu_idle_s: float
    member_gpu_overlap_s: float
    first_actionable_evidence_s: float | None
    best_verified_score: float
    best_version_id: str | None
    best_artifact_id: str | None
    verification_level: str
    model_cost_used: float
    versions_closed: int
    stop_reason: str | None
    time_to_target_score: dict[str, dict[str, float | None]] = field(
        default_factory=dict
    )

    def to_dict(self) -> dict[str, Any]:
        return {
            "scheduler_mode": self.scheduler_mode,
            "protocol": self.protocol,
            "total_wall_s": round(self.total_wall_s, 3),
            "gpu_busy_s": round(self.gpu_busy_s, 3),
            "gpu_idle_s": round(self.gpu_idle_s, 3),
            "member_gpu_overlap_s": round(self.member_gpu_overlap_s, 3),
            "first_actionable_evidence_s": (
                round(self.first_actionable_evidence_s, 3)
                if self.first_actionable_evidence_s is not None
                else None
            ),
            "best_verified_score": round(self.best_verified_score, 4),
            "best_version_id": self.best_version_id,
            "best_artifact_id": self.best_artifact_id,
            "verification_level": self.verification_level,
            "model_cost_used": round(self.model_cost_used, 3),
            "versions_closed": self.versions_closed,
            "stop_reason": self.stop_reason,
            "time_to_target_score": self.time_to_target_score,
        }


def _union_interval_length(intervals: list[tuple[float, float]]) -> float:
    if not intervals:
        return 0.0
    sorted_iv = sorted(intervals)
    total = 0.0
    cur_s, cur_e = sorted_iv[0]
    for s, e in sorted_iv[1:]:
        if s <= cur_e:
            cur_e = max(cur_e, e)
        else:
            total += max(0.0, cur_e - cur_s)
            cur_s, cur_e = s, e
    total += max(0.0, cur_e - cur_s)
    return total


def _intersect_intervals(
    a_list: list[tuple[float, float]], b_list: list[tuple[float, float]]
) -> list[tuple[float, float]]:
    out: list[tuple[float, float]] = []
    for as_, ae in a_list:
        for bs, be in b_list:
            s = max(as_, bs)
            e = min(ae, be)
            if e > s:
                out.append((s, e))
    return out


class KernelLabRuntime:
    """Wires Leader, Coder, Controller, Operator, and Judge for a single-GPU run."""

    def __init__(
        self,
        run_dir: str | Path,
        config: RuntimeConfig | None = None,
        clock: Clock | None = None,
        backend: ExecutionBackend | None = None,
    ) -> None:
        self.config = config or RuntimeConfig()
        self.clock = clock or SimClock()
        self.store = RunStore(run_dir)
        self.timeline = PipelineTimeline(self.clock)
        self.evaluator = TrustedEvaluator(
            protocol=self.config.protocol_name,
            device_id=f"{self.config.gpu_name}:0",
            total_cases=self.config.num_benchmark_blocks * self.config.cases_per_block,
            num_blocks=self.config.num_benchmark_blocks,
            min_gain_threshold=self.config.min_gain_threshold,
        )
        self.backend = backend or SimulatedExecutionBackend(self.evaluator)
        self.wiki = KernelWiki(
            hardware_plugin=HardwareKnowledgePlugin(
                gpu_arch=self.config.gpu_name,
                toolchain=self.config.toolchain,
            )
        )
        self.wiki.route_tree.min_gain_threshold = self.config.min_gain_threshold
        self.wiki.route_tree.stagnation_limit = self.config.stagnation_limit

        self.leader = Leader(
            run_id=self.config.run_id,
            wiki=self.wiki,
            gpu_name=self.config.gpu_name,
            clock=self.clock,
        )
        self.coder = Coder(
            packages_dir=self.store.packages_dir,
            run_id=self.config.run_id,
            clock=self.clock,
            wiki=self.wiki,
        )
        self.controller = Controller(
            config=self.config,
            store=self.store,
            clock=self.clock,
        )
        self.operator = Operator(
            run_id=self.config.run_id,
            backend=self.backend,
            evaluator=self.evaluator,
            store=self.store,
            clock=self.clock,
        )
        self.judge = Judge(
            run_id=self.config.run_id,
            evaluator=self.evaluator,
            clock=self.clock,
            min_gain_threshold=self.config.min_gain_threshold,
            require_abba_on_first_candidate=self.config.require_abba_on_first_candidate,
        )

        # Telemetry & logs
        self.packages_by_id: dict[str, CodePackage] = {}
        self.deliveries_log: list[EvidenceDelivery] = []
        self.gpu_intervals: list[tuple[float, float]] = []
        self.non_gpu_intervals: list[tuple[float, float]] = []
        self.first_actionable_evidence_s: float | None = None
        self.score_milestones: list[tuple[float, float, float, float]] = []  # (wall_s, gpu_s, model_cost, score)

        # Discrete-event heap for async simulation: (time, seq, kind, payload)
        self._heap: list[tuple[float, int, str, Any]] = []
        self._heap_seq = 0

    def register_route(
        self,
        direction_id: str,
        hypothesis: str,
        priority: float = 1.0,
        researcher_id: str = "researcher",
        parent_direction_id: str | None = None,
        reference_score: float = 1.0,
    ) -> None:
        self.wiki.route_tree.ensure_route(
            direction_id=direction_id,
            hypothesis=hypothesis,
            parent_direction_id=parent_direction_id,
            priority=priority,
            reference_score=reference_score,
        )
        self.leader.assign_route_researcher(direction_id, researcher_id)
        self.operator.register_route_researcher(direction_id, researcher_id)
        evt = self.leader.set_route_priority_event(direction_id, priority)
        self.controller.process_event(evt)

    def submit_command_and_package(
        self,
        command: Command,
        package_spec: dict[str, Any] | None = None,
    ) -> tuple[Event, CodePackage | None]:
        evt, pkg = self.coder.execute_command(command, provided_spec=package_spec)
        if pkg is not None:
            self.packages_by_id[pkg.manifest.package_id] = pkg
        self.controller.process_event(evt)
        return evt, pkg

    def dispatch_gpu_if_ready(
        self,
        extra_inputs_fn: Callable[[TaskSpec], dict[str, Any]] | None = None,
    ) -> tuple[JobHandle | None, SchedulingDecision]:
        task, decision = self.controller.select_next_gpu_task()
        if task is None:
            return None, decision

        pkg = (
            self.packages_by_id.get(task.package_id)
            if task.package_id
            else None
        )
        inputs: dict[str, Any] = {
            "task_id": task.task_id,
            "kind": task.kind,
            "resource": "gpu",
            "version_id": task.version_id,
            "artifact_id": task.artifact_id,
            "direction_id": task.direction_id,
            "block_id": task.block_id,
            "command_id": task.command_id,
            "reply_to": task.reply_to,
            "timeout_s": task.timeout_s or self.config.budgets.default_job_timeout_s,
            "required_blocks": (
                self.controller.active_version.required_blocks
                if self.controller.active_version
                else ["W1", "W2", "W3", "W4"]
            ),
            **task.metadata,
        }
        if extra_inputs_fn is not None:
            inputs.update(extra_inputs_fn(task))

        useful_until = task.metadata.get("useful_until_s")
        if useful_until is not None:
            inputs["timeout_s"] = min(inputs["timeout_s"], useful_until - self.clock.now())

        attempt_num = task.retry_count + 1
        handle = self.operator.start(pkg, inputs, attempt_num=attempt_num)
        self.controller.mark_gpu_dispatched(task, handle.job_id, handle.attempt_id)
        return handle, decision

    def complete_gpu_job(
        self, handle: JobHandle, advance_sim_clock: bool = True
    ) -> OperatorDispatchResult:
        if advance_sim_clock and isinstance(self.clock, SimClock):
            if handle.expected_finish_at > self.clock.now():
                self.clock.advance_to(handle.expected_finish_at)

        op_res = self.operator.finalize_handle(handle)
        end_t = self.clock.now()
        self.gpu_intervals.append((handle.started_at, end_t))

        for evt in op_res.scheduling_events:
            self.controller.process_event(evt)
        self.process_deliveries(op_res.deliveries)
        return op_res

    def process_deliveries(
        self,
        deliveries: list[EvidenceDelivery],
        auto_adjudicate: bool = True,
        on_research_delivery: Callable[[EvidenceDelivery], None] | None = None,
    ) -> None:
        for d in deliveries:
            self.deliveries_log.append(d)
            if d.recipient == "judge":
                if auto_adjudicate:
                    j_evt, leader_deliv = self.judge.adjudicate(d)
                    self.deliveries_log.append(leader_deliv)
                    self.controller.process_event(j_evt)
                    pause_evts = self.leader._handle_judge_delivery(leader_deliv)
                    for pe in pause_evts:
                        self.controller.process_event(pe)
                    self.leader.judge_deliveries.append(leader_deliv)
                    if leader_deliv.payload.get("verdict") == "promoted":
                        self.score_milestones.append(
                            (
                                self.clock.now(),
                                self.controller.ledger.gpu_seconds_used,
                                self.controller.ledger.model_cost_used,
                                self.judge.incumbent.geomean_speedup,
                            )
                        )
            else:
                if (
                    self.first_actionable_evidence_s is None
                    and d.result_kind
                    in (
                        ResultKind.BENCHMARK_PARTIAL.value,
                        ResultKind.NCU_PARSED.value,
                        ResultKind.PROBE_RESULT.value,
                        ResultKind.COMPILER_ANALYSIS.value,
                    )
                ):
                    self.first_actionable_evidence_s = self.clock.now()
                self.leader.deliver_evidence(d)
                if on_research_delivery is not None:
                    on_research_delivery(d)

    # ------------------------------------------------------------------
    # Discrete-event async simulation runner (for §8.5 comparison & E2E runs)
    # ------------------------------------------------------------------
    def _push_sim(self, at_time: float, kind: str, payload: Any) -> None:
        self._heap_seq += 1
        heapq.heappush(self._heap, (at_time, self._heap_seq, kind, payload))

    def run_simulated_campaign(
        self,
        candidate_plan: list[dict[str, Any]],
        max_wall_s: float | None = None,
    ) -> RunMetrics:
        """Run a complete simulated campaign comparing scheduler modes on identical workload.

        Each item in candidate_plan defines a route candidate:
          {
            "direction_id": str,
            "hypothesis": str,
            "priority": float,
            "researcher_id": str,
            "research_duration_s": float,
            "code_duration_s": float,
            "model_cost": float,
            "block_duration_s": float,
            "speedup": float,
            "diagnostic": {...} | None,   # optional NCU or probe for this candidate
          }
        """
        if not isinstance(self.clock, SimClock):
            raise RuntimeError("run_simulated_campaign requires SimClock")

        mode = self.config.scheduler_mode
        limit_wall = max_wall_s or self.config.budgets.total_wall_budget_s

        # Seed initial routes & schedule initial research rounds concurrently on CPU/Model
        idx_counter = 0
        for item in candidate_plan:
            idx_counter += 1
            d_id = item["direction_id"]
            if d_id not in self.wiki.route_tree.routes:
                self.register_route(
                    direction_id=d_id,
                    hypothesis=item["hypothesis"],
                    priority=float(item.get("priority", 1.0)),
                    researcher_id=item.get("researcher_id", "researcher"),
                )

        # Group items per route sequentially so each route advances round-by-round
        per_route_queues: dict[str, list[dict[str, Any]]] = {}
        for item in candidate_plan:
            per_route_queues.setdefault(item["direction_id"], []).append(item)
        in_flight_research: set[str] = set()

        def _kickoff_route_research(d_id: str) -> None:
            if d_id in in_flight_research:
                return
            if d_id in self.controller.pending_candidates:
                return
            q = per_route_queues.get(d_id, [])
            if not q:
                return
            route_node = self.wiki.route_tree.routes.get(d_id)
            if route_node and route_node.status != "active":
                return
            if self.controller.ledger.admission_frozen:
                return
            spec = q.pop(0)
            in_flight_research.add(d_id)
            now = self.clock.now()
            res_dur = float(spec.get("research_duration_s", 20.0))
            self.controller.ledger.model_cost_used += float(spec.get("model_cost", 1.0)) * 0.5
            self.non_gpu_intervals.append((now, now + res_dur))
            self._push_sim(now + res_dur, "research_done", spec)

        # Start initial research in parallel for all routes
        for d_id in list(per_route_queues.keys()):
            _kickoff_route_research(d_id)

        def _maybe_dispatch_gpu() -> None:
            if self.controller.gpu_busy or not self.controller.gpu_cleanup_confirmed:
                return
            if self.clock.now() >= limit_wall:
                return

            # In whole_round mode, wait for research/coding to settle only when no GPU task ready;
            # and don't deliver partial blocks early (handled below on delivery).
            handle, _ = self.dispatch_gpu_if_ready()
            if handle is not None:
                self._push_sim(handle.expected_finish_at, "gpu_done", handle)

        held_partial_deliveries: dict[str, list[EvidenceDelivery]] = {}

        def _deliver_with_mode(deliveries: list[EvidenceDelivery]) -> None:
            if mode == "whole_round":
                # In whole_round baseline, Researchers wait until full round ends before acting
                to_judge = [d for d in deliveries if d.recipient == "judge"]
                to_res = [d for d in deliveries if d.recipient != "judge"]
                for d in to_res:
                    held_partial_deliveries.setdefault(d.version_id or "", []).append(d)
                self.process_deliveries(to_judge, auto_adjudicate=False)
                for d in to_judge:
                    # Flush held partial deliveries only at BENCHMARK_COMPLETE
                    if d.result_kind == ResultKind.BENCHMARK_COMPLETE.value:
                        self.process_deliveries(
                            held_partial_deliveries.pop(d.version_id or "", [])
                        )
                    j_evt, leader_deliv = self.judge.adjudicate(d)
                    j_dur = 25.0
                    self.non_gpu_intervals.append((self.clock.now(), self.clock.now() + j_dur))
                    self._push_sim(
                        self.clock.now() + j_dur,
                        "judge_done",
                        (j_evt, leader_deliv, d.version_id),
                    )
                return

            # chunked_fcfs / chunked_dynamic: partial deliveries go immediately to Researcher!
            for d in deliveries:
                if d.recipient == "judge":
                    self.deliveries_log.append(d)
                    j_dur = 25.0
                    self.non_gpu_intervals.append((self.clock.now(), self.clock.now() + j_dur))
                    j_evt, leader_deliv = self.judge.adjudicate(d)
                    self._push_sim(
                        self.clock.now() + j_dur,
                        "judge_done",
                        (j_evt, leader_deliv, d.version_id),
                    )
                else:
                    self.process_deliveries([d])
                    # Researcher consumes chunked evidence immediately if route has next item waiting!
                    if (
                        d.direction_id
                        and per_route_queues.get(d.direction_id)
                        and (
                            d.result_kind
                            in (
                                ResultKind.PROBE_RESULT.value,
                                ResultKind.NCU_PARSED.value,
                            )
                            or (
                                d.result_kind == ResultKind.BENCHMARK_PARTIAL.value
                                and d.block_id == "W1"
                                and not any(
                                    rel["active"] and rel["direction_id"] == d.direction_id
                                    for rel in self.controller.waiting_relations.values()
                                )
                            )
                        )
                    ):
                        _kickoff_route_research(d.direction_id)

        # Main discrete-event loop
        while self._heap:
            ev_time, _, ev_kind, payload = heapq.heappop(self._heap)
            if ev_time > limit_wall:
                self.clock.advance_to(limit_wall)
                break
            self.clock.advance_to(ev_time)

            if ev_kind == "research_done":
                spec = payload
                d_id = spec["direction_id"]
                researcher = self.leader.ensure_researcher(
                    spec.get("researcher_id", "researcher")
                )
                rid = self.leader.next_research_id()
                lookup, _ = researcher.start_research_round(
                    research_id=rid,
                    direction_id=d_id,
                    evidence_terms=(d_id, "register", "layout"),
                    observation=f"Baseline/partial obs for {spec['artifact_id']}",
                    explanation=spec["hypothesis"],
                    prediction=f"Expected speedup {spec.get('speedup', 1.0)}",
                    hypotheses=[],
                )
                cmd = self.leader.build_command(
                    kind=JobKind.BENCHMARK_BLOCK.value.replace("benchmark_block", "implement_kernel"),
                    research_id=rid,
                    direction_id=d_id,
                    target=TargetRef(version_id="V0", artifact_id="fla-baseline"),
                    hypothesis=spec["hypothesis"],
                    evidence_refs=(lookup.lookup_id,),
                    instruction=f"Implement {spec['artifact_id']}",
                    deliverable="kernel package",
                    reply_to=spec.get("researcher_id", "researcher"),
                )
                code_dur = float(spec.get("code_duration_s", 25.0))
                self.controller.ledger.model_cost_used += float(spec.get("model_cost", 1.0)) * 0.5
                self.non_gpu_intervals.append((ev_time, ev_time + code_dur))
                self._push_sim(ev_time + code_dur, "code_done", (cmd, spec))

            elif ev_kind == "code_done":
                cmd, spec = payload
                in_flight_research.discard(spec["direction_id"])
                block_dur = float(spec.get("block_duration_s", 45.0))
                speedup = float(spec.get("speedup", 1.0))
                # Pre-register per-block duration/speedup in SimulatedExecutionBackend
                if isinstance(self.backend, SimulatedExecutionBackend):
                    for b in ("W1", "W2", "W3", "W4"):
                        self.backend.set_behavior(
                            f"{spec['artifact_id']}:benchmark_block:{b}",
                            {"duration_s": block_dur, "speedup": speedup},
                        )
                    self.backend.set_behavior(
                        f"{spec['artifact_id']}:abba_retest",
                        {"duration_s": 70.0, "speedup": speedup},
                    )
                    for fb in ("F1", "F2", "F3"):
                        self.backend.set_behavior(
                            f"{spec['artifact_id']}:final_eval:{fb}",
                            {"duration_s": 50.0, "speedup": speedup},
                        )
                self.submit_command_and_package(
                    cmd,
                    package_spec={"artifact_id": spec["artifact_id"]},
                )

                # Optional diagnostic attached to this candidate (in chunked modes)
                diag = spec.get("diagnostic")
                if diag and mode != "whole_round":
                    diag_cmd = self.leader.build_command(
                        kind=diag.get("command_kind", "write_probe"),
                        research_id=cmd.head.research_id,
                        direction_id=spec["direction_id"],
                        target=TargetRef(version_id="pending", artifact_id=spec["artifact_id"]),
                        hypothesis=spec["hypothesis"],
                        evidence_refs=cmd.body.evidence_refs,
                        instruction="Targeted diagnostic",
                        deliverable="diag result",
                        reply_to=spec.get("researcher_id", "researcher"),
                    )
                    diag_tid = f"diag-{spec['artifact_id']}"
                    self.controller.process_event(
                        self.leader.propose_task_event(
                            task_id=diag_tid,
                            kind=diag.get("job_kind", JobKind.PROBE.value),
                            resource="gpu",
                            direction_id=spec["direction_id"],
                            artifact_id=spec["artifact_id"],
                            command_id=diag_cmd.head.command_id,
                            estimated_duration_s=float(diag.get("gpu_duration_s", 15.0)),
                            post_cpu_kind=diag.get("post_cpu_kind"),
                            post_cpu_duration_s=diag.get("post_cpu_duration_s"),
                        )
                    )
                    if isinstance(self.backend, SimulatedExecutionBackend):
                        self.backend.set_behavior(
                            diag_tid,
                            {"duration_s": float(diag.get("gpu_duration_s", 15.0))},
                        )
                    self.submit_command_and_package(
                        diag_cmd,
                        package_spec={
                            "artifact_id": spec["artifact_id"],
                            "task_id": diag_tid,
                        },
                    )
                    if diag.get("leader_waiting_s"):
                        self.controller.process_event(
                            self.leader.declare_waiting_event(
                                member=spec.get("researcher_id", "researcher"),
                                waiting_for=f"result_ready:{diag_tid}",
                                unblocks_work="hypothesis_refinement",
                                estimated_duration_s=float(diag["leader_waiting_s"]),
                                direction_id=spec["direction_id"],
                            )
                        )

            elif ev_kind == "gpu_done":
                handle = payload
                op_res = self.operator.finalize_handle(handle)
                self.gpu_intervals.append((handle.started_at, ev_time))
                for evt in op_res.scheduling_events:
                    self.controller.process_event(evt)
                for cpu_job in op_res.pending_cpu_jobs:
                    cpu_handle = self.operator.start(None, cpu_job)
                    self.non_gpu_intervals.append(
                        (cpu_handle.started_at, cpu_handle.expected_finish_at)
                    )
                    self._push_sim(cpu_handle.expected_finish_at, "cpu_done", cpu_handle)
                _deliver_with_mode(op_res.deliveries)

            elif ev_kind == "cpu_done":
                cpu_handle = payload
                op_res = self.operator.finalize_handle(cpu_handle)
                for evt in op_res.scheduling_events:
                    self.controller.process_event(evt)
                _deliver_with_mode(op_res.deliveries)

            elif ev_kind == "judge_done":
                j_evt, leader_deliv, _ = payload
                j_evt.sim_time = ev_time
                j_evt.timestamp = self.clock.now_iso()
                self.deliveries_log.append(leader_deliv)
                self.controller.process_event(j_evt)
                pause_evts = self.leader._handle_judge_delivery(leader_deliv)
                for pe in pause_evts:
                    self.controller.process_event(pe)
                self.leader.judge_deliveries.append(leader_deliv)
                if leader_deliv.payload.get("verdict") == "promoted":
                    self.score_milestones.append(
                        (
                            ev_time,
                            self.controller.ledger.gpu_seconds_used,
                            self.controller.ledger.model_cost_used,
                            self.judge.incumbent.geomean_speedup,
                        )
                    )
                if (
                    mode == "whole_round"
                    and leader_deliv.direction_id
                    and per_route_queues.get(leader_deliv.direction_id)
                ):
                    _kickoff_route_research(leader_deliv.direction_id)
                elif (
                    mode != "whole_round"
                    and leader_deliv.direction_id
                    and per_route_queues.get(leader_deliv.direction_id)
                    and not any(
                        rel["active"] and rel["direction_id"] == leader_deliv.direction_id
                        for rel in self.controller.waiting_relations.values()
                    )
                ):
                    # If a route's diagnostic was skipped when version closed, unblock next round on judge verdict
                    d_id = leader_deliv.direction_id
                    if not any(
                        c.get("direction_id") == d_id
                        for c in [self.controller.pending_candidates.get(d_id) or {}]
                        if c
                    ):
                        _kickoff_route_research(d_id)

            _maybe_dispatch_gpu()

            # Trigger final eval once all work is done (or admissions frozen) and incumbent exists
            if (
                not self._heap
                and not self.controller.gpu_busy
                and self.controller.active_version is None
                and (
                    not self.controller.pending_candidates
                    or self.controller.ledger.admission_frozen
                )
                and self.controller.global_best_version_id is not None
                and not self.controller.final_eval_requested
            ):
                self.controller.schedule_final_eval()
                _maybe_dispatch_gpu()

        stop_reason = (
            self.controller.ledger.freeze_reason
            or ("completed_all_routes" if not per_route_queues or all(not q for q in per_route_queues.values()) else "wall_budget_reached")
        )
        self.controller.stop_reason = stop_reason
        self.controller.run_finished = True

        wall_s = self.clock.now()
        gpu_busy_s = _union_interval_length(self.gpu_intervals)
        gpu_idle_s = max(0.0, wall_s - gpu_busy_s)
        overlap_s = _union_interval_length(
            _intersect_intervals(self.gpu_intervals, self.non_gpu_intervals)
        )

        # Compute time/cost to reach target score thresholds (e.g., 1.10x, 1.20x)
        milestones: dict[str, dict[str, float | None]] = {}
        for target in (1.10, 1.20):
            hit = next((m for m in self.score_milestones if m[3] >= target), None)
            milestones[f"{target:.2f}x"] = {
                "wall_s": hit[0] if hit else None,
                "gpu_s": hit[1] if hit else None,
                "model_cost": hit[2] if hit else None,
            }

        return RunMetrics(
            scheduler_mode=mode,
            protocol=self.config.protocol_name,
            total_wall_s=wall_s,
            gpu_busy_s=gpu_busy_s,
            gpu_idle_s=gpu_idle_s,
            member_gpu_overlap_s=overlap_s,
            first_actionable_evidence_s=self.first_actionable_evidence_s,
            best_verified_score=self.judge.incumbent.geomean_speedup,
            best_version_id=self.judge.incumbent.version_id,
            best_artifact_id=self.judge.incumbent.artifact_id,
            verification_level=self.judge.incumbent.verification_level,
            model_cost_used=self.controller.ledger.model_cost_used,
            versions_closed=len(self.controller.closed_versions),
            stop_reason=stop_reason,
            time_to_target_score=milestones,
        )
