"""Dynamic lookahead scheduler extracted from Controller (§1, §6)."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable

from kernel_agent_v1.duration import DurationModel
from kernel_agent_v1.protocol import (
    BudgetLedger,
    JobKind,
    RuntimeConfig,
    TaskSpec,
    TaskStatus,
    VersionRecord,
    VersionStatus,
)


@dataclass
class CandidateScoreRecord:
    task_id: str
    kind: str
    is_required: bool
    typical_score: float
    slow_score: float | None
    expected_benchmark_complete_s: float
    ready_at: float


@dataclass
class SchedulingDecision:
    state_seq: int
    sim_time: float
    scheduler_mode: str
    selected_task_id: str | None
    reason: str
    candidates: list[dict[str, Any]] = field(default_factory=list)
    excluded: dict[str, str] = field(default_factory=dict)
    tie_breaker_used: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "state_seq": self.state_seq,
            "sim_time": self.sim_time,
            "scheduler_mode": self.scheduler_mode,
            "selected_task_id": self.selected_task_id,
            "reason": self.reason,
            "candidates": self.candidates,
            "excluded": self.excluded,
            "tie_breaker_used": self.tie_breaker_used,
        }


class DynamicScheduler:
    """Stateless Filter + Horizon-H member-active-seconds scorer & sensitivity checker."""

    def __init__(
        self,
        config: RuntimeConfig,
        duration_model: DurationModel,
    ) -> None:
        self.config = config
        self.duration_model = duration_model

    def estimate_task_duration(
        self, task: TaskSpec, use_slow: bool = False
    ) -> tuple[float, float]:
        """Return (gpu_or_primary_duration_s, post_cpu_duration_s)."""
        if task.kind == JobKind.BENCHMARK_BLOCK.value:
            k, fb = DurationModel.benchmark_keys(
                self.config.gpu_name,
                self.config.protocol_name,
                task.block_id or "W1",
            )
            est = self.duration_model.estimate(
                k, fb, default_typical_s=task.estimated_duration_s or 45.0
            )
            val = est.slow_s if use_slow else est.typical_s
            return (
                task.estimated_duration_s
                if (task.estimated_duration_s and est.sample_count == 0)
                else val
            ), 0.0

        if task.kind in (
            JobKind.NCU_CAPTURE.value,
            JobKind.PROBE.value,
            JobKind.GPU_COMPILER_ANALYSIS.value,
        ):
            k, fb = DurationModel.diagnostic_keys(
                task.template_id or task.kind,
                task.workload_count,
                self.config.gpu_name,
            )
            est = self.duration_model.estimate(
                k, fb, default_typical_s=task.estimated_duration_s or 20.0
            )
            primary = (
                est.slow_s
                if use_slow
                else (
                    task.estimated_duration_s
                    if (task.estimated_duration_s is not None and est.sample_count == 0)
                    else est.typical_s
                )
            )
            if use_slow and task.estimated_duration_s is not None and est.sample_count == 0:
                primary = max(est.slow_s, task.estimated_duration_s * 1.5)

            post_s = 0.0
            if task.post_cpu_kind:
                ck, cfb = DurationModel.cpu_keys(
                    self.config.toolchain, "warm", task.post_cpu_kind
                )
                cest = self.duration_model.estimate(
                    ck, cfb, default_typical_s=task.post_cpu_duration_s or 10.0
                )
                if use_slow:
                    post_s = (
                        max(cest.slow_s, (task.post_cpu_duration_s or 10.0) * 1.5)
                        if cest.sample_count == 0
                        else cest.slow_s
                    )
                else:
                    post_s = (
                        task.post_cpu_duration_s
                        if (task.post_cpu_duration_s is not None and cest.sample_count == 0)
                        else cest.typical_s
                    )
            return primary, post_s

        if task.resource == "model":
            k, fb = DurationModel.model_keys(
                task.assigned_member or "leader", task.kind, task.duration_category
            )
            est = self.duration_model.estimate(
                k, fb, default_typical_s=task.estimated_duration_s or 30.0
            )
            val = (
                task.estimated_duration_s
                if (task.estimated_duration_s is not None and est.sample_count == 0)
                else (est.slow_s if use_slow else est.typical_s)
            )
            return val, 0.0

        return task.estimated_duration_s or 20.0, 0.0

    def filter_gpu_candidates(
        self,
        now: float,
        tasks: dict[str, TaskSpec],
        active_version: VersionRecord | None,
        final_eval_requested: bool,
        ledger: BudgetLedger,
        deps_met_fn: Callable[[list[str]], tuple[bool, str | None]],
    ) -> tuple[list[TaskSpec], dict[str, str]]:
        """Step 1 & 2: Filter compliant ready GPU tasks for current version or final eval."""
        eligible: list[TaskSpec] = []
        excluded: dict[str, str] = {}
        b = self.config.budgets
        gpu_rem = b.total_gpu_budget_s - ledger.gpu_seconds_used
        wall_rem = b.total_wall_budget_s - (now - ledger.wall_start_s)

        for t in tasks.values():
            if t.resource != "gpu" or t.status != TaskStatus.PENDING.value:
                continue

            if t.kind == JobKind.FINAL_EVAL.value:
                if active_version is None and final_eval_requested:
                    eligible.append(t)
                else:
                    excluded[t.task_id] = "waiting_version_close_for_final_eval"
                continue

            if active_version is None:
                excluded[t.task_id] = "no_active_version"
                continue
            if t.version_id != active_version.version_id:
                excluded[t.task_id] = (
                    f"version_mismatch:{t.version_id}!={active_version.version_id}"
                )
                continue

            v = active_version
            if v.status == VersionStatus.CLOSING.value and not t.is_required:
                excluded[t.task_id] = "version_closing"
                continue
            if v.status == VersionStatus.CLOSED.value:
                excluded[t.task_id] = "version_closed"
                continue

            deps_ok, dep_reason = deps_met_fn(t.depends_on)
            if not deps_ok:
                excluded[t.task_id] = dep_reason or "deps_unmet"
                continue

            if not t.is_required:
                diag_gpu_s, _ = self.estimate_task_duration(t, use_slow=True)
                if v.diag_spent_s + diag_gpu_s > v.diag_budget_s:
                    excluded[t.task_id] = "version_diag_budget_exceeded"
                    continue
                rem_blocks = len(set(v.required_blocks) - set(v.completed_blocks))
                reserved_needed = rem_blocks * b.reserve_required_block_s + (
                    b.reserve_retest_s if v.conditional_retest_registered else 0.0
                )
                if (
                    diag_gpu_s + reserved_needed > gpu_rem
                    or diag_gpu_s + reserved_needed > wall_rem
                ):
                    excluded[t.task_id] = "insufficient_budget_for_required_eval"
                    continue
                if now + diag_gpu_s + reserved_needed > v.deadline_at:
                    excluded[t.task_id] = "would_exceed_version_deadline"
                    continue

            eligible.append(t)

        return eligible, excluded

    def simulate_first_step(
        self,
        now: float,
        first_task: TaskSpec,
        tasks: dict[str, TaskSpec],
        active_version: VersionRecord | None,
        satisfied_tokens: set[str],
        waiting_relations: dict[str, dict[str, Any]],
        worker_states: dict[str, dict[str, Any]],
        use_slow: bool = False,
    ) -> CandidateScoreRecord:
        """Step 3 & 4: Simulate first_task in window H and sum active member-seconds."""
        H = self.config.horizon_h_s
        horizon_end = now + H

        rem_required = sorted(
            [
                t
                for t in tasks.values()
                if t.resource == "gpu"
                and t.status == TaskStatus.PENDING.value
                and t.version_id == first_task.version_id
                and t.is_required
                and t.task_id != first_task.task_id
            ],
            key=lambda t: (t.block_id or "ZZ", t.ready_at or 0.0, t.task_id),
        )

        gpu_t = now
        cpu_free = now
        token_ready_at: dict[str, float] = {tok: now for tok in satisfied_tokens}

        gpu_sequence = [first_task] + rem_required
        expected_bench_complete_s = float("inf")
        completed_req_blocks: set[str] = set(
            active_version.completed_blocks if active_version else ()
        )

        for gt in gpu_sequence:
            slow_this = use_slow and (not gt.is_required)
            g_dur, post_cpu_dur = self.estimate_task_duration(gt, use_slow=slow_this)
            gpu_start = gpu_t
            gpu_end = gpu_start + g_dur
            gpu_t = gpu_end

            if post_cpu_dur > 0 and gt.post_cpu_kind:
                c_start = max(cpu_free, gpu_end)
                c_end = c_start + post_cpu_dur
                cpu_free = c_end
                ready_t = c_end
            else:
                ready_t = gpu_end

            token_ready_at[f"result_ready:{gt.task_id}"] = ready_t
            for prod in gt.produces:
                token_ready_at[prod] = ready_t

            if gt.kind == JobKind.BENCHMARK_BLOCK.value and gt.block_id:
                completed_req_blocks.add(gt.block_id)
                if (
                    active_version is not None
                    and completed_req_blocks >= set(active_version.required_blocks)
                    and expected_bench_complete_s == float("inf")
                ):
                    expected_bench_complete_s = gpu_end
                    token_ready_at[f"BENCHMARK_COMPLETE:{active_version.version_id}"] = (
                        gpu_end
                    )

        if (
            active_version is not None
            and expected_bench_complete_s == float("inf")
            and set(active_version.completed_blocks) >= set(active_version.required_blocks)
        ):
            expected_bench_complete_s = now

        member_tasks: list[tuple[str, float, float]] = []

        for t in tasks.values():
            if t.resource == "gpu" or t.status != TaskStatus.PENDING.value:
                continue
            dep_times: list[float] = [now]
            possible = True
            for d in t.depends_on:
                if d in token_ready_at:
                    dep_times.append(token_ready_at[d])
                else:
                    possible = False
                    break
            if not possible:
                continue
            dur, _ = self.estimate_task_duration(t, use_slow=False)
            member = t.assigned_member or ("cpu" if t.resource == "cpu" else "leader")
            member_tasks.append((member, max(dep_times), dur))

        for rel in waiting_relations.values():
            if not rel.get("active"):
                continue
            wf = rel["waiting_for"]
            if wf not in token_ready_at:
                continue
            member_tasks.append(
                (rel["member"], token_ready_at[wf], float(rel["estimated_duration_s"]))
            )

        member_intervals: dict[str, list[tuple[float, float]]] = {}
        member_busy_until: dict[str, float] = {
            m: max(now, float(ws.get("busy_until", now)))
            for m, ws in worker_states.items()
            if ws.get("state") == "busy"
        }

        member_tasks.sort(key=lambda item: (item[1], item[0]))
        for member, avail_at, dur in member_tasks:
            start_t = max(now, avail_at, member_busy_until.get(member, now))
            end_t = start_t + dur
            member_busy_until[member] = end_t
            cs = max(now, start_t)
            ce = min(horizon_end, end_t)
            if ce > cs:
                member_intervals.setdefault(member, []).append((cs, ce))

        total_active_s = 0.0
        for intervals in member_intervals.values():
            intervals.sort()
            cur_s, cur_e = intervals[0]
            for s, e in intervals[1:]:
                if s <= cur_e:
                    cur_e = max(cur_e, e)
                else:
                    total_active_s += cur_e - cur_s
                    cur_s, cur_e = s, e
            total_active_s += cur_e - cur_s

        return CandidateScoreRecord(
            task_id=first_task.task_id,
            kind=first_task.kind,
            is_required=first_task.is_required,
            typical_score=total_active_s,
            slow_score=None,
            expected_benchmark_complete_s=expected_bench_complete_s,
            ready_at=first_task.ready_at or now,
        )

    def recompute_plan(
        self,
        state_seq: int,
        now: float,
        tasks: dict[str, TaskSpec],
        active_version: VersionRecord | None,
        final_eval_requested: bool,
        ledger: BudgetLedger,
        satisfied_tokens: set[str],
        waiting_relations: dict[str, dict[str, Any]],
        worker_states: dict[str, dict[str, Any]],
        deps_met_fn: Callable[[list[str]], tuple[bool, str | None]],
    ) -> SchedulingDecision:
        """Select tentative or dispatchable GPU task with explainable scores & P90 check."""
        eligible, excluded = self.filter_gpu_candidates(
            now=now,
            tasks=tasks,
            active_version=active_version,
            final_eval_requested=final_eval_requested,
            ledger=ledger,
            deps_met_fn=deps_met_fn,
        )
        if not eligible:
            return SchedulingDecision(
                state_seq=state_seq,
                sim_time=now,
                scheduler_mode=self.config.scheduler_mode,
                selected_task_id=None,
                reason="no_eligible_gpu_task",
                excluded=excluded,
            )

        mode = self.config.scheduler_mode
        if mode in ("chunked_fcfs", "whole_round"):
            ordered = sorted(
                eligible,
                key=lambda t: (
                    0 if (mode == "whole_round" and t.is_required) else 1,
                    t.ready_at or 0.0,
                    t.task_id,
                ),
            )
            chosen = ordered[0]
            return SchedulingDecision(
                state_seq=state_seq,
                sim_time=now,
                scheduler_mode=mode,
                selected_task_id=chosen.task_id,
                reason=f"{mode}_fifo",
                candidates=[
                    {"task_id": t.task_id, "kind": t.kind, "ready_at": t.ready_at}
                    for t in ordered
                ],
                excluded=excluded,
            )

        records = [
            self.simulate_first_step(
                now=now,
                first_task=t,
                tasks=tasks,
                active_version=active_version,
                satisfied_tokens=satisfied_tokens,
                waiting_relations=waiting_relations,
                worker_states=worker_states,
                use_slow=False,
            )
            for t in eligible
        ]
        records.sort(
            key=lambda r: (
                -r.typical_score,
                r.expected_benchmark_complete_s,
                r.ready_at,
                r.task_id,
            )
        )
        winner = records[0]
        tie_breaker = None

        top_tied = [r for r in records if abs(r.typical_score - winner.typical_score) < 1e-9]
        if len(top_tied) > 1:
            tie_breaker = "earliest_benchmark_complete_then_ready_order"

        req_records = [r for r in records if r.is_required]
        reason = "highest_member_active_seconds"
        if not winner.is_required and req_records:
            slow_recs: dict[str, CandidateScoreRecord] = {}
            for t in eligible:
                slow_recs[t.task_id] = self.simulate_first_step(
                    now=now,
                    first_task=t,
                    tasks=tasks,
                    active_version=active_version,
                    satisfied_tokens=satisfied_tokens,
                    waiting_relations=waiting_relations,
                    worker_states=worker_states,
                    use_slow=True,
                )
            for r in records:
                r.slow_score = slow_recs[r.task_id].typical_score

            winner_slow = slow_recs[winner.task_id]
            best_req_slow = max(
                (slow_recs[r.task_id] for r in req_records),
                key=lambda r: (r.typical_score, -r.expected_benchmark_complete_s),
            )
            squeezes_deadline = (
                active_version is not None
                and winner_slow.expected_benchmark_complete_s > active_version.deadline_at
            )
            loses_lead = winner_slow.typical_score <= best_req_slow.typical_score
            if loses_lead or squeezes_deadline:
                excluded[winner.task_id] = (
                    "sensitivity_squeezed_deadline"
                    if squeezes_deadline
                    else "sensitivity_slow_score_lost_lead"
                )
                winner = req_records[0]
                reason = "sensitivity_reverted_to_required_benchmark"

        return SchedulingDecision(
            state_seq=state_seq,
            sim_time=now,
            scheduler_mode=mode,
            selected_task_id=winner.task_id,
            reason=reason,
            candidates=[
                {
                    "task_id": r.task_id,
                    "kind": r.kind,
                    "is_required": r.is_required,
                    "typical_score": r.typical_score,
                    "slow_score": r.slow_score,
                    "expected_benchmark_complete_s": r.expected_benchmark_complete_s,
                }
                for r in records
            ],
            excluded=excluded,
            tie_breaker_used=tie_breaker,
        )
