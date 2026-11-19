"""True asyncio concurrent runtime coordinating GPU, CPU, and Model/Agent workers."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable

from kernel_agent_v1.backends import JobHandle
from kernel_agent_v1.protocol import (
    Event,
    EvidenceDelivery,
    JobKind,
    TaskSpec,
)
from kernel_agent_v1.runtime import (
    KernelLabRuntime,
    RunMetrics,
    _intersect_intervals,
    _union_interval_length,
)


@dataclass
class AsyncRunnerHooks:
    """Optional async callbacks for Research/Coding/Judge overlapping with GPU execution."""

    on_research_evidence: Callable[
        [KernelLabRuntime, EvidenceDelivery], Awaitable[None]
    ] | None = None
    on_idle_check: Callable[[KernelLabRuntime], Awaitable[bool]] | None = None


class AsyncKernelLabRunner:
    """Non-blocking asyncio coordinator running GPU jobs, CPU workers, and Agent tasks concurrently."""

    def __init__(
        self,
        runtime: KernelLabRuntime,
        hooks: AsyncRunnerHooks | None = None,
    ) -> None:
        self.runtime = runtime
        self.hooks = hooks or AsyncRunnerHooks()
        self._event_queue: asyncio.Queue[tuple[str, Any]] | None = None
        self._inflight_bg_tasks: set[asyncio.Task[Any]] = set()
        self._gpu_task: asyncio.Task[None] | None = None

    def _spawn_bg(self, coro: Awaitable[Any]) -> asyncio.Task[Any]:
        task = asyncio.create_task(coro)
        self._inflight_bg_tasks.add(task)
        task.add_done_callback(self._inflight_bg_tasks.discard)
        return task

    async def _run_gpu_handle(self, handle: JobHandle) -> None:
        rt = self.runtime
        assert self._event_queue is not None
        try:
            outcome = await rt.backend.async_wait(handle, rt.clock)
            end_t = rt.clock.now()
            rt.gpu_intervals.append((handle.started_at, end_t))
            op_res = rt.operator.finalize_handle(handle, outcome=outcome)
            await self._event_queue.put(("gpu_result", op_res))
        except Exception as exc:
            await self._event_queue.put(("bg_error", exc))

    async def _run_cpu_job(self, cpu_inputs: dict[str, Any]) -> None:
        rt = self.runtime
        assert self._event_queue is not None
        try:
            ch = rt.operator.start(None, cpu_inputs)
            start_t = rt.clock.now()
            outcome = await rt.backend.async_wait(ch, rt.clock)
            end_t = rt.clock.now()
            rt.non_gpu_intervals.append((start_t, end_t))
            op_res = rt.operator.finalize_handle(ch, outcome=outcome)
            await self._event_queue.put(("cpu_result", op_res))
        except Exception as exc:
            await self._event_queue.put(("bg_error", exc))

    async def _run_delivery_async(self, delivery: EvidenceDelivery) -> None:
        rt = self.runtime
        assert self._event_queue is not None
        try:
            start_t = rt.clock.now()
            if delivery.recipient == "judge":
                j_evt, leader_deliv = rt.judge.adjudicate(delivery)
                end_t = rt.clock.now()
                rt.non_gpu_intervals.append((start_t, end_t))
                await self._event_queue.put(("judge_done", (j_evt, leader_deliv)))
                return

            rt.leader.deliver_evidence(delivery)
            if rt.first_actionable_evidence_s is None:
                rt.first_actionable_evidence_s = rt.clock.now()
            if self.hooks.on_research_evidence is not None:
                await self.hooks.on_research_evidence(rt, delivery)
            end_t = rt.clock.now()
            if end_t > start_t:
                rt.non_gpu_intervals.append((start_t, end_t))
            await self._event_queue.put(("research_done", delivery.delivery_id))
        except Exception as exc:
            await self._event_queue.put(("bg_error", exc))

    async def submit_external_event(self, event: Event) -> None:
        """Thread/task-safe enqueue of external scheduling header (e.g. code_ready from Coder)."""
        if self._event_queue is None:
            self._event_queue = asyncio.Queue()
        await self._event_queue.put(("event", event))

    def _try_dispatch_gpu(self) -> bool:
        rt = self.runtime
        if rt.controller.gpu_busy or not rt.controller.gpu_cleanup_confirmed:
            return False
        handle, _ = rt.dispatch_gpu_if_ready()
        if handle is None:
            return False
        self._gpu_task = asyncio.create_task(self._run_gpu_handle(handle))
        return True

    async def run(self, max_wall_s: float | None = None) -> RunMetrics:
        """Run concurrent GPU + CPU + Agent event loop until all work completes or budget ends."""
        if self._event_queue is None:
            self._event_queue = asyncio.Queue()
        rt = self.runtime
        limit = max_wall_s or rt.config.budgets.total_wall_budget_s
        start_wall = rt.clock.now()

        self._try_dispatch_gpu()

        try:
            while True:
                if rt.clock.now() - start_wall >= limit:
                    rt.controller.stop_reason = "wall_budget_reached"
                    if rt.operator.active_gpu_handle is not None:
                        rt.operator.stop(rt.operator.active_gpu_handle, "wall_budget_reached")
                    break

                # Check termination when GPU is idle & no background work in flight
                if (
                    not rt.controller.gpu_busy
                    and self._gpu_task is None
                    and not self._inflight_bg_tasks
                    and self._event_queue.empty()
                ):
                    if (
                        rt.controller.active_version is None
                        and (
                            not rt.controller.pending_candidates
                            or rt.controller.ledger.admission_frozen
                        )
                        and rt.controller.global_best_version_id is not None
                        and not rt.controller.final_eval_requested
                    ):
                        rt.controller.schedule_final_eval()
                        if self._try_dispatch_gpu():
                            continue

                    if self.hooks.on_idle_check is not None:
                        spawned = await self.hooks.on_idle_check(rt)
                        if spawned:
                            self._try_dispatch_gpu()
                            continue

                    if not self._try_dispatch_gpu():
                        break

                try:
                    msg_kind, payload = await asyncio.wait_for(
                        self._event_queue.get(), timeout=0.5
                    )
                except asyncio.TimeoutError:
                    continue

                if msg_kind == "bg_error":
                    raise payload
                elif msg_kind == "event":
                    rt.controller.process_event(payload)
                elif msg_kind in ("gpu_result", "cpu_result"):
                    if msg_kind == "gpu_result":
                        self._gpu_task = None
                    for evt in payload.scheduling_events:
                        rt.controller.process_event(evt)
                    for cpu_job in payload.pending_cpu_jobs:
                        self._spawn_bg(self._run_cpu_job(cpu_job))
                    for deliv in payload.deliveries:
                        rt.deliveries_log.append(deliv)
                        self._spawn_bg(self._run_delivery_async(deliv))
                elif msg_kind == "judge_done":
                    j_evt, leader_deliv = payload
                    rt.deliveries_log.append(leader_deliv)
                    rt.controller.process_event(j_evt)
                    pause_evts = rt.leader._handle_judge_delivery(leader_deliv)
                    for pe in pause_evts:
                        rt.controller.process_event(pe)
                    rt.leader.judge_deliveries.append(leader_deliv)
                elif msg_kind == "research_done":
                    pass

                self._try_dispatch_gpu()
        finally:
            if self._gpu_task is not None and not self._gpu_task.done():
                self._gpu_task.cancel()
            for bg in list(self._inflight_bg_tasks):
                if not bg.done():
                    bg.cancel()
            self._gpu_task = None

        rt.controller.run_finished = True
        if not rt.controller.stop_reason:
            rt.controller.stop_reason = (
                rt.controller.ledger.freeze_reason or "completed"
            )
        rt.controller.flush_snapshot()

        wall_s = rt.clock.now() - start_wall
        gpu_busy_s = _union_interval_length(rt.gpu_intervals)
        overlap_s = _union_interval_length(
            _intersect_intervals(rt.gpu_intervals, rt.non_gpu_intervals)
        )
        return RunMetrics(
            scheduler_mode=rt.config.scheduler_mode,
            protocol=rt.config.protocol_name,
            total_wall_s=wall_s,
            gpu_busy_s=gpu_busy_s,
            gpu_idle_s=max(0.0, wall_s - gpu_busy_s),
            member_gpu_overlap_s=overlap_s,
            first_actionable_evidence_s=rt.first_actionable_evidence_s,
            best_verified_score=rt.judge.incumbent.geomean_speedup,
            best_version_id=rt.judge.incumbent.version_id,
            best_artifact_id=rt.judge.incumbent.artifact_id,
            verification_level=rt.judge.incumbent.verification_level,
            model_cost_used=rt.controller.ledger.model_cost_used,
            versions_closed=len(rt.controller.closed_versions),
            stop_reason=rt.controller.stop_reason,
        )
