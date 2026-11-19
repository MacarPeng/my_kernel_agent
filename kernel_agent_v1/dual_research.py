"""Optional diagnosis work and explicitly requested design dependencies."""

from __future__ import annotations

import asyncio
import json
import math
import sys
from typing import Any, Awaitable, Callable
from urllib.error import URLError

from kernel_agent_v1.llm import LLMClient, LLMMessage, LLMResponseFormatError, _complete_json
from kernel_agent_v1.protocol import Event, EvidenceDelivery, JobKind, ResultKind, TaskStatus, VersionStatus
from kernel_agent_v1.runtime import KernelLabRuntime


class DualResearch:
    def __init__(
        self,
        runtime: KernelLabRuntime,
        client: LLMClient,
        submit_event: Callable[[Event], Awaitable[None]],
        *,
        ncu_available: bool,
    ) -> None:
        self.runtime = runtime
        self.client = client
        self.submit_event = submit_event
        self.ncu_available = ncu_available
        self.request_keys: set[tuple[str, ...]] = set()
        self.requests: dict[str, dict[str, Any]] = {}
        self.ready: dict[str, asyncio.Event] = {}

    async def diagnose(self, runtime: KernelLabRuntime, delivery: EvidenceDelivery) -> None:
        wiki = runtime.wiki
        controller = runtime.controller
        task_id = delivery.payload.get("diagnostic_task_id")
        try:
            if controller.ledger.model_cost_used >= runtime.config.budgets.total_model_budget_cost:
                return
            lookup = wiki.query(
                runtime.config.gpu_name,
                delivery.direction_id,
                (delivery.result_kind, delivery.block_id or ""),
                runtime.clock.now_iso(),
            )
            measurement = wiki.measurements[delivery.receipt_id]
            kinds = []
            if delivery.result_kind == ResultKind.BENCHMARK_PARTIAL.value:
                kinds.append(JobKind.PROBE.value)
                if self.ncu_available:
                    kinds.append(JobKind.NCU_CAPTURE.value)
            version = controller.active_version
            deadline = (
                version.deadline_at
                if version and version.version_id == delivery.version_id and version.status == VersionStatus.ACTIVE.value
                else None
            )
            related = [
                entry for entry in wiki.measurements.values()
                if entry["receipt_id"] != delivery.receipt_id
                and entry["direction_id"] == delivery.direction_id
                and entry["version_id"] == delivery.version_id
                and entry["artifact_id"] == delivery.artifact_id
            ]
            diagnostics = [
                entry for entry in related
                if entry["kind"] in ("ncu_parsed", "probe_result", "compiler_analysis")
            ]
            payload = {
                "wiki_lookup": lookup.to_dict(),
                "measurement": measurement,
                "evidence": delivery.payload,
                "related_measurements": related,
                "prior_diagnostic_evidence": [
                    runtime.store.read_evidence(entry["receipt_id"])
                    for entry in diagnostics[-4:]
                ],
                "now_s": runtime.clock.now(),
                "version_deadline_s": deadline,
                "available_diagnostics": kinds if deadline is not None else [],
            }
            try:
                parsed, usage = await _complete_json(self.client, [
                    LLMMessage("system", (
                        "You are the Diagnosis Researcher. Use the existing measurement and compile "
                        "evidence first. Attribute NCU counters only to the kernel named in the "
                        "report. A copy or helper kernel cannot establish the recurrence kernel's "
                        "performance. Missing or null counters are unknown, including spill. "
                        "Return JSON with observation, explanation, limitations, "
                        "and design_impact strings. Distinguish measurements from hypotheses. "
                        "If additional GPU evidence distinguishes competing explanations and would "
                        "change a design decision, optionally return diagnostic with kind, "
                        "workload_id from measurement, question, competing_explanations (at least "
                        "two distinct strings), decision_impact, and useful_until_s on the run "
                        "clock before version_deadline_s. Use only available_diagnostics. "
                        "probe is a targeted correctness and latency check, ncu_capture profiles "
                        "the tested candidate on that workload. Omit diagnostic otherwise."
                    )),
                    LLMMessage("user", json.dumps(payload, ensure_ascii=False)),
                ])
            except LLMResponseFormatError as exc:
                controller.ledger.model_cost_used += exc.usage.cost_usd
                print(f"[diagnosis-error] receipt={delivery.receipt_id} error={exc}", file=sys.stderr)
                return
            except (URLError, TimeoutError, json.JSONDecodeError) as exc:
                print(f"[diagnosis-error] receipt={delivery.receipt_id} error={exc}", file=sys.stderr)
                return

            controller.ledger.model_cost_used += usage.cost_usd
            fields = {
                field: str(parsed.get(field) or "").strip()
                for field in ("observation", "explanation", "limitations", "design_impact")
            }
            if any(fields.values()):
                receipt = f"diagnosis-{delivery.receipt_id}"
                conclusion = {
                    **measurement,
                    **fields,
                    "receipt_id": receipt,
                    "kind": "diagnosis_conclusion",
                    "evidence_refs": [delivery.receipt_id, lookup.lookup_id],
                    "ready_at_s": runtime.clock.now(),
                }
                conclusion["evidence_path"] = str(runtime.store.evidence_dir / f"{receipt}.json")
                runtime.store.write_evidence(receipt, conclusion)
                wiki.diagnoses[receipt] = conclusion

            request = parsed.get("diagnostic")
            if deadline is not None and kinds and isinstance(request, dict):
                await self._request_gpu(delivery, request, kinds)
        finally:
            if isinstance(task_id, str) and task_id in self.ready:
                self.ready[task_id].set()

    async def _request_gpu(
        self, delivery: EvidenceDelivery, request: dict[str, Any], kinds: list[str]
    ) -> None:
        rt = self.runtime
        controller = rt.controller
        version = controller.active_version
        if (
            version is None or version.version_id != delivery.version_id
            or version.status != VersionStatus.ACTIVE.value or version.artifact_id != delivery.artifact_id
        ):
            return
        kind = request.get("kind")
        uid = request.get("workload_id")
        measurement = rt.wiki.measurements[delivery.receipt_id]
        if kind not in kinds or not isinstance(uid, str) or uid not in measurement["workload_ids"]:
            return
        question = str(request.get("question") or "").strip()
        impact = str(request.get("decision_impact") or "").strip()
        explanations = request.get("competing_explanations")
        if not question or not impact or not isinstance(explanations, list) or len({
            item.strip().casefold() for item in explanations if isinstance(item, str) and item.strip()
        }) < 2:
            return
        deadline = request.get("useful_until_s")
        if not isinstance(deadline, (int, float)) or not math.isfinite(deadline):
            return
        deadline = min(float(deadline), version.deadline_at)
        now = rt.clock.now()
        if deadline <= now:
            return
        key = (version.direction_id, version.version_id, version.artifact_id, uid, kind)
        if key in self.request_keys:
            return
        package_id = controller._benchmark_package_id(version.version_id, version.artifact_id)
        block_id = next((b for b, ids in rt.evaluator.blocks.items() if uid in ids), None)
        if not package_id or not block_id:
            return
        task_id = f"diagnostic-{len(self.requests) + 1:04d}"
        budget = rt.config.budgets
        limit = min(budget.first_trial_diag_timeout_s, deadline - now)
        post_cpu_kind = (
            JobKind.NCU_PARSE.value if kind == JobKind.NCU_CAPTURE.value else None
        )
        event = rt.leader.propose_task_event(
            task_id, kind, "gpu", version.direction_id, version.version_id,
            version.artifact_id, reply_to="diagnosis",
            post_cpu_kind=post_cpu_kind,
            produces=[f"diagnosis_ready:{task_id}"],
        )
        event.block_id = block_id
        event.extra.update({
            "package_id": package_id, "timeout_s": limit,
            "metadata": {"workload_id": uid, "useful_until_s": deadline},
        })
        details = {
            "task_id": task_id, "receipt_id": delivery.receipt_id, "kind": kind,
            "version_id": version.version_id, "artifact_id": version.artifact_id,
            "workload_id": uid, "useful_until_s": deadline, "question": question,
            "competing_explanations": explanations, "decision_impact": impact,
        }
        rt.store.write_evidence(task_id, details)
        self.request_keys.add(key)
        self.requests[task_id] = details
        self.ready[task_id] = asyncio.Event()
        await self.submit_event(event)
        wait = rt.leader.declare_waiting_event(
            "diagnosis", f"result_ready:{task_id}", "analyze_diagnostic",
            estimated_duration_s=10.0, direction_id=version.direction_id,
        )
        wait.extra["deadline_s"] = deadline
        await self.submit_event(wait)

    def relevant_diagnoses(self, delivery: EvidenceDelivery) -> list[dict[str, Any]]:
        controller = self.runtime.controller
        lineage = {delivery.version_id: delivery.artifact_id}
        version_id = delivery.version_id
        while version_id is not None:
            version = controller.closed_versions.get(version_id)
            active = controller.active_version
            if active is not None and active.version_id == version_id:
                version = active
            if version is None:
                break
            version_id = version.source_parent_version
            parent = controller.closed_versions.get(version_id)
            if parent is None or version_id in lineage:
                break
            lineage[version_id] = parent.artifact_id
        return [
            row for row in self.runtime.wiki.diagnoses.values()
            if row["direction_id"] == delivery.direction_id
            and row["version_id"] in lineage
            and lineage[row["version_id"]] == row["artifact_id"]
        ]

    def pending_for(self, delivery: EvidenceDelivery) -> list[dict[str, Any]]:
        return [
            request for task_id, request in self.requests.items()
            if request["version_id"] == delivery.version_id
            and request["artifact_id"] == delivery.artifact_id
            and request["useful_until_s"] > self.runtime.clock.now()
            and not self.ready[task_id].is_set()
        ]

    async def wait_if_requested(self, plan: dict[str, Any], delivery: EvidenceDelivery) -> None:
        task_id = plan.get("wait_for_diagnostic_task_id")
        seconds = plan.get("wait_seconds")
        if not isinstance(task_id, str) or not isinstance(seconds, (int, float)) or not math.isfinite(seconds):
            return
        request = next((r for r in self.pending_for(delivery) if r["task_id"] == task_id), None)
        rt = self.runtime
        task = rt.controller.tasks.get(task_id)
        if request is None or (task and task.status in (
            TaskStatus.SKIPPED.value, TaskStatus.CANCELLED.value, TaskStatus.FAILED.value,
            TaskStatus.TIMED_OUT.value,
        )):
            return
        now = rt.clock.now()
        timeout = min(seconds, request["useful_until_s"] - now,
                      rt.config.budgets.total_wall_budget_s + rt.controller.ledger.wall_start_s - now)
        if timeout <= 0:
            return
        wait = rt.leader.declare_waiting_event(
            delivery.recipient, f"diagnosis_ready:{task_id}", "design_revision",
            estimated_duration_s=1.0, direction_id=delivery.direction_id,
            wait_id=f"design-wait:{delivery.receipt_id}:{task_id}",
        )
        wait.extra["deadline_s"] = now + timeout
        await self.submit_event(wait)
        try:
            await asyncio.wait_for(self.ready[task_id].wait(), timeout)
        except asyncio.TimeoutError:
            pass
        finally:
            await self.submit_event(rt.leader.withdraw_waiting_event(wait_id=wait.extra["wait_id"]))
