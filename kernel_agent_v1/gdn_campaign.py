"""Route selection for version-capped GDN optimization campaigns."""

from __future__ import annotations

from collections import Counter
from dataclasses import replace
from pathlib import Path
import re
from typing import Any, Awaitable, Callable

from kernel_agent_v1.protocol import EvidenceDelivery, ResultKind, RouteStatus, TargetRef
from kernel_agent_v1.runtime import KernelLabRuntime
from kernel_agent_v1.llm import DeepSeekResearcherAdapter, LLMResponseFormatError


GDN_ROOT_ROUTE = "gdn-prefill-main"
GDN_SUBROUTES = {
    "gdn-state-tiling": "Build and tune a fused Triton recurrent kernel with independent V-axis state tiles.",
    "gdn-short-prefill": "Improve short-prefill latency with a low-overhead size-specialized recurrent path.",
    "gdn-chunked-recurrence": "Parallelize long sequences over time using an exact chunked delta-rule recurrence.",
    "gdn-ragged-mapping": "Improve work assignment for variable-length sequences and uneven batch shapes.",
    "gdn-state-layout": "Improve state memory access and register pressure by changing internal layout or staging.",
    "gdn-gate-fusion": "Reduce gating overhead and repeated decay/beta work without changing the recurrence.",
}
GDN_CANDIDATE_FAILURES = frozenset({"compile_failed", "candidate_failed", "numerical_failed"})


class GdnRouteCampaign:
    def __init__(
        self,
        runtime: KernelLabRuntime,
        generate: Callable[
            [str, list[EvidenceDelivery], TargetRef, dict[str, Any] | None, dict[str, str] | None],
            Awaitable[bool],
        ],
        researcher: DeepSeekResearcherAdapter,
    ) -> None:
        self.runtime = runtime
        self.generate = generate
        self.researcher = researcher
        self.discovered: dict[str, dict[str, str]] = {}
        self.evidence_by_version: dict[str, EvidenceDelivery] = {}
        self.repairs_in_chain: Counter[str] = Counter()
        self.opened_routes: set[str] = set()
        self.last_processed_version: str | None = None
        for route, hypothesis in GDN_SUBROUTES.items():
            runtime.register_route(
                route, hypothesis, parent_direction_id=GDN_ROOT_ROUTE, researcher_id="res_main",
            )
        self.save_routes()

    def save_routes(self) -> None:
        self.runtime.store.write_json("run_routes", {
            "routes": {key: node.to_dict() for key, node in self.runtime.wiki.route_tree.routes.items()},
            "repairs_in_chain": dict(self.repairs_in_chain),
            "discovered": self.discovered,
        })

    async def _discover_branch(self, rt: KernelLabRuntime) -> bool:
        routes = rt.wiki.route_tree.routes
        evidence = {row.receipt_id: row for row in self.evidence_by_version.values()}
        if not evidence:
            return False
        context = {
            "direction_id": GDN_ROOT_ROUTE,
            "routes": {
                key: {
                    "hypothesis": node.hypothesis, "parent": node.parent_direction_id,
                    "pause_reason": node.pause_reason, "best_score": node.best_score,
                    **self.discovered.get(key, {}),
                } for key, node in routes.items()
            },
            "evidence": {
                key: {
                    "direction_id": row.direction_id, "version_id": row.version_id,
                    "status": row.payload.get("status"), "error": row.payload.get("error"),
                    "geomean_speedup": row.payload.get("geomean_speedup"),
                } for key, row in evidence.items()
            },
        }
        fields = ("parent_direction_id", "hypothesis", "mechanism", "falsification_test", "receipt_id")
        def normalize(value: str) -> str:
            return " ".join(re.findall(r"\w+", value.casefold()))

        known = {normalize(node.hypothesis) for node in routes.values()}
        known.update(normalize(row["mechanism"]) for row in self.discovered.values())
        known.update(normalize(row["hypothesis"]) for row in self.discovered.values())
        for _ in range(3):
            if rt.controller.ledger.model_cost_used >= rt.config.budgets.total_model_budget_cost:
                rt.controller.freeze_admissions("budget_near_exhaustion")
                return False
            try:
                with rt.timeline.measure("Design", "Discover a new optimization branch"):
                    proposal, usage = await self.researcher.propose_branch(context)
            except LLMResponseFormatError as exc:
                rt.controller.ledger.model_cost_used += exc.usage.cost_usd
                context["previous_rejection"] = str(exc)
                continue
            rt.controller.ledger.model_cost_used += usage.cost_usd
            if not all(isinstance(proposal.get(key), str) and proposal[key].strip() for key in fields):
                context["previous_rejection"] = "All five fields must be nonempty strings."
                continue
            branch = {key: proposal[key].strip() for key in fields}
            parent = branch["parent_direction_id"]
            receipt = evidence.get(branch["receipt_id"])
            if parent not in routes or receipt is None or receipt.direction_id != parent:
                context["previous_rejection"] = "Cite a supplied receipt belonging to your chosen parent."
                continue
            if any(not normalize(branch[key]) or normalize(branch[key]) in known
                   for key in ("hypothesis", "mechanism")):
                context["previous_rejection"] = "Choose a distinct hypothesis and optimization mechanism."
                continue
            route = f"gdn-discovered-{len(self.discovered) + 1:04d}"
            objective = (
                f"{branch['hypothesis']} Mechanism: {branch['mechanism']} "
                f"Falsification test: {branch['falsification_test']}"
            )
            rt.register_route(route, objective, parent_direction_id=parent, researcher_id="res_main")
            self.discovered[route] = branch
            rt.store.append_lifecycle({
                "event": "route_discovered", "timestamp": rt.clock.now_iso(), "direction_id": route, **branch,
            })
            self.save_routes()
            return True
        rt.store.append_lifecycle({
            "event": "route_discovery_exhausted", "timestamp": rt.clock.now_iso(),
            "reason": context["previous_rejection"],
        })
        return False

    def _pause_route(self, direction_id: str, reason: str) -> None:
        rt = self.runtime
        node = rt.wiki.route_tree.routes[direction_id]
        node.status = RouteStatus.PAUSED.value
        node.pause_reason = reason
        event = rt.leader.set_route_priority_event(direction_id, node.priority, node.status)
        event.extra["reason"] = reason
        rt.controller.process_event(event)

    def _collect_evidence(self, version_id: str) -> EvidenceDelivery | None:
        first_block = None
        for delivery in reversed(self.runtime.deliveries_log):
            if delivery.version_id != version_id or delivery.producer != "operator":
                continue
            if delivery.result_kind == ResultKind.BENCHMARK_PARTIAL.value:
                if delivery.payload.get("status") in GDN_CANDIDATE_FAILURES:
                    return delivery
                if delivery.block_id == "W1" and delivery.payload.get("valid"):
                    first_block = delivery
            if delivery.result_kind == ResultKind.ABBA_RETEST.value:
                if delivery.payload.get("status") in GDN_CANDIDATE_FAILURES:
                    return delivery
                failed_cases = [
                    case["uuid"] for case in delivery.payload.get("paired_cases", [])
                    if case.get("correct") is False
                ]
                if failed_cases:
                    return replace(delivery, payload={
                        "status": "numerical_failed", "error": "ABBA correctness failed.",
                        "workload_ids": failed_cases,
                    })
        return first_block

    async def advance(self, rt: KernelLabRuntime) -> bool:
        controller, tree = rt.controller, rt.wiki.route_tree
        if controller.active_version is not None or controller.pending_candidates:
            return False
        previous = controller.closed_versions.get(f"V{controller.version_seq}")
        if previous is None:
            return False
        newly_closed = previous.version_id != self.last_processed_version
        evidence = (
            self._collect_evidence(previous.version_id) if newly_closed
            else self.evidence_by_version.get(previous.version_id)
        )
        failed = evidence is not None and evidence.payload.get("status") in GDN_CANDIDATE_FAILURES
        if newly_closed:
            self.last_processed_version = previous.version_id
            if evidence is not None:
                self.evidence_by_version[previous.version_id] = evidence
            if previous.direction_id != GDN_ROOT_ROUTE and failed:
                assert evidence is not None
                status = evidence.payload["status"]
                failure_kind = {
                    "compile_failed": "compile_failure",
                    "numerical_failed": "correctness_failure",
                    "candidate_failed": "execution_failure",
                }[status]
                tree.record_failure_repair(
                    previous.direction_id, failure_kind, previous.version_id,
                    previous.artifact_id, evidence.receipt_id,
                )
                if self.repairs_in_chain[previous.direction_id] >= rt.config.budgets.max_compile_correctness_repairs:
                    self._pause_route(previous.direction_id, "repair_limit_reached")
            elif previous.close_reason == "completed":
                self.repairs_in_chain.pop(previous.direction_id, None)
            self.save_routes()

        if controller.version_search_finished or controller.gpu_halt_reason:
            return False
        route = max(
            (key for key, node in tree.routes.items()
             if node.parent_direction_id is not None and node.status == RouteStatus.ACTIVE.value),
            key=lambda key: tree.routes[key].priority, default=None,
        )
        if route is None:
            if not await self._discover_branch(rt):
                controller.freeze_admissions("routes_exhausted")
            return True

        node = tree.routes[route]
        if route not in self.opened_routes:
            self.opened_routes.add(route)
            if controller.global_best_version_id and controller.global_best_artifact_id:
                node.reference_version_id = controller.global_best_version_id
                node.reference_artifact_id = controller.global_best_artifact_id
                node.reference_score = rt.judge.incumbent.geomean_speedup

        repair = failed and route == previous.direction_id
        if repair:
            version_id = previous.version_id
        elif node.reference_version_id in controller.closed_versions:
            version_id = node.reference_version_id
        else:
            version_id = node.best_version_id or controller.global_best_version_id or "V1"
        target = controller.closed_versions[version_id]
        package_id = controller._benchmark_package_id(version_id, target.artifact_id)
        package = rt.packages_by_id[package_id]
        files = {
            name: (Path(package.root_dir) / name).read_text(encoding="utf-8")
            for name in package.manifest.file_digests
        }
        parent_evidence = self.evidence_by_version.get(version_id)
        deliveries = [item for item in (parent_evidence, evidence) if item is not None]
        if len(deliveries) == 2 and deliveries[0].receipt_id == deliveries[1].receipt_id:
            deliveries.pop()
        if route in self.discovered:
            cited = self.discovered[route]["receipt_id"]
            if all(item.receipt_id != cited for item in deliveries):
                deliveries.extend(item for item in self.evidence_by_version.values() if item.receipt_id == cited)
        repair_context = None
        if repair:
            assert evidence is not None
            repair_context = {"diagnostic": evidence.payload, "gpu": rt.config.gpu_name, "files": files}
        submitted = await self.generate(
            route, deliveries, TargetRef(version_id, target.artifact_id),
            repair_context, None if repair else files,
        )
        if submitted and repair:
            self.repairs_in_chain[route] += 1
            node.repair_count += 1
        elif not submitted and not controller.ledger.admission_frozen:
            self._pause_route(route, "candidate_generation_exhausted")
        self.save_routes()
        return True
