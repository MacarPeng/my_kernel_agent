"""Leader & Researchers: Wiki-backed hypothesis generation, commands, and route control (§2, §3, §4)."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Callable

from kernel_agent_v1.clock import Clock
from kernel_agent_v1.protocol import (
    Command,
    CommandBody,
    CommandConstraints,
    CommandHead,
    CommandKind,
    Event,
    EventType,
    EvidenceDelivery,
    RouteStatus,
    TargetRef,
)
from kernel_agent_v1.wiki import KernelWiki, WikiLookupResult


@dataclass
class Hypothesis:
    hypothesis_id: str
    direction_id: str
    statement: str
    applicable_shapes: tuple[str, ...]
    evidence_refs: tuple[str, ...]
    falsification_criteria: str
    next_action: str


@dataclass
class ResearchNote:
    research_id: str
    researcher_id: str
    direction_id: str
    wiki_lookup_id: str
    observation: str
    explanation: str
    prediction: str
    hypotheses: list[Hypothesis] = field(default_factory=list)
    reused_evidence_scope: str | None = None
    timestamp: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "research_id": self.research_id,
            "researcher_id": self.researcher_id,
            "direction_id": self.direction_id,
            "wiki_lookup_id": self.wiki_lookup_id,
            "observation": self.observation,
            "explanation": self.explanation,
            "prediction": self.prediction,
            "hypotheses": [asdict(h) for h in self.hypotheses],
            "reused_evidence_scope": self.reused_evidence_scope,
            "timestamp": self.timestamp,
        }


class Researcher:
    """Specialized researcher under Leader handling assigned evidence & hypotheses."""

    def __init__(
        self,
        researcher_id: str,
        wiki: KernelWiki,
        gpu_name: str,
        clock: Clock,
    ) -> None:
        self.researcher_id = researcher_id
        self.wiki = wiki
        self.gpu_name = gpu_name
        self.clock = clock
        self.inbox: list[EvidenceDelivery] = []
        self.notes: list[ResearchNote] = []

    def receive_evidence(self, delivery: EvidenceDelivery) -> None:
        self.inbox.append(delivery)

    def start_research_round(
        self,
        research_id: str,
        direction_id: str,
        evidence_terms: tuple[str, ...],
        observation: str,
        explanation: str,
        prediction: str,
        hypotheses: list[Hypothesis],
        reused_evidence_scope: str | None = None,
    ) -> tuple[WikiLookupResult, ResearchNote]:
        # Mandatory Wiki lookup at the start of every research round (§3, §9)
        lookup = self.wiki.query(
            gpu_name=self.gpu_name,
            direction_id=direction_id,
            evidence_terms=evidence_terms,
            timestamp=self.clock.now_iso(),
        )
        note = ResearchNote(
            research_id=research_id,
            researcher_id=self.researcher_id,
            direction_id=direction_id,
            wiki_lookup_id=lookup.lookup_id,
            observation=observation,
            explanation=explanation,
            prediction=prediction,
            hypotheses=hypotheses,
            reused_evidence_scope=reused_evidence_scope,
            timestamp=self.clock.now_iso(),
        )
        self.notes.append(note)
        return lookup, note


class Leader:
    """Coordinates Researchers, Kernel Wiki, commands to Coder, and route priorities."""

    def __init__(
        self,
        run_id: str,
        wiki: KernelWiki,
        gpu_name: str,
        clock: Clock,
        researcher_ids: tuple[str, ...] = ("researcher",),
    ) -> None:
        self.run_id = run_id
        self.wiki = wiki
        self.gpu_name = gpu_name
        self.clock = clock
        self.researchers: dict[str, Researcher] = {
            rid: Researcher(rid, wiki, gpu_name, clock) for rid in researcher_ids
        }
        self.direction_researcher: dict[str, str] = {}
        self.judge_deliveries: list[EvidenceDelivery] = []
        self.commands_issued: list[Command] = []
        self._cmd_seq = 0
        self._res_seq = 0
        self._evt_seq = 0

    def ensure_researcher(self, researcher_id: str) -> Researcher:
        if researcher_id not in self.researchers:
            self.researchers[researcher_id] = Researcher(
                researcher_id, self.wiki, self.gpu_name, self.clock
            )
        return self.researchers[researcher_id]

    def assign_route_researcher(self, direction_id: str, researcher_id: str) -> None:
        self.ensure_researcher(researcher_id)
        self.direction_researcher[direction_id] = researcher_id

    def next_research_id(self) -> str:
        self._res_seq += 1
        return f"r-{self._res_seq:03d}"

    def next_command_id(self) -> str:
        self._cmd_seq += 1
        return f"cmd-{self._cmd_seq:03d}"

    def _next_event_id(self) -> str:
        self._evt_seq += 1
        return f"evt-leader-{self._evt_seq:05d}"

    def deliver_evidence(self, delivery: EvidenceDelivery) -> None:
        """Route direct evidence delivery from Operator or Judge."""
        if delivery.producer == "judge" or delivery.recipient == "leader":
            self.judge_deliveries.append(delivery)
            self._handle_judge_delivery(delivery)
            return
        rid = delivery.recipient
        researcher = self.ensure_researcher(rid)
        researcher.receive_evidence(delivery)
        if delivery.direction_id:
            self.wiki.route_tree.record_reusable_evidence(
                direction_id=delivery.direction_id,
                receipt_id=delivery.receipt_id,
                version_id=delivery.version_id,
                kind=delivery.result_kind,
                scope=f"workloads:{delivery.block_id or 'targeted'}",
                summary={
                    "geomean_speedup": delivery.payload.get("geomean_speedup"),
                    "observations": delivery.payload.get("observations"),
                },
            )

    def _handle_judge_delivery(self, delivery: EvidenceDelivery) -> list[Event]:
        p = delivery.payload
        verdict = p.get("verdict")
        direction_id = delivery.direction_id or p.get("direction_id")
        ver_id = delivery.version_id or p.get("version_id")
        art_id = delivery.artifact_id or p.get("artifact_id")
        if not direction_id or direction_id not in self.wiki.route_tree.routes:
            return []

        events: list[Event] = []
        if verdict in ("promoted", "rejected_no_gain", "rejected_abba") and p.get("all_correct", True):
            # Only complete comparable verdicts count as a route round (§3);
            # ExperimentRouteTree deduplicates by artifact_id so ABBA won't double-count.
            if delivery.result_kind in ("judge_benchmark_complete", "judge_abba_retest"):
                node = self.wiki.route_tree.record_trusted_verdict(
                    direction_id=direction_id,
                    version_id=str(ver_id or ""),
                    artifact_id=str(art_id or ""),
                    geomean_speedup=float(p.get("geomean_speedup", 1.0)),
                    verdict=str(verdict),
                    receipt_id=delivery.receipt_id,
                )
                if node.status == RouteStatus.PAUSED.value:
                    events.append(
                        Event(
                            event=EventType.PRIORITY_CHANGED.value,
                            event_id=self._next_event_id(),
                            run_id=self.run_id,
                            producer="leader",
                            timestamp=self.clock.now_iso(),
                            sim_time=self.clock.now(),
                            direction_id=direction_id,
                            extra={
                                "route_status": "paused",
                                "reason": node.pause_reason,
                            },
                        )
                    )
        return events

    def build_command(
        self,
        kind: str,
        research_id: str,
        direction_id: str,
        target: TargetRef,
        hypothesis: str,
        evidence_refs: tuple[str, ...],
        instruction: str,
        deliverable: str,
        reply_to: str = "researcher",
        workload_ids: tuple[str, ...] = (),
        allowed_paths: tuple[str, ...] = (),
        preserve_target: bool = True,
        required_files: tuple[str, ...] = (),
    ) -> Command:
        if not any(ref.startswith("wiki-lookup-") for ref in evidence_refs):
            raise ValueError(
                "Command must cite at least one valid wiki-lookup-* credential from this research round"
            )
        cmd = Command(
            head=CommandHead(
                command_id=self.next_command_id(),
                kind=kind,
                research_id=research_id,
                direction_id=direction_id,
                target=target,
                reply_to=reply_to,
            ),
            body=CommandBody(
                hypothesis=hypothesis,
                evidence_refs=evidence_refs,
                instruction=instruction,
                deliverable=deliverable,
                constraints=CommandConstraints(
                    workload_ids=workload_ids,
                    allowed_paths=allowed_paths,
                    preserve_target=preserve_target,
                    required_files=required_files,
                ),
            ),
        )
        self.commands_issued.append(cmd)
        return cmd

    def propose_task_event(
        self,
        task_id: str,
        kind: str,
        resource: str,
        direction_id: str,
        version_id: str | None = None,
        artifact_id: str | None = None,
        command_id: str | None = None,
        priority: float = 1.0,
        assigned_member: str | None = None,
        reply_to: str = "researcher",
        depends_on: list[str] | None = None,
        produces: list[str] | None = None,
        estimated_duration_s: float | None = None,
        duration_category: str = "normal",
        post_cpu_kind: str | None = None,
        post_cpu_duration_s: float | None = None,
        is_required: bool = False,
        replaces_task_id: str | None = None,
    ) -> Event:
        return Event(
            event=EventType.TASK_PROPOSED.value,
            event_id=self._next_event_id(),
            run_id=self.run_id,
            producer="leader",
            timestamp=self.clock.now_iso(),
            sim_time=self.clock.now(),
            command_id=command_id,
            direction_id=direction_id,
            version_id=version_id,
            artifact_id=artifact_id,
            kind=kind,
            extra={
                "task_id": task_id,
                "resource": resource,
                "priority": priority,
                "assigned_member": assigned_member,
                "reply_to": reply_to,
                "depends_on": depends_on or [],
                "produces": produces or [],
                "estimated_duration_s": estimated_duration_s,
                "duration_category": duration_category,
                "post_cpu_kind": post_cpu_kind,
                "post_cpu_duration_s": post_cpu_duration_s,
                "is_required": is_required,
                "replaces_task_id": replaces_task_id,
            },
        )

    def declare_waiting_event(
        self,
        member: str,
        waiting_for: str,
        unblocks_work: str,
        estimated_duration_s: float,
        direction_id: str | None = None,
        wait_id: str | None = None,
    ) -> Event:
        return Event(
            event=EventType.TASK_WAITING.value,
            event_id=self._next_event_id(),
            run_id=self.run_id,
            producer="leader",
            timestamp=self.clock.now_iso(),
            sim_time=self.clock.now(),
            direction_id=direction_id,
            extra={
                "wait_id": wait_id or f"wait:{member}:{waiting_for}",
                "member": member,
                "waiting_for": waiting_for,
                "unblocks_work": unblocks_work,
                "estimated_duration_s": estimated_duration_s,
            },
        )

    def withdraw_waiting_event(
        self,
        member: str | None = None,
        waiting_for: str | None = None,
        wait_id: str | None = None,
        task_id: str | None = None,
    ) -> Event:
        return Event(
            event=EventType.TASK_WITHDRAWN.value,
            event_id=self._next_event_id(),
            run_id=self.run_id,
            producer="leader",
            timestamp=self.clock.now_iso(),
            sim_time=self.clock.now(),
            extra={
                "wait_id": wait_id,
                "member": member,
                "waiting_for": waiting_for,
                "task_id": task_id,
            },
        )

    def set_route_priority_event(
        self, direction_id: str, priority: float, route_status: str | None = None
    ) -> Event:
        if direction_id in self.wiki.route_tree.routes:
            self.wiki.route_tree.routes[direction_id].priority = priority
        extra: dict[str, Any] = {"priority": priority}
        if route_status:
            extra["route_status"] = route_status
        return Event(
            event=EventType.PRIORITY_CHANGED.value,
            event_id=self._next_event_id(),
            run_id=self.run_id,
            producer="leader",
            timestamp=self.clock.now_iso(),
            sim_time=self.clock.now(),
            direction_id=direction_id,
            extra=extra,
        )
