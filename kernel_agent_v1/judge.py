"""Judge: fixed adjudication pipeline for complete evals, ABBA retest, and final 3-block eval (§2, §7)."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from kernel_agent_v1.clock import Clock
from kernel_agent_v1.evaluator import TrustedEvaluator
from kernel_agent_v1.protocol import (
    Event,
    EventType,
    EvidenceDelivery,
    ResultKind,
    VerificationLevel,
)


@dataclass
class IncumbentState:
    version_id: str | None = None
    artifact_id: str | None = None
    direction_id: str | None = None
    geomean_speedup: float = 1.0
    verification_level: str = VerificationLevel.BASELINE_ONLY.value
    final_eval_score: float | None = None
    final_eval_blocks: dict[str, dict[str, Any]] = field(default_factory=dict)


class Judge:
    """Fixed adjudication role verifying evidence and deciding promotion/retest/final eval."""

    def __init__(
        self,
        run_id: str,
        evaluator: TrustedEvaluator,
        clock: Clock,
        min_gain_threshold: float = 0.01,
        require_abba_on_first_candidate: bool = False,
        final_eval_blocks: tuple[str, ...] = ("F1", "F2", "F3"),
    ) -> None:
        self.run_id = run_id
        self.evaluator = evaluator
        self.clock = clock
        self.min_gain_threshold = min_gain_threshold
        self.require_abba_on_first_candidate = require_abba_on_first_candidate
        self.final_eval_blocks = final_eval_blocks
        self.incumbent = IncumbentState()
        self._evt_seq = 0
        self._deliv_seq = 0
        self._committed_promotions: set[tuple[str, str]] = set()
        self.verdict_history: list[dict[str, Any]] = []
        # Store complete eval score per (version_id, artifact_id)
        self.verified_scores: dict[tuple[str, str], float] = {}

    def _next_event_id(self) -> str:
        self._evt_seq += 1
        return f"evt-judge-{self._evt_seq:05d}"

    def _next_delivery_id(self) -> str:
        self._deliv_seq += 1
        return f"deliv-judge-{self._deliv_seq:05d}"

    def adjudicate(
        self, delivery: EvidenceDelivery
    ) -> tuple[Event, EvidenceDelivery]:
        """Adjudicate complete benchmark, ABBA retest, or final eval delivery."""
        if delivery.result_kind == ResultKind.BENCHMARK_COMPLETE.value:
            return self._adjudicate_complete(delivery)
        if delivery.result_kind == ResultKind.ABBA_RETEST.value:
            return self._adjudicate_abba(delivery)
        if delivery.result_kind == ResultKind.FINAL_EVAL.value:
            return self._adjudicate_final_block(delivery)
        raise ValueError(f"Judge does not handle result_kind={delivery.result_kind}")

    def _build_outputs(
        self,
        delivery: EvidenceDelivery,
        verdict: str,
        needs_retest: bool,
        full_payload: dict[str, Any],
        extra_header: dict[str, Any] | None = None,
    ) -> tuple[Event, EvidenceDelivery]:
        self.verdict_history.append(full_payload)
        header = Event(
            event=EventType.JUDGE_RESULT.value,
            event_id=self._next_event_id(),
            run_id=self.run_id,
            producer="judge",
            timestamp=self.clock.now_iso(),
            sim_time=self.clock.now(),
            direction_id=delivery.direction_id,
            version_id=delivery.version_id,
            artifact_id=delivery.artifact_id,
            receipt_id=delivery.receipt_id,
            result_kind=delivery.result_kind,
            status=verdict,
            extra={
                "verdict": verdict,
                "needs_retest": needs_retest,
                "incumbent_id": self.incumbent.version_id,
                "incumbent_artifact_id": self.incumbent.artifact_id,
                "verification_level": self.incumbent.verification_level,
                **(extra_header or {}),
            },
        )
        leader_delivery = EvidenceDelivery(
            delivery_id=self._next_delivery_id(),
            receipt_id=delivery.receipt_id,
            recipient="leader",
            result_kind=f"judge_{delivery.result_kind}",
            producer="judge",
            timestamp=self.clock.now_iso(),
            version_id=delivery.version_id,
            artifact_id=delivery.artifact_id,
            direction_id=delivery.direction_id,
            command_id=delivery.command_id,
            block_id=delivery.block_id,
            evidence_path=delivery.evidence_path,
            payload=full_payload,
        )
        return header, leader_delivery

    def _adjudicate_complete(
        self, delivery: EvidenceDelivery
    ) -> tuple[Event, EvidenceDelivery]:
        p = delivery.payload
        ver_id = str(p.get("version_id") or delivery.version_id or "")
        art_id = str(p.get("artifact_id") or delivery.artifact_id or "")
        blocks = p.get("blocks", {})

        agg = self.evaluator.aggregate_complete(
            block_payloads=blocks,
            expected_version_id=ver_id,
            expected_artifact_id=art_id,
        )
        if not agg.valid:
            return self._build_outputs(
                delivery=delivery,
                verdict="rejected_invalid",
                needs_retest=False,
                full_payload={
                    "version_id": ver_id,
                    "artifact_id": art_id,
                    "direction_id": delivery.direction_id,
                    "receipt_id": delivery.receipt_id,
                    "verdict": "rejected_invalid",
                    "needs_retest": False,
                    "issues": agg.issues,
                    "all_correct": agg.all_correct,
                    "geomean_speedup": agg.geomean_speedup,
                },
            )

        self.verified_scores[(ver_id, art_id)] = agg.geomean_speedup
        ref_score = self.incumbent.geomean_speedup
        rel_gain = (agg.geomean_speedup - ref_score) / max(ref_score, 1e-9)

        if self.incumbent.version_id is None and not self.require_abba_on_first_candidate:
            if rel_gain > self.min_gain_threshold:
                key = (ver_id, art_id)
                if key not in self._committed_promotions:
                    self._committed_promotions.add(key)
                    self.incumbent.version_id = ver_id
                    self.incumbent.artifact_id = art_id
                    self.incumbent.direction_id = delivery.direction_id
                    self.incumbent.geomean_speedup = agg.geomean_speedup
                    self.incumbent.verification_level = (
                        VerificationLevel.BLOCK_VERIFIED.value
                    )
                return self._build_outputs(
                    delivery=delivery,
                    verdict="promoted",
                    needs_retest=False,
                    full_payload={
                        "version_id": ver_id,
                        "artifact_id": art_id,
                        "direction_id": delivery.direction_id,
                        "receipt_id": delivery.receipt_id,
                        "verdict": "promoted",
                        "needs_retest": False,
                        "geomean_speedup": agg.geomean_speedup,
                        "rel_gain_vs_baseline": rel_gain,
                        "incumbent_id": self.incumbent.version_id,
                    },
                )
            return self._build_outputs(
                delivery=delivery,
                verdict="rejected_no_gain",
                needs_retest=False,
                full_payload={
                    "version_id": ver_id,
                    "artifact_id": art_id,
                    "direction_id": delivery.direction_id,
                    "receipt_id": delivery.receipt_id,
                    "verdict": "rejected_no_gain",
                    "needs_retest": False,
                    "geomean_speedup": agg.geomean_speedup,
                    "rel_gain": rel_gain,
                },
            )

        # Incumbent exists (or first candidate requires ABBA)
        if rel_gain > self.min_gain_threshold:
            return self._build_outputs(
                delivery=delivery,
                verdict="needs_retest",
                needs_retest=True,
                full_payload={
                    "version_id": ver_id,
                    "artifact_id": art_id,
                    "direction_id": delivery.direction_id,
                    "receipt_id": delivery.receipt_id,
                    "verdict": "needs_retest",
                    "needs_retest": True,
                    "geomean_speedup": agg.geomean_speedup,
                    "rel_gain_vs_incumbent": rel_gain,
                    "expected_incumbent_id": self.incumbent.version_id or "V0",
                    "expected_incumbent_artifact_id": self.incumbent.artifact_id
                    or "fla-baseline",
                },
                extra_header={
                    "expected_incumbent_id": self.incumbent.version_id or "V0",
                    "expected_incumbent_artifact_id": self.incumbent.artifact_id
                    or "fla-baseline",
                    "incumbent_speedup": self.incumbent.geomean_speedup,
                    "candidate_speedup": agg.geomean_speedup,
                },
            )

        return self._build_outputs(
            delivery=delivery,
            verdict="rejected_no_gain",
            needs_retest=False,
            full_payload={
                "version_id": ver_id,
                "artifact_id": art_id,
                "direction_id": delivery.direction_id,
                "receipt_id": delivery.receipt_id,
                "verdict": "rejected_no_gain",
                "needs_retest": False,
                "geomean_speedup": agg.geomean_speedup,
                "rel_gain_vs_incumbent": rel_gain,
            },
        )

    def _adjudicate_abba(
        self, delivery: EvidenceDelivery
    ) -> tuple[Event, EvidenceDelivery]:
        p = delivery.payload
        if p.get("status") not in (None, "completed"):
            return self._build_outputs(
                delivery=delivery,
                verdict="rejected_abba_failed",
                needs_retest=False,
                full_payload={
                    "version_id": delivery.version_id,
                    "artifact_id": delivery.artifact_id,
                    "direction_id": delivery.direction_id,
                    "receipt_id": delivery.receipt_id,
                    "verdict": "rejected_abba_failed",
                    "error": p.get("error"),
                },
            )

        inc_ver = p.get("incumbent_version_id") or "V0"
        inc_art = p.get("incumbent_artifact_id") or "fla-baseline"
        cand_ver = str(p.get("candidate_version_id") or delivery.version_id or "")
        cand_art = str(p.get("candidate_artifact_id") or delivery.artifact_id or "")
        abba = self.evaluator.evaluate_abba(
            incumbent_version_id=inc_ver,
            incumbent_artifact_id=inc_art,
            candidate_version_id=cand_ver,
            candidate_artifact_id=cand_art,
            paired_cases=p.get("paired_cases", []),
            protocol=str(p.get("protocol", self.evaluator.protocol)),
            device_id=str(p.get("device_id", self.evaluator.device_id)),
        )

        # Verify commit-time incumbent identity matches active incumbent!
        current_inc_ver = self.incumbent.version_id or "V0"
        incumbent_still_valid = inc_ver == current_inc_ver
        key = (cand_ver, cand_art)

        if abba.passed and incumbent_still_valid and key not in self._committed_promotions:
            self._committed_promotions.add(key)
            cand_score = self.verified_scores.get(
                key, self.incumbent.geomean_speedup * (1.0 + abba.geomean_gain)
            )
            self.incumbent.version_id = cand_ver
            self.incumbent.artifact_id = cand_art
            self.incumbent.direction_id = delivery.direction_id
            self.incumbent.geomean_speedup = cand_score
            self.incumbent.verification_level = VerificationLevel.ABBA_VERIFIED.value
            return self._build_outputs(
                delivery=delivery,
                verdict="promoted",
                needs_retest=False,
                full_payload={
                    "version_id": cand_ver,
                    "artifact_id": cand_art,
                    "direction_id": delivery.direction_id,
                    "receipt_id": delivery.receipt_id,
                    "verdict": "promoted",
                    "needs_retest": False,
                    "geomean_speedup": cand_score,
                    "abba_gain": abba.geomean_gain,
                    "incumbent_id": cand_ver,
                },
            )

        issues = list(abba.issues)
        if not incumbent_still_valid:
            issues.append(
                f"stale incumbent at commit: compared against {inc_ver}, active is {current_inc_ver}"
            )
        if key in self._committed_promotions:
            issues.append("duplicate promotion blocked")
        cand_score = self.verified_scores.get(key, 0.0)
        return self._build_outputs(
            delivery=delivery,
            verdict="rejected_abba",
            needs_retest=False,
            full_payload={
                "version_id": cand_ver,
                "artifact_id": cand_art,
                "direction_id": delivery.direction_id,
                "receipt_id": delivery.receipt_id,
                "verdict": "rejected_abba",
                "needs_retest": False,
                "geomean_speedup": cand_score,
                "abba_gain": abba.geomean_gain,
                "issues": issues,
            },
        )

    def _adjudicate_final_block(
        self, delivery: EvidenceDelivery
    ) -> tuple[Event, EvidenceDelivery]:
        p = delivery.payload
        ver_id = str(p.get("version_id") or delivery.version_id or "")
        art_id = str(p.get("artifact_id") or delivery.artifact_id or "")
        block_id = str(p.get("final_block_id") or delivery.block_id or "F1")

        if p.get("status") not in (None, "completed") or ver_id != self.incumbent.version_id:
            return self._build_outputs(
                delivery=delivery,
                verdict="final_eval_failed",
                needs_retest=False,
                full_payload={
                    "version_id": ver_id,
                    "artifact_id": art_id,
                    "block_id": block_id,
                    "receipt_id": delivery.receipt_id,
                    "verdict": "final_eval_failed",
                },
                extra_header={"final_eval_done": False},
            )

        self.incumbent.final_eval_blocks[block_id] = p
        all_present = set(self.incumbent.final_eval_blocks.keys()) >= set(
            self.final_eval_blocks
        )
        if not all_present:
            return self._build_outputs(
                delivery=delivery,
                verdict="final_block_recorded",
                needs_retest=False,
                full_payload={
                    "version_id": ver_id,
                    "artifact_id": art_id,
                    "block_id": block_id,
                    "receipt_id": delivery.receipt_id,
                    "verdict": "final_block_recorded",
                    "blocks_collected": sorted(self.incumbent.final_eval_blocks.keys()),
                },
                extra_header={"final_eval_done": False},
            )

        block_scores = [
            float(self.incumbent.final_eval_blocks[b]["geomean_speedup"])
            for b in self.final_eval_blocks
        ]
        final_score = sum(block_scores) / len(block_scores)
        self.incumbent.final_eval_score = final_score
        self.incumbent.verification_level = VerificationLevel.FINAL_VERIFIED.value
        return self._build_outputs(
            delivery=delivery,
            verdict="final_verified",
            needs_retest=False,
            full_payload={
                "version_id": ver_id,
                "artifact_id": art_id,
                "receipt_id": delivery.receipt_id,
                "verdict": "final_verified",
                "final_eval_score": final_score,
                "block_scores": block_scores,
                "verification_level": self.incumbent.verification_level,
            },
            extra_header={"final_eval_done": True},
        )
