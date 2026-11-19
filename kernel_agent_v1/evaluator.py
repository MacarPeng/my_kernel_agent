"""Trusted GDN evaluator implementing gdn-unified-v2, geomean speedup, ABBA & final eval (§7)."""

from __future__ import annotations

from dataclasses import dataclass, field
import math
from typing import Any


GDN_PROTOCOL = "gdn-unified-v2"
BLOCK_IDS = ("W1", "W2", "W3", "W4")
FINAL_BLOCK_IDS = ("F1", "F2", "F3")


def generate_fixed_uuids(total_cases: int = 100) -> tuple[str, ...]:
    return tuple(f"gdn-uuid-{i:03d}" for i in range(total_cases))


def partition_blocks(
    uuids: tuple[str, ...], num_blocks: int = 4
) -> dict[str, tuple[str, ...]]:
    if len(uuids) % num_blocks != 0:
        raise ValueError("UUID count must divide evenly across benchmark blocks")
    per_block = len(uuids) // num_blocks
    return {
        f"W{idx + 1}": uuids[idx * per_block : (idx + 1) * per_block]
        for idx in range(num_blocks)
    }


@dataclass(frozen=True)
class CaseMeasurement:
    uuid: str
    block_id: str
    correct: bool
    candidate_ms: float
    baseline_ms: float
    artifact_id: str
    version_id: str
    protocol: str
    device_id: str
    attempt_id: str

    @property
    def speedup(self) -> float:
        if self.candidate_ms <= 0:
            return 0.0
        return self.baseline_ms / self.candidate_ms


@dataclass
class BlockEvaluation:
    block_id: str
    version_id: str
    artifact_id: str
    protocol: str
    device_id: str
    attempt_id: str
    cases: list[CaseMeasurement]
    all_correct: bool
    geomean_speedup: float
    valid: bool
    issues: list[str] = field(default_factory=list)


@dataclass
class CompleteEvaluation:
    version_id: str
    artifact_id: str
    protocol: str
    device_id: str
    total_cases: int
    unique_uuids: int
    all_correct: bool
    geomean_speedup: float
    valid: bool
    issues: list[str] = field(default_factory=list)
    block_receipts: dict[str, str] = field(default_factory=dict)


@dataclass
class ABBAEvaluation:
    incumbent_version_id: str
    incumbent_artifact_id: str
    candidate_version_id: str
    candidate_artifact_id: str
    protocol: str
    device_id: str
    all_correct: bool
    geomean_gain: float  # e.g. 0.05 means +5% faster than incumbent A
    passed: bool
    issues: list[str] = field(default_factory=list)


def geometric_mean(values: list[float]) -> float:
    if not values:
        return 0.0
    if any(v <= 0 for v in values):
        return 0.0
    log_sum = sum(math.log(v) for v in values)
    return math.exp(log_sum / len(values))


class TrustedEvaluator:
    """Trusted platform-side evaluator for GDN protocol verification & scoring."""

    def __init__(
        self,
        protocol: str = GDN_PROTOCOL,
        device_id: str = "NVIDIA-H100-SXM:0",
        total_cases: int = 100,
        num_blocks: int = 4,
        min_gain_threshold: float = 0.01,
    ) -> None:
        self.protocol = protocol
        self.device_id = device_id
        self.uuids = generate_fixed_uuids(total_cases)
        self.expected_uuid_set = frozenset(self.uuids)
        self.blocks = partition_blocks(self.uuids, num_blocks)
        self.min_gain_threshold = min_gain_threshold
        # Frozen FLA baseline latencies (ms) deterministic per UUID
        self.frozen_baseline_ms: dict[str, float] = {
            uuid: 1.0 + (idx % 17) * 0.12 for idx, uuid in enumerate(self.uuids)
        }

    def verify_block(
        self,
        block_id: str,
        raw_cases: list[dict[str, Any]],
        expected_version_id: str,
        expected_artifact_id: str,
        expected_attempt_id: str,
    ) -> BlockEvaluation:
        issues: list[str] = []
        expected_uuids = self.blocks.get(block_id)
        if expected_uuids is None:
            issues.append(f"unknown block_id: {block_id}")

        parsed: list[CaseMeasurement] = []
        seen_uuids: set[str] = set()
        for row in raw_cases:
            uuid = str(row["uuid"])
            if uuid in seen_uuids:
                issues.append(f"duplicate uuid in block {block_id}: {uuid}")
            seen_uuids.add(uuid)

            art = str(row.get("artifact_id", expected_artifact_id))
            ver = str(row.get("version_id", expected_version_id))
            proto = str(row.get("protocol", self.protocol))
            dev = str(row.get("device_id", self.device_id))
            att = str(row.get("attempt_id", expected_attempt_id))
            if art != expected_artifact_id or ver != expected_version_id:
                issues.append(
                    f"identity mismatch on {uuid}: got ({ver},{art}), expected ({expected_version_id},{expected_artifact_id})"
                )
            if proto != self.protocol:
                issues.append(f"protocol mismatch on {uuid}: {proto} != {self.protocol}")
            if dev != self.device_id:
                issues.append(f"device mismatch on {uuid}: {dev} != {self.device_id}")
            if att != expected_attempt_id:
                issues.append(f"attempt mismatch on {uuid}: {att} != {expected_attempt_id}")

            baseline_ms = self.frozen_baseline_ms.get(
                uuid, float(row.get("baseline_ms", 1.0))
            )
            cand_ms = float(row["candidate_ms"])
            correct = bool(row["correct"])
            if cand_ms <= 0:
                issues.append(f"non-positive candidate_ms on {uuid}: {cand_ms}")

            parsed.append(
                CaseMeasurement(
                    uuid=uuid,
                    block_id=block_id,
                    correct=correct,
                    candidate_ms=cand_ms,
                    baseline_ms=baseline_ms,
                    artifact_id=art,
                    version_id=ver,
                    protocol=proto,
                    device_id=dev,
                    attempt_id=att,
                )
            )

        if expected_uuids is not None:
            missing = set(expected_uuids) - seen_uuids
            extra = seen_uuids - set(expected_uuids)
            if missing:
                issues.append(f"missing {len(missing)} uuid(s) in {block_id}")
            if extra:
                issues.append(f"unexpected {len(extra)} uuid(s) in {block_id}")

        all_correct = all(c.correct for c in parsed) and len(parsed) > 0
        if not all_correct:
            issues.append(f"numerical/correctness failure in {block_id}")

        speedups = [c.speedup for c in parsed] if parsed else []
        geomean = geometric_mean(speedups)
        valid = len(issues) == 0

        return BlockEvaluation(
            block_id=block_id,
            version_id=expected_version_id,
            artifact_id=expected_artifact_id,
            protocol=self.protocol,
            device_id=self.device_id,
            attempt_id=expected_attempt_id,
            cases=parsed,
            all_correct=all_correct,
            geomean_speedup=geomean,
            valid=valid,
            issues=issues,
        )

    def aggregate_complete(
        self,
        block_payloads: dict[str, dict[str, Any]],
        expected_version_id: str,
        expected_artifact_id: str,
    ) -> CompleteEvaluation:
        """Aggregate W1..W4 and strictly validate coverage, uniqueness, and provenance."""
        issues: list[str] = []
        all_cases: list[CaseMeasurement] = []
        seen_uuids: set[str] = set()
        block_receipts: dict[str, str] = {}

        expected_blocks = set(self.blocks.keys())
        present_blocks = set(block_payloads.keys())
        if present_blocks != expected_blocks:
            missing_b = sorted(expected_blocks - present_blocks)
            issues.append(f"missing blocks: {missing_b}")

        for block_id in sorted(block_payloads.keys()):
            payload = block_payloads[block_id]
            receipt_id = str(payload.get("receipt_id", ""))
            block_receipts[block_id] = receipt_id

            if payload.get("version_id") != expected_version_id:
                issues.append(
                    f"mixed version in {block_id}: {payload.get('version_id')} != {expected_version_id}"
                )
            if payload.get("artifact_id") != expected_artifact_id:
                issues.append(
                    f"mixed artifact in {block_id}: {payload.get('artifact_id')} != {expected_artifact_id}"
                )
            if payload.get("protocol") != self.protocol:
                issues.append(
                    f"mixed protocol in {block_id}: {payload.get('protocol')} != {self.protocol}"
                )
            if payload.get("device_id") != self.device_id:
                issues.append(
                    f"mixed device in {block_id}: {payload.get('device_id')} != {self.device_id}"
                )

            raw_cases = payload.get("cases", [])
            for row in raw_cases:
                uuid = str(row["uuid"])
                if uuid in seen_uuids:
                    issues.append(f"duplicate uuid across blocks: {uuid}")
                seen_uuids.add(uuid)

                row_art = str(row.get("artifact_id", payload.get("artifact_id", "")))
                row_ver = str(row.get("version_id", payload.get("version_id", "")))
                row_proto = str(row.get("protocol", payload.get("protocol", "")))
                row_dev = str(row.get("device_id", payload.get("device_id", "")))
                if row_art != expected_artifact_id or row_ver != expected_version_id:
                    issues.append(f"mixed source on {uuid} in {block_id}")
                if row_proto != self.protocol or row_dev != self.device_id:
                    issues.append(f"mixed env/protocol on {uuid} in {block_id}")

                baseline_ms = self.frozen_baseline_ms.get(
                    uuid, float(row.get("baseline_ms", 1.0))
                )
                cand_ms = float(row["candidate_ms"])
                correct = bool(row["correct"])
                if cand_ms <= 0:
                    issues.append(f"invalid timing on {uuid}: {cand_ms}")
                all_cases.append(
                    CaseMeasurement(
                        uuid=uuid,
                        block_id=block_id,
                        correct=correct,
                        candidate_ms=cand_ms,
                        baseline_ms=baseline_ms,
                        artifact_id=row_art,
                        version_id=row_ver,
                        protocol=row_proto,
                        device_id=row_dev,
                        attempt_id=str(row.get("attempt_id", payload.get("attempt_id", ""))),
                    )
                )

        missing_global = self.expected_uuid_set - seen_uuids
        extra_global = seen_uuids - self.expected_uuid_set
        if missing_global:
            issues.append(f"missing {len(missing_global)} required UUIDs")
        if extra_global:
            issues.append(f"unexpected {len(extra_global)} extra UUIDs")

        all_correct = len(all_cases) > 0 and all(c.correct for c in all_cases)
        if not all_correct:
            issues.append("not all 100 cases passed correctness")

        geomean = geometric_mean([c.speedup for c in all_cases]) if all_cases else 0.0
        valid = (
            len(issues) == 0
            and len(seen_uuids) == len(self.expected_uuid_set)
            and all_correct
        )
        return CompleteEvaluation(
            version_id=expected_version_id,
            artifact_id=expected_artifact_id,
            protocol=self.protocol,
            device_id=self.device_id,
            total_cases=len(all_cases),
            unique_uuids=len(seen_uuids),
            all_correct=all_correct,
            geomean_speedup=geomean,
            valid=valid,
            issues=issues,
            block_receipts=block_receipts,
        )

    def evaluate_abba(
        self,
        incumbent_version_id: str,
        incumbent_artifact_id: str,
        candidate_version_id: str,
        candidate_artifact_id: str,
        paired_cases: list[dict[str, Any]],
        protocol: str,
        device_id: str,
    ) -> ABBAEvaluation:
        """Verify ABBA paired timing inside one GPU lease (A1, B1, B2, A2)."""
        issues: list[str] = []
        if protocol != self.protocol:
            issues.append(f"protocol mismatch: {protocol} != {self.protocol}")
        if device_id != self.device_id:
            issues.append(f"device mismatch: {device_id} != {self.device_id}")

        seen: set[str] = set()
        ratios: list[float] = []
        all_correct = True
        for row in paired_cases:
            uuid = str(row["uuid"])
            if uuid in seen:
                issues.append(f"duplicate uuid in ABBA: {uuid}")
            seen.add(uuid)
            if not bool(row.get("correct", False)):
                all_correct = False
            a1 = float(row["a1_ms"])
            b1 = float(row["b1_ms"])
            b2 = float(row["b2_ms"])
            a2 = float(row["a2_ms"])
            if min(a1, b1, b2, a2) <= 0:
                issues.append(f"invalid ABBA timing on {uuid}")
                continue
            a_mean = 0.5 * (a1 + a2)
            b_mean = 0.5 * (b1 + b2)
            ratios.append(a_mean / b_mean)

        if seen != self.expected_uuid_set:
            issues.append(
                f"ABBA coverage incomplete: {len(seen)}/{len(self.expected_uuid_set)}"
            )
        if not all_correct:
            issues.append("ABBA correctness failed")

        geomean_ratio = geometric_mean(ratios) if ratios else 0.0
        gain = geomean_ratio - 1.0
        passed = (
            len(issues) == 0
            and all_correct
            and gain > self.min_gain_threshold
        )
        return ABBAEvaluation(
            incumbent_version_id=incumbent_version_id,
            incumbent_artifact_id=incumbent_artifact_id,
            candidate_version_id=candidate_version_id,
            candidate_artifact_id=candidate_artifact_id,
            protocol=protocol,
            device_id=device_id,
            all_correct=all_correct,
            geomean_gain=gain,
            passed=passed,
            issues=issues,
        )
