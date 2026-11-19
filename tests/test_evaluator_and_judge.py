"""Tests for TrustedEvaluator (gdn-unified-v2) and Judge (ABBA, commit check, 3-block final)."""

from __future__ import annotations

import unittest

from kernel_agent_v1.clock import SimClock
from kernel_agent_v1.evaluator import TrustedEvaluator
from kernel_agent_v1.judge import Judge
from kernel_agent_v1.protocol import EvidenceDelivery, ResultKind, VerificationLevel


def _make_block_payload(
    ev: TrustedEvaluator,
    block_id: str,
    version_id: str,
    artifact_id: str,
    speedup: float,
    drop_last: bool = False,
    dup_first: bool = False,
    mixed_first_art: str | None = None,
) -> dict:
    uuids = list(ev.blocks[block_id])
    if drop_last:
        uuids = uuids[:-1]
    if dup_first:
        uuids[1] = uuids[0]
    cases = []
    for idx, uid in enumerate(uuids):
        base_ms = ev.frozen_baseline_ms[uid]
        cases.append(
            {
                "uuid": uid,
                "correct": True,
                "candidate_ms": base_ms / speedup,
                "baseline_ms": base_ms,
                "artifact_id": mixed_first_art if (idx == 0 and mixed_first_art) else artifact_id,
                "version_id": version_id,
                "protocol": ev.protocol,
                "device_id": ev.device_id,
                "attempt_id": f"job-{block_id}/1",
            }
        )
    return {
        "receipt_id": f"rec-{version_id}-{block_id}",
        "version_id": version_id,
        "artifact_id": artifact_id,
        "protocol": ev.protocol,
        "device_id": ev.device_id,
        "cases": cases,
    }


class TestEvaluatorAndJudge(unittest.TestCase):
    def test_measured_baseline_is_used_for_block_and_complete_scores(self) -> None:
        ev = TrustedEvaluator(total_cases=4, use_measured_baselines=True)
        blocks = {b: _make_block_payload(ev, b, "V1", "art-1", 1.0) for b in ev.blocks}
        for payload in blocks.values():
            for row in payload["cases"]:
                row.update(baseline_ms=0.6, candidate_ms=0.2)

        block = ev.verify_block("W1", blocks["W1"]["cases"], "V1", "art-1", "job-W1/1")
        self.assertTrue(block.valid)
        self.assertAlmostEqual(block.geomean_speedup, 3.0)
        complete = ev.aggregate_complete(blocks, "V1", "art-1")
        self.assertTrue(complete.valid)
        self.assertAlmostEqual(complete.geomean_speedup, 3.0)

        blocks["W1"]["cases"][0].pop("baseline_ms")
        self.assertFalse(ev.verify_block("W1", blocks["W1"]["cases"], "V1", "art-1", "job-W1/1").valid)
        self.assertFalse(ev.aggregate_complete(blocks, "V1", "art-1").valid)

    def test_missing_duplicate_and_mixed_source_rejected(self) -> None:
        ev = TrustedEvaluator()
        # Missing UUID
        blocks_missing = {
            b: _make_block_payload(ev, b, "V1", "art-1", 1.2, drop_last=(b == "W4"))
            for b in ("W1", "W2", "W3", "W4")
        }
        res_missing = ev.aggregate_complete(blocks_missing, "V1", "art-1")
        self.assertFalse(res_missing.valid)

        # Duplicate UUID
        blocks_dup = {
            b: _make_block_payload(ev, b, "V1", "art-1", 1.2, dup_first=(b == "W2"))
            for b in ("W1", "W2", "W3", "W4")
        }
        res_dup = ev.aggregate_complete(blocks_dup, "V1", "art-1")
        self.assertFalse(res_dup.valid)

        # Mixed artifact source
        blocks_mix = {
            b: _make_block_payload(
                ev, b, "V1", "art-1", 1.2, mixed_first_art=("art-OTHER" if b == "W1" else None)
            )
            for b in ("W1", "W2", "W3", "W4")
        }
        res_mix = ev.aggregate_complete(blocks_mix, "V1", "art-1")
        self.assertFalse(res_mix.valid)

        blocks_retry = {b: _make_block_payload(ev, b, "V1", "art-1", 1.2) for b in ev.blocks}
        blocks_retry["W2"]["attempt_id"] = "job-W2/2"
        res_retry = ev.aggregate_complete(blocks_retry, "V1", "art-1")
        self.assertFalse(res_retry.valid)
        self.assertTrue(any("attempt mismatch" in issue for issue in res_retry.issues))

    def test_first_retention_abba_and_final_eval_three_blocks(self) -> None:
        ev = TrustedEvaluator(min_gain_threshold=0.01)
        clock = SimClock()
        judge = Judge("run-1", ev, clock, min_gain_threshold=0.01)

        # V1 beats baseline by 1.15x (> 0.01) -> promoted as first incumbent
        b1 = {b: _make_block_payload(ev, b, "V1", "art-1", 1.15) for b in ("W1", "W2", "W3", "W4")}
        d1 = EvidenceDelivery(
            delivery_id="d1",
            receipt_id="r1",
            recipient="judge",
            result_kind=ResultKind.BENCHMARK_COMPLETE.value,
            producer="operator",
            timestamp=clock.now_iso(),
            version_id="V1",
            artifact_id="art-1",
            direction_id="dir-1",
            command_id=None,
            block_id=None,
            evidence_path="/tmp/r1.json",
            payload={"version_id": "V1", "artifact_id": "art-1", "blocks": b1},
        )
        evt1, _ = judge.adjudicate(d1)
        self.assertEqual(evt1.extra["verdict"], "promoted")
        self.assertEqual(judge.incumbent.version_id, "V1")

        # V2 beats V1 by +8% -> needs_retest (ABBA)
        b2 = {b: _make_block_payload(ev, b, "V2", "art-2", 1.24) for b in ("W1", "W2", "W3", "W4")}
        d2 = EvidenceDelivery(
            delivery_id="d2",
            receipt_id="r2",
            recipient="judge",
            result_kind=ResultKind.BENCHMARK_COMPLETE.value,
            producer="operator",
            timestamp=clock.now_iso(),
            version_id="V2",
            artifact_id="art-2",
            direction_id="dir-1",
            command_id=None,
            block_id=None,
            evidence_path="/tmp/r2.json",
            payload={"version_id": "V2", "artifact_id": "art-2", "blocks": b2},
        )
        evt2, _ = judge.adjudicate(d2)
        self.assertTrue(evt2.extra["needs_retest"])

        # ABBA paired retest confirms V2 > V1
        paired = [
            {
                "uuid": uid,
                "correct": True,
                "a1_ms": ev.frozen_baseline_ms[uid] / 1.15,
                "b1_ms": ev.frozen_baseline_ms[uid] / 1.24,
                "b2_ms": ev.frozen_baseline_ms[uid] / 1.24,
                "a2_ms": ev.frozen_baseline_ms[uid] / 1.15,
            }
            for uid in ev.uuids
        ]
        d_abba = EvidenceDelivery(
            delivery_id="d3",
            receipt_id="r3",
            recipient="judge",
            result_kind=ResultKind.ABBA_RETEST.value,
            producer="operator",
            timestamp=clock.now_iso(),
            version_id="V2",
            artifact_id="art-2",
            direction_id="dir-1",
            command_id=None,
            block_id=None,
            evidence_path="/tmp/r3.json",
            payload={
                "incumbent_version_id": "V1",
                "incumbent_artifact_id": "art-1",
                "candidate_version_id": "V2",
                "candidate_artifact_id": "art-2",
                "protocol": ev.protocol,
                "device_id": ev.device_id,
                "paired_cases": paired,
            },
        )
        evt_abba, _ = judge.adjudicate(d_abba)
        self.assertEqual(evt_abba.extra["verdict"], "promoted")
        self.assertEqual(judge.incumbent.version_id, "V2")
        self.assertEqual(judge.incumbent.verification_level, VerificationLevel.ABBA_VERIFIED.value)

        # Stale ABBA comparing against V1 after V2 already promoted must fail commit check
        d_stale = EvidenceDelivery(
            delivery_id="d4",
            receipt_id="r4",
            recipient="judge",
            result_kind=ResultKind.ABBA_RETEST.value,
            producer="operator",
            timestamp=clock.now_iso(),
            version_id="V3",
            artifact_id="art-3",
            direction_id="dir-1",
            command_id=None,
            block_id=None,
            evidence_path="/tmp/r4.json",
            payload={
                "incumbent_version_id": "V1",
                "incumbent_artifact_id": "art-1",
                "candidate_version_id": "V3",
                "candidate_artifact_id": "art-3",
                "protocol": ev.protocol,
                "device_id": ev.device_id,
                "paired_cases": paired,
            },
        )
        evt_stale, _ = judge.adjudicate(d_stale)
        self.assertEqual(evt_stale.extra["verdict"], "rejected_abba")
        self.assertEqual(judge.incumbent.version_id, "V2")

        # 3 independent final eval blocks
        for fb in ("F1", "F2", "F3"):
            df = EvidenceDelivery(
                delivery_id=f"df-{fb}",
                receipt_id=f"rf-{fb}",
                recipient="judge",
                result_kind=ResultKind.FINAL_EVAL.value,
                producer="operator",
                timestamp=clock.now_iso(),
                version_id="V2",
                artifact_id="art-2",
                direction_id="final_eval",
                command_id=None,
                block_id=fb,
                evidence_path=f"/tmp/rf-{fb}.json",
                payload={
                    "final_block_id": fb,
                    "version_id": "V2",
                    "artifact_id": "art-2",
                    "geomean_speedup": 1.24,
                    "status": "completed",
                },
            )
            ef, _ = judge.adjudicate(df)
        self.assertEqual(ef.extra["verdict"], "final_verified")
        self.assertEqual(judge.incumbent.verification_level, VerificationLevel.FINAL_VERIFIED.value)


if __name__ == "__main__":
    unittest.main()
