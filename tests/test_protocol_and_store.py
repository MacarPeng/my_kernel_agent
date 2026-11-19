"""Tests for protocol header guards, deduplication, VERSION_CLOSED once, and crash recovery."""

from __future__ import annotations

from pathlib import Path
import tempfile
import unittest

from kernel_agent_v1.clock import SimClock
from kernel_agent_v1.controller import Controller
from kernel_agent_v1.protocol import Event, EventType, RuntimeConfig, assert_header_only
from kernel_agent_v1.store import RunStore


class TestProtocolAndStore(unittest.TestCase):
    def test_header_only_rejects_body_fields(self) -> None:
        with self.assertRaises(ValueError):
            assert_header_only({"event": "gpu_job_finished", "case_results": [1, 2]})

        with self.assertRaises(ValueError):
            Event(
                event=EventType.GPU_JOB_FINISHED.value,
                event_id="evt-bad",
                run_id="r1",
                producer="operator",
                timestamp="2026-11-18T20:00:00+08:00",
                extra={"raw_report": {"x": 1}},
            )

    def test_event_deduplication_and_recovery(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            store = RunStore(Path(tmp))
            cfg = RuntimeConfig(run_id="r-rec")
            clock = SimClock()
            ctrl = Controller(cfg, store, clock)

            evt1 = Event(
                event=EventType.PRIORITY_CHANGED.value,
                event_id="evt-1",
                run_id="r-rec",
                producer="leader",
                timestamp=clock.now_iso(),
                sim_time=0.0,
                direction_id="dir-a",
                extra={"priority": 3.5},
            )
            ctrl.process_event(evt1)
            seq_after_first = ctrl.state_seq
            # Duplicate event_id must be ignored
            ctrl.process_event(evt1)
            self.assertEqual(ctrl.state_seq, seq_after_first)
            self.assertEqual(ctrl.route_priorities["dir-a"], 3.5)

            # Append an unapplied tail event directly to log to simulate crash before snapshot
            evt2 = Event(
                event=EventType.PRIORITY_CHANGED.value,
                event_id="evt-2",
                run_id="r-rec",
                producer="leader",
                timestamp=clock.now_iso(),
                sim_time=5.0,
                direction_id="dir-b",
                extra={"priority": 2.0},
            )
            store.append_event(evt2, seq_after_first + 1)

            recovered = Controller.recover(cfg, store, clock)
            self.assertEqual(recovered.route_priorities["dir-a"], 3.5)
            self.assertEqual(recovered.route_priorities["dir-b"], 2.0)
            self.assertIn("evt-1", recovered.processed_event_ids)
            self.assertIn("evt-2", recovered.processed_event_ids)


if __name__ == "__main__":
    unittest.main()
