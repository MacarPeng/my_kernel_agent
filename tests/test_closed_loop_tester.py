"""Tests for the reactive ClosedLoopTester across both sim and real subprocess backends."""

from __future__ import annotations

from pathlib import Path
import tempfile
import unittest

from kernel_agent_v1.closed_loop_tester import ClosedLoopTester


class TestClosedLoopTester(unittest.TestCase):
    def test_closed_loop_sim_backend(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            report = ClosedLoopTester(backend_kind="sim").run(work_dir=Path(tmp), verbose=True)
            self.assertTrue(report.passed, report.failures)
            self.assertTrue(all(report.checks.values()))
            self.assertEqual(report.best_artifact_id, "art-reg-v2-fix")
            self.assertEqual(report.verification_level, "final_verified")
            self.assertGreater(len(report.trace), 10)

    def test_closed_loop_subprocess_backend(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            report = ClosedLoopTester(backend_kind="subprocess").run(work_dir=Path(tmp))
            self.assertTrue(report.passed, report.failures)
            self.assertTrue(all(report.checks.values()))
            self.assertEqual(report.best_artifact_id, "art-reg-v2-fix")
            self.assertEqual(report.verification_level, "final_verified")


if __name__ == "__main__":
    unittest.main()
