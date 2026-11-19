"""Persistent append-only event log, atomic state snapshots, and evidence store."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

from kernel_agent_v1.protocol import Event, assert_header_only


class RunStore:
    """Local disk persistence for Controller recovery, audit, and Operator evidence."""

    def __init__(self, run_dir: str | Path) -> None:
        self.root = Path(run_dir)
        self.root.mkdir(parents=True, exist_ok=True)
        self.events_path = self.root / "events.jsonl"
        self.snapshot_path = self.root / "controller_state.json"
        self.decisions_path = self.root / "scheduler_decisions.jsonl"
        self.evidence_dir = self.root / "evidence"
        self.packages_dir = self.root / "packages"
        self.evidence_dir.mkdir(parents=True, exist_ok=True)
        self.packages_dir.mkdir(parents=True, exist_ok=True)

    def append_event(self, event: Event, state_seq: int) -> None:
        record = {
            "state_seq": state_seq,
            "event": event.to_dict(),
        }
        assert_header_only(record["event"])
        with self.events_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
            f.flush()

    def load_events(self) -> list[tuple[int, Event]]:
        if not self.events_path.exists():
            return []
        out: list[tuple[int, Event]] = []
        with self.events_path.open("r", encoding="utf-8") as f:
            for line in f:
                stripped = line.strip()
                if not stripped:
                    continue
                obj = json.loads(stripped)
                out.append((int(obj["state_seq"]), Event.from_dict(obj["event"])))
        return out

    def save_snapshot(self, state: dict[str, Any]) -> None:
        tmp_path = self.snapshot_path.with_suffix(".json.tmp")
        with tmp_path.open("w", encoding="utf-8") as f:
            json.dump(state, f, ensure_ascii=False, indent=2)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp_path, self.snapshot_path)

    def load_snapshot(self) -> dict[str, Any] | None:
        if not self.snapshot_path.exists():
            return None
        with self.snapshot_path.open("r", encoding="utf-8") as f:
            return json.load(f)

    def append_decision(self, decision: dict[str, Any]) -> None:
        with self.decisions_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(decision, ensure_ascii=False) + "\n")
            f.flush()

    def load_decisions(self) -> list[dict[str, Any]]:
        if not self.decisions_path.exists():
            return []
        out: list[dict[str, Any]] = []
        with self.decisions_path.open("r", encoding="utf-8") as f:
            for line in f:
                stripped = line.strip()
                if stripped:
                    out.append(json.loads(stripped))
        return out

    def write_evidence(self, receipt_id: str, payload: dict[str, Any]) -> Path:
        path = self.evidence_dir / f"{receipt_id}.json"
        if path.exists():
            return path
        tmp_path = path.with_suffix(".json.tmp")
        with tmp_path.open("w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp_path, path)
        return path

    def read_evidence(self, receipt_id: str) -> dict[str, Any]:
        path = self.evidence_dir / f"{receipt_id}.json"
        with path.open("r", encoding="utf-8") as f:
            return json.load(f)
