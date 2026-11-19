"""Coder role: constraint-checked immutable code package builder (§4)."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import stat
from typing import Any, Callable

from kernel_agent_v1.clock import Clock
from kernel_agent_v1.protocol import (
    CodePackage,
    Command,
    CommandKind,
    ConstraintCheckResult,
    Event,
    EventType,
    JobKind,
    PackageManifest,
)
from kernel_agent_v1.wiki import KernelWiki


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def verify_package_integrity(package: CodePackage) -> tuple[bool, str]:
    """Verify an immutable code package has not been altered on disk after sealing."""
    root = Path(package.root_dir)
    manifest_path = root / "manifest.json"
    if not manifest_path.exists():
        return False, "missing manifest.json"
    for rel_path, expected_digest in package.manifest.file_digests.items():
        fp = root / rel_path
        if not fp.exists():
            return False, f"missing packaged file: {rel_path}"
        actual = _sha256_bytes(fp.read_bytes())
        if actual != expected_digest:
            return False, f"digest mismatch on {rel_path}: {actual} != {expected_digest}"
    return True, "ok"


class Coder:
    """Builds and seals immutable, constraint-verified code packages."""

    def __init__(
        self,
        packages_dir: str | Path,
        run_id: str,
        clock: Clock,
        wiki: KernelWiki | None = None,
        implementation_fn: Callable[[Command], dict[str, Any]] | None = None,
    ) -> None:
        self.packages_dir = Path(packages_dir)
        self.packages_dir.mkdir(parents=True, exist_ok=True)
        self.run_id = run_id
        self.clock = clock
        self.wiki = wiki
        self.implementation_fn = implementation_fn
        self._pkg_seq = 0
        self._evt_seq = 0
        self.packages: dict[str, CodePackage] = {}

    def _next_event_id(self) -> str:
        self._evt_seq += 1
        return f"evt-coder-{self._evt_seq:05d}"

    @staticmethod
    def _default_stages(kind: str) -> tuple[str, ...]:
        if kind == CommandKind.IMPLEMENT_KERNEL.value:
            return (JobKind.CPU_COMPILE.value, JobKind.BENCHMARK_BLOCK.value)
        if kind == CommandKind.WRITE_NCU_CAPTURE.value:
            return (JobKind.NCU_CAPTURE.value, JobKind.NCU_PARSE.value)
        if kind == CommandKind.WRITE_PROBE.value:
            return (JobKind.PROBE.value,)
        if kind == CommandKind.WRITE_COMPILER_ANALYSIS.value:
            return (JobKind.COMPILER_ANALYSIS.value,)
        return (JobKind.PROBE.value,)

    def _check_constraints(
        self,
        command: Command,
        files: dict[str, str],
        entrypoint: str,
        target_digest_after: str | None,
        workload_ids: tuple[str, ...],
    ) -> ConstraintCheckResult:
        violations: list[str] = []
        constraints = command.body.constraints

        # 1. Wiki lookup credential check (§3, §9)
        wiki_refs = [r for r in command.body.evidence_refs if r.startswith("wiki-lookup-")]
        if not wiki_refs:
            violations.append("missing wiki-lookup credential in evidence_refs")
        elif self.wiki is not None:
            if not any(self.wiki.has_valid_lookup(ref) for ref in wiki_refs):
                violations.append(f"unregistered wiki lookup credential(s): {wiki_refs}")

        # 2. Target preservation & digest check
        if constraints.preserve_target and command.head.target.digest:
            if (
                target_digest_after is not None
                and target_digest_after != command.head.target.digest
            ):
                violations.append(
                    f"preserve_target violated: {target_digest_after} != {command.head.target.digest}"
                )

        # 3. Allowed paths check
        checked_paths = tuple(sorted(files.keys()))
        if constraints.allowed_paths:
            for path in checked_paths:
                norm = path.lstrip("./")
                if not any(
                    norm == ap.rstrip("/") or norm.startswith(ap.rstrip("/") + "/")
                    for ap in constraints.allowed_paths
                ):
                    violations.append(
                        f"path out of bounds: '{path}' not in allowed_paths {constraints.allowed_paths}"
                    )

        # 4. Required files & entrypoint check
        if entrypoint not in files:
            violations.append(f"missing entrypoint '{entrypoint}' in package files")
        for req in constraints.required_files:
            if req not in files:
                violations.append(f"missing required file '{req}'")

        # 5. Fixed workload_ids check
        if constraints.workload_ids and set(workload_ids) != set(constraints.workload_ids):
            violations.append(
                f"workload_ids mismatch: {workload_ids} != {constraints.workload_ids}"
            )

        return ConstraintCheckResult(
            passed=len(violations) == 0,
            violations=tuple(violations),
            checked_paths=checked_paths,
        )

    def execute_command(
        self,
        command: Command,
        provided_spec: dict[str, Any] | None = None,
    ) -> tuple[Event, CodePackage | None]:
        """Build an immutable package or return code_blocked on constraint/input error."""
        spec = dict(provided_spec) if provided_spec is not None else (
            self.implementation_fn(command) if self.implementation_fn else {}
        )

        if spec.get("blocked_reason"):
            evt = Event(
                event=EventType.CODE_BLOCKED.value,
                event_id=self._next_event_id(),
                run_id=self.run_id,
                producer="coder",
                timestamp=self.clock.now_iso(),
                sim_time=self.clock.now(),
                command_id=command.head.command_id,
                direction_id=command.head.direction_id,
                parent_version_id=command.head.target.version_id,
                status="blocked",
                extra={"reason": spec["blocked_reason"]},
            )
            return evt, None

        self._pkg_seq += 1
        package_id = spec.get("package_id") or f"pkg-{self._pkg_seq:04d}"
        artifact_id = spec.get("artifact_id") or f"art-{command.head.command_id}"
        entrypoint = spec.get("entrypoint") or self._default_entrypoint(command)
        files: dict[str, str] = spec.get("files") or self._default_files(command, entrypoint)
        stages = tuple(spec.get("executable_stages") or self._default_stages(command.head.kind))
        workload_ids = tuple(
            spec.get("workload_ids") or command.body.constraints.workload_ids
        )
        target_digest_after = spec.get("target_digest_after", command.head.target.digest)

        check = self._check_constraints(
            command=command,
            files=files,
            entrypoint=entrypoint,
            target_digest_after=target_digest_after,
            workload_ids=workload_ids,
        )
        if not check.passed:
            evt = Event(
                event=EventType.CODE_BLOCKED.value,
                event_id=self._next_event_id(),
                run_id=self.run_id,
                producer="coder",
                timestamp=self.clock.now_iso(),
                sim_time=self.clock.now(),
                command_id=command.head.command_id,
                direction_id=command.head.direction_id,
                artifact_id=artifact_id,
                parent_version_id=command.head.target.version_id,
                status="constraint_violation",
                extra={"violations": list(check.violations)},
            )
            return evt, None

        pkg_dir = self.packages_dir / package_id
        pkg_dir.mkdir(parents=True, exist_ok=True)
        file_digests: dict[str, str] = {}
        for rel_path, content in sorted(files.items()):
            fp = pkg_dir / rel_path
            fp.parent.mkdir(parents=True, exist_ok=True)
            raw = content.encode("utf-8")
            fp.write_bytes(raw)
            file_digests[rel_path] = _sha256_bytes(raw)

        pkg_digest_src = json.dumps(
            {
                "artifact_id": artifact_id,
                "command_id": command.head.command_id,
                "entrypoint": entrypoint,
                "file_digests": file_digests,
            },
            sort_keys=True,
        ).encode("utf-8")
        pkg_digest = _sha256_bytes(pkg_digest_src)

        manifest = PackageManifest(
            package_id=package_id,
            artifact_id=artifact_id,
            command_id=command.head.command_id,
            task_kind=command.head.kind,
            direction_id=command.head.direction_id,
            parent_version_id=command.head.target.version_id,
            target_identity=command.head.target,
            entrypoint=entrypoint,
            file_digests=file_digests,
            executable_stages=stages,
            workload_ids=workload_ids,
            reply_to=command.head.reply_to,
            package_digest=pkg_digest,
            constraint_check=check,
            created_at=self.clock.now_iso(),
        )
        manifest_path = pkg_dir / "manifest.json"
        manifest_path.write_text(
            json.dumps(manifest.to_dict(), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

        # Seal files read-only so in-place mutation is blocked at OS level too.
        for rel_path in files:
            os.chmod(pkg_dir / rel_path, stat.S_IRUSR | stat.S_IRGRP | stat.S_IROTH)
        os.chmod(manifest_path, stat.S_IRUSR | stat.S_IRGRP | stat.S_IROTH)

        metadata = dict(spec.get("metadata", {}))
        pkg = CodePackage(manifest=manifest, root_dir=str(pkg_dir), metadata=metadata)
        self.packages[package_id] = pkg

        evt = Event(
            event=EventType.CODE_READY.value,
            event_id=self._next_event_id(),
            run_id=self.run_id,
            producer="coder",
            timestamp=self.clock.now_iso(),
            sim_time=self.clock.now(),
            command_id=command.head.command_id,
            direction_id=command.head.direction_id,
            artifact_id=artifact_id,
            parent_version_id=command.head.target.version_id,
            kind=command.head.kind,
            status="ready",
            extra={
                "package_id": package_id,
                "package_digest": pkg_digest,
                "executable_stages": list(stages),
                "reply_to": command.head.reply_to,
                "task_id": spec.get("task_id"),
            },
        )
        return evt, pkg

    @staticmethod
    def _default_entrypoint(command: Command) -> str:
        allowed = command.body.constraints.allowed_paths
        prefix = (allowed[0].rstrip("/") + "/") if allowed else "src/"
        if command.head.kind == CommandKind.IMPLEMENT_KERNEL.value:
            return f"{prefix}kernel.py"
        if command.head.kind == CommandKind.WRITE_NCU_CAPTURE.value:
            return f"{prefix}ncu_capture.py"
        if command.head.kind == CommandKind.WRITE_PROBE.value:
            return f"{prefix}probe.py"
        return f"{prefix}analyze.py"

    @staticmethod
    def _default_files(command: Command, entrypoint: str) -> dict[str, str]:
        return {
            entrypoint: (
                f"# Auto-generated immutable deliverable for {command.head.command_id}\n"
                f"# Target: {command.head.target.version_id}/{command.head.target.artifact_id}\n"
                f"# Hypothesis: {command.body.hypothesis}\n"
                "def run(inputs):\n"
                "    return {'status': 'ok', 'inputs': inputs}\n"
            )
        }
