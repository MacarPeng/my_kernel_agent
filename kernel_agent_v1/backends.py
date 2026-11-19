"""Execution backends for Operator: simulated discrete-event backend & local subprocess backend."""

from __future__ import annotations

from abc import ABC, abstractmethod
import asyncio
import csv
from dataclasses import dataclass, field
import io
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import tempfile
from typing import Any

from kernel_agent_v1.clock import Clock
from kernel_agent_v1.evaluator import TrustedEvaluator
from kernel_agent_v1.protocol import CodePackage, JobKind


@dataclass
class JobHandle:
    job_id: str
    attempt_id: str
    task_id: str
    kind: str
    resource: str  # "gpu" | "cpu"
    version_id: str | None
    artifact_id: str | None
    direction_id: str
    block_id: str | None
    reply_to: str
    started_at: float
    expected_finish_at: float
    timeout_at: float
    status: str = "running"  # running | completed | failed | cancelled | timed_out
    pid: int | None = None
    pgid: int | None = None
    workdir: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class RawOutcome:
    job_id: str
    attempt_id: str
    status: str  # completed | compile_failed | numerical_failed | infra_failed | timed_out | cancelled
    gpu_released: bool
    duration_s: float
    payload: dict[str, Any] = field(default_factory=dict)
    error: str | None = None


@dataclass
class CleanupReceipt:
    job_id: str
    attempt_id: str
    gpu_released: bool
    reason: str
    already_stopped: bool = False


class ExecutionBackend(ABC):
    """Backend abstraction for Operator start/wait/stop."""

    @abstractmethod
    def start(
        self,
        job_id: str,
        attempt_id: str,
        task_id: str,
        kind: str,
        resource: str,
        package: CodePackage | None,
        inputs: dict[str, Any],
        clock: Clock,
    ) -> JobHandle:
        """Launch a GPU or CPU job and return its handle."""

    @abstractmethod
    def wait(self, handle: JobHandle, clock: Clock) -> RawOutcome:
        """Wait for or collect the outcome of a started handle."""

    @abstractmethod
    def stop(self, handle: JobHandle, reason: str, clock: Clock) -> CleanupReceipt:
        """Idempotently cancel and confirm GPU/process cleanup."""

    async def async_wait(self, handle: JobHandle, clock: Clock) -> RawOutcome:
        """Non-blocking awaitable wait; subclasses override for true async I/O."""
        return self.wait(handle, clock)


class SimulatedExecutionBackend(ExecutionBackend):
    """Deterministic simulation backend supporting fault injection and exact timing."""

    def __init__(self, evaluator: TrustedEvaluator) -> None:
        self.evaluator = evaluator
        self._stopped_attempts: set[str] = set()
        # Optional overrides keyed by attempt_id, (artifact_id, kind, block_id), or task_id
        self.scripted_behaviors: dict[str, dict[str, Any]] = {}
        self.unconfirmed_cleanup_attempts: set[str] = set()

    def set_behavior(self, key: str, behavior: dict[str, Any]) -> None:
        self.scripted_behaviors[key] = behavior

    def _lookup_behavior(
        self,
        task_id: str,
        attempt_id: str,
        artifact_id: str | None,
        kind: str,
        block_id: str | None,
        inputs: dict[str, Any],
    ) -> dict[str, Any]:
        for key in (
            attempt_id,
            task_id,
            f"{artifact_id}:{kind}:{block_id}",
            f"{artifact_id}:{kind}",
            f"{kind}:{block_id}",
            kind,
        ):
            if key in self.scripted_behaviors:
                return dict(self.scripted_behaviors[key])
        return dict(inputs.get("sim_behavior", {}))

    def start(
        self,
        job_id: str,
        attempt_id: str,
        task_id: str,
        kind: str,
        resource: str,
        package: CodePackage | None,
        inputs: dict[str, Any],
        clock: Clock,
    ) -> JobHandle:
        now = clock.now()
        artifact_id = inputs.get("artifact_id") or (
            package.manifest.artifact_id if package else None
        )
        version_id = inputs.get("version_id")
        direction_id = str(inputs.get("direction_id", ""))
        block_id = inputs.get("block_id")
        reply_to = str(inputs.get("reply_to", "researcher"))

        beh = self._lookup_behavior(
            task_id=task_id,
            attempt_id=attempt_id,
            artifact_id=artifact_id,
            kind=kind,
            block_id=block_id,
            inputs=inputs,
        )
        if package and not beh:
            beh = dict(package.metadata.get("sim_behavior", {}))

        duration_s = float(
            beh.get("duration_s", inputs.get("duration_s", 20.0))
        )
        timeout_s = float(inputs.get("timeout_s", 180.0))

        return JobHandle(
            job_id=job_id,
            attempt_id=attempt_id,
            task_id=task_id,
            kind=kind,
            resource=resource,
            version_id=version_id,
            artifact_id=artifact_id,
            direction_id=direction_id,
            block_id=block_id,
            reply_to=reply_to,
            started_at=now,
            expected_finish_at=now + duration_s,
            timeout_at=now + timeout_s,
            metadata={"behavior": beh, "inputs": dict(inputs)},
        )

    def wait(self, handle: JobHandle, clock: Clock) -> RawOutcome:
        if handle.attempt_id in self._stopped_attempts:
            released = handle.attempt_id not in self.unconfirmed_cleanup_attempts
            return RawOutcome(
                job_id=handle.job_id,
                attempt_id=handle.attempt_id,
                status="cancelled",
                gpu_released=released,
                duration_s=max(0.0, clock.now() - handle.started_at),
                error="cancelled",
            )

        beh = handle.metadata.get("behavior", {})
        inputs = handle.metadata.get("inputs", {})
        dur = handle.expected_finish_at - handle.started_at

        if beh.get("timeout"):
            return RawOutcome(
                job_id=handle.job_id,
                attempt_id=handle.attempt_id,
                status="timed_out",
                gpu_released=handle.attempt_id not in self.unconfirmed_cleanup_attempts,
                duration_s=float(inputs.get("timeout_s", dur)),
                error="job_timed_out",
            )

        if beh.get("infra_fail"):
            return RawOutcome(
                job_id=handle.job_id,
                attempt_id=handle.attempt_id,
                status="infra_failed",
                gpu_released=handle.attempt_id not in self.unconfirmed_cleanup_attempts,
                duration_s=dur,
                error=str(beh.get("error", "transient_cuda_error")),
            )

        if beh.get("compile_fail"):
            return RawOutcome(
                job_id=handle.job_id,
                attempt_id=handle.attempt_id,
                status="compile_failed",
                gpu_released=True,
                duration_s=dur,
                payload={"compiler_stderr": beh.get("error", "nvcc compilation error")},
                error="compile_failed",
            )

        payload = self._build_payload(handle, beh, inputs)
        status = "completed"
        if beh.get("numerical_fail"):
            status = "numerical_failed"
        return RawOutcome(
            job_id=handle.job_id,
            attempt_id=handle.attempt_id,
            status=status,
            gpu_released=True,
            duration_s=dur,
            payload=payload,
        )

    def _build_payload(
        self, handle: JobHandle, beh: dict[str, Any], inputs: dict[str, Any]
    ) -> dict[str, Any]:
        kind = handle.kind
        if kind == JobKind.BENCHMARK_BLOCK.value:
            block_id = handle.block_id or "W1"
            uuids = list(self.evaluator.blocks.get(block_id, ()))
            if beh.get("drop_last_uuid") and uuids:
                uuids = uuids[:-1]
            if beh.get("duplicate_first_uuid") and len(uuids) >= 2:
                uuids[1] = uuids[0]
            speedup = float(beh.get("speedup", inputs.get("speedup", 1.0)))
            bad_uuids = set(beh.get("incorrect_uuids", []))
            if beh.get("numerical_fail") and uuids:
                bad_uuids.add(uuids[0])
            override_artifact = beh.get("override_case_artifact_id", handle.artifact_id)
            override_version = beh.get("override_case_version_id", handle.version_id)
            cases = []
            for idx, uid in enumerate(uuids):
                base_ms = self.evaluator.frozen_baseline_ms[uid]
                cases.append(
                    {
                        "uuid": uid,
                        "correct": uid not in bad_uuids,
                        "candidate_ms": base_ms / max(speedup, 1e-6),
                        "baseline_ms": base_ms,
                        "artifact_id": override_artifact if idx == 0 else handle.artifact_id,
                        "version_id": override_version if idx == 0 else handle.version_id,
                        "protocol": beh.get("protocol", self.evaluator.protocol),
                        "device_id": beh.get("device_id", self.evaluator.device_id),
                        "attempt_id": handle.attempt_id,
                    }
                )
            return {
                "block_id": block_id,
                "version_id": handle.version_id,
                "artifact_id": handle.artifact_id,
                "protocol": beh.get("protocol", self.evaluator.protocol),
                "device_id": beh.get("device_id", self.evaluator.device_id),
                "attempt_id": handle.attempt_id,
                "cases": cases,
            }

        if kind == JobKind.NCU_CAPTURE.value:
            return {
                "raw_ncu_report": {
                    "version_id": handle.version_id,
                    "artifact_id": handle.artifact_id,
                    "workloads": inputs.get("workload_ids", ["case-07"]),
                    "raw_counters": beh.get(
                        "raw_counters",
                        {"lmem_spill_bytes": 128, "achieved_occupancy": 0.34},
                    ),
                }
            }

        if kind == JobKind.NCU_PARSE.value:
            raw = inputs.get("raw_ncu_report", {})
            return {
                "version_id": handle.version_id,
                "artifact_id": handle.artifact_id,
                "observations": beh.get(
                    "observations",
                    {
                        "register_spill_detected": True,
                        "counters": raw.get("raw_counters", {}),
                        "recommendation": "reduce register count via tile split",
                    },
                ),
            }

        if kind in (JobKind.PROBE.value, JobKind.CPU_PROBE.value):
            return {
                "version_id": handle.version_id,
                "artifact_id": handle.artifact_id,
                "workloads": inputs.get("workload_ids", ["case-07"]),
                "observations": beh.get(
                    "observations",
                    {"probe_metric": 0.42, "hypothesis_supported": True},
                ),
            }

        if kind in (
            JobKind.COMPILER_ANALYSIS.value,
            JobKind.GPU_COMPILER_ANALYSIS.value,
            JobKind.CPU_COMPILE.value,
        ):
            return {
                "version_id": handle.version_id,
                "artifact_id": handle.artifact_id,
                "compiler_artifacts": beh.get(
                    "compiler_artifacts",
                    {"ptx_registers": 144, "lmem_bytes": 64, "smem_bytes": 49152},
                ),
            }

        if kind == JobKind.ABBA_RETEST.value:
            incumbent_speedup = float(inputs.get("incumbent_speedup", 1.0))
            cand_speedup = float(beh.get("speedup", inputs.get("speedup", 1.05)))
            bad_uuids = set(beh.get("incorrect_uuids", []))
            if beh.get("numerical_fail"):
                bad_uuids.add(self.evaluator.uuids[0])
            paired_cases = []
            for uid in self.evaluator.uuids:
                base_ms = self.evaluator.frozen_baseline_ms[uid]
                a_ms = base_ms / max(incumbent_speedup, 1e-6)
                b_ms = base_ms / max(cand_speedup, 1e-6)
                paired_cases.append(
                    {
                        "uuid": uid,
                        "correct": uid not in bad_uuids,
                        "a1_ms": a_ms,
                        "b1_ms": b_ms,
                        "b2_ms": b_ms,
                        "a2_ms": a_ms,
                    }
                )
            return {
                "incumbent_version_id": inputs.get("incumbent_version_id"),
                "incumbent_artifact_id": inputs.get("incumbent_artifact_id"),
                "candidate_version_id": handle.version_id,
                "candidate_artifact_id": handle.artifact_id,
                "protocol": self.evaluator.protocol,
                "device_id": self.evaluator.device_id,
                "paired_cases": paired_cases,
            }

        if kind == JobKind.FINAL_EVAL.value:
            block_id = handle.block_id or "F1"
            speedup = float(beh.get("speedup", inputs.get("speedup", 1.05)))
            cases = []
            for uid in self.evaluator.uuids:
                base_ms = self.evaluator.frozen_baseline_ms[uid]
                cases.append(
                    {
                        "uuid": uid,
                        "correct": not beh.get("numerical_fail", False),
                        "candidate_ms": base_ms / max(speedup, 1e-6),
                        "baseline_ms": base_ms,
                    }
                )
            return {
                "final_block_id": block_id,
                "version_id": handle.version_id,
                "artifact_id": handle.artifact_id,
                "protocol": self.evaluator.protocol,
                "device_id": self.evaluator.device_id,
                "geomean_speedup": speedup,
                "cases": cases,
            }

        return {}

    def stop(self, handle: JobHandle, reason: str, clock: Clock) -> CleanupReceipt:
        already = handle.attempt_id in self._stopped_attempts
        self._stopped_attempts.add(handle.attempt_id)
        handle.status = "cancelled"
        released = handle.attempt_id not in self.unconfirmed_cleanup_attempts
        return CleanupReceipt(
            job_id=handle.job_id,
            attempt_id=handle.attempt_id,
            gpu_released=released,
            reason=reason,
            already_stopped=already,
        )


class LocalSubprocessBackend(ExecutionBackend):
    """Real subprocess execution backend with process-group isolation & idempotent stop."""

    def __init__(self) -> None:
        self._processes: dict[str, subprocess.Popen[bytes]] = {}
        self._stopped: set[str] = set()

    def start(
        self,
        job_id: str,
        attempt_id: str,
        task_id: str,
        kind: str,
        resource: str,
        package: CodePackage | None,
        inputs: dict[str, Any],
        clock: Clock,
    ) -> JobHandle:
        now = clock.now()
        timeout_s = float(inputs.get("timeout_s", 180.0))
        workdir = tempfile.mkdtemp(prefix=f"kernellab-{job_id}-")
        input_path = Path(workdir) / "inputs.json"
        output_path = Path(workdir) / "output.json"
        input_path.write_text(
            json.dumps({**inputs, "attempt_id": attempt_id}, ensure_ascii=False),
            encoding="utf-8",
        )

        cmd = inputs.get("command")
        if package is not None:
            entry = (Path(package.root_dir) / package.manifest.entrypoint).resolve()
            if cmd is None:
                cmd = [
                    sys.executable,
                    str(entry),
                    "--inputs",
                    str(input_path),
                    "--output",
                    str(output_path),
                ]
        elif cmd is None:
            raise ValueError("LocalSubprocessBackend requires package or inputs['command']")

        env = os.environ.copy()
        repo_root = str(Path(__file__).resolve().parent.parent)
        existing_pp = env.get("PYTHONPATH", "")
        if repo_root not in existing_pp.split(os.pathsep):
            env["PYTHONPATH"] = (
                f"{repo_root}{os.pathsep}{existing_pp}" if existing_pp else repo_root
            )
        if "env" in inputs and isinstance(inputs["env"], dict):
            env.update({str(k): str(v) for k, v in inputs["env"].items()})

        proc = subprocess.Popen(
            cmd,
            cwd=workdir,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            env=env,
            start_new_session=True,
        )
        self._processes[attempt_id] = proc
        return JobHandle(
            job_id=job_id,
            attempt_id=attempt_id,
            task_id=task_id,
            kind=kind,
            resource=resource,
            version_id=inputs.get("version_id"),
            artifact_id=inputs.get("artifact_id")
            or (package.manifest.artifact_id if package else None),
            direction_id=str(inputs.get("direction_id", "")),
            block_id=inputs.get("block_id"),
            reply_to=str(inputs.get("reply_to", "researcher")),
            started_at=now,
            expected_finish_at=now,
            timeout_at=now + timeout_s,
            pid=proc.pid,
            pgid=os.getpgid(proc.pid),
            workdir=workdir,
            metadata={"output_path": str(output_path), "timeout_s": timeout_s},
        )

    def wait(self, handle: JobHandle, clock: Clock) -> RawOutcome:
        proc = self._processes.get(handle.attempt_id)
        if proc is None:
            return RawOutcome(
                job_id=handle.job_id,
                attempt_id=handle.attempt_id,
                status="infra_failed",
                gpu_released=False,
                duration_s=0.0,
                error="missing process handle",
            )
        timeout_s = float(handle.metadata.get("timeout_s", 180.0))
        try:
            stdout_b, stderr_b = proc.communicate(timeout=timeout_s)
        except subprocess.TimeoutExpired:
            receipt = self.stop(handle, "timeout", clock)
            return RawOutcome(
                job_id=handle.job_id,
                attempt_id=handle.attempt_id,
                status="timed_out",
                gpu_released=receipt.gpu_released,
                duration_s=max(0.0, clock.now() - handle.started_at),
                error="subprocess timed out",
            )

        dur = max(0.0, clock.now() - handle.started_at)
        if handle.attempt_id in self._stopped:
            return RawOutcome(
                job_id=handle.job_id,
                attempt_id=handle.attempt_id,
                status="cancelled",
                gpu_released=True,
                duration_s=dur,
                error="cancelled",
            )

        out_path = Path(handle.metadata["output_path"])
        payload: dict[str, Any] = {}
        if out_path.exists():
            payload = json.loads(out_path.read_text(encoding="utf-8"))
        status = "completed" if proc.returncode == 0 else "infra_failed"
        self._processes.pop(handle.attempt_id, None)
        return RawOutcome(
            job_id=handle.job_id,
            attempt_id=handle.attempt_id,
            status=status,
            gpu_released=True,
            duration_s=dur,
            payload=payload,
            error=stderr_b.decode("utf-8", errors="replace") if proc.returncode != 0 else None,
        )

    def stop(self, handle: JobHandle, reason: str, clock: Clock) -> CleanupReceipt:
        already = handle.attempt_id in self._stopped
        self._stopped.add(handle.attempt_id)
        proc = self._processes.get(handle.attempt_id)
        if proc is None or proc.poll() is not None:
            return CleanupReceipt(
                job_id=handle.job_id,
                attempt_id=handle.attempt_id,
                gpu_released=True,
                reason=reason,
                already_stopped=already,
            )
        pgid = handle.pgid or proc.pid
        try:
            os.killpg(pgid, signal.SIGTERM)
            proc.wait(timeout=2.0)
        except subprocess.TimeoutExpired:
            try:
                os.killpg(pgid, signal.SIGKILL)
                proc.wait(timeout=2.0)
            except Exception:
                pass
        except ProcessLookupError:
            pass
        released = proc.poll() is not None
        for stream in (proc.stdout, proc.stderr):
            if stream is not None and not stream.closed:
                stream.close()
        return CleanupReceipt(
            job_id=handle.job_id,
            attempt_id=handle.attempt_id,
            gpu_released=released,
            reason=reason,
            already_stopped=already,
        )

    async def async_wait(
        self, handle: JobHandle, clock: Clock, poll_interval_s: float = 0.01
    ) -> RawOutcome:
        """Non-blocking async wait with timeout watchdog & group kill."""
        proc = self._processes.get(handle.attempt_id)
        if proc is None:
            return RawOutcome(
                job_id=handle.job_id,
                attempt_id=handle.attempt_id,
                status="infra_failed",
                gpu_released=False,
                duration_s=0.0,
                error="missing process handle",
            )
        timeout_s = float(handle.metadata.get("timeout_s", 180.0))
        deadline = handle.started_at + timeout_s
        while proc.poll() is None:
            if clock.now() >= deadline:
                receipt = self.stop(handle, "timeout", clock)
                return RawOutcome(
                    job_id=handle.job_id,
                    attempt_id=handle.attempt_id,
                    status="timed_out",
                    gpu_released=receipt.gpu_released,
                    duration_s=max(0.0, clock.now() - handle.started_at),
                    error="subprocess timed out",
                )
            await asyncio.sleep(poll_interval_s)

        stdout_b, stderr_b = proc.communicate()
        dur = max(0.0, clock.now() - handle.started_at)
        if handle.attempt_id in self._stopped:
            return RawOutcome(
                job_id=handle.job_id,
                attempt_id=handle.attempt_id,
                status="cancelled",
                gpu_released=True,
                duration_s=dur,
                error="cancelled",
            )

        out_path = Path(handle.metadata["output_path"])
        payload: dict[str, Any] = {}
        if out_path.exists():
            payload = json.loads(out_path.read_text(encoding="utf-8"))
        status = "completed" if proc.returncode == 0 else "infra_failed"
        self._processes.pop(handle.attempt_id, None)
        return RawOutcome(
            job_id=handle.job_id,
            attempt_id=handle.attempt_id,
            status=status,
            gpu_released=True,
            duration_s=dur,
            payload=payload,
            error=stderr_b.decode("utf-8", errors="replace") if proc.returncode != 0 else None,
        )


def parse_ncu_csv(csv_text: str) -> dict[str, Any]:
    """Parse NVIDIA Nsight Compute (`ncu --csv`) counter output into structured observations."""
    metrics: dict[str, float] = {}
    reader = csv.DictReader(io.StringIO(csv_text.strip()))
    for row in reader:
        name = (row.get("Metric Name") or "").strip()
        val_raw = (row.get("Metric Value") or "").replace(",", "").strip()
        if not name or not val_raw:
            continue
        try:
            metrics[name] = float(val_raw)
        except ValueError:
            continue

    lmem_spill = metrics.get(
        "lmem_spill_bytes",
        metrics.get("smsp__sass_lmem_bytes_per_thread.sum", 0.0),
    )
    occupancy = metrics.get(
        "achieved_occupancy",
        metrics.get("sm__warps_active.avg.pct_of_peak_sustained_active", 0.0) / 100.0,
    )
    regs = metrics.get("launch__registers_per_thread", 0.0)
    return {
        "register_spill_detected": lmem_spill > 0.0,
        "counters": {
            "lmem_spill_bytes": lmem_spill,
            "achieved_occupancy": round(occupancy, 4),
            "registers_per_thread": regs,
            **metrics,
        },
    }


class RealSingleGpuBackend(LocalSubprocessBackend):
    """Single-GPU hardware execution adapter with CUDA_VISIBLE_DEVICES & nvidia-smi cleanup checks."""

    def __init__(
        self,
        gpu_index: int = 0,
        enable_gpu_probe: bool = True,
        nvidia_smi_cmd: tuple[str, ...] = (
            "nvidia-smi",
            "--query-compute-apps=pid",
            "--format=csv,noheader,nounits",
        ),
        eval_cmd_template: list[str] | None = None,
        ncu_cmd_template: list[str] | None = None,
    ) -> None:
        super().__init__()
        self.gpu_index = gpu_index
        self.enable_gpu_probe = enable_gpu_probe
        self.nvidia_smi_cmd = list(nvidia_smi_cmd)
        self.eval_cmd_template = eval_cmd_template
        self.ncu_cmd_template = ncu_cmd_template

    def query_active_gpu_pids(self) -> tuple[bool, set[int]]:
        """Query active compute PIDs on the target GPU via nvidia-smi (safe on non-GPU hosts)."""
        if not self.enable_gpu_probe:
            return True, set()
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = str(self.gpu_index)
        try:
            proc = subprocess.run(
                self.nvidia_smi_cmd,
                env=env,
                capture_output=True,
                text=True,
                timeout=5.0,
                check=False,
            )
            if proc.returncode != 0:
                return False, set()
            pids: set[int] = set()
            for line in proc.stdout.splitlines():
                s = line.strip()
                if s.isdigit():
                    pids.add(int(s))
            return True, pids
        except (FileNotFoundError, subprocess.TimeoutExpired, OSError):
            return False, set()

    def start(
        self,
        job_id: str,
        attempt_id: str,
        task_id: str,
        kind: str,
        resource: str,
        package: CodePackage | None,
        inputs: dict[str, Any],
        clock: Clock,
    ) -> JobHandle:
        enriched = dict(inputs)
        env = dict(enriched.get("env", {}))
        if resource == "gpu":
            env["CUDA_VISIBLE_DEVICES"] = str(self.gpu_index)
        enriched["env"] = env

        if "command" not in enriched:
            tmpl = (
                self.ncu_cmd_template
                if kind == JobKind.NCU_CAPTURE.value
                else self.eval_cmd_template
            )
            if tmpl and package is not None:
                enriched["command"] = [
                    tok.format(
                        package_dir=package.root_dir,
                        entrypoint=package.manifest.entrypoint,
                        block_id=inputs.get("block_id", ""),
                        version_id=inputs.get("version_id", ""),
                        artifact_id=inputs.get("artifact_id", ""),
                        gpu_index=self.gpu_index,
                    )
                    for tok in tmpl
                ]
        return super().start(job_id, attempt_id, task_id, kind, resource, package, enriched, clock)

    def stop(self, handle: JobHandle, reason: str, clock: Clock) -> CleanupReceipt:
        receipt = super().stop(handle, reason, clock)
        if not receipt.gpu_released:
            return receipt
        if handle.resource == "gpu" and self.enable_gpu_probe and handle.pid is not None:
            ok, active_pids = self.query_active_gpu_pids()
            if ok and handle.pid in active_pids:
                return CleanupReceipt(
                    job_id=handle.job_id,
                    attempt_id=handle.attempt_id,
                    gpu_released=False,
                    reason=f"{reason}:gpu_pid_still_active({handle.pid})",
                    already_stopped=receipt.already_stopped,
                )
        return receipt
