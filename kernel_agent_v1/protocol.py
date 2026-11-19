"""Core message headers, commands, manifests, tasks, budgets, and configs for KernelLab V1."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any


class CommandKind(str, Enum):
    IMPLEMENT_KERNEL = "implement_kernel"
    WRITE_NCU_CAPTURE = "write_ncu_capture"
    WRITE_PROBE = "write_probe"
    WRITE_COMPILER_ANALYSIS = "write_compiler_analysis"


class JobKind(str, Enum):
    BENCHMARK_BLOCK = "benchmark_block"
    NCU_CAPTURE = "ncu_capture"
    NCU_PARSE = "ncu_parse"
    PROBE = "probe"
    CPU_PROBE = "cpu_probe"
    CPU_COMPILE = "cpu_compile"
    COMPILER_ANALYSIS = "compiler_analysis"
    GPU_COMPILER_ANALYSIS = "gpu_compiler_analysis"
    ABBA_RETEST = "abba_retest"
    FINAL_EVAL = "final_eval"
    RESEARCH = "research"
    CODING = "coding"
    JUDGE_EVAL = "judge_eval"
    JUDGE_FINAL = "judge_final"


GPU_JOB_KINDS = frozenset(
    {
        JobKind.BENCHMARK_BLOCK.value,
        JobKind.NCU_CAPTURE.value,
        JobKind.PROBE.value,
        JobKind.GPU_COMPILER_ANALYSIS.value,
        JobKind.ABBA_RETEST.value,
        JobKind.FINAL_EVAL.value,
    }
)

REQUIRED_VERSION_GPU_KINDS = frozenset(
    {
        JobKind.BENCHMARK_BLOCK.value,
        JobKind.ABBA_RETEST.value,
        JobKind.FINAL_EVAL.value,
    }
)

OPTIONAL_DIAGNOSTIC_KINDS = frozenset(
    {
        JobKind.NCU_CAPTURE.value,
        JobKind.PROBE.value,
        JobKind.GPU_COMPILER_ANALYSIS.value,
    }
)


class ResultKind(str, Enum):
    BENCHMARK_PARTIAL = "benchmark_partial"
    BENCHMARK_COMPLETE = "benchmark_complete"
    NCU_RAW = "ncu_raw"
    NCU_PARSED = "ncu_parsed"
    PROBE_RESULT = "probe_result"
    COMPILER_ANALYSIS = "compiler_analysis"
    ABBA_RETEST = "abba_retest"
    FINAL_EVAL = "final_eval"


class EventType(str, Enum):
    TASK_PROPOSED = "task_proposed"
    TASK_WAITING = "task_waiting"
    PRIORITY_CHANGED = "priority_changed"
    TASK_WITHDRAWN = "task_withdrawn"
    CODE_READY = "code_ready"
    CODE_BLOCKED = "code_blocked"
    WORKER_STATE_CHANGED = "worker_state_changed"
    GPU_JOB_FINISHED = "gpu_job_finished"
    RESULT_READY = "result_ready"
    BENCHMARK_COMPLETE = "BENCHMARK_COMPLETE"
    JUDGE_RESULT = "judge_result"
    VERSION_CLOSED = "VERSION_CLOSED"


class VersionStatus(str, Enum):
    ACTIVE = "active"
    CLOSING = "closing"
    CLOSED = "closed"


class RouteStatus(str, Enum):
    ACTIVE = "active"
    PAUSED = "paused"
    CLOSED = "closed"


class TaskStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    SKIPPED = "skipped"
    TIMED_OUT = "timed_out"


class WorkerState(str, Enum):
    IDLE = "idle"
    BUSY = "busy"
    BLOCKED = "blocked"


class VerificationLevel(str, Enum):
    FINAL_VERIFIED = "final_verified"
    ABBA_VERIFIED = "abba_verified"
    BLOCK_VERIFIED = "block_verified"
    BASELINE_ONLY = "baseline_only"


PROHIBITED_HEADER_FIELDS = frozenset(
    {
        "case_results",
        "per_case",
        "raw_report",
        "parsed_report",
        "observations",
        "source_code",
        "ptx",
        "sass",
        "body",
        "metrics_payload",
    }
)


def assert_header_only(payload: dict[str, Any]) -> None:
    """Ensure scheduling events sent to Controller never carry result bodies."""
    forbidden = PROHIBITED_HEADER_FIELDS.intersection(payload.keys())
    if forbidden:
        raise ValueError(
            f"Scheduling event header contains prohibited body fields: {sorted(forbidden)}"
        )


@dataclass
class Event:
    """Unified scheduling event header (§5)."""

    event: str
    event_id: str
    run_id: str
    producer: str
    timestamp: str
    sim_time: float = 0.0
    command_id: str | None = None
    direction_id: str | None = None
    version_id: str | None = None
    artifact_id: str | None = None
    parent_version_id: str | None = None
    job_id: str | None = None
    attempt_id: str | None = None
    kind: str | None = None
    block_id: str | None = None
    status: str | None = None
    gpu_released: bool | None = None
    receipt_id: str | None = None
    result_kind: str | None = None
    extra: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        assert_header_only(self.extra)

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        extra = data.pop("extra", {})
        data.update(extra)
        return {k: v for k, v in data.items() if v is not None}

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Event":
        assert_header_only(data)
        known_keys = {
            "event",
            "event_id",
            "run_id",
            "producer",
            "timestamp",
            "sim_time",
            "command_id",
            "direction_id",
            "version_id",
            "artifact_id",
            "parent_version_id",
            "job_id",
            "attempt_id",
            "kind",
            "block_id",
            "status",
            "gpu_released",
            "receipt_id",
            "result_kind",
        }
        base = {k: data[k] for k in known_keys if k in data}
        extra = {k: v for k, v in data.items() if k not in known_keys}
        return cls(**base, extra=extra)


@dataclass(frozen=True)
class TargetRef:
    version_id: str
    artifact_id: str
    digest: str | None = None


@dataclass(frozen=True)
class CommandHead:
    command_id: str
    kind: str
    research_id: str
    direction_id: str
    target: TargetRef
    reply_to: str = "researcher"


@dataclass(frozen=True)
class CommandConstraints:
    workload_ids: tuple[str, ...] = ()
    allowed_paths: tuple[str, ...] = ()
    preserve_target: bool = True
    required_files: tuple[str, ...] = ()


@dataclass(frozen=True)
class CommandBody:
    hypothesis: str
    evidence_refs: tuple[str, ...]
    instruction: str
    deliverable: str
    constraints: CommandConstraints = field(default_factory=CommandConstraints)


@dataclass(frozen=True)
class Command:
    """Leader -> Coder command (§4)."""

    head: CommandHead
    body: CommandBody

    def to_dict(self) -> dict[str, Any]:
        return {
            "head": {
                "command_id": self.head.command_id,
                "kind": self.head.kind,
                "research_id": self.head.research_id,
                "direction_id": self.head.direction_id,
                "target": {
                    "version_id": self.head.target.version_id,
                    "artifact_id": self.head.target.artifact_id,
                    "digest": self.head.target.digest,
                },
                "reply_to": self.head.reply_to,
            },
            "body": {
                "hypothesis": self.body.hypothesis,
                "evidence_refs": list(self.body.evidence_refs),
                "instruction": self.body.instruction,
                "deliverable": self.body.deliverable,
                "constraints": {
                    "workload_ids": list(self.body.constraints.workload_ids),
                    "allowed_paths": list(self.body.constraints.allowed_paths),
                    "preserve_target": self.body.constraints.preserve_target,
                    "required_files": list(self.body.constraints.required_files),
                },
            },
        }


@dataclass(frozen=True)
class ConstraintCheckResult:
    passed: bool
    violations: tuple[str, ...] = ()
    checked_paths: tuple[str, ...] = ()


@dataclass(frozen=True)
class PackageManifest:
    """Immutable code package manifest (§4)."""

    package_id: str
    artifact_id: str
    command_id: str
    task_kind: str
    direction_id: str
    parent_version_id: str
    target_identity: TargetRef
    entrypoint: str
    file_digests: dict[str, str]
    executable_stages: tuple[str, ...]
    workload_ids: tuple[str, ...]
    reply_to: str
    package_digest: str
    constraint_check: ConstraintCheckResult
    created_at: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "package_id": self.package_id,
            "artifact_id": self.artifact_id,
            "command_id": self.command_id,
            "task_kind": self.task_kind,
            "direction_id": self.direction_id,
            "parent_version_id": self.parent_version_id,
            "target_identity": {
                "version_id": self.target_identity.version_id,
                "artifact_id": self.target_identity.artifact_id,
                "digest": self.target_identity.digest,
            },
            "entrypoint": self.entrypoint,
            "file_digests": dict(self.file_digests),
            "executable_stages": list(self.executable_stages),
            "workload_ids": list(self.workload_ids),
            "reply_to": self.reply_to,
            "package_digest": self.package_digest,
            "constraint_check": {
                "passed": self.constraint_check.passed,
                "violations": list(self.constraint_check.violations),
                "checked_paths": list(self.constraint_check.checked_paths),
            },
            "created_at": self.created_at,
        }


@dataclass
class CodePackage:
    """In-memory and on-disk handle to a sealed, immutable package."""

    manifest: PackageManifest
    root_dir: str
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class TaskSpec:
    """Schedulable task node tracked by Controller (§6)."""

    task_id: str
    kind: str
    resource: str  # "gpu" | "cpu" | "model"
    direction_id: str
    version_id: str | None = None
    artifact_id: str | None = None
    parent_version_id: str | None = None
    command_id: str | None = None
    package_id: str | None = None
    block_id: str | None = None
    is_required: bool = False
    priority: float = 1.0
    assigned_member: str | None = None
    reply_to: str = "researcher"
    depends_on: list[str] = field(default_factory=list)
    produces: list[str] = field(default_factory=list)
    estimated_duration_s: float | None = None
    duration_category: str = "normal"  # short | normal | long
    timeout_s: float | None = None
    template_id: str | None = None
    workload_count: int = 1
    post_cpu_kind: str | None = None
    post_cpu_duration_s: float | None = None
    status: str = TaskStatus.PENDING.value
    ready_at: float | None = None
    started_at: float | None = None
    finished_at: float | None = None
    skip_count: int = 0
    retry_count: int = 0
    max_retries: int = 1
    unblocks_work: str | None = None
    replaces_task_id: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class VersionRecord:
    """Version lifecycle state owned exclusively by Controller (§2, §7)."""

    version_id: str
    artifact_id: str
    source_parent_version: str
    direction_id: str
    priority: float
    reply_to: str
    admitted_at: float
    deadline_at: float
    diag_budget_s: float
    diag_spent_s: float = 0.0
    required_blocks: list[str] = field(default_factory=lambda: ["W1", "W2", "W3", "W4"])
    completed_blocks: list[str] = field(default_factory=list)
    failed_blocks: list[str] = field(default_factory=list)
    block_receipts: dict[str, str] = field(default_factory=dict)
    conditional_retest_registered: bool = True
    retest_required: bool | None = None  # None while awaiting Judge decision
    retest_status: str = "pending"  # pending | not_needed | scheduled | completed | failed
    judge_decision_deadline_at: float | None = None
    benchmark_complete_sent: bool = False
    benchmark_complete_receipt_id: str | None = None
    status: str = VersionStatus.ACTIVE.value
    closed_at: float | None = None
    close_reason: str | None = None
    close_receipt_ids: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class EvidenceDelivery:
    """Direct evidence delivery from Operator -> Researcher/Judge or Judge -> Leader."""

    delivery_id: str
    receipt_id: str
    recipient: str
    result_kind: str
    producer: str
    timestamp: str
    version_id: str | None
    artifact_id: str | None
    direction_id: str | None
    command_id: str | None
    block_id: str | None
    evidence_path: str
    payload: dict[str, Any]


@dataclass
class BudgetConfig:
    """Frozen run and per-version budgets (§7)."""

    total_wall_budget_s: float = 3600.0
    total_gpu_budget_s: float = 1800.0
    total_model_budget_cost: float = 100.0
    version_diag_budget_s: float = 120.0
    version_deadline_s: float = 600.0
    judge_decision_timeout_s: float = 30.0
    reserve_required_block_s: float = 60.0
    reserve_retest_s: float = 90.0
    reserve_final_eval_s: float = 180.0
    first_trial_diag_timeout_s: float = 45.0
    default_job_timeout_s: float = 180.0
    max_compile_correctness_repairs: int = 2
    max_infra_retries: int = 1
    final_eval_job_timeout_s: float | None = None
    abba_retest_job_timeout_s: float | None = None


@dataclass
class BudgetLedger:
    """Tracked consumption against BudgetConfig."""

    gpu_seconds_used: float = 0.0
    model_cost_used: float = 0.0
    wall_start_s: float = 0.0
    admission_frozen: bool = False
    freeze_reason: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class RuntimeConfig:
    """Top-level KernelLab V1 configuration."""

    run_id: str = "run-001"
    gpu_name: str = "NVIDIA-H100-SXM"
    protocol_name: str = "gdn-unified-v2"
    toolchain: str = "cuda-12.6/triton-3.1"
    scheduler_mode: str = "chunked_dynamic"  # whole_round | chunked_fcfs | chunked_dynamic
    horizon_h_s: float = 120.0
    min_gain_threshold: float = 0.01
    stagnation_limit: int = 3
    max_pending_per_route: int = 1
    num_benchmark_blocks: int = 4
    cases_per_block: int = 25
    final_eval_blocks: int = 3
    require_abba_on_first_candidate: bool = False
    snapshot_interval: int = 10
    budgets: BudgetConfig = field(default_factory=BudgetConfig)
