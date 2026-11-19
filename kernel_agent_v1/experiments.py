"""Three-way baseline comparison harness (§8.5, §9): whole_round vs chunked_fcfs vs chunked_dynamic."""

from __future__ import annotations

from pathlib import Path
import tempfile
from typing import Any

from kernel_agent_v1.clock import SimClock
from kernel_agent_v1.protocol import BudgetConfig, RuntimeConfig
from kernel_agent_v1.runtime import KernelLabRuntime, RunMetrics


def default_comparison_campaign() -> list[dict[str, Any]]:
    """Multi-route campaign with identical candidates across all 3 scheduler modes."""
    return [
        {
            "direction_id": "register-pressure",
            "hypothesis": "Short-sequence slowdown comes from register spill",
            "priority": 2.0,
            "researcher_id": "researcher_reg",
            "artifact_id": "art-reg-1",
            "research_duration_s": 25.0,
            "code_duration_s": 25.0,
            "model_cost": 2.0,
            "block_duration_s": 50.0,
            "speedup": 1.12,
            "diagnostic": {
                "command_kind": "write_probe",
                "job_kind": "probe",
                "gpu_duration_s": 15.0,
                "post_cpu_kind": "cpu_probe",
                "post_cpu_duration_s": 5.0,
                "leader_waiting_s": 50.0,
            },
        },
        {
            "direction_id": "smem-layout",
            "hypothesis": "Swizzled 128B layout eliminates bank conflicts",
            "priority": 1.5,
            "researcher_id": "researcher_mem",
            "artifact_id": "art-mem-1",
            "research_duration_s": 25.0,
            "code_duration_s": 25.0,
            "model_cost": 2.0,
            "block_duration_s": 50.0,
            "speedup": 1.22,
            "diagnostic": None,
        },
        {
            "direction_id": "register-pressure",
            "hypothesis": "Tune warp tile split after partial W1 + probe evidence",
            "priority": 2.0,
            "researcher_id": "researcher_reg",
            "artifact_id": "art-reg-2",
            "research_duration_s": 30.0,
            "code_duration_s": 25.0,
            "model_cost": 2.0,
            "block_duration_s": 50.0,
            "speedup": 1.28,
            "diagnostic": None,
        },
        {
            "direction_id": "smem-layout",
            "hypothesis": "Vectorize cp.async tile loads",
            "priority": 1.5,
            "researcher_id": "researcher_mem",
            "artifact_id": "art-mem-2",
            "research_duration_s": 30.0,
            "code_duration_s": 25.0,
            "model_cost": 2.0,
            "block_duration_s": 50.0,
            "speedup": 1.24,
            "diagnostic": None,
        },
    ]


def run_three_way_comparison(
    base_dir: str | Path | None = None,
    total_wall_budget_s: float = 20_000.0,
    total_gpu_budget_s: float = 16_000.0,
) -> dict[str, RunMetrics]:
    """Run whole_round, chunked_fcfs, and chunked_dynamic under identical protocol & budget."""
    results: dict[str, RunMetrics] = {}
    root = Path(base_dir) if base_dir else Path(tempfile.mkdtemp(prefix="kl-compare-"))

    for mode in ("whole_round", "chunked_fcfs", "chunked_dynamic"):
        cfg = RuntimeConfig(
            run_id=f"compare-{mode}",
            scheduler_mode=mode,
            horizon_h_s=120.0,
            budgets=BudgetConfig(
                total_wall_budget_s=total_wall_budget_s,
                total_gpu_budget_s=total_gpu_budget_s,
                total_model_budget_cost=50.0,
                version_diag_budget_s=1000.0,
            ),
        )
        rt = KernelLabRuntime(
            run_dir=root / mode,
            config=cfg,
            clock=SimClock(),
        )
        metrics = rt.run_simulated_campaign(default_comparison_campaign())
        results[mode] = metrics

    return results


def format_comparison_report(results: dict[str, RunMetrics]) -> str:
    lines = [
        "KernelLab V1 Baseline Comparison Report (protocol: gdn-unified-v2)",
        "=" * 74,
    ]
    header = (
        f"{'Mode':<16} {'1stEv(s)':>9} {'Overlap(s)':>11} {'GPUBusy(s)':>11} "
        f"{'GPUIdle(s)':>11} {'Wall(s)':>8} {'BestScore':>10} {'Level':<15}"
    )
    lines.append(header)
    lines.append("-" * 74)
    for mode in ("whole_round", "chunked_fcfs", "chunked_dynamic"):
        m = results[mode]
        fe = f"{m.first_actionable_evidence_s:.1f}" if m.first_actionable_evidence_s is not None else "N/A"
        lines.append(
            f"{mode:<16} {fe:>9} {m.member_gpu_overlap_s:>11.1f} {m.gpu_busy_s:>11.1f} "
            f"{m.gpu_idle_s:>11.1f} {m.total_wall_s:>8.1f} {m.best_verified_score:>9.3f}x {m.verification_level:<15}"
        )
    return "\n".join(lines)
