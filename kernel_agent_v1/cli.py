"""CLI entrypoint for KernelLab V1."""

from __future__ import annotations

import argparse
import asyncio
from dataclasses import asdict
import json
import math
import shutil
from pathlib import Path
import sys
import tempfile
import traceback
from typing import Any
from urllib.error import HTTPError

from kernel_agent_v1.async_runtime import AsyncKernelLabRunner, AsyncRunnerHooks
from kernel_agent_v1.backends import RealSingleGpuBackend, SimulatedExecutionBackend
from kernel_agent_v1.clock import RealClock, SimClock
from kernel_agent_v1.closed_loop_tester import ClosedLoopTester
from kernel_agent_v1.dual_research import DualResearch
from kernel_agent_v1.evaluator import FINAL_BLOCK_IDS, TrustedEvaluator
from kernel_agent_v1.experiments import format_comparison_report, run_three_way_comparison
from kernel_agent_v1.gdn_campaign import GDN_CANDIDATE_FAILURES, GDN_ROOT_ROUTE, GdnRouteCampaign
from kernel_agent_v1.gdn_task import load_official_definition, seed_gdn_candidate_source
from kernel_agent_v1.llm import (
    DeepSeekClient,
    DeepSeekCoderAdapter,
    DeepSeekConfig,
    DeepSeekResearcherAdapter,
    LLMClient,
    LLMResponseFormatError,
)
from kernel_agent_v1.protocol import (
    BudgetConfig,
    CommandKind,
    EvidenceDelivery,
    ResultKind,
    RuntimeConfig,
    TargetRef,
)
from kernel_agent_v1.runtime import KernelLabRuntime


def _positive_seconds(value: str) -> float:
    seconds = float(value)
    if not math.isfinite(seconds) or seconds <= 0:
        raise argparse.ArgumentTypeError("expected a positive, finite number of seconds")
    return seconds


class _AuditedLLMClient(LLMClient):
    def __init__(self, client: LLMClient, runtime: KernelLabRuntime) -> None:
        self.client = client
        self.runtime = runtime
        self.sequence = 0

    async def complete(self, messages, *, json_mode=True):
        store, clock = self.runtime.store, self.runtime.clock
        for attempt in range(1, 4):
            self.sequence += 1
            call_id = f"llm-{self.sequence:05d}"
            # HTTP credentials never pass through this interface.
            store.write_evidence(f"{call_id}-request", {
                "started_at": clock.now_iso(), "json_mode": json_mode, "attempt": attempt,
                "messages": [asdict(message) for message in messages],
            })
            try:
                response = await self.client.complete(messages, json_mode=json_mode)
            except (Exception, asyncio.CancelledError) as exc:
                store.write_evidence(f"{call_id}-response", {
                    "finished_at": clock.now_iso(), "error": str(exc), "type": type(exc).__name__,
                })
                transient = (
                    exc.code == 429 or 500 <= exc.code < 600
                    if isinstance(exc, HTTPError) else isinstance(exc, OSError)
                )
                if not transient or attempt == 3:
                    raise
                await asyncio.sleep(2 ** attempt)
            else:
                store.write_evidence(f"{call_id}-response", {
                    "finished_at": clock.now_iso(), **asdict(response),
                })
                return response


def _run_end_to_end(args: argparse.Namespace) -> int:
    run_dir = Path(args.run_dir) if args.run_dir else Path(tempfile.mkdtemp(prefix="kl-run-"))
    if run_dir.exists():
        existing_data = any(
            path.name != "console.log" and (not path.is_dir() or any(path.iterdir()))
            for path in run_dir.iterdir()
        )
        if existing_data:
            print(f"Run directory already contains experiment data: {run_dir}", file=sys.stderr)
            return 2
    budgets = BudgetConfig(total_wall_budget_s=18_000.0, total_gpu_budget_s=12_000.0)
    final_timeout_s = args.final_eval_timeout_s
    if final_timeout_s is None and args.backend == "real-gpu":
        final_timeout_s = 36_000.0
    if final_timeout_s is not None:
        budgets.final_eval_job_timeout_s = final_timeout_s
        budgets.reserve_final_eval_s = len(FINAL_BLOCK_IDS) * final_timeout_s
        budgets.total_wall_budget_s += budgets.reserve_final_eval_s
        budgets.total_gpu_budget_s += budgets.reserve_final_eval_s
    abba_timeout_s = args.abba_retest_timeout_s
    if abba_timeout_s is None and args.backend == "real-gpu":
        abba_timeout_s = 18_000.0
    if abba_timeout_s is not None:
        budgets.abba_retest_job_timeout_s = abba_timeout_s
        budgets.reserve_retest_s = abba_timeout_s
        budgets.total_wall_budget_s += budgets.reserve_retest_s
        budgets.total_gpu_budget_s += budgets.reserve_retest_s
    if args.wall_budget_s is not None:
        budgets.total_wall_budget_s = args.wall_budget_s
    if args.gpu_budget_s is not None:
        budgets.total_gpu_budget_s = args.gpu_budget_s

    route_campaign = args.max_versions is not None and args.deepseek_live
    cfg = RuntimeConfig(
        run_id=args.run_id,
        gpu_name=args.gpu,
        scheduler_mode=args.scheduler,
        max_versions=args.max_versions,
        finalize_when_admission_frozen=route_campaign,
        budgets=budgets,
    )
    evaluator = TrustedEvaluator.from_official_gdn_dataset(
        device_id=f"{cfg.gpu_name}:0",
        min_gain_threshold=cfg.min_gain_threshold,
        use_measured_baselines=args.backend == "real-gpu",
    )
    if args.backend == "real-gpu":
        clock = RealClock()
        backend = RealSingleGpuBackend(
            gpu_index=args.gpu_index,
            enable_gpu_probe=True,
            eval_cmd_template=[
                sys.executable,
                "-m",
                "kernel_agent_v1.gdn_gpu_runner",
                "--inputs",
                "inputs.json",
                "--output",
                "output.json",
            ],
            ncu_cmd_template=[
                sys.executable,
                "-m",
                "kernel_agent_v1.gdn_gpu_runner",
                "--inputs",
                "inputs.json",
                "--output",
                "output.json",
            ],
        )
    else:
        clock = SimClock()
        backend = SimulatedExecutionBackend(evaluator)

    rt = KernelLabRuntime(run_dir=run_dir, config=cfg, clock=clock, backend=backend)
    rt.store.write_json("run_config", {
        "args": vars(args),
        "runtime_config": asdict(cfg),
        "started_at": clock.now_iso(),
    })
    rt.evaluator = evaluator
    rt.operator.evaluator = evaluator
    rt.judge.evaluator = evaluator
    rt.coder.enforce_gdn_abi = True

    rt.register_route(
        direction_id=GDN_ROOT_ROUTE,
        hypothesis="Optimize GDN prefill k-last kernel on target GPU using official gdn-unified-v2 workloads",
        priority=2.0,
        researcher_id="res_main",
    )

    # Seed Round 0 candidate matching exact GDN ABI
    lookup = rt.wiki.query(cfg.gpu_name, GDN_ROOT_ROUTE, ("gdn", "l4", "sm89"), clock.now_iso())
    seed_cmd = rt.leader.build_command(
        kind=CommandKind.IMPLEMENT_KERNEL.value,
        research_id=rt.leader.next_research_id(),
        direction_id=GDN_ROOT_ROUTE,
        target=TargetRef("V0", "fla-baseline"),
        hypothesis="Seed verified GDN prefill baseline candidate",
        evidence_refs=(lookup.lookup_id,),
        instruction="Package seed GDN prefill kernel.py matching fixed ABI",
        deliverable="kernel.py",
        reply_to="res_main",
        allowed_paths=("kernel.py",),
        preserve_target=False,
    )
    rt.store.write_evidence(f"command-{seed_cmd.head.command_id}", seed_cmd.to_dict())
    seed_evt, seed_pkg = rt.submit_command_and_package(
        seed_cmd,
        {
            "artifact_id": "art-seed-v0",
            "entrypoint": "kernel.py",
            "files": {"kernel.py": seed_gdn_candidate_source()},
        },
    )
    if seed_pkg is None:
        print(f"Failed to package seed candidate: {seed_evt.to_dict()}", file=sys.stderr)
        return 2
    known_sources = {tuple(sorted(seed_pkg.manifest.file_digests.items()))}

    # Inject candidate_path (and incumbent_path for ABBA) into RealSingleGpuBackend inputs
    if args.backend == "real-gpu":
        orig_dispatch = rt.dispatch_gpu_if_ready

        def _dispatch_with_paths(extra_inputs_fn=None):
            def _extra(task):
                extra = extra_inputs_fn(task) if extra_inputs_fn else {}
                pkg = rt.packages_by_id.get(task.package_id) if task.package_id else None
                if pkg is not None:
                    candidate_path = Path(pkg.root_dir) / pkg.manifest.entrypoint
                    extra["candidate_path"] = str(candidate_path.resolve())
                inc_pkg = rt.packages_by_id.get(task.metadata.get("incumbent_package_id"))
                if inc_pkg is not None:
                    incumbent_path = Path(inc_pkg.root_dir) / inc_pkg.manifest.entrypoint
                    extra["incumbent_path"] = str(incumbent_path.resolve())
                extra["device_id"] = evaluator.device_id
                extra["timing_baseline"] = args.timing_baseline
                if args.smoke_cases > 0:
                    extra["max_cases"] = args.smoke_cases
                return extra

            return orig_dispatch(extra_inputs_fn=_extra)

        rt.dispatch_gpu_if_ready = _dispatch_with_paths  # type: ignore[method-assign]

    rounds_remaining = [int(args.max_rounds)]
    llm_client = DeepSeekClient(
        DeepSeekConfig(
            model=args.deepseek_model,
            live_enabled=bool(args.deepseek_live),
        )
    )
    if args.audit_llm or args.max_versions is not None:
        llm_client = _AuditedLLMClient(llm_client, rt)
    task_definition = load_official_definition()
    res_adapter = DeepSeekResearcherAdapter(llm_client, rt.leader, task_definition=task_definition)
    coder_adapter = DeepSeekCoderAdapter(llm_client, rt.coder, task_definition=task_definition)

    dual_research = (
        DualResearch(
            rt, llm_client, lambda event: runner.submit_external_event(event),
            ncu_available=args.backend == "sim" or shutil.which("ncu") is not None,
        )
        if args.dual_researcher else None
    )

    async def _generate_candidate(
        direction_id: str,
        deliveries: list[EvidenceDelivery],
        target: TargetRef,
        repair_context: dict[str, Any] | None,
        parent_files: dict[str, str] | None,
    ) -> bool:
        feedback = None
        delivery = deliveries[0] if deliveries else None
        trigger = delivery.receipt_id if delivery else target.version_id
        for _ in range(3 if route_campaign else 1):
            if rt.controller.version_cap_reached or rt.controller.ledger.admission_frozen:
                return False
            try:
                design_role = "Design" if dual_research is not None else "Leader"
                with rt.timeline.measure(design_role, f"Plan from {trigger}"):
                    cmd, plan, usage = await res_adapter.plan_from_evidence(
                        researcher_id="res_main",
                        direction_id=direction_id,
                        deliveries=deliveries,
                        repair_context=repair_context,
                        parent_files=parent_files,
                        dual_research=dual_research,
                        generation_feedback=feedback,
                        default_target=target,
                    )
            except LLMResponseFormatError as exc:
                rt.controller.ledger.model_cost_used += exc.usage.cost_usd
                print(
                    f"[model-error] stage=researcher version={target.version_id} "
                    f"receipt={trigger} error={exc}",
                    file=sys.stderr,
                )
                feedback = str(exc)
                continue
            rt.controller.ledger.model_cost_used += usage.cost_usd
            rt.store.write_evidence(f"command-{cmd.head.command_id}", cmd.to_dict())
            if "diagnosis_feedback" in plan:
                rt.store.write_evidence(f"design-{cmd.head.command_id}", {
                    "command_id": cmd.head.command_id,
                    "trigger_receipt_id": trigger,
                    "version_id": target.version_id,
                    "issued_at_s": rt.clock.now(),
                    **plan["diagnosis_feedback"],
                })
            try:
                with rt.timeline.measure("Coder", f"Implement {cmd.head.command_id}") as span:
                    evt, pkg, usage = await coder_adapter.implement(
                        cmd, repair_context=repair_context, parent_files=parent_files,
                        content_addressed=route_campaign,
                    )
                    span.status = evt.event
            except LLMResponseFormatError as exc:
                usage = exc.usage
                evt, pkg = rt.coder.execute_command(cmd, {"blocked_reason": str(exc)})
            rt.controller.ledger.model_cost_used += usage.cost_usd
            if pkg is not None:
                source_key = tuple(sorted(pkg.manifest.file_digests.items()))
                if route_campaign and source_key in known_sources:
                    feedback = "The proposed source is byte-identical to an earlier candidate. Produce a distinct change."
                    rt.store.write_evidence(f"duplicate-{cmd.head.command_id}", {
                        "direction_id": direction_id, "package_id": pkg.manifest.package_id,
                        "file_digests": pkg.manifest.file_digests,
                    })
                    continue
                known_sources.add(source_key)
                rt.packages_by_id[pkg.manifest.package_id] = pkg
                if route_campaign:
                    rt.wiki.route_tree.record_experiment_submission(
                        cmd.head.command_id, direction_id, target.version_id, pkg.manifest.artifact_id,
                        cmd.body.instruction, cmd.body.constraints.workload_ids,
                    )
            else:
                print(
                    f"[coder-blocked] cmd={cmd.head.command_id} details={evt.extra}",
                    file=sys.stderr,
                )
            await runner.submit_external_event(evt)
            if pkg is not None:
                return True
            feedback = json.dumps(evt.extra, ensure_ascii=False)
        return False

    async def _on_evidence(runtime: KernelLabRuntime, delivery: EvidenceDelivery) -> None:
        if not args.deepseek_live or rounds_remaining[0] <= 0:
            return
        if delivery.result_kind not in (
            ResultKind.BENCHMARK_PARTIAL.value, ResultKind.NCU_PARSED.value, ResultKind.PROBE_RESULT.value,
        ):
            return
        status = delivery.payload.get("status")
        repair_context = None
        if delivery.result_kind == ResultKind.BENCHMARK_PARTIAL.value and status in GDN_CANDIDATE_FAILURES:
            failed_package = runtime.packages_by_id.get(delivery.payload.get("package_id"))
            if failed_package is None:
                print(f"[repair] missing source for {delivery.receipt_id}", file=sys.stderr)
                return
            repair_context = {
                "diagnostic": delivery.payload,
                "gpu": runtime.config.gpu_name,
                "files": {
                    name: (Path(failed_package.root_dir) / name).read_text(encoding="utf-8")
                    for name in failed_package.manifest.file_digests
                },
            }
        elif status in ("infra_failed", "timed_out", "cancelled") or (
            delivery.block_id and delivery.block_id != "W1"
        ):
            return
        rounds_remaining[0] -= 1
        target = TargetRef(delivery.version_id or "V1", delivery.artifact_id or "art-seed-v0")
        await _generate_candidate(GDN_ROOT_ROUTE, [delivery], target, repair_context, None)

    campaign = GdnRouteCampaign(rt, _generate_candidate, res_adapter) if route_campaign else None
    runner = AsyncKernelLabRunner(
        rt, hooks=AsyncRunnerHooks(
            on_research_evidence=None if campaign else _on_evidence,
            on_idle_check=campaign.advance if campaign else None,
            on_diagnosis_evidence=dual_research.diagnose if dual_research else None,
        )
    )
    timeline_path = (run_dir / "pipeline_timeline.html").resolve()
    try:
        metrics = asyncio.run(runner.run())
    except (Exception, KeyboardInterrupt) as exc:
        rt.controller.flush_snapshot()
        rt.store.write_json("run_result", {
            "status": "interrupted" if isinstance(exc, KeyboardInterrupt) else "failed",
            "target_version": args.max_versions,
            "last_admitted_version": rt.controller.version_seq,
            "error": traceback.format_exc(),
        })
        raise
    finally:
        rt.timeline.write_html(timeline_path, cfg.run_id)
    # Surface any job execution errors clearly in stderr for immediate diagnostics
    for _, evt in rt.store.load_events():
        err = evt.extra.get("error")
        if err:
            print(
                f"[job-error] event={evt.event} ver={evt.version_id} block={evt.block_id} "
                f"status={evt.status} error={str(err)[:300]}",
                file=sys.stderr,
            )
    result = metrics.to_dict()
    result.update({
        "status": "completed" if rt.controller.final_eval_completed else "incomplete",
        "target_version": args.max_versions,
        "version_cap_reached": rt.controller.version_cap_reached,
        "last_admitted_version": rt.controller.version_seq,
        "last_admitted_version_closed": f"V{rt.controller.version_seq}" in rt.controller.closed_versions,
        "final_eval_score": rt.judge.incumbent.final_eval_score,
        "final_eval_version_id": rt.controller.final_eval_version_id,
        "final_eval_completed": rt.controller.final_eval_completed,
    })
    result["role_active_s"] = rt.timeline.active_seconds()
    result["pipeline_timeline"] = str(timeline_path)
    if dual_research is not None:
        result["diagnosis"] = {
            "conclusions": len(rt.wiki.diagnoses),
            "gpu_requests": len(dual_research.requests),
            "gpu_seconds": sum(
                float(evt.extra.get("duration_s", 0))
                for _, evt in rt.store.load_events()
                if evt.extra.get("task_id") in dual_research.requests
                and evt.event == "gpu_job_finished"
            ),
        }
    rt.store.write_json("run_result", result)
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return 0 if args.max_versions is None or rt.controller.final_eval_completed else 2


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="kernel-agent-v1",
        description="KernelLab V1 single-GPU asynchronous optimization runtime",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    cmp_p = sub.add_parser(
        "compare",
        help="Run 3-way comparison: whole_round vs chunked_fcfs vs chunked_dynamic",
    )
    cmp_p.add_argument("--out-dir", type=str, default=None)
    cmp_p.add_argument("--json", action="store_true", dest="as_json")

    cl_p = sub.add_parser(
        "closed-loop",
        help="Run reactive end-to-end closed-loop verification without a physical GPU",
    )
    cl_p.add_argument("--backend", choices=("sim", "subprocess"), default="sim")
    cl_p.add_argument("--out-dir", type=str, default=None)
    cl_p.add_argument("--quiet", action="store_true")

    run_p = sub.add_parser(
        "run",
        help="Run end-to-end GDN optimization loop (real-gpu for Colab L4, or sim dry-run)",
    )
    run_p.add_argument("--backend", choices=("real-gpu", "sim"), default="sim")
    run_p.add_argument("--gpu", type=str, default="NVIDIA-L4")
    run_p.add_argument("--gpu-index", type=int, default=0)
    run_p.add_argument("--run-id", type=str, default="gdn-l4-run-001")
    run_p.add_argument("--run-dir", type=str, default=None)
    run_p.add_argument(
        "--scheduler",
        choices=("chunked_dynamic", "chunked_fcfs", "whole_round"),
        default="chunked_dynamic",
    )
    run_p.add_argument(
        "--wall-budget-s", type=float,
        help="Total wall-clock budget in seconds. Default: 18000 plus ABBA and final-eval reserves on real GPU.",
    )
    run_p.add_argument(
        "--gpu-budget-s", type=float,
        help="Total GPU scheduling budget in seconds. Default: 12000 plus ABBA and final-eval reserves on real GPU.",
    )
    run_p.add_argument(
        "--final-eval-timeout-s", type=_positive_seconds,
        help="Timeout per final-eval block in seconds. Default: 36000 on real GPU.",
    )
    run_p.add_argument(
        "--abba-retest-timeout-s", type=_positive_seconds,
        help="Timeout per full ABBA retest in seconds. Default: 18000 on real GPU.",
    )
    limits = run_p.add_mutually_exclusive_group()
    limits.add_argument("--max-rounds", type=int, default=3, help="Maximum model proposal rounds.")
    limits.add_argument(
        "--max-versions", type=int,
        help="Global version cap including seed V1. Route exhaustion can end the search earlier.",
    )
    run_p.add_argument("--audit-llm", action="store_true", help="Persist model requests and responses.")
    run_p.add_argument(
        "--timing-baseline",
        choices=("fla", "reference"),
        default="fla",
        help="GPU timing baseline: pinned FLA v0.5.2 (default) or pure-Python reference fallback.",
    )
    run_p.add_argument("--smoke-cases", type=int, default=0)
    run_p.add_argument("--deepseek-live", action="store_true")
    run_p.add_argument(
        "--dual-researcher", action="store_true",
        help="Run a parallel Diagnosis Researcher alongside Design (requires --deepseek-live).",
    )
    run_p.add_argument("--deepseek-model", type=str, default="deepseek-chat")

    args = parser.parse_args(argv)
    if args.command == "compare":
        out_dir = Path(args.out_dir) if args.out_dir else Path(tempfile.mkdtemp(prefix="kl-cli-"))
        results = run_three_way_comparison(base_dir=out_dir)
        if args.as_json:
            print(json.dumps({k: v.to_dict() for k, v in results.items()}, indent=2))
        else:
            print(format_comparison_report(results))
        return 0
    if args.command == "closed-loop":
        out_dir = Path(args.out_dir) if args.out_dir else Path(tempfile.mkdtemp(prefix="kl-cl-"))
        report = ClosedLoopTester(backend_kind=args.backend).run(
            work_dir=out_dir, verbose=not args.quiet
        )
        print(report.format_text(verbose=not args.quiet))
        return 0 if report.passed else 2
    if args.command == "run":
        if args.dual_researcher and not args.deepseek_live:
            run_p.error("--dual-researcher requires --deepseek-live")
        if args.max_versions is not None and args.max_versions < 1:
            run_p.error("--max-versions must be at least 1")
        if args.max_versions is not None and args.max_versions > 1 and not args.deepseek_live:
            run_p.error("--max-versions above 1 requires --deepseek-live")
        return _run_end_to_end(args)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
