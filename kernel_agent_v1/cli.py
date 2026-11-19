"""CLI entrypoint for KernelLab V1."""

from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path
import sys
import tempfile

from kernel_agent_v1.async_runtime import AsyncKernelLabRunner, AsyncRunnerHooks
from kernel_agent_v1.backends import RealSingleGpuBackend, SimulatedExecutionBackend
from kernel_agent_v1.clock import RealClock, SimClock
from kernel_agent_v1.closed_loop_tester import ClosedLoopTester
from kernel_agent_v1.evaluator import TrustedEvaluator
from kernel_agent_v1.experiments import format_comparison_report, run_three_way_comparison
from kernel_agent_v1.gdn_task import seed_gdn_candidate_source
from kernel_agent_v1.llm import (
    DeepSeekClient,
    DeepSeekCoderAdapter,
    DeepSeekConfig,
    DeepSeekResearcherAdapter,
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


def _run_end_to_end(args: argparse.Namespace) -> int:
    run_dir = Path(args.run_dir) if args.run_dir else Path(tempfile.mkdtemp(prefix="kl-run-"))
    cfg = RuntimeConfig(
        run_id=args.run_id,
        gpu_name=args.gpu,
        scheduler_mode=args.scheduler,
        budgets=BudgetConfig(
            total_wall_budget_s=float(args.wall_budget_s),
            total_gpu_budget_s=float(args.gpu_budget_s),
        ),
    )
    evaluator = TrustedEvaluator.from_official_gdn_dataset(
        device_id=f"{cfg.gpu_name}:0",
        min_gain_threshold=cfg.min_gain_threshold,
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
    rt.evaluator = evaluator
    rt.operator.evaluator = evaluator
    rt.judge.evaluator = evaluator
    rt.coder.enforce_gdn_abi = True

    rt.register_route(
        direction_id="gdn-prefill-main",
        hypothesis="Optimize GDN prefill k-last kernel on target GPU using official gdn-unified-v2 workloads",
        priority=2.0,
        researcher_id="res_main",
    )

    # Seed Round 0 candidate matching exact GDN ABI
    lookup = rt.wiki.query(cfg.gpu_name, "gdn-prefill-main", ("gdn", "l4", "sm89"), clock.now_iso())
    seed_cmd = rt.leader.build_command(
        kind=CommandKind.IMPLEMENT_KERNEL.value,
        research_id=rt.leader.next_research_id(),
        direction_id="gdn-prefill-main",
        target=TargetRef("V0", "fla-baseline"),
        hypothesis="Seed verified GDN prefill baseline candidate",
        evidence_refs=(lookup.lookup_id,),
        instruction="Package seed GDN prefill kernel.py matching fixed ABI",
        deliverable="kernel.py",
        reply_to="res_main",
        allowed_paths=("kernel.py",),
        preserve_target=False,
    )
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

    # Inject candidate_path into RealSingleGpuBackend inputs
    if args.backend == "real-gpu":
        orig_dispatch = rt.dispatch_gpu_if_ready

        def _dispatch_with_paths(extra_inputs_fn=None):
            def _extra(task):
                extra = extra_inputs_fn(task) if extra_inputs_fn else {}
                pkg = rt.packages_by_id.get(task.package_id) if task.package_id else None
                if pkg is not None:
                    extra["candidate_path"] = str(Path(pkg.root_dir) / pkg.manifest.entrypoint)
                extra["device_id"] = evaluator.device_id
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
    res_adapter = DeepSeekResearcherAdapter(llm_client, rt.leader)
    coder_adapter = DeepSeekCoderAdapter(llm_client, rt.coder)

    async def _on_evidence(runtime: KernelLabRuntime, delivery: EvidenceDelivery) -> None:
        if not args.deepseek_live or rounds_remaining[0] <= 0:
            return
        if delivery.result_kind not in (
            ResultKind.BENCHMARK_PARTIAL.value,
            ResultKind.NCU_PARSED.value,
            ResultKind.PROBE_RESULT.value,
        ):
            return
        if delivery.block_id and delivery.block_id != "W1":
            return
        rounds_remaining[0] -= 1
        cmd, _, u1 = await res_adapter.plan_from_evidence(
            researcher_id="res_main",
            direction_id="gdn-prefill-main",
            deliveries=[delivery],
            default_target=TargetRef(
                delivery.version_id or "V1", delivery.artifact_id or "art-seed-v0"
            ),
        )
        evt, pkg, u2 = await coder_adapter.implement(cmd)
        runtime.controller.ledger.model_cost_used += u1.cost_usd + u2.cost_usd
        if pkg is not None:
            runtime.packages_by_id[pkg.manifest.package_id] = pkg
        await runner.submit_external_event(evt)

    runner = AsyncKernelLabRunner(
        rt, hooks=AsyncRunnerHooks(on_research_evidence=_on_evidence)
    )
    metrics = asyncio.run(runner.run(max_wall_s=float(args.wall_budget_s)))
    # Surface any job execution errors clearly in stderr for immediate diagnostics
    for _, evt in rt.store.load_events():
        err = evt.extra.get("error")
        if err:
            print(
                f"[job-error] event={evt.event} ver={evt.version_id} block={evt.block_id} "
                f"status={evt.status} error={str(err)[:300]}",
                file=sys.stderr,
            )
    print(json.dumps(metrics.to_dict(), indent=2, ensure_ascii=False))
    return 0


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
    run_p.add_argument("--wall-budget-s", type=float, default=1800.0)
    run_p.add_argument("--gpu-budget-s", type=float, default=1200.0)
    run_p.add_argument("--max-rounds", type=int, default=3)
    run_p.add_argument("--smoke-cases", type=int, default=0)
    run_p.add_argument("--deepseek-live", action="store_true")
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
        return _run_end_to_end(args)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
