"""CLI entrypoint for KernelLab V1."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import tempfile

from kernel_agent_v1.closed_loop_tester import ClosedLoopTester
from kernel_agent_v1.experiments import format_comparison_report, run_three_way_comparison


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
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
