"""Real single-GPU GDN evaluator runner for Colab L4 / Ada / Ampere / Hopper.

Invoked by RealSingleGpuBackend as:
    python3 -m kernel_agent_v1.gdn_gpu_runner --inputs inputs.json --output output.json
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
from pathlib import Path
import subprocess
import sys
from typing import Any

from kernel_agent_v1.backends import parse_ncu_csv
from kernel_agent_v1.gdn_task import (
    load_official_definition,
    load_official_workloads,
    stratified_split_workblocks,
)


def _load_python_entrypoint(py_path: Path, module_name: str = "candidate_kernel"):
    spec = importlib.util.spec_from_file_location(module_name, str(py_path))
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load module from {py_path}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = mod
    spec.loader.exec_module(mod)
    return mod


def _build_reference_fn(defn: dict[str, Any]):
    ns: dict[str, Any] = {}
    exec(defn["reference"], ns)  # Trusted frozen definition reference
    return ns["run"]


def _make_deterministic_gdn_inputs(row: dict[str, Any], torch_mod, device: str):
    """Synthesize deterministic GDN inputs matching the official workload axes."""
    axes = row["axes"]
    total_seq_len = int(axes["total_seq_len"])
    num_seqs = int(axes["num_seqs"])
    hq, hk, hv, d = 4, 4, 8, 128

    seed = int(row["uuid"].replace("-", "")[:8], 16) % (2**31 - 1)
    gen = torch_mod.Generator(device="cpu")
    gen.manual_seed(seed)

    # Deterministic cu_seqlens summing to total_seq_len
    base_len = total_seq_len // num_seqs
    rem = total_seq_len % num_seqs
    lens = [base_len + (1 if i < rem else 0) for i in range(num_seqs)]
    cu = [0]
    for ln in lens:
        cu.append(cu[-1] + ln)

    q = torch_mod.randn((total_seq_len, hq, d), generator=gen, dtype=torch_mod.float32).to(
        device=device, dtype=torch_mod.bfloat16
    ) * 0.1
    k = torch_mod.randn((total_seq_len, hk, d), generator=gen, dtype=torch_mod.float32).to(
        device=device, dtype=torch_mod.bfloat16
    ) * 0.1
    v = torch_mod.randn((total_seq_len, hv, d), generator=gen, dtype=torch_mod.float32).to(
        device=device, dtype=torch_mod.bfloat16
    ) * 0.1

    has_state = row.get("inputs", {}).get("state") is not None
    state = (
        torch_mod.randn((num_seqs, hv, d, d), generator=gen, dtype=torch_mod.float32).to(
            device=device
        )
        * 0.05
        if has_state
        else None
    )
    A_log = torch_mod.randn((hv,), generator=gen, dtype=torch_mod.float32).to(device=device) * 0.1
    a = torch_mod.randn((total_seq_len, hv), generator=gen, dtype=torch_mod.float32).to(
        device=device, dtype=torch_mod.bfloat16
    ) * 0.1
    dt_bias = torch_mod.randn((hv,), generator=gen, dtype=torch_mod.float32).to(device=device) * 0.1
    b = torch_mod.randn((total_seq_len, hv), generator=gen, dtype=torch_mod.float32).to(
        device=device, dtype=torch_mod.bfloat16
    ) * 0.1
    cu_seqlens = torch_mod.tensor(cu, dtype=torch_mod.int64, device=device)
    scale = 1.0 / math.sqrt(d)
    return (q, k, v, state, A_log, a, dt_bias, b, cu_seqlens, scale)


def _time_cuda_fn(fn, args, torch_mod, warmup: int = 1, iters: int = 3) -> float:
    for _ in range(warmup):
        fn(*args)
    torch_mod.cuda.synchronize()
    start = torch_mod.cuda.Event(enable_timing=True)
    end = torch_mod.cuda.Event(enable_timing=True)
    times: list[float] = []
    for _ in range(iters):
        start.record()
        fn(*args)
        end.record()
        torch_mod.cuda.synchronize()
        times.append(float(start.elapsed_time(end)))
    times.sort()
    return times[len(times) // 2]


def _check_correctness(cand_out, ref_out, torch_mod) -> bool:
    c_o, c_s = cand_out
    r_o, r_s = ref_out
    ok_o = bool(torch_mod.allclose(c_o.float(), r_o.float(), atol=2e-2, rtol=2e-2))
    ok_s = bool(torch_mod.allclose(c_s.float(), r_s.float(), atol=2e-2, rtol=2e-2))
    return ok_o and ok_s


def _run_benchmark_block(inputs: dict[str, Any], torch_mod) -> dict[str, Any]:
    defn = load_official_definition()
    rows = load_official_workloads()
    blocks = stratified_split_workblocks(rows)
    by_uuid = {r["uuid"]: r for r in rows}

    block_id = str(inputs.get("block_id", "W1"))
    uuids = blocks[block_id]
    device = "cuda:0" if torch_mod.cuda.is_available() else "cpu"
    device_name = (
        f"{torch_mod.cuda.get_device_name(0)}:0" if torch_mod.cuda.is_available() else "cpu:0"
    )

    ref_fn = _build_reference_fn(defn)
    cand_path = Path(inputs["candidate_path"])
    cand_mod = _load_python_entrypoint(cand_path, "cand_kernel")
    cand_fn = cand_mod.run

    # Optional case cap for quick smoke tests (default 0 = full 25-case official block)
    max_cases = int(inputs.get("max_cases", 0))
    eval_uuids = uuids[:max_cases] if max_cases > 0 else uuids

    cases: list[dict[str, Any]] = []
    for uid in eval_uuids:
        row = by_uuid[uid]
        args = _make_deterministic_gdn_inputs(row, torch_mod, device)
        if torch_mod.cuda.is_available():
            # Time reference in a single pass to halve pure-Python reference overhead
            torch_mod.cuda.synchronize()
            s_ref = torch_mod.cuda.Event(enable_timing=True)
            e_ref = torch_mod.cuda.Event(enable_timing=True)
            s_ref.record()
            ref_out = ref_fn(*args)
            e_ref.record()
            torch_mod.cuda.synchronize()
            base_ms = float(s_ref.elapsed_time(e_ref))
            cand_out = cand_fn(*args)
            correct = _check_correctness(cand_out, ref_out, torch_mod)
            cand_ms = _time_cuda_fn(cand_fn, args, torch_mod, warmup=1, iters=2)
        else:
            ref_out = ref_fn(*args)
            cand_out = cand_fn(*args)
            correct = _check_correctness(cand_out, ref_out, torch_mod)
            base_ms, cand_ms = 1.0, 1.0

        cases.append(
            {
                "uuid": uid,
                "correct": correct,
                "candidate_ms": cand_ms,
                "baseline_ms": base_ms,
                "artifact_id": inputs.get("artifact_id"),
                "version_id": inputs.get("version_id"),
                "protocol": "gdn-unified-v2",
                "device_id": inputs.get("device_id", device_name),
                "attempt_id": inputs.get("attempt_id", "gpu/1"),
            }
        )

    return {
        "block_id": block_id,
        "version_id": inputs.get("version_id"),
        "artifact_id": inputs.get("artifact_id"),
        "protocol": "gdn-unified-v2",
        "device_id": inputs.get("device_id", device_name),
        "attempt_id": inputs.get("attempt_id", "gpu/1"),
        "cases": cases,
    }


def _run_ncu_capture(inputs: dict[str, Any]) -> dict[str, Any]:
    cand_path = Path(inputs["candidate_path"])
    probe_code = (
        f"import torch; from pathlib import Path; "
        f"from kernel_agent_v1.gdn_gpu_runner import _load_python_entrypoint, _make_deterministic_gdn_inputs; "
        f"from kernel_agent_v1.gdn_task import load_official_workloads; "
        f"row = load_official_workloads()[0]; "
        f"args = _make_deterministic_gdn_inputs(row, torch, 'cuda:0'); "
        f"mod = _load_python_entrypoint(Path({str(cand_path)!r}), 'ncu_cand'); "
        f"mod.run(*args); torch.cuda.synchronize()"
    )
    cmd = [
        "ncu",
        "--csv",
        "--metrics",
        "launch__registers_per_thread,smsp__sass_lmem_bytes_per_thread.sum,sm__warps_active.avg.pct_of_peak_sustained_active",
        "--launch-count",
        "1",
        sys.executable,
        "-c",
        probe_code,
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=180, check=False)
    if proc.returncode != 0:
        return {
            "raw_ncu_report": {
                "version_id": inputs.get("version_id"),
                "artifact_id": inputs.get("artifact_id"),
                "error": proc.stderr[-1000:],
                "raw_counters": {},
            }
        }
    parsed = parse_ncu_csv(proc.stdout)
    return {
        "raw_ncu_report": {
            "version_id": inputs.get("version_id"),
            "artifact_id": inputs.get("artifact_id"),
            "raw_counters": parsed["counters"],
            "register_spill_detected": parsed["register_spill_detected"],
        }
    }


def _run_abba_retest(inputs: dict[str, Any], torch_mod) -> dict[str, Any]:
    defn = load_official_definition()
    rows = load_official_workloads()
    device = "cuda:0" if torch_mod.cuda.is_available() else "cpu"
    device_name = (
        f"{torch_mod.cuda.get_device_name(0)}:0" if torch_mod.cuda.is_available() else "cpu:0"
    )
    ref_fn = _build_reference_fn(defn)
    cand_fn = _load_python_entrypoint(Path(inputs["candidate_path"]), "abba_cand").run
    inc_path = inputs.get("incumbent_path")
    inc_fn = (
        _load_python_entrypoint(Path(inc_path), "abba_inc").run
        if inc_path
        else ref_fn
    )

    paired_cases: list[dict[str, Any]] = []
    for row in rows:
        args = _make_deterministic_gdn_inputs(row, torch_mod, device)
        ref_out = ref_fn(*args)
        cand_out = cand_fn(*args)
        correct = _check_correctness(cand_out, ref_out, torch_mod)
        if torch_mod.cuda.is_available():
            a1 = _time_cuda_fn(inc_fn, args, torch_mod, warmup=1, iters=3)
            b1 = _time_cuda_fn(cand_fn, args, torch_mod, warmup=1, iters=3)
            b2 = _time_cuda_fn(cand_fn, args, torch_mod, warmup=1, iters=3)
            a2 = _time_cuda_fn(inc_fn, args, torch_mod, warmup=1, iters=3)
        else:
            a1 = b1 = b2 = a2 = 1.0
        paired_cases.append(
            {
                "uuid": row["uuid"],
                "correct": correct,
                "a1_ms": a1,
                "b1_ms": b1,
                "b2_ms": b2,
                "a2_ms": a2,
            }
        )
    return {
        "incumbent_version_id": inputs.get("incumbent_version_id"),
        "incumbent_artifact_id": inputs.get("incumbent_artifact_id"),
        "candidate_version_id": inputs.get("version_id"),
        "candidate_artifact_id": inputs.get("artifact_id"),
        "protocol": "gdn-unified-v2",
        "device_id": inputs.get("device_id", device_name),
        "paired_cases": paired_cases,
    }


def _run_final_eval(inputs: dict[str, Any], torch_mod) -> dict[str, Any]:
    defn = load_official_definition()
    rows = load_official_workloads()
    device = "cuda:0" if torch_mod.cuda.is_available() else "cpu"
    device_name = (
        f"{torch_mod.cuda.get_device_name(0)}:0" if torch_mod.cuda.is_available() else "cpu:0"
    )
    ref_fn = _build_reference_fn(defn)
    cand_fn = _load_python_entrypoint(Path(inputs["candidate_path"]), "final_cand").run

    cases: list[dict[str, Any]] = []
    ratios: list[float] = []
    for row in rows:
        args = _make_deterministic_gdn_inputs(row, torch_mod, device)
        ref_out = ref_fn(*args)
        cand_out = cand_fn(*args)
        correct = _check_correctness(cand_out, ref_out, torch_mod)
        if torch_mod.cuda.is_available():
            base_ms = _time_cuda_fn(ref_fn, args, torch_mod, warmup=3, iters=10)
            cand_ms = _time_cuda_fn(cand_fn, args, torch_mod, warmup=3, iters=10)
        else:
            base_ms = cand_ms = 1.0
        ratios.append(base_ms / max(cand_ms, 1e-9))
        cases.append(
            {
                "uuid": row["uuid"],
                "correct": correct,
                "candidate_ms": cand_ms,
                "baseline_ms": base_ms,
            }
        )
    geomean = math.exp(sum(math.log(max(r, 1e-9)) for r in ratios) / max(len(ratios), 1))
    return {
        "final_block_id": inputs.get("block_id", "F1"),
        "version_id": inputs.get("version_id"),
        "artifact_id": inputs.get("artifact_id"),
        "protocol": "gdn-unified-v2",
        "device_id": inputs.get("device_id", device_name),
        "geomean_speedup": geomean,
        "cases": cases,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Real single-GPU GDN evaluator runner")
    parser.add_argument("--inputs", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args(argv)

    inp = json.loads(Path(args.inputs).read_text(encoding="utf-8"))
    kind = str(inp.get("kind", "benchmark_block"))

    if kind == "ncu_capture":
        out = _run_ncu_capture(inp)
    else:
        import torch  # Imported lazily so CPU-only hosts don't require torch at import time

        if kind == "abba_retest":
            out = _run_abba_retest(inp, torch)
        elif kind == "final_eval":
            out = _run_final_eval(inp, torch)
        else:
            out = _run_benchmark_block(inp, torch)

    Path(args.output).write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
