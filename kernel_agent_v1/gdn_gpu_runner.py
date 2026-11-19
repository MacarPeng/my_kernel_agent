"""Real single-GPU GDN evaluator runner for Colab L4 / Ada / Ampere / Hopper.

Invoked by RealSingleGpuBackend as:
    python3 -m kernel_agent_v1.gdn_gpu_runner --inputs inputs.json --output output.json
"""

from __future__ import annotations

import argparse
import ast
import importlib
import importlib.util
import json
import math
import os
import re
from pathlib import Path
import subprocess
import sys
import traceback
from typing import Any

from kernel_agent_v1.backends import parse_ncu_csv
from kernel_agent_v1.gdn_task import (
    load_official_definition,
    load_official_workloads,
    stratified_split_workblocks,
)
from kernel_agent_v1.protocol import BudgetConfig

FLA_COMMIT = "9c8e42e762fce087c27b673af4922795d9edb85e"
FLA_BASELINE_NAME = "fla-v0-5-2-gdn-adapter"
FLA_ENVIRONMENT = {
    "FLA_DISABLE_BACKEND_DISPATCH": "1",
    "FLA_USE_TMA": "0",
    "FLA_USE_FAST_OPS": "0",
    "FLA_TRIL_PRECISION": "ieee",
}


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


def _build_timing_baseline_fn(
    mode: str, defn: dict[str, Any], cuda_available: bool, ref_fn=None
):
    """Return (baseline_fn, baseline_name); uses pinned FLA v0.5.2 adapter on CUDA by default."""
    if mode == "reference" or not cuda_available:
        return (ref_fn if ref_fn is not None else _build_reference_fn(defn)), "reference"
    if mode != "fla":
        raise ValueError(f"Unsupported timing_baseline mode: {mode}")

    os.environ.update(FLA_ENVIRONMENT)
    try:
        fla_mod = importlib.import_module("fla.ops.gated_delta_rule")
    except ImportError as exc:
        raise RuntimeError(
            "FLA baseline requested (timing_baseline='fla') but 'flash-linear-attention' is not installed. "
            "Install pinned FLA v0.5.2 via: "
            f"pip install --no-deps git+https://github.com/fla-org/flash-linear-attention.git@{FLA_COMMIT} "
            "or pass --timing-baseline reference explicitly."
        ) from exc

    chunk_gdn = fla_mod.chunk_gated_delta_rule
    recurrent_gdn = fla_mod.fused_recurrent_gated_delta_rule

    def _run_fla(q, k, v, state, A_log, a, dt_bias, b, cu_seqlens, scale):
        op = recurrent_gdn if q.shape[0] <= 64 else chunk_gdn
        output, new_state = op(
            q=q.unsqueeze(0),
            k=k.unsqueeze(0),
            v=v.unsqueeze(0),
            g=a.unsqueeze(0),
            beta=b.unsqueeze(0),
            A_log=A_log,
            dt_bias=dt_bias,
            initial_state=state,
            output_final_state=True,
            scale=128 ** -0.5 if scale is None or scale == 0.0 else scale,
            use_qk_l2norm_in_kernel=False,
            use_gate_in_kernel=True,
            use_beta_sigmoid_in_kernel=True,
            allow_neg_eigval=False,
            state_v_first=True,
            cu_seqlens=cu_seqlens,
        )
        return output.squeeze(0), new_state

    return _run_fla, FLA_BASELINE_NAME


def _make_deterministic_gdn_inputs(
    row: dict[str, Any], torch_mod, device: str, definition: dict[str, Any] | None = None
):
    """Synthesize deterministic GDN inputs matching the official workload axes."""
    axes = row["axes"]
    total_seq_len = int(axes["total_seq_len"])
    num_seqs = int(axes["num_seqs"])
    task_axes = (definition if definition is not None else load_official_definition())["axes"]
    hq, hk, hv, d = (
        task_axes[name]["value"]
        for name in ("num_q_heads", "num_k_heads", "num_v_heads", "head_size")
    )

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
    if c_o.shape != r_o.shape or c_s.shape != r_s.shape:
        return False
    ok_o = bool(torch_mod.allclose(c_o.float(), r_o.float(), atol=2e-2, rtol=2e-2))
    ok_s = bool(torch_mod.allclose(c_s.float(), r_s.float(), atol=2e-2, rtol=2e-2))
    return ok_o and ok_s


def _passes_decay_regression(cand_fn, ref_fn, torch_mod, device: str, definition: dict[str, Any]) -> bool:
    """One untimed token distinguishes projection before and after state decay."""
    axes = definition["axes"]
    hq, hk, hv, dim = (
        axes[name]["value"]
        for name in ("num_q_heads", "num_k_heads", "num_v_heads", "head_size")
    )
    q = torch_mod.zeros((1, hq, dim), dtype=torch_mod.bfloat16, device=device)
    k = torch_mod.zeros((1, hk, dim), dtype=torch_mod.bfloat16, device=device)
    v = torch_mod.zeros((1, hv, dim), dtype=torch_mod.bfloat16, device=device)
    state = torch_mod.zeros((1, hv, dim, dim), dtype=torch_mod.float32, device=device)
    q[..., 0] = 1
    k[..., 0] = 1
    v[..., 1] = 1
    state[..., 0, 0] = 1
    per_head = torch_mod.zeros(hv, dtype=torch_mod.float32, device=device)
    per_token = torch_mod.zeros((1, hv), dtype=torch_mod.bfloat16, device=device)
    cu = torch_mod.tensor([0, 1], dtype=torch_mod.int64, device=device)
    args = (q, k, v, state, per_head, per_token, per_head, per_token, cu, 1 / math.sqrt(dim))
    return _check_correctness(cand_fn(*args), ref_fn(*args), torch_mod)


_DECAY_REGRESSION_ERROR = (
    "GDN state-decay regression failed: project the decayed state (g * S) onto k "
    "before computing the gated delta update, and store the outer product in [V, K] order."
)


def _candidate_failure(status: str, stage: str, error: str, workload_id: str | None = None):
    return {
        "candidate_failure": status,
        "failure_stage": stage,
        "error": error,
        "workload_id": workload_id,
    }


def _run_benchmark_block(inputs: dict[str, Any], torch_mod) -> dict[str, Any]:
    defn = load_official_definition()
    rows = load_official_workloads()
    blocks = stratified_split_workblocks(rows)
    by_uuid = {r["uuid"]: r for r in rows}

    block_id = str(inputs.get("block_id", "W1"))
    attempt_id = inputs["attempt_id"]
    uuids = blocks[block_id]
    if inputs.get("kind") == "probe" and inputs.get("workload_id"):
        workload_id = inputs["workload_id"]
        if workload_id not in uuids:
            raise ValueError(f"Probe workload {workload_id} outside {block_id}")
        uuids = (workload_id,)
    device = "cuda:0" if torch_mod.cuda.is_available() else "cpu"
    device_name = (
        f"{torch_mod.cuda.get_device_name(0)}:0" if torch_mod.cuda.is_available() else "cpu:0"
    )

    ref_fn = _build_reference_fn(defn)
    baseline_mode = str(
        inputs.get("timing_baseline", "fla" if torch_mod.cuda.is_available() else "reference")
    )
    baseline_fn, baseline_name = _build_timing_baseline_fn(
        baseline_mode, defn, torch_mod.cuda.is_available(), ref_fn=ref_fn
    )
    cand_path = Path(inputs["candidate_path"])
    try:
        cand_fn = _load_python_entrypoint(cand_path, "cand_kernel").run
    except Exception:
        return _candidate_failure("compile_failed", "load", traceback.format_exc())

    # Optional case cap for quick smoke tests (default 0 = full 25-case official block)
    max_cases = int(inputs.get("max_cases", 0))
    eval_uuids = uuids[:max_cases] if max_cases > 0 else uuids

    cases: list[dict[str, Any]] = []
    for uid in eval_uuids:
        row = by_uuid[uid]
        args = _make_deterministic_gdn_inputs(row, torch_mod, device, defn)
        if torch_mod.cuda.is_available():
            if baseline_name == FLA_BASELINE_NAME:
                ref_out = ref_fn(*args)
                base_ms = _time_cuda_fn(baseline_fn, args, torch_mod, warmup=1, iters=2)
            else:
                # Single-pass timing when explicit reference fallback is requested
                torch_mod.cuda.synchronize()
                s_ref = torch_mod.cuda.Event(enable_timing=True)
                e_ref = torch_mod.cuda.Event(enable_timing=True)
                s_ref.record()
                ref_out = ref_fn(*args)
                e_ref.record()
                torch_mod.cuda.synchronize()
                base_ms = float(s_ref.elapsed_time(e_ref))
        else:
            ref_out = ref_fn(*args)
            base_ms = 1.0

        try:
            cand_out = cand_fn(*args)
            correct = _check_correctness(cand_out, ref_out, torch_mod)
            if not correct:
                actual_shapes = [tuple(t.shape) for t in cand_out]
                expected_shapes = [tuple(t.shape) for t in ref_out]
                return _candidate_failure(
                    "numerical_failed", "correctness",
                    f"Output/state comparison failed; candidate shapes {actual_shapes}, "
                    f"reference shapes {expected_shapes}.",
                    uid,
                )
            if not cases and not _passes_decay_regression(cand_fn, ref_fn, torch_mod, device, defn):
                return _candidate_failure("numerical_failed", "decay_regression", _DECAY_REGRESSION_ERROR)
            cand_ms = (
                _time_cuda_fn(cand_fn, args, torch_mod, warmup=1, iters=2)
                if torch_mod.cuda.is_available() else 1.0
            )
        except Exception as exc:
            is_compile_error = any(
                cls.__module__.startswith("triton.compiler.") for cls in type(exc).__mro__
            )
            return _candidate_failure(
                "compile_failed" if is_compile_error else "candidate_failed",
                "compile" if is_compile_error else "execution",
                traceback.format_exc(),
                uid,
            )

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
                "attempt_id": attempt_id,
            }
        )

    return {
        "block_id": block_id,
        "version_id": inputs.get("version_id"),
        "artifact_id": inputs.get("artifact_id"),
        "protocol": "gdn-unified-v2",
        "timing_baseline": baseline_name,
        "device_id": inputs.get("device_id", device_name),
        "attempt_id": attempt_id,
        "cases": cases,
    }


def _run_ncu_capture(inputs: dict[str, Any]) -> dict[str, Any]:
    cand_path = Path(inputs["candidate_path"])
    tree = ast.parse(cand_path.read_text(encoding="utf-8"))
    triton_kernels = [
        node.name for node in tree.body
        if isinstance(node, ast.FunctionDef)
        and any(
            isinstance(decorator, ast.Attribute)
            and isinstance(decorator.value, ast.Name)
            and decorator.value.id == "triton" and decorator.attr == "jit"
            for decorator in node.decorator_list
        )
    ]
    target_kernel = triton_kernels[0] if len(triton_kernels) == 1 else None
    kernel_filter = (
        ["--kernel-name-base", "function", "--kernel-name", f"regex:^{re.escape(target_kernel)}$"]
        if target_kernel else []
    )
    probe_code = (
        f"import torch; from pathlib import Path; "
        f"from kernel_agent_v1.gdn_gpu_runner import _load_python_entrypoint, _make_deterministic_gdn_inputs; "
        f"from kernel_agent_v1.gdn_task import load_official_workloads; "
        f"rows = load_official_workloads(); "
        f"uid = {inputs.get('workload_id')!r}; "
        f"row = rows[0] if uid is None else next(r for r in rows if r['uuid'] == uid); "
        f"args = _make_deterministic_gdn_inputs(row, torch, 'cuda:0'); "
        f"mod = _load_python_entrypoint(Path({str(cand_path)!r}), 'ncu_cand'); "
        f"mod.run(*args); torch.cuda.synchronize(); "
        f"torch.cuda.cudart().cudaProfilerStart(); mod.run(*args); "
        f"torch.cuda.synchronize(); torch.cuda.cudart().cudaProfilerStop()"
    )
    cmd = [
        "ncu",
        "--csv",
        "--profile-from-start",
        "off",
        "--metrics",
        "launch__registers_per_thread,smsp__sass_lmem_bytes_per_thread.sum,sm__warps_active.avg.pct_of_peak_sustained_active",
        "--launch-count",
        "1",
        *kernel_filter,
        sys.executable,
        "-c",
        probe_code,
    ]
    timeout_s = float(inputs.get("timeout_s", BudgetConfig.default_job_timeout_s))
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout_s, check=False)
    if proc.returncode != 0:
        return {
            "raw_ncu_report": {
                "version_id": inputs.get("version_id"),
                "artifact_id": inputs.get("artifact_id"),
                "error": proc.stderr[-1000:],
                "raw_counters": {},
            }
        }
    return {"raw_ncu_report": {
        "csv": proc.stdout,
        "stderr": proc.stderr,
        "requested_kernel": target_kernel,
    }}


def _parse_ncu_report(inputs: dict[str, Any]) -> dict[str, Any]:
    raw = inputs["raw_ncu_report"]
    if raw.get("error"):
        return {"observations": {"error": raw["error"]}}
    if "csv" in raw:
        observations = parse_ncu_csv(raw["csv"])
        observations["requested_kernel"] = raw.get("requested_kernel")
        observations["capture_scope"] = "named_kernel" if raw.get("requested_kernel") else "first_launch"
    else:
        observations = dict(raw)
    if not observations.get("counters", observations.get("raw_counters")):
        observations.setdefault("error", "NCU report contains no usable counters")
    return {"observations": observations}


def _run_abba_retest(inputs: dict[str, Any], torch_mod) -> dict[str, Any]:
    defn = load_official_definition()
    rows = load_official_workloads()
    device = "cuda:0" if torch_mod.cuda.is_available() else "cpu"
    device_name = (
        f"{torch_mod.cuda.get_device_name(0)}:0" if torch_mod.cuda.is_available() else "cpu:0"
    )
    ref_fn = _build_reference_fn(defn)
    baseline_mode = str(
        inputs.get("timing_baseline", "fla" if torch_mod.cuda.is_available() else "reference")
    )
    baseline_fn, baseline_name = _build_timing_baseline_fn(
        baseline_mode, defn, torch_mod.cuda.is_available(), ref_fn=ref_fn
    )
    cand_fn = _load_python_entrypoint(Path(inputs["candidate_path"]), "abba_cand").run
    inc_path = inputs.get("incumbent_path")
    if not inc_path and inputs.get("incumbent_version_id") != "V0":
        raise ValueError(f"ABBA incumbent path missing for {inputs.get('incumbent_version_id')}")
    inc_fn = (
        _load_python_entrypoint(Path(inc_path), "abba_inc").run
        if inc_path
        else baseline_fn
    )

    if not _passes_decay_regression(cand_fn, ref_fn, torch_mod, device, defn):
        return _candidate_failure("numerical_failed", "decay_regression", _DECAY_REGRESSION_ERROR)
    paired_cases: list[dict[str, Any]] = []
    for row in rows:
        args = _make_deterministic_gdn_inputs(row, torch_mod, device, defn)
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
        "timing_baseline": baseline_name,
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
    baseline_mode = str(
        inputs.get("timing_baseline", "fla" if torch_mod.cuda.is_available() else "reference")
    )
    baseline_fn, baseline_name = _build_timing_baseline_fn(
        baseline_mode, defn, torch_mod.cuda.is_available(), ref_fn=ref_fn
    )
    cand_fn = _load_python_entrypoint(Path(inputs["candidate_path"]), "final_cand").run

    if not _passes_decay_regression(cand_fn, ref_fn, torch_mod, device, defn):
        return _candidate_failure("numerical_failed", "decay_regression", _DECAY_REGRESSION_ERROR)
    cases: list[dict[str, Any]] = []
    ratios: list[float] = []
    for row in rows:
        args = _make_deterministic_gdn_inputs(row, torch_mod, device, defn)
        ref_out = ref_fn(*args)
        cand_out = cand_fn(*args)
        correct = _check_correctness(cand_out, ref_out, torch_mod)
        if torch_mod.cuda.is_available():
            base_ms = _time_cuda_fn(baseline_fn, args, torch_mod, warmup=3, iters=10)
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
        "timing_baseline": baseline_name,
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
    elif kind == "ncu_parse":
        out = _parse_ncu_report(inp)
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
