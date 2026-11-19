"""Official GDN prefill task assets, stratified 4x25 workblocks, and AST/ABI validation."""

from __future__ import annotations

import ast
from collections import Counter
import hashlib
import json
from pathlib import Path, PurePosixPath
from typing import Any


DEFINITION_NAME = "gdn_prefill_qk4_v8_d128_k_last"
PARTITION_VERSION = "gdn-workblocks-v1"
REQUIRED_GDN_ARGS = [
    "q",
    "k",
    "v",
    "state",
    "A_log",
    "a",
    "dt_bias",
    "b",
    "cu_seqlens",
    "scale",
]
ALLOWED_TOP_LEVEL_IMPORTS = frozenset({"triton", "torch", "math", "typing", "functools"})
FORBIDDEN_CALLS = frozenset(
    {"eval", "exec", "compile", "__import__", "open", "input", "breakpoint"}
)


def assets_root() -> Path:
    return Path(__file__).resolve().parent / "assets" / "gdn"


def load_official_definition(root: Path | None = None) -> dict[str, Any]:
    base = root or assets_root()
    path = base / "definitions" / f"{DEFINITION_NAME}.json"
    return json.loads(path.read_text(encoding="utf-8"))


def load_official_workloads(root: Path | None = None) -> list[dict[str, Any]]:
    base = root or assets_root()
    path = base / "workloads" / f"{DEFINITION_NAME}.jsonl"
    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        s = line.strip()
        if not s:
            continue
        obj = json.loads(s)
        rows.append(obj.get("workload", obj))
    return rows


def workload_features(row: dict[str, Any]) -> set[str]:
    axes = row["axes"]
    t, n = int(axes["total_seq_len"]), int(axes["num_seqs"])
    tb = "tiny" if t <= 64 else "small" if t <= 512 else "medium" if t <= 4096 else "large"
    nb = "single" if n == 1 else "few" if n <= 4 else "many"
    avg = t / max(n, 1)
    ab = "short" if avg <= 64 else "mid" if avg <= 512 else "long"
    feats = {
        f"tokens:{tb}",
        f"sequences:{nb}",
        f"mean_length:{ab}",
        f"joint:{tb}:{nb}",
        f"tail:{t % 64 != 0}",
        "dispatch:" + ("recurrent" if t <= 64 else "chunk"),
        "state:" + ("none" if row.get("inputs", {}).get("state") is None else "present"),
    }
    return feats


def stratified_split_workblocks(
    rows: list[dict[str, Any]],
) -> dict[str, tuple[str, ...]]:
    """Deterministic, disjoint 4x25 stratification adapted from kernel_lab/evaluation/gdn/workblocks.py."""
    if len(rows) != 100 or len({r["uuid"] for r in rows}) != 100:
        raise ValueError("GDN protocol requires exactly 100 distinct official workloads")

    ordered = sorted(rows, key=lambda r: r["uuid"])
    tags = {r["uuid"]: workload_features(r) for r in ordered}
    totals = Counter(t for vals in tags.values() for t in vals)
    target_load = sum(r["axes"]["total_seq_len"] for r in ordered) / 4.0

    groups: list[list[dict[str, Any]]] = [[] for _ in range(4)]
    counts: list[Counter[str]] = [Counter() for _ in range(4)]
    loads = [0] * 4

    pool = sorted(
        ordered,
        key=lambda r: (
            min(totals[t] for t in tags[r["uuid"]]),
            -r["axes"]["total_seq_len"],
            r["uuid"],
        ),
    )
    for row in pool:
        ts = tags[row["uuid"]]
        eligible = [i for i in range(4) if len(groups[i]) < 25]

        def rank(i: int) -> tuple[float, int, int, int]:
            spread = sum((counts[i][t] + 1) ** 2 / totals[t] for t in ts)
            load_pen = 0.2 * ((loads[i] + row["axes"]["total_seq_len"]) / target_load) ** 2
            return (spread + load_pen, len(groups[i]), loads[i], i)

        idx = min(eligible, key=rank)
        groups[idx].append(row)
        counts[idx].update(ts)
        loads[idx] += row["axes"]["total_seq_len"]

    def score() -> float:
        coverage = sum(
            min(4, total) - sum(c[t] > 0 for c in counts)
            for t, total in totals.items()
        )
        balance = sum(
            sum((c[t] - total / 4.0) ** 2 for c in counts) / total
            for t, total in totals.items()
        )
        cost = sum((load / target_load - 1.0) ** 2 for load in loads)
        return coverage * 10000.0 + balance + 2.0 * cost

    for _ in range(8):
        before = score()
        best = before
        chosen: tuple[int, int, int, int] | None = None
        for i in range(4):
            for j in range(i + 1, 4):
                for a, left in enumerate(groups[i]):
                    for b, right in enumerate(groups[j]):
                        lt, rt = tags[left["uuid"]], tags[right["uuid"]]
                        delta = right["axes"]["total_seq_len"] - left["axes"]["total_seq_len"]
                        counts[i].subtract(lt)
                        counts[i].update(rt)
                        counts[j].subtract(rt)
                        counts[j].update(lt)
                        loads[i] += delta
                        loads[j] -= delta
                        val = score()
                        if val < best - 1e-9:
                            best = val
                            chosen = (i, j, a, b)
                        counts[i].subtract(rt)
                        counts[i].update(lt)
                        counts[j].subtract(lt)
                        counts[j].update(rt)
                        loads[i] -= delta
                        loads[j] += delta
        if chosen is None:
            break
        i, j, a, b = chosen
        left, right = groups[i][a], groups[j][b]
        counts[i].subtract(tags[left["uuid"]])
        counts[i].update(tags[right["uuid"]])
        counts[j].subtract(tags[right["uuid"]])
        counts[j].update(tags[left["uuid"]])
        delta = right["axes"]["total_seq_len"] - left["axes"]["total_seq_len"]
        loads[i] += delta
        loads[j] -= delta
        groups[i][a], groups[j][b] = right, left

    result: dict[str, tuple[str, ...]] = {}
    for idx, g in enumerate(groups):
        sorted_g = sorted(
            g, key=lambda r: (r["axes"]["total_seq_len"], r["axes"]["num_seqs"], r["uuid"])
        )
        result[f"W{idx + 1}"] = tuple(r["uuid"] for r in sorted_g)
    return result


def validate_gdn_candidate_sources(files: dict[str, str]) -> list[str]:
    """Validate GDN candidate Python sources against fixed ABI & AST safety boundary."""
    violations: list[str] = []
    if not files:
        return ["candidate package contains no source files"]

    local_modules = {PurePosixPath(p).stem for p in files}
    kernel_entry: ast.FunctionDef | None = None
    has_kernel_py = False

    for rel_path, content in sorted(files.items()):
        p = PurePosixPath(rel_path)
        if p.is_absolute() or ".." in p.parts or p.suffix != ".py" or ".git" in p.parts:
            violations.append(f"invalid candidate source path: {rel_path}")
            continue
        try:
            tree = ast.parse(content, filename=rel_path)
        except SyntaxError as exc:
            violations.append(f"syntax error in {rel_path}: {exc}")
            continue

        if p.name == "kernel.py":
            has_kernel_py = True
            kernel_entry = next(
                (n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "run"),
                None,
            )

        for node in ast.walk(tree):
            if isinstance(node, (ast.Import, ast.ImportFrom)):
                modules = (
                    [a.name for a in node.names]
                    if isinstance(node, ast.Import)
                    else [node.module or ""]
                )
                for mod in modules:
                    top = mod.split(".")[0]
                    if top not in (ALLOWED_TOP_LEVEL_IMPORTS | local_modules):
                        violations.append(
                            f"disallowed import '{mod}' in {rel_path} (allowed: {sorted(ALLOWED_TOP_LEVEL_IMPORTS)})"
                        )
            if (
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Name)
                and node.func.id in FORBIDDEN_CALLS
            ):
                violations.append(
                    f"forbidden host/code call '{node.func.id}' in {rel_path}"
                )

    if not has_kernel_py:
        violations.append("missing required kernel.py file")
    elif kernel_entry is None:
        violations.append("kernel.py must define run(q, k, v, state, A_log, a, dt_bias, b, cu_seqlens, scale)")
    else:
        arg_names = [a.arg for a in kernel_entry.args.args]
        if (
            arg_names != REQUIRED_GDN_ARGS
            or kernel_entry.args.vararg
            or kernel_entry.args.kwarg
        ):
            violations.append(
                f"kernel.py::run signature {arg_names} != required GDN ABI {REQUIRED_GDN_ARGS}"
            )
        body = [
            n
            for n in kernel_entry.body
            if not (
                isinstance(n, ast.Expr)
                and isinstance(n.value, ast.Constant)
                and isinstance(n.value.value, str)
            )
        ]
        if not body or all(isinstance(n, (ast.Pass, ast.Raise)) for n in body):
            violations.append("kernel.py::run body is empty or unfinished")

    return violations


def load_curated_knowledge_files(root: Path | None = None) -> list[dict[str, str]]:
    base = (root or assets_root()) / "knowledge"
    if not base.exists():
        return []
    docs: list[dict[str, str]] = []
    for md in sorted(base.glob("*.md")):
        docs.append({"name": md.stem, "path": str(md), "content": md.read_text(encoding="utf-8")})
    return docs


def seed_gdn_candidate_source() -> str:
    """Return a valid, self-contained PyTorch/Triton-compatible GDN seed kernel matching exact ABI."""
    defn = load_official_definition()
    return str(defn["reference"])
