"""Kernel Wiki: Hardware Knowledge Plugin and Experiment Route Tree (§3)."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import re
from typing import Any

from kernel_agent_v1.protocol import EvidenceDelivery, RouteStatus


def _normalize_hypothesis(text: str) -> str:
    return re.sub(r"\s+", " ", text.strip().lower())


@dataclass(frozen=True)
class KnowledgeEntry:
    entry_id: str
    gpu_arch: str
    toolchain: str
    topic: str
    source: str
    scope: str
    version: str
    guidance: str
    keywords: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class WikiLookupResult:
    lookup_id: str
    gpu_name: str
    direction_id: str | None
    query_terms: tuple[str, ...]
    hardware_entries: tuple[KnowledgeEntry, ...]
    route_context: dict[str, Any]
    timestamp: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "lookup_id": self.lookup_id,
            "gpu_name": self.gpu_name,
            "direction_id": self.direction_id,
            "query_terms": list(self.query_terms),
            "hardware_entries": [e.to_dict() for e in self.hardware_entries],
            "route_context": self.route_context,
            "timestamp": self.timestamp,
        }


class HardwareKnowledgePlugin:
    """Architecture- and toolchain-scoped hardware & compiler knowledge plugin."""

    def __init__(
        self,
        gpu_arch: str = "NVIDIA-H100-SXM",
        toolchain: str = "cuda-12.6/triton-3.1",
        entries: list[KnowledgeEntry] | None = None,
    ) -> None:
        self.gpu_arch = gpu_arch
        self.toolchain = toolchain
        self.entries: list[KnowledgeEntry] = (
            entries if entries is not None else self._default_entries(gpu_arch, toolchain)
        )

    @staticmethod
    def _default_entries(gpu_arch: str, toolchain: str) -> list[KnowledgeEntry]:
        entries = [
            KnowledgeEntry(
                entry_id="hw-l4-00",
                gpu_arch="NVIDIA-L4",
                toolchain=toolchain,
                topic="l4_ada_hardware_boundary",
                source="NVIDIA L4 Datasheet & Ada Lovelace Tuning Guide",
                scope="SM89 / 24 GB GDDR6 (~300 GB/s), 48 MB L2, 58 SMs, 100 KiB SMEM/SM",
                version=toolchain,
                guidance=(
                    "NVIDIA L4 (SM89 Ada) supports BF16/TF32/FP8 Tensor Cores and SM80 cp.async "
                    "(no TMA/WGMMA). Prioritize L2 cache reuse (48 MB) and fuse gates/epilogues "
                    "to reduce GDDR6 traffic (~300 GB/s); keep per-CTA static SMEM <= 48 KiB."
                ),
                keywords=("l4", "sm89", "ada", "l2", "bandwidth", "cp.async", "gdn", "bf16"),
            ),
            KnowledgeEntry(
                entry_id="hw-reg-01",
                gpu_arch="NVIDIA-H100-SXM",
                toolchain=toolchain,
                topic="register_pressure_occupancy",
                source="NVIDIA Hopper Tuning Guide & NCU Profiling Playbook",
                scope="SM90 / short-sequence & high-state-dim GDN kernels",
                version=toolchain,
                guidance=(
                    "Keep per-thread register count <= 128 to sustain >= 50% warp occupancy "
                    "on SM90; local memory spills in short-sequence blocks sharply inflate latency."
                ),
                keywords=("register", "spill", "occupancy", "short", "probe", "ncu"),
            ),
            KnowledgeEntry(
                entry_id="hw-layout-02",
                gpu_arch="NVIDIA-H100-SXM",
                toolchain=toolchain,
                topic="smem_layout_swizzle",
                source="CUDA C++ Best Practices & Triton Shared Memory Layout Notes",
                scope="SM90 / chunked matrix & state recurrence tiles",
                version=toolchain,
                guidance=(
                    "Use 128B swizzled shared memory layouts and vectorized 16B cp.async/TMA "
                    "loads to eliminate shared-memory bank conflicts in chunkwise GDN scans."
                ),
                keywords=("layout", "shared", "bank", "swizzle", "tma", "memory", "coalesce"),
            ),
            KnowledgeEntry(
                entry_id="hw-tma-03",
                gpu_arch="NVIDIA-H100-SXM",
                toolchain=toolchain,
                topic="wgmma_tma_pipeline",
                source="NVIDIA Hopper Architecture Whitepaper",
                scope="SM90 / Hopper async TMA + WGMMA warpgroup pipeline",
                version=toolchain,
                guidance=(
                    "Align tensor tiles to 64x16 FP16/BF16 WGMMA shapes and overlap TMA bulk "
                    "async copies with warpgroup MMA using 2-3 producer/consumer stages."
                ),
                keywords=("tma", "wgmma", "hopper", "sm90", "pipeline", "stage", "async"),
            ),
            KnowledgeEntry(
                entry_id="hw-dataflow-04",
                gpu_arch=gpu_arch,
                toolchain=toolchain,
                topic="work_partitioning_dataflow",
                source="GDN KernelLab Architecture Notes",
                scope="gdn-unified-v2 / varlen & multi-head workloads",
                version=toolchain,
                guidance=(
                    "Split long sequences across persistent CTAs and fuse intra-chunk state "
                    "updates in registers/SMEM before global writeback to cut HBM roundtrips."
                ),
                keywords=("partition", "dataflow", "fusion", "persistent", "chunk", "long"),
            ),
            KnowledgeEntry(
                entry_id="hw-compiler-05",
                gpu_arch=gpu_arch,
                toolchain=toolchain,
                topic="ptx_sass_compiler_flags",
                source="NVCC/PTXAS & Triton Compiler Behavior Reference",
                scope=toolchain,
                version=toolchain,
                guidance=(
                    "Inspect PTX/SASS lmem spill bytes and unroll pragmas off-GPU via CPU "
                    "compiler analysis before allocating a GPU profiling slot."
                ),
                keywords=("compiler", "ptx", "sass", "ir", "unroll", "spill"),
            ),
            KnowledgeEntry(
                entry_id="hw-a100-06",
                gpu_arch="NVIDIA-A100-SXM",
                toolchain=toolchain,
                topic="ampere_cp_async_smem",
                source="NVIDIA Ampere Tuning Guide",
                scope="SM80 / 164KB shared memory & cp.async.cg",
                version=toolchain,
                guidance=(
                    "Use cp.async.cg 16B bypass L1 for streaming K/V chunks and limit SMEM "
                    "per CTA <= 80KB when targeting 2 CTAs/SM for latency hiding."
                ),
                keywords=("a100", "sm80", "cp.async", "shared", "ampere"),
            ),
            KnowledgeEntry(
                entry_id="hw-l40s-07",
                gpu_arch="NVIDIA-L40S",
                toolchain=toolchain,
                topic="ada_l2_bandwidth_balance",
                source="NVIDIA Ada Lovelace Tuning Guide",
                scope="SM89 / lower HBM bandwidth, 96MB L2 cache",
                version=toolchain,
                guidance=(
                    "Prioritize L2 cache persistence and CTA swizzling on SM89 to compensate "
                    "for lower HBM bandwidth relative to SXM Hopper/Ampere."
                ),
                keywords=("l40s", "sm89", "ada", "l2", "bandwidth", "swizzle"),
            ),
            KnowledgeEntry(
                entry_id="hw-precision-08",
                gpu_arch=gpu_arch,
                toolchain=toolchain,
                topic="gdn_numerical_stability",
                source="GDN Protocol & Numerical Verification Notes",
                scope="gdn-unified-v2 / bf16/fp16 decay & state accumulation",
                version=toolchain,
                guidance=(
                    "Accumulate gated recurrence decay exp(g) and hidden state updates in FP32 "
                    "before downcasting outputs to BF16/FP16 to avoid long-sequence drift."
                ),
                keywords=("numerical", "precision", "fp32", "bf16", "decay", "correctness"),
            ),
        ]
        # Load curated Ada/GDN/NCU hardware knowledge docs adapted from KernelLab assets
        try:
            from kernel_agent_v1.gdn_task import load_curated_knowledge_files

            for doc in load_curated_knowledge_files():
                name = doc["name"]
                arch = (
                    "NVIDIA-L4"
                    if "ada" in name
                    else ("NVIDIA-A100-SXM" if "ampere" in name else gpu_arch)
                )
                entries.append(
                    KnowledgeEntry(
                        entry_id=f"hw-doc-{name}",
                        gpu_arch=arch,
                        toolchain=toolchain,
                        topic=name.replace("-", "_"),
                        source=f"KernelLab curated knowledge ({name}.md)",
                        scope="gdn-unified-v2 / curated hardware & profiling boundary",
                        version=toolchain,
                        guidance=doc["content"][:600],
                        keywords=tuple(name.split("-")) + ("gdn", "sm89", "ncu", "register"),
                    )
                )
        except Exception:
            pass
        return entries

    def add_verified_lesson(
        self,
        topic: str,
        scope: str,
        guidance: str,
        source: str,
        keywords: tuple[str, ...] = (),
        gpu_arch: str | None = None,
    ) -> KnowledgeEntry:
        """Append a verified hardware/compiler lesson from trusted run evidence."""
        entry = KnowledgeEntry(
            entry_id=f"hw-learned-{len(self.entries) + 1:03d}",
            gpu_arch=gpu_arch or self.gpu_arch,
            toolchain=self.toolchain,
            topic=topic,
            source=source,
            scope=scope,
            version=self.toolchain,
            guidance=guidance,
            keywords=keywords,
        )
        self.entries.append(entry)
        return entry

    def query(
        self, gpu_name: str, terms: tuple[str, ...], limit: int = 4
    ) -> tuple[KnowledgeEntry, ...]:
        lowered = [t.lower() for t in terms if t]
        scored: list[tuple[int, KnowledgeEntry]] = []
        for entry in self.entries:
            arch_match = 1 if (entry.gpu_arch == gpu_name or gpu_name in entry.gpu_arch) else 0
            text = " ".join(
                [entry.topic, entry.guidance, entry.scope, *entry.keywords]
            ).lower()
            kw_hits = sum(1 for t in lowered if t in text)
            score = arch_match * 2 + kw_hits * 3
            if score > 0 or not lowered:
                scored.append((score, entry))
        scored.sort(key=lambda item: (-item[0], item[1].entry_id))
        if not scored and self.entries:
            return tuple(self.entries[:limit])
        return tuple(entry for _, entry in scored[:limit])


@dataclass
class ExperimentRecord:
    experiment_id: str
    direction_id: str
    source_parent_version: str
    artifact_id: str
    version_id: str | None
    changes: str
    workload_ids: tuple[str, ...]
    trusted_results: dict[str, Any] = field(default_factory=dict)
    final_verdict: str | None = None
    retry_conditions: str | None = None
    interpretations: list[dict[str, Any]] = field(default_factory=list)


@dataclass
class RouteNode:
    direction_id: str
    parent_direction_id: str | None
    major_direction_id: str
    hypothesis: str
    hypothesis_fingerprint: str
    priority: float = 1.0
    status: str = RouteStatus.ACTIVE.value
    reference_version_id: str | None = "V0"
    reference_artifact_id: str | None = "fla-baseline"
    reference_score: float = 1.0
    best_version_id: str | None = None
    best_artifact_id: str | None = None
    best_score: float | None = None
    stagnation_count: int = 0
    compile_failure_count: int = 0
    correctness_failure_count: int = 0
    execution_failure_count: int = 0
    repair_count: int = 0
    pause_reason: str | None = None
    reusable_evidence: list[dict[str, Any]] = field(default_factory=list)
    experiments: list[ExperimentRecord] = field(default_factory=list)
    seen_complete_artifacts: set[str] = field(default_factory=set)

    def to_dict(self) -> dict[str, Any]:
        return {
            "direction_id": self.direction_id,
            "parent_direction_id": self.parent_direction_id,
            "major_direction_id": self.major_direction_id,
            "hypothesis": self.hypothesis,
            "priority": self.priority,
            "status": self.status,
            "reference_version_id": self.reference_version_id,
            "reference_artifact_id": self.reference_artifact_id,
            "reference_score": self.reference_score,
            "best_version_id": self.best_version_id,
            "best_artifact_id": self.best_artifact_id,
            "best_score": self.best_score,
            "stagnation_count": self.stagnation_count,
            "compile_failure_count": self.compile_failure_count,
            "correctness_failure_count": self.correctness_failure_count,
            "execution_failure_count": self.execution_failure_count,
            "repair_count": self.repair_count,
            "pause_reason": self.pause_reason,
            "reusable_evidence": list(self.reusable_evidence),
            "experiments": [asdict(e) for e in self.experiments],
        }


class ExperimentRouteTree:
    """Hierarchical experiment route tree with strict stagnation & trusted write rules (§3)."""

    def __init__(
        self,
        min_gain_threshold: float = 0.01,
        stagnation_limit: int = 3,
    ) -> None:
        self.min_gain_threshold = min_gain_threshold
        self.stagnation_limit = stagnation_limit
        self.routes: dict[str, RouteNode] = {}
        self._hypothesis_stagnation: dict[str, int] = {}

    def ensure_route(
        self,
        direction_id: str,
        hypothesis: str,
        parent_direction_id: str | None = None,
        priority: float = 1.0,
        reference_version_id: str | None = "V0",
        reference_artifact_id: str | None = "fla-baseline",
        reference_score: float = 1.0,
    ) -> RouteNode:
        if direction_id in self.routes:
            node = self.routes[direction_id]
            new_fp = _normalize_hypothesis(hypothesis)
            # Renaming the same hypothesis never resets stagnation.
            if new_fp and new_fp != node.hypothesis_fingerprint:
                node.hypothesis = hypothesis
                node.hypothesis_fingerprint = new_fp
            return node

        major_id = direction_id
        if parent_direction_id and parent_direction_id in self.routes:
            major_id = self.routes[parent_direction_id].major_direction_id

        fp = _normalize_hypothesis(hypothesis)
        # Preserve prior stagnation if an identical hypothesis was merely renamed into a new id.
        inherited_stagnation = self._hypothesis_stagnation.get(fp, 0)
        node = RouteNode(
            direction_id=direction_id,
            parent_direction_id=parent_direction_id,
            major_direction_id=major_id,
            hypothesis=hypothesis,
            hypothesis_fingerprint=fp,
            priority=priority,
            reference_version_id=reference_version_id,
            reference_artifact_id=reference_artifact_id,
            reference_score=reference_score,
            stagnation_count=inherited_stagnation,
        )
        if inherited_stagnation >= self.stagnation_limit:
            node.status = RouteStatus.PAUSED.value
            node.pause_reason = "stagnation_limit_reached"
        self.routes[direction_id] = node
        return node

    def rename_hypothesis(self, direction_id: str, new_hypothesis: str) -> RouteNode:
        """Update hypothesis label without ever resetting stagnation_count."""
        node = self.routes[direction_id]
        node.hypothesis = new_hypothesis
        node.hypothesis_fingerprint = _normalize_hypothesis(new_hypothesis)
        return node

    def record_experiment_submission(
        self,
        experiment_id: str,
        direction_id: str,
        source_parent_version: str,
        artifact_id: str,
        changes: str,
        workload_ids: tuple[str, ...] = (),
        version_id: str | None = None,
    ) -> ExperimentRecord:
        node = self.routes[direction_id]
        for exp in node.experiments:
            if exp.artifact_id == artifact_id:
                if version_id is not None:
                    exp.version_id = version_id
                return exp
        rec = ExperimentRecord(
            experiment_id=experiment_id,
            direction_id=direction_id,
            source_parent_version=source_parent_version,
            artifact_id=artifact_id,
            version_id=version_id,
            changes=changes,
            workload_ids=workload_ids,
        )
        node.experiments.append(rec)
        return rec

    def record_reusable_evidence(
        self,
        direction_id: str,
        receipt_id: str,
        version_id: str | None,
        kind: str,
        scope: str,
        summary: dict[str, Any],
    ) -> None:
        """Record partial benchmark or diagnostic evidence (never counts as a stagnation round)."""
        if direction_id not in self.routes:
            return
        node = self.routes[direction_id]
        if any(ev.get("receipt_id") == receipt_id for ev in node.reusable_evidence):
            return
        node.reusable_evidence.append(
            {
                "receipt_id": receipt_id,
                "version_id": version_id,
                "kind": kind,
                "scope": scope,
                "summary": summary,
            }
        )

    def record_failure_repair(
        self,
        direction_id: str,
        failure_kind: str,
        version_id: str | None,
        artifact_id: str,
        receipt_id: str | None = None,
        is_repair: bool = False,
    ) -> None:
        """Record candidate failures outside the performance stagnation counter."""
        if direction_id not in self.routes:
            return
        node = self.routes[direction_id]
        if failure_kind == "compile_failure":
            node.compile_failure_count += 1
        elif failure_kind == "correctness_failure":
            node.correctness_failure_count += 1
        elif failure_kind == "execution_failure":
            node.execution_failure_count += 1
        if is_repair:
            node.repair_count += 1
        for exp in node.experiments:
            if exp.artifact_id == artifact_id:
                exp.version_id = version_id
                exp.final_verdict = failure_kind
                if receipt_id:
                    exp.trusted_results["failure_receipt_id"] = receipt_id
                break

    def record_trusted_verdict(
        self,
        direction_id: str,
        version_id: str,
        artifact_id: str,
        geomean_speedup: float,
        verdict: str,
        receipt_id: str,
        retry_conditions: str | None = None,
    ) -> RouteNode:
        """Update route reference & stagnation ONLY from trusted complete comparable Judge verdict."""
        node = self.routes[direction_id]

        # Same-artifact retries or duplicate receipts never count twice as a round.
        if artifact_id in node.seen_complete_artifacts:
            return node
        node.seen_complete_artifacts.add(artifact_id)

        matched_exp: ExperimentRecord | None = None
        for exp in node.experiments:
            if exp.artifact_id == artifact_id:
                matched_exp = exp
                break
        if matched_exp is None:
            matched_exp = ExperimentRecord(
                experiment_id=f"exp-{artifact_id}",
                direction_id=direction_id,
                source_parent_version=node.reference_version_id or "V0",
                artifact_id=artifact_id,
                version_id=version_id,
                changes="recorded from trusted verdict",
                workload_ids=(),
            )
            node.experiments.append(matched_exp)

        matched_exp.version_id = version_id
        matched_exp.trusted_results = {
            "receipt_id": receipt_id,
            "geomean_speedup": geomean_speedup,
            "protocol_comparable": True,
        }
        matched_exp.final_verdict = verdict
        if retry_conditions:
            matched_exp.retry_conditions = retry_conditions

        if node.best_score is None or geomean_speedup > node.best_score:
            node.best_score = geomean_speedup
            node.best_version_id = version_id
            node.best_artifact_id = artifact_id

        # Relative improvement over route's most recent confirmed significant reference
        rel_gain = (geomean_speedup - node.reference_score) / max(node.reference_score, 1e-9)
        if rel_gain > self.min_gain_threshold:
            node.stagnation_count = 0
            node.reference_score = geomean_speedup
            node.reference_version_id = version_id
            node.reference_artifact_id = artifact_id
        else:
            node.stagnation_count += 1

        self._hypothesis_stagnation[node.hypothesis_fingerprint] = node.stagnation_count
        if node.stagnation_count >= self.stagnation_limit:
            node.status = RouteStatus.PAUSED.value
            node.pause_reason = f"stagnation_count={node.stagnation_count}"
        return node

    def record_interpretation(
        self,
        direction_id: str,
        artifact_id: str,
        researcher_id: str,
        observation: str,
        explanation: str,
        prediction: str,
    ) -> None:
        """Store Researcher interpretation separately from trusted metrics."""
        if direction_id not in self.routes:
            return
        node = self.routes[direction_id]
        for exp in node.experiments:
            if exp.artifact_id == artifact_id:
                exp.interpretations.append(
                    {
                        "researcher_id": researcher_id,
                        "observation": observation,
                        "explanation": explanation,
                        "prediction": prediction,
                    }
                )
                return

    def close_route(self, direction_id: str, reason: str) -> None:
        if direction_id in self.routes:
            node = self.routes[direction_id]
            node.status = RouteStatus.CLOSED.value
            node.pause_reason = reason

    def active_routes(self) -> list[RouteNode]:
        return [r for r in self.routes.values() if r.status == RouteStatus.ACTIVE.value]


class KernelWiki:
    """Unified Kernel Wiki interface combining Hardware Plugin and Experiment Route Tree."""

    def __init__(
        self,
        hardware_plugin: HardwareKnowledgePlugin | None = None,
        route_tree: ExperimentRouteTree | None = None,
    ) -> None:
        self.hardware_plugin = hardware_plugin or HardwareKnowledgePlugin()
        self.route_tree = route_tree or ExperimentRouteTree()
        self._lookup_counter = 0
        self.valid_lookups: dict[str, WikiLookupResult] = {}
        self.measurements: dict[str, dict[str, Any]] = {}
        self.diagnoses: dict[str, dict[str, Any]] = {}

    def register_measurement(self, run_id: str, delivery: EvidenceDelivery) -> bool:
        """Index the trusted receipt before any researcher is notified."""
        if delivery.receipt_id in self.measurements:
            return False
        workload_ids = [case["uuid"] for case in delivery.payload.get("cases", [])]
        if not workload_ids and delivery.payload.get("workload_id"):
            workload_ids = [delivery.payload["workload_id"]]
        if not workload_ids:
            workload_ids = delivery.payload.get("workloads") or []
        self.measurements[delivery.receipt_id] = {
            "run_id": run_id,
            "receipt_id": delivery.receipt_id,
            "direction_id": delivery.direction_id,
            "version_id": delivery.version_id,
            "artifact_id": delivery.artifact_id,
            "block_id": delivery.block_id,
            "workload_ids": workload_ids,
            "kind": delivery.result_kind,
            "evidence_path": delivery.evidence_path,
        }
        return True

    def query(
        self,
        gpu_name: str,
        direction_id: str | None = None,
        evidence_terms: tuple[str, ...] = (),
        timestamp: str = "",
    ) -> WikiLookupResult:
        self._lookup_counter += 1
        lookup_id = f"wiki-lookup-{self._lookup_counter:03d}"

        terms = list(evidence_terms)
        route_ctx: dict[str, Any] = {}
        if direction_id and direction_id in self.route_tree.routes:
            node = self.route_tree.routes[direction_id]
            route_ctx = node.to_dict()
            terms.extend(node.hypothesis.split())
            terms.append(direction_id)

        hw_entries = self.hardware_plugin.query(gpu_name, tuple(terms))
        res = WikiLookupResult(
            lookup_id=lookup_id,
            gpu_name=gpu_name,
            direction_id=direction_id,
            query_terms=tuple(terms),
            hardware_entries=hw_entries,
            route_context=route_ctx,
            timestamp=timestamp,
        )
        self.valid_lookups[lookup_id] = res
        return res

    def has_valid_lookup(self, lookup_id: str) -> bool:
        return lookup_id in self.valid_lookups
