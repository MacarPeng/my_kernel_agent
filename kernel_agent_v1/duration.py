"""Duration tracking, grouped estimation, P90 sensitivity, and censored timeout handling."""

from __future__ import annotations

from dataclasses import dataclass, field
import math
from typing import Any


@dataclass
class TimingObservation:
    """Recorded lifecycle timestamps and derived durations for a job/task."""

    task_id: str
    group_key: tuple[str, ...]
    fallback_keys: tuple[tuple[str, ...], ...]
    ready_at: float | None = None
    started_at: float | None = None
    gpu_released_at: float | None = None
    result_ready_at: float | None = None
    duration_s: float = 0.0
    cpu_post_s: float = 0.0
    timed_out: bool = False

    @property
    def queue_time_s(self) -> float | None:
        if self.ready_at is not None and self.started_at is not None:
            return max(0.0, self.started_at - self.ready_at)
        return None

    @property
    def gpu_lease_s(self) -> float | None:
        if self.started_at is not None and self.gpu_released_at is not None:
            return max(0.0, self.gpu_released_at - self.started_at)
        return None

    @property
    def background_post_s(self) -> float | None:
        if self.gpu_released_at is not None and self.result_ready_at is not None:
            return max(0.0, self.result_ready_at - self.gpu_released_at)
        return None


@dataclass
class DurationEstimate:
    typical_s: float
    slow_s: float
    sample_count: int
    source_key: tuple[str, ...]


def _quantile(sorted_vals: list[float], q: float) -> float:
    if not sorted_vals:
        raise ValueError("Cannot compute quantile of empty list")
    if len(sorted_vals) == 1:
        return sorted_vals[0]
    idx = (len(sorted_vals) - 1) * q
    lo = int(math.floor(idx))
    hi = int(math.ceil(idx))
    if lo == hi:
        return sorted_vals[lo]
    frac = idx - lo
    return sorted_vals[lo] * (1.0 - frac) + sorted_vals[hi] * frac


class DurationModel:
    """Grouped duration estimator with median typical & P90 slow quantiles (§6)."""

    def __init__(self, min_samples: int = 3, slow_quantile: float = 0.90) -> None:
        self.min_samples = min_samples
        self.slow_quantile = slow_quantile
        # group_key -> list of (duration_s, timed_out)
        self._samples: dict[tuple[str, ...], list[tuple[float, bool]]] = {}
        self.observations: list[TimingObservation] = []

    def record(self, obs: TimingObservation) -> None:
        self.observations.append(obs)
        val = max(0.001, float(obs.duration_s))
        self._samples.setdefault(obs.group_key, []).append((val, obs.timed_out))
        for fb in obs.fallback_keys:
            self._samples.setdefault(fb, []).append((val, obs.timed_out))

    def record_sample(
        self,
        group_key: tuple[str, ...],
        duration_s: float,
        timed_out: bool = False,
        fallback_keys: tuple[tuple[str, ...], ...] = (),
        task_id: str = "",
        ready_at: float | None = None,
        started_at: float | None = None,
        gpu_released_at: float | None = None,
        result_ready_at: float | None = None,
    ) -> TimingObservation:
        obs = TimingObservation(
            task_id=task_id,
            group_key=group_key,
            fallback_keys=fallback_keys,
            ready_at=ready_at,
            started_at=started_at,
            gpu_released_at=gpu_released_at,
            result_ready_at=result_ready_at,
            duration_s=duration_s,
            timed_out=timed_out,
        )
        self.record(obs)
        return obs

    def estimate(
        self,
        group_key: tuple[str, ...],
        fallback_keys: tuple[tuple[str, ...], ...] = (),
        default_typical_s: float = 30.0,
        default_slow_s: float | None = None,
    ) -> DurationEstimate:
        slow_fallback = (
            default_slow_s if default_slow_s is not None else default_typical_s * 1.5
        )
        candidates = (group_key,) + tuple(fallback_keys)
        for key in candidates:
            samples = self._samples.get(key, [])
            if len(samples) >= self.min_samples:
                # Timed-out samples are lower-bounded at timeout_s (so we never drop
                # them and never let only successful fast runs skew the distribution).
                vals = sorted(v for v, _ in samples)
                typical = _quantile(vals, 0.50)
                slow = _quantile(vals, self.slow_quantile)
                # Ensure slow >= typical and account for high timeout rate
                timeout_rate = sum(1 for _, to in samples if to) / len(samples)
                if timeout_rate >= 0.5:
                    typical = max(typical, slow)
                slow = max(slow, typical)
                return DurationEstimate(
                    typical_s=typical,
                    slow_s=slow,
                    sample_count=len(samples),
                    source_key=key,
                )

        # Insufficient samples: if we have 1-2 samples in exact/fallback key, blend
        # conservatively with explicit default/timeout bound instead of ignoring timeouts.
        for key in candidates:
            samples = self._samples.get(key, [])
            if samples:
                vals = sorted(v for v, _ in samples)
                max_seen = vals[-1]
                typical = max(vals[len(vals) // 2], default_typical_s * 0.5)
                slow = max(max_seen, slow_fallback)
                return DurationEstimate(
                    typical_s=typical,
                    slow_s=slow,
                    sample_count=len(samples),
                    source_key=key,
                )

        return DurationEstimate(
            typical_s=default_typical_s,
            slow_s=slow_fallback,
            sample_count=0,
            source_key=("default",),
        )

    @staticmethod
    def benchmark_keys(
        gpu_name: str, protocol: str, block_id: str
    ) -> tuple[tuple[str, ...], tuple[tuple[str, ...], ...]]:
        return (
            ("benchmark", gpu_name, protocol, block_id),
            (
                ("benchmark", gpu_name, protocol, "*"),
                ("benchmark", gpu_name, "*", "*"),
            ),
        )

    @staticmethod
    def diagnostic_keys(
        template_id: str, workload_count: int, gpu_name: str
    ) -> tuple[tuple[str, ...], tuple[tuple[str, ...], ...]]:
        wc = str(workload_count)
        return (
            ("diag", template_id, wc, gpu_name),
            (
                ("diag", template_id, "*", gpu_name),
                ("diag", "*", wc, gpu_name),
            ),
        )

    @staticmethod
    def cpu_keys(
        toolchain: str, cache_state: str, report_kind: str
    ) -> tuple[tuple[str, ...], tuple[tuple[str, ...], ...]]:
        return (
            ("cpu", toolchain, cache_state, report_kind),
            (
                ("cpu", toolchain, "*", report_kind),
                ("cpu", "*", "*", report_kind),
            ),
        )

    @staticmethod
    def model_keys(
        role: str, task_kind: str, size_bucket: str = "normal"
    ) -> tuple[tuple[str, ...], tuple[tuple[str, ...], ...]]:
        return (
            ("model", role, task_kind, size_bucket),
            (
                ("model", role, "*", size_bucket),
                ("model", role, "*", "*"),
            ),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "min_samples": self.min_samples,
            "slow_quantile": self.slow_quantile,
            "samples": {
                "|".join(k): [[v, to] for v, to in vals]
                for k, vals in self._samples.items()
            },
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "DurationModel":
        dm = cls(
            min_samples=int(data.get("min_samples", 3)),
            slow_quantile=float(data.get("slow_quantile", 0.90)),
        )
        for key_str, vals in data.get("samples", {}).items():
            key = tuple(key_str.split("|"))
            dm._samples[key] = [(float(v), bool(to)) for v, to in vals]
        return dm
