"""Deterministic virtual clock and real wall-clock abstractions."""

from __future__ import annotations

from abc import ABC, abstractmethod
from datetime import datetime, timedelta, timezone
import time


TZ_SHANGHAI = timezone(timedelta(hours=8))


class Clock(ABC):
    """Abstract clock providing elapsed simulation/run seconds and ISO timestamps."""

    @abstractmethod
    def now(self) -> float:
        """Return elapsed seconds since run start."""

    @abstractmethod
    def now_iso(self) -> str:
        """Return ISO-8601 timestamp string."""


class SimClock(Clock):
    """Deterministic virtual clock for simulation and replayable tests."""

    def __init__(
        self,
        start_seconds: float = 0.0,
        epoch: datetime | None = None,
    ) -> None:
        self._now = float(start_seconds)
        self._epoch = epoch or datetime(2026, 11, 18, 20, 0, 0, tzinfo=TZ_SHANGHAI)

    def now(self) -> float:
        return self._now

    def now_iso(self) -> str:
        dt = self._epoch + timedelta(seconds=self._now)
        return dt.isoformat(timespec="milliseconds")

    def advance(self, seconds: float) -> float:
        if seconds < 0:
            raise ValueError(f"Cannot advance SimClock backwards by {seconds}s")
        self._now += float(seconds)
        return self._now

    def advance_to(self, target_seconds: float) -> float:
        if target_seconds < self._now:
            raise ValueError(
                f"Cannot move SimClock backwards from {self._now} to {target_seconds}"
            )
        self._now = float(target_seconds)
        return self._now


class RealClock(Clock):
    """Wall-clock implementation backed by monotonic timer."""

    def __init__(self, tz: timezone = TZ_SHANGHAI) -> None:
        self._start_mono = time.monotonic()
        self._start_wall = datetime.now(tz)
        self._tz = tz

    def now(self) -> float:
        return time.monotonic() - self._start_mono

    def now_iso(self) -> str:
        return datetime.now(self._tz).isoformat(timespec="milliseconds")
