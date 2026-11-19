"""Wall-clock spans and a standalone pipeline timeline for CLI runs."""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from html import escape
from pathlib import Path

from kernel_agent_v1.clock import Clock


@dataclass
class WorkSpan:
    role: str
    activity: str
    start_s: float
    end_s: float
    status: str


class PipelineTimeline:
    roles = ("Controller", "Leader", "Coder", "Operator", "Judge")

    def __init__(self, clock: Clock) -> None:
        self.clock = clock
        self.spans: list[WorkSpan] = []

    @contextmanager
    def measure(
        self, role: str, activity: str, *, start_s: float | None = None
    ) -> Iterator[WorkSpan]:
        start = self.clock.now() if start_s is None else start_s
        span = WorkSpan(role, activity, start, start, "completed")
        try:
            yield span
        except BaseException as exc:
            span.status = type(exc).__name__
            raise
        finally:
            span.end_s = self.clock.now()
            self.spans.append(span)

    def active_seconds(self) -> dict[str, float]:
        totals = {}
        roles = dict.fromkeys((*self.roles, *(span.role for span in self.spans)))
        for role in roles:
            intervals = sorted((s.start_s, s.end_s) for s in self.spans if s.role == role)
            covered_to = 0.0
            total = 0.0
            for start, end in intervals:
                total += max(0.0, end - max(start, covered_to))
                covered_to = max(covered_to, end)
            totals[role] = round(total, 6)
        return totals

    def write_html(self, path: Path, run_id: str) -> None:
        end = max(self.clock.now(), 0.001)
        totals = self.active_seconds()
        sections = []
        for role in totals:
            spans = sorted(
                (s for s in self.spans if s.role == role), key=lambda s: s.start_s
            )
            rows = []
            for span in spans:
                start_pct = 100 * span.start_s / end
                width_pct = 100 * (span.end_s - span.start_s) / end
                rows.append(
                    f"<tr><td>{escape(span.activity)}</td>"
                    f'<td><div class="track"><span class="bar" '
                    f'style="left:{start_pct:.6f}%;width:{width_pct:.6f}%"></span></div></td>'
                    f"<td>{span.start_s:.6f}</td><td>{span.end_s:.6f}</td>"
                    f"<td>{span.end_s - span.start_s:.6f}</td><td>{escape(span.status)}</td></tr>"
                )
            if not rows:
                rows.append('<tr><td colspan="6">No recorded work</td></tr>')
            sections.append(
                f'<tbody><tr><th class="role" colspan="6">{role} · {totals[role]:.6f} s active</th></tr>'
                + "".join(rows) + "</tbody>"
            )
        path.write_text(
            f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><title>Pipeline · {escape(run_id)}</title>
<style>
body {{ font: 14px system-ui,sans-serif; color: #172237; margin: 32px; }}
table {{ width: 100%; border-collapse: collapse; font-variant-numeric: tabular-nums; }}
td,th {{ text-align: left; border-bottom: 1px solid #e0e6ec; padding: 6px 10px; }}
.role {{ background: #e7eef5; }}
.axis {{ width: 44%; }}
.track {{ position: relative; height: 14px; background: #eef2f6;
  background-image: repeating-linear-gradient(to right, transparent, transparent calc(25% - 1px), #c9d5e2 calc(25% - 1px), #c9d5e2 25%); }}
.bar {{ position: absolute; height: 100%; min-width: 1px; background: #147c87; }}
.ticks {{ display: flex; justify-content: space-between; gap: 6px; }}
</style></head><body><h1>Pipeline · {escape(run_id)}</h1>
<p>Times are seconds on the run clock. Active time merges overlapping spans for each role.
Model spans include response latency. Operator GPU spans show job wall time.</p>
<table><thead><tr><th>Activity</th><th class="axis"><div class="ticks">
{''.join(f'<span>{end * i / 4:.2f} s</span>' for i in range(5))}
</div></th><th>Start (s)</th><th>End (s)</th><th>Duration (s)</th><th>Span status</th></tr></thead>
{''.join(sections)}
</table></body></html>""",
            encoding="utf-8",
        )
