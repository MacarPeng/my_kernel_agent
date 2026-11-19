"""DeepSeek LLM interface and structured Researcher/Coder adapters (interface-ready, offline-safe)."""

from __future__ import annotations

from abc import ABC, abstractmethod
import asyncio
from dataclasses import dataclass, field
import json
import os
from typing import TYPE_CHECKING, Any, Callable
import urllib.request

from kernel_agent_v1.coder import Coder
from kernel_agent_v1.leader import Hypothesis, Leader, ResearchNote
from kernel_agent_v1.protocol import (
    CodePackage,
    Command,
    CommandKind,
    Event,
    EvidenceDelivery,
    TargetRef,
)


if TYPE_CHECKING:
    from kernel_agent_v1.dual_research import DualResearch


class LLMNotConfiguredError(RuntimeError):
    """Raised when live DeepSeek API is invoked without enabling live mode or API key."""


@dataclass(frozen=True)
class DeepSeekConfig:
    """Configuration for DeepSeek OpenAI-compatible API."""

    model: str = "deepseek-chat"  # e.g. "deepseek-chat" or "deepseek-reasoner"
    base_url: str = "https://api.deepseek.com"
    api_key_env: str = "DEEPSEEK_API_KEY"
    temperature: float = 0.2
    max_tokens: int = 4096
    timeout_s: float = 60.0
    live_enabled: bool = False  # Intentionally False until user chooses to wire live API
    input_cost_per_1m: float = 0.27
    output_cost_per_1m: float = 1.10


@dataclass(frozen=True)
class LLMMessage:
    role: str  # "system" | "user" | "assistant"
    content: str


@dataclass(frozen=True)
class LLMUsage:
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    cost_usd: float = 0.0

    def __add__(self, other: LLMUsage) -> LLMUsage:
        return LLMUsage(
            prompt_tokens=self.prompt_tokens + other.prompt_tokens,
            completion_tokens=self.completion_tokens + other.completion_tokens,
            total_tokens=self.total_tokens + other.total_tokens,
            cost_usd=self.cost_usd + other.cost_usd,
        )


class LLMResponseFormatError(ValueError):
    """A structured response could not be decoded after retrying."""

    def __init__(self, message: str, usage: LLMUsage) -> None:
        super().__init__(message)
        self.usage = usage


@dataclass(frozen=True)
class LLMResponse:
    content: str | None
    model: str
    usage: LLMUsage = field(default_factory=LLMUsage)
    raw: dict[str, Any] = field(default_factory=dict)

    def parse_json(self) -> dict[str, Any]:
        choices = self.raw.get("choices") or []
        finish_reason = choices[0].get("finish_reason") if choices else None
        if finish_reason == "length":
            raise ValueError("JSON response truncated at the token limit")
        text = self.content.strip() if isinstance(self.content, str) else ""
        if not text:
            raise ValueError("empty response content")
        lines = text.splitlines()
        if (
            len(lines) >= 2
            and lines[0].strip().lower() in ("```", "```json")
            and lines[-1].strip() == "```"
        ):
            text = "\n".join(lines[1:-1]).strip()
        parsed = json.loads(text)
        if not isinstance(parsed, dict):
            raise ValueError("expected a JSON object")
        return parsed


class LLMClient(ABC):
    """Abstract async LLM client interface."""

    @abstractmethod
    async def complete(
        self, messages: list[LLMMessage], *, json_mode: bool = True
    ) -> LLMResponse:
        """Send chat completion request and return structured response + token cost."""


async def _complete_json(
    llm: LLMClient, messages: list[LLMMessage]
) -> tuple[dict[str, Any], LLMUsage]:
    usage = LLMUsage()
    for attempt in range(2):
        response = await llm.complete(messages, json_mode=True)
        usage += response.usage
        try:
            return response.parse_json(), usage
        except ValueError as exc:
            if attempt == 1:
                raise LLMResponseFormatError(
                    f"invalid JSON after 2 attempts ({exc})", usage
                ) from exc
            system, *rest = messages
            messages = [
                LLMMessage(
                    system.role,
                    system.content + "\nReturn exactly one complete, compact JSON object "
                    "matching the example. Omit Markdown and surrounding prose.",
                ),
                *rest,
            ]


class DeepSeekClient(LLMClient):
    """DeepSeek API client skeleton (disabled by default until live_enabled=True & API key set)."""

    def __init__(
        self,
        config: DeepSeekConfig | None = None,
        api_key: str | None = None,
    ) -> None:
        self.config = config or DeepSeekConfig()
        self._explicit_api_key = api_key

    def resolve_api_key(self) -> str | None:
        return self._explicit_api_key or os.environ.get(self.config.api_key_env)

    def is_ready(self) -> bool:
        return bool(self.config.live_enabled and self.resolve_api_key())

    def compute_cost(self, prompt_tokens: int, completion_tokens: int) -> float:
        return (
            prompt_tokens * self.config.input_cost_per_1m
            + completion_tokens * self.config.output_cost_per_1m
        ) / 1_000_000.0

    def _sync_request(
        self, messages: list[LLMMessage], json_mode: bool
    ) -> LLMResponse:
        if not self.config.live_enabled:
            raise LLMNotConfiguredError(
                "DeepSeekClient live_enabled=False; interface reserved for offline/manual wiring."
            )
        api_key = self.resolve_api_key()
        if not api_key:
            raise LLMNotConfiguredError(
                f"Missing DeepSeek API key in environment variable {self.config.api_key_env}"
            )

        url = f"{self.config.base_url.rstrip('/')}/chat/completions"
        payload: dict[str, Any] = {
            "model": self.config.model,
            "messages": [{"role": m.role, "content": m.content} for m in messages],
            "temperature": self.config.temperature,
            "max_tokens": self.config.max_tokens,
        }
        if json_mode and self.config.model != "deepseek-reasoner":
            payload["response_format"] = {"type": "json_object"}

        req = urllib.request.Request(
            url=url,
            data=json.dumps(payload).encode("utf-8"),
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {api_key}",
            },
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=self.config.timeout_s) as resp:
            raw = json.loads(resp.read().decode("utf-8"))

        choices = raw.get("choices") or []
        content = (choices[0].get("message") or {}).get("content") if choices else None
        u = raw.get("usage", {})
        pt = int(u.get("prompt_tokens", 0))
        ct = int(u.get("completion_tokens", 0))
        usage = LLMUsage(
            prompt_tokens=pt,
            completion_tokens=ct,
            total_tokens=int(u.get("total_tokens", pt + ct)),
            cost_usd=self.compute_cost(pt, ct),
        )
        return LLMResponse(content=content, model=self.config.model, usage=usage, raw=raw)

    async def complete(
        self, messages: list[LLMMessage], *, json_mode: bool = True
    ) -> LLMResponse:
        return await asyncio.to_thread(self._sync_request, messages, json_mode)


class ScriptedDeepSeekClient(LLMClient):
    """Deterministic offline DeepSeek mock client for tests and no-API simulations."""

    def __init__(
        self,
        handler: Callable[[list[LLMMessage]], dict[str, Any] | str],
        model: str = "deepseek-chat",
        latency_s: float = 0.0,
    ) -> None:
        self.handler = handler
        self.model = model
        self.latency_s = latency_s
        self.calls: list[list[LLMMessage]] = []

    async def complete(
        self, messages: list[LLMMessage], *, json_mode: bool = True
    ) -> LLMResponse:
        self.calls.append(messages)
        if self.latency_s > 0:
            await asyncio.sleep(self.latency_s)
        out = self.handler(messages)
        text = out if isinstance(out, str) else json.dumps(out, ensure_ascii=False)
        usage = LLMUsage(
            prompt_tokens=sum(len(m.content) // 4 for m in messages),
            completion_tokens=max(1, len(text) // 4),
            total_tokens=sum(len(m.content) // 4 for m in messages) + max(1, len(text) // 4),
            cost_usd=0.0005,
        )
        return LLMResponse(content=text, model=self.model, usage=usage)


class DeepSeekResearcherAdapter:
    """Translates EvidenceDelivery + Wiki lookup into validated Leader/Researcher commands."""

    def __init__(
        self, llm: LLMClient, leader: Leader, *, task_definition: dict[str, Any] | None = None
    ) -> None:
        self.llm = llm
        self.leader = leader
        self.task_definition = task_definition

    async def plan_from_evidence(
        self,
        researcher_id: str,
        direction_id: str,
        deliveries: list[EvidenceDelivery],
        default_target: TargetRef,
        *,
        repair_context: dict[str, Any] | None = None,
        dual_research: DualResearch | None = None,
    ) -> tuple[Command, dict[str, Any], LLMUsage]:
        researcher = self.leader.ensure_researcher(researcher_id)
        rid = self.leader.next_research_id()
        terms = tuple(
            [direction_id]
            + [d.result_kind for d in deliveries]
            + [d.block_id for d in deliveries if d.block_id]
        )
        # Mandatory Wiki first-lookup before calling LLM (§3)
        lookup = self.leader.wiki.query(
            gpu_name=self.leader.gpu_name,
            direction_id=direction_id,
            evidence_terms=terms,
            timestamp=self.leader.clock.now_iso(),
        )

        sys_prompt = (
            "You are a KernelLab Researcher optimizing kernels. "
            "Separate observation, explanation, prediction, and emit valid JSON with keys: "
            "observation, explanation, prediction, hypothesis, command_kind "
            "(one of implement_kernel, write_ncu_capture, write_probe, write_compiler_analysis), "
            "instruction, deliverable, allowed_paths (use ['kernel.py'] for implement_kernel), "
            "workload_ids, preserve_target (false for implement_kernel). "
            "When provided, task_definition specifies the required kernel interface and semantics. "
            'Example JSON: {"observation":"W1 long cases are slow","explanation":"launch overhead",'
            '"prediction":"fewer launches reduce latency","hypothesis":"fuse the gating step",'
            '"command_kind":"implement_kernel","instruction":"fuse gating and update",'
            '"deliverable":"kernel.py","allowed_paths":["kernel.py"],"workload_ids":[],"preserve_target":false}'
        )
        user_payload = {
            "research_id": rid,
            "direction_id": direction_id,
            "wiki_lookup": lookup.to_dict(),
            "evidence": [
                {
                    "receipt_id": d.receipt_id,
                    "result_kind": d.result_kind,
                    "version_id": d.version_id,
                    "artifact_id": d.artifact_id,
                    "block_id": d.block_id,
                    "payload": d.payload,
                }
                for d in deliveries
            ],
            "target": {
                "version_id": default_target.version_id,
                "artifact_id": default_target.artifact_id,
            },
        }
        if self.task_definition is not None:
            user_payload["task_definition"] = self.task_definition
        if repair_context is not None:
            user_payload["repair_context"] = repair_context
            sys_prompt += (
                " Diagnose the candidate failure from the evidence and source. "
                "Explain the cause and issue an implement_kernel instruction to repair the target "
                "while preserving the task interface."
            )
        design_evidence = deliveries[0] if dual_research is not None and deliveries else None
        initial_diagnoses = []
        if design_evidence is not None:
            initial_diagnoses = dual_research.relevant_diagnoses(design_evidence)
            sys_prompt += " Conclusions for ancestor versions are historical, not measurements of this target."
            user_payload["available_diagnoses"] = initial_diagnoses
            user_payload["pending_diagnostics"] = dual_research.pending_for(design_evidence)
            sys_prompt += (
                " You are the Design Researcher. Issue implement_kernel; Diagnosis owns GPU "
                "diagnostics. Use the evidence now available and cite used conclusion IDs in "
                "diagnosis_refs. Omit waiting by default. Only if you need a listed pending "
                "diagnostic to make the design, return wait_for_diagnostic_task_id and finite "
                "positive wait_seconds, along with your fallback instruction for expiry."
            )
        parsed, usage = await _complete_json(
            self.llm,
            [
                LLMMessage("system", sys_prompt),
                LLMMessage("user", json.dumps(user_payload, ensure_ascii=False)),
            ],
        )

        used_diagnoses: tuple[str, ...] = ()
        if design_evidence is not None:
            await dual_research.wait_if_requested(parsed, design_evidence)
            current = dual_research.relevant_diagnoses(design_evidence)
            initial_ids = {item["receipt_id"] for item in initial_diagnoses}
            new_diagnoses = [
                item for item in current
                if item["receipt_id"] not in initial_ids and item["design_impact"]
            ]
            refs = parsed.get("diagnosis_refs")
            used_diagnoses = tuple(
                item["receipt_id"] for item in initial_diagnoses
                if isinstance(refs, list) and item["receipt_id"] in refs
            )
            changed = False
            if new_diagnoses:
                try:
                    revision, revision_usage = await _complete_json(self.llm, [
                        LLMMessage("system", (
                            "Review the proposed kernel design against the newly available "
                            "diagnoses. Return JSON with the instruction and diagnosis_refs only "
                            "if evidence changes the design. Cite the relevant diagnosis IDs. "
                            "Otherwise return an empty JSON object."
                        )),
                        LLMMessage("user", json.dumps({
                            "plan": parsed, "new_diagnoses": new_diagnoses,
                            "task_definition": self.task_definition,
                        }, ensure_ascii=False)),
                    ])
                    usage += revision_usage
                    instruction = revision.get("instruction")
                    refs = revision.get("diagnosis_refs")
                    new_refs = tuple(
                        item["receipt_id"] for item in new_diagnoses
                        if isinstance(refs, list) and item["receipt_id"] in refs
                    )
                    if (
                        isinstance(instruction, str) and instruction.strip() and new_refs
                        and instruction.strip() != str(parsed.get("instruction", "")).strip()
                    ):
                        parsed["instruction"] = instruction.strip()
                        used_diagnoses += new_refs
                        changed = True
                except LLMResponseFormatError as exc:
                    usage += exc.usage
                except (urllib.error.URLError, TimeoutError, json.JSONDecodeError):
                    pass  # A failed optional review leaves the original plan available.
            parsed["diagnosis_feedback"] = {
                "ready_before_command": [
                    item["receipt_id"] for item in dual_research.relevant_diagnoses(design_evidence)
                ],
                "late_diagnoses_reviewed": [item["receipt_id"] for item in new_diagnoses],
                "cited_by_design": list(used_diagnoses),
                "late_diagnosis_changed_instruction": changed,
            }

        evidence_refs = (lookup.lookup_id,) + tuple(d.receipt_id for d in deliveries) + used_diagnoses
        # Record structured research note once, reusing the mandatory initial Wiki lookup
        researcher.notes.append(
            ResearchNote(
                research_id=rid,
                researcher_id=researcher_id,
                direction_id=direction_id,
                wiki_lookup_id=lookup.lookup_id,
                observation=str(parsed.get("observation", "")),
                explanation=str(parsed.get("explanation", "")),
                prediction=str(parsed.get("prediction", "")),
                hypotheses=[
                    Hypothesis(
                        hypothesis_id=f"hyp-{rid}",
                        direction_id=direction_id,
                        statement=str(parsed.get("hypothesis", "")),
                        applicable_shapes=tuple(parsed.get("workload_ids", [])),
                        evidence_refs=evidence_refs,
                        falsification_criteria=str(
                            parsed.get("falsification_criteria", "no speedup gain")
                        ),
                        next_action=str(parsed.get("instruction", "")),
                    )
                ],
                timestamp=self.leader.clock.now_iso(),
            )
        )

        cmd = self.leader.build_command(
            kind=(
                CommandKind.IMPLEMENT_KERNEL.value if repair_context is not None or dual_research is not None
                else str(parsed.get("command_kind", CommandKind.IMPLEMENT_KERNEL.value))
            ),
            research_id=rid,
            direction_id=direction_id,
            target=default_target,
            hypothesis=str(parsed.get("hypothesis", "optimization hypothesis")),
            evidence_refs=evidence_refs,
            instruction=str(parsed.get("instruction", "implement candidate")),
            deliverable=str(parsed.get("deliverable", "kernel package")),
            reply_to=researcher_id,
            workload_ids=tuple(parsed.get("workload_ids", ())),
            allowed_paths=(
                tuple(repair_context["files"]) if repair_context is not None
                else tuple(parsed.get("allowed_paths", ()))
            ),
            preserve_target=(
                False if repair_context is not None
                else bool(parsed.get("preserve_target", True))
            ),
        )
        return cmd, parsed, usage


class DeepSeekCoderAdapter:
    """Translates Leader Command into constraint-checked immutable CodePackage via DeepSeek."""

    def __init__(
        self, llm: LLMClient, coder: Coder, *, task_definition: dict[str, Any] | None = None
    ) -> None:
        self.llm = llm
        self.coder = coder
        self.task_definition = task_definition

    async def implement(
        self, command: Command, *, repair_context: dict[str, Any] | None = None
    ) -> tuple[Event, CodePackage | None, LLMUsage]:
        sys_prompt = (
            "You are KernelLab Coder. Output valid JSON with keys: artifact_id, entrypoint "
            "('kernel.py' for implement_kernel), files (dict of rel_path -> complete Python code), "
            "executable_stages. Strictly stay inside allowed_paths. Only import from "
            "triton, torch, math, typing, functools (no os/sys/eval/open). For implement_kernel, "
            "follow task_definition when provided, including the Python callable, tensor shapes, "
            "dtypes, layouts and implementation rules. Use inputs and outputs in listed order. "
            'Example JSON shape: {"artifact_id":"optimized-kernel","entrypoint":"kernel.py",'
            '"files":{"kernel.py":"<complete implementation>"}}'
        )
        user_payload = command.to_dict()
        if self.task_definition is not None:
            user_payload["task_definition"] = self.task_definition
        if repair_context is not None:
            user_payload["repair_context"] = repair_context
            sys_prompt += (
                " Use the failed source and diagnostic in repair_context to implement "
                "the Leader's repair instruction. Return the complete corrected files."
            )
        parsed, usage = await _complete_json(
            self.llm,
            [
                LLMMessage("system", sys_prompt),
                LLMMessage("user", json.dumps(user_payload, ensure_ascii=False)),
            ],
        )
        if repair_context is not None:
            parsed.pop("package_id", None)
            parsed.pop("artifact_id", None)
        evt, pkg = self.coder.execute_command(command, provided_spec=parsed)
        return evt, pkg, usage
