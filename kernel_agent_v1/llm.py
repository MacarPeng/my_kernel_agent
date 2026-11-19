"""DeepSeek LLM interface and structured Researcher/Coder adapters (interface-ready, offline-safe)."""

from __future__ import annotations

from abc import ABC, abstractmethod
import asyncio
from dataclasses import dataclass, field
import json
import os
from typing import Any, Callable
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


@dataclass(frozen=True)
class LLMResponse:
    content: str
    model: str
    usage: LLMUsage = field(default_factory=LLMUsage)
    raw: dict[str, Any] = field(default_factory=dict)

    def parse_json(self) -> dict[str, Any]:
        text = self.content.strip()
        if text.startswith("```"):
            lines = text.splitlines()
            if lines[0].startswith("```"):
                lines = lines[1:]
            if lines and lines[-1].strip().startswith("```"):
                lines = lines[:-1]
            text = "\n".join(lines).strip()
        return json.loads(text)


class LLMClient(ABC):
    """Abstract async LLM client interface."""

    @abstractmethod
    async def complete(
        self, messages: list[LLMMessage], *, json_mode: bool = True
    ) -> LLMResponse:
        """Send chat completion request and return structured response + token cost."""


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

        content = raw["choices"][0]["message"]["content"]
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

    def __init__(self, llm: LLMClient, leader: Leader) -> None:
        self.llm = llm
        self.leader = leader

    async def plan_from_evidence(
        self,
        researcher_id: str,
        direction_id: str,
        deliveries: list[EvidenceDelivery],
        default_target: TargetRef,
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
            "You are a KernelLab Researcher optimizing GDN prefill Triton/PyTorch kernels. "
            "Separate observation, explanation, prediction, and emit valid JSON with keys: "
            "observation, explanation, prediction, hypothesis, command_kind "
            "(one of implement_kernel, write_ncu_capture, write_probe, write_compiler_analysis), "
            "instruction, deliverable, allowed_paths (use ['kernel.py'] for implement_kernel), "
            "workload_ids, preserve_target (false for implement_kernel). "
            "For implement_kernel, kernel.py MUST define "
            "run(q, k, v, state, A_log, a, dt_bias, b, cu_seqlens, scale) -> (output, new_state)."
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
        resp = await self.llm.complete(
            [
                LLMMessage("system", sys_prompt),
                LLMMessage("user", json.dumps(user_payload, ensure_ascii=False)),
            ],
            json_mode=True,
        )
        parsed = resp.parse_json()

        evidence_refs = (lookup.lookup_id,) + tuple(d.receipt_id for d in deliveries)
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
            kind=str(parsed.get("command_kind", CommandKind.IMPLEMENT_KERNEL.value)),
            research_id=rid,
            direction_id=direction_id,
            target=default_target,
            hypothesis=str(parsed.get("hypothesis", "optimization hypothesis")),
            evidence_refs=evidence_refs,
            instruction=str(parsed.get("instruction", "implement candidate")),
            deliverable=str(parsed.get("deliverable", "kernel package")),
            reply_to=researcher_id,
            workload_ids=tuple(parsed.get("workload_ids", ())),
            allowed_paths=tuple(parsed.get("allowed_paths", ())),
            preserve_target=bool(parsed.get("preserve_target", True)),
        )
        return cmd, parsed, resp.usage


class DeepSeekCoderAdapter:
    """Translates Leader Command into constraint-checked immutable CodePackage via DeepSeek."""

    def __init__(self, llm: LLMClient, coder: Coder) -> None:
        self.llm = llm
        self.coder = coder

    async def implement(
        self, command: Command
    ) -> tuple[Event, CodePackage | None, LLMUsage]:
        sys_prompt = (
            "You are KernelLab Coder. Output valid JSON with keys: artifact_id, entrypoint "
            "('kernel.py' for implement_kernel), files (dict of rel_path -> complete Python code), "
            "executable_stages. Strictly stay inside allowed_paths. Only import from "
            "triton, torch, math, typing, functools (no os/sys/eval/open). For implement_kernel, "
            "kernel.py MUST define run(q, k, v, state, A_log, a, dt_bias, b, cu_seqlens, scale) "
            "returning (output_bf16, new_state_fp32)."
        )
        resp = await self.llm.complete(
            [
                LLMMessage("system", sys_prompt),
                LLMMessage("user", json.dumps(command.to_dict(), ensure_ascii=False)),
            ],
            json_mode=True,
        )
        parsed = resp.parse_json()
        evt, pkg = self.coder.execute_command(command, provided_spec=parsed)
        return evt, pkg, resp.usage
