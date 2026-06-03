#!/usr/bin/env python3
"""Audit Codex rollout JSONL files for large retained tool outputs."""

from __future__ import annotations

import argparse
import dataclasses
import datetime as dt
import json
import math
import os
from pathlib import Path
import re
import sys
from typing import Any


ORIGINAL_TOKEN_RE = re.compile(r"Original token count:\s*([0-9][0-9,]*)")
TRUNCATION_RE = re.compile(
    r"(output truncated|truncated.*tokens|additional .* omitted|omitted .* output|token limit)",
    re.IGNORECASE,
)
SECRET_RE = re.compile(
    r"(?i)\b(api[_-]?key|access[_-]?token|auth[_-]?token|bearer|password|secret|token)=([^\s'\";]+)"
)
BEARER_RE = re.compile(r"(?i)\bBearer\s+[A-Za-z0-9._~+/=-]+")
DEFAULT_MODEL = "gpt-5.3-codex"
OPENAI_CODEX_RATE_SOURCE = "https://help.openai.com/en/articles/20001106"
OPENAI_API_RATE_SOURCE = (
    "https://developers.openai.com/api/docs/models/gpt-5.5, "
    "https://developers.openai.com/api/docs/models/gpt-5.4, "
    "https://developers.openai.com/api/docs/models/gpt-5.3-codex"
)


@dataclasses.dataclass(frozen=True)
class RateEntry:
    model: str
    input_rate: float
    cached_input_rate: float
    output_rate: float
    unit: str
    source: str


CODEX_CREDIT_RATES: dict[str, RateEntry] = {
    "gpt-5.5": RateEntry("gpt-5.5", 125.0, 12.5, 750.0, "credits", "OpenAI Codex rate card, 2026-06-03"),
    "gpt-5.4": RateEntry("gpt-5.4", 62.5, 6.25, 375.0, "credits", "OpenAI Codex rate card, 2026-06-03"),
    "gpt-5.4-mini": RateEntry("gpt-5.4-mini", 18.75, 1.875, 113.0, "credits", "OpenAI Codex rate card, 2026-06-03"),
    "gpt-5.3-codex": RateEntry("gpt-5.3-codex", 43.75, 4.375, 350.0, "credits", "OpenAI Codex rate card, 2026-06-03"),
    "gpt-5.2": RateEntry("gpt-5.2", 43.75, 4.375, 350.0, "credits", "OpenAI Codex rate card, 2026-06-03"),
    "gpt-5.2-codex": RateEntry("gpt-5.2-codex", 43.75, 4.375, 350.0, "credits", "OpenAI Codex rate card, 2026-06-03"),
}

API_USD_RATES: dict[str, RateEntry] = {
    "gpt-5.5": RateEntry("gpt-5.5", 5.0, 0.5, 30.0, "USD", "OpenAI API model pricing, 2026-06-03"),
    "gpt-5.4": RateEntry("gpt-5.4", 2.5, 0.25, 15.0, "USD", "OpenAI API model pricing, 2026-06-03"),
    "gpt-5.4-mini": RateEntry("gpt-5.4-mini", 0.75, 0.075, 4.5, "USD", "OpenAI API model pricing, 2026-06-03"),
    "gpt-5.3-codex": RateEntry("gpt-5.3-codex", 1.75, 0.175, 14.0, "USD", "OpenAI API model pricing, 2026-06-03"),
    "gpt-5.2": RateEntry("gpt-5.2", 1.75, 0.175, 14.0, "USD", "OpenAI API model pricing, 2026-06-03"),
    "gpt-5.2-codex": RateEntry("gpt-5.2-codex", 1.75, 0.175, 14.0, "USD", "OpenAI API model pricing, 2026-06-03"),
}


@dataclasses.dataclass
class TokenUsage:
    input_tokens: int | None = None
    cached_input_tokens: int | None = None
    output_tokens: int | None = None
    reasoning_output_tokens: int | None = None
    total_tokens: int | None = None
    model_context_window: int | None = None

    @property
    def noncached_input_tokens(self) -> int | None:
        if self.input_tokens is None or self.cached_input_tokens is None:
            return None
        return max(0, self.input_tokens - self.cached_input_tokens)

    @property
    def is_real_turn(self) -> bool:
        return any(
            (value or 0) > 0
            for value in (self.input_tokens, self.cached_input_tokens, self.output_tokens)
        )


@dataclasses.dataclass
class UsageRecord:
    rollout_path: Path
    session_id: str | None
    timestamp: str | None
    line_no: int
    usage: TokenUsage
    model: str | None

    @property
    def is_real_turn(self) -> bool:
        return self.usage.is_real_turn


@dataclasses.dataclass
class CompactionRecord:
    rollout_path: Path
    session_id: str | None
    timestamp: str | None
    line_no: int
    marker: str
    model: str | None
    replacement_history_len: int | None = None
    context_estimate_tokens: int | None = None


@dataclasses.dataclass
class ToolCall:
    call_id: str
    tool_type: str
    name: str
    timestamp: str | None
    line_no: int
    turn_id: str | None
    turn_cwd: str | None
    command: str | None = None
    workdir: str | None = None
    max_output_tokens: int | None = None
    invocation_label: str | None = None


@dataclasses.dataclass
class ToolEvent:
    timestamp: str | None = None
    line_no: int | None = None
    command: str | None = None
    cwd: str | None = None
    status: str | None = None
    exit_code: int | None = None
    invocation_label: str | None = None


@dataclasses.dataclass
class OutputRecord:
    rollout_path: Path
    session_id: str | None
    timestamp: str | None
    line_no: int
    call_id: str
    tool_name: str
    tool_type: str
    command: str | None
    workdir: str | None
    turn_cwd: str | None
    max_output_tokens: int | None
    retained_chars: int
    retained_tokens_est: int
    original_tokens: int | None
    truncated: bool
    pattern: str
    next_usage: TokenUsage | None = None

    @property
    def rank_tokens(self) -> int:
        return self.retained_tokens_est


@dataclasses.dataclass
class RolloutScan:
    path: Path
    session_id: str | None
    cwds: list[str]
    outputs: list[OutputRecord]
    parse_errors: int
    usages: list[UsageRecord] = dataclasses.field(default_factory=list)
    compactions: list[CompactionRecord] = dataclasses.field(default_factory=list)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Audit Codex rollout JSONL files for large retained tool outputs."
    )
    parser.add_argument(
        "--analysis",
        choices=("tool-output", "compactions"),
        default="tool-output",
        help="Analyze large tool outputs or compaction cost around compact events.",
    )
    parser.add_argument(
        "--scope",
        choices=("current-cwd", "cwd", "global"),
        required=True,
        help="Audit current cwd sessions, a named cwd, or all sessions.",
    )
    parser.add_argument(
        "--cwd",
        default=os.getcwd(),
        help="Cwd path for --scope current-cwd or --scope cwd. Defaults to current directory.",
    )
    parser.add_argument(
        "--sessions-root",
        default=str(Path.home() / ".codex" / "sessions"),
        help="Root containing Codex rollout JSONL files.",
    )
    parser.add_argument(
        "--since-days",
        type=int,
        default=30,
        help="Only scan rollout files modified within this many days. Use 0 for all files.",
    )
    parser.add_argument(
        "--top",
        type=int,
        default=20,
        help="Maximum top findings to print.",
    )
    parser.add_argument(
        "--large-output-tokens",
        type=int,
        default=2000,
        help="Approximate retained/original token threshold for findings.",
    )
    parser.add_argument(
        "--format",
        choices=("markdown", "json"),
        default="markdown",
        help="Output format.",
    )
    parser.add_argument(
        "--apply-suggestions",
        choices=("none", "print-patch"),
        default="none",
        help="Print AGENTS.md patch suggestions; never edits files directly.",
    )
    parser.add_argument(
        "--rate-card",
        choices=("codex-credits", "api-usd"),
        default="codex-credits",
        help="Built-in rate table used for compaction cost estimates.",
    )
    parser.add_argument(
        "--model",
        default="auto",
        help=f"Model for cost estimates. Use 'auto' to read turn_context, defaulting unknown turns to {DEFAULT_MODEL}.",
    )
    parser.add_argument("--input-rate", type=float, default=None, help="Override uncached input rate per 1M tokens.")
    parser.add_argument("--cached-input-rate", type=float, default=None, help="Override cached input rate per 1M tokens.")
    parser.add_argument("--output-rate", type=float, default=None, help="Override output rate per 1M tokens.")
    parser.add_argument(
        "--compaction-before",
        type=int,
        default=3,
        help="Real token_count turns to show before each compaction.",
    )
    parser.add_argument(
        "--compaction-after",
        type=int,
        default=8,
        help="Real token_count turns to show after each compaction.",
    )
    return parser.parse_args()


def load_json_object(line: str) -> dict[str, Any] | None:
    try:
        value = json.loads(line)
    except json.JSONDecodeError:
        return None
    return value if isinstance(value, dict) else None


def parse_maybe_json(value: Any) -> Any:
    if not isinstance(value, str):
        return value
    try:
        return json.loads(value)
    except json.JSONDecodeError:
        return value


def compact_label(value: Any, *, limit: int = 160) -> str:
    if value is None:
        return ""
    if not isinstance(value, str):
        value = json.dumps(value, sort_keys=True, ensure_ascii=False)
    value = " ".join(value.split())
    value = SECRET_RE.sub(r"\1=<redacted>", value)
    value = BEARER_RE.sub("Bearer <redacted>", value)
    if len(value) > limit:
        return value[: limit - 1].rstrip() + "..."
    return value


def token_estimate_from_chars(char_count: int) -> int:
    return max(0, math.ceil(char_count / 4))


def extract_original_tokens(output: str) -> int | None:
    match = ORIGINAL_TOKEN_RE.search(output)
    if not match:
        return None
    return int(match.group(1).replace(",", ""))


def output_text_and_size(value: Any) -> tuple[str, int]:
    if isinstance(value, str):
        return value, len(value)
    text = json.dumps(value, sort_keys=True, ensure_ascii=False)
    return text, len(text)


def int_or_none(value: Any) -> int | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, str) and value.isdigit():
        return int(value)
    return None


def path_matches(candidate: str | None, target: Path | None) -> bool:
    if target is None or not candidate:
        return False
    try:
        path = Path(candidate).expanduser().resolve()
    except OSError:
        return False
    return path == target or target in path.parents


def command_from_args(tool_name: str, args: Any) -> tuple[str | None, str | None, int | None, str | None]:
    if not isinstance(args, dict):
        return None, None, None, None
    command = args.get("cmd") or args.get("command")
    workdir = args.get("workdir") or args.get("cwd")
    max_output_tokens = int_or_none(args.get("max_output_tokens"))

    invocation_label = None
    if tool_name in {"write_stdin"}:
        invocation_label = "interactive session output"
    elif command:
        invocation_label = str(command)
    elif args:
        invocation_label = json.dumps(args, sort_keys=True, ensure_ascii=False)

    return (
        str(command) if command is not None else None,
        str(workdir) if workdir is not None else None,
        max_output_tokens,
        invocation_label,
    )


def event_from_payload(payload: dict[str, Any], line_no: int, timestamp: str | None) -> tuple[str | None, ToolEvent | None]:
    event_type = payload.get("type")
    call_id = payload.get("call_id")
    if not isinstance(call_id, str):
        return None, None

    if event_type == "exec_command_end":
        return call_id, ToolEvent(
            timestamp=timestamp,
            line_no=line_no,
            command=str(payload.get("command")) if payload.get("command") is not None else None,
            cwd=str(payload.get("cwd")) if payload.get("cwd") is not None else None,
            status=str(payload.get("status")) if payload.get("status") is not None else None,
            exit_code=int_or_none(payload.get("exit_code")),
        )

    if event_type == "mcp_tool_call_end":
        invocation = payload.get("invocation")
        label = None
        if isinstance(invocation, dict):
            server = invocation.get("server")
            tool = invocation.get("tool")
            if server or tool:
                label = f"mcp:{server or '?'}/{tool or '?'}"
        return call_id, ToolEvent(timestamp=timestamp, line_no=line_no, invocation_label=label)

    if event_type == "patch_apply_end":
        return call_id, ToolEvent(timestamp=timestamp, line_no=line_no, invocation_label="apply_patch")

    return None, None


def token_usage_from_payload(payload: dict[str, Any]) -> TokenUsage | None:
    info = payload.get("info")
    if not isinstance(info, dict):
        return None
    usage = info.get("last_token_usage")
    if not isinstance(usage, dict):
        return None
    return TokenUsage(
        input_tokens=int_or_none(usage.get("input_tokens")),
        cached_input_tokens=int_or_none(usage.get("cached_input_tokens")),
        output_tokens=int_or_none(usage.get("output_tokens")),
        reasoning_output_tokens=int_or_none(usage.get("reasoning_output_tokens")),
        total_tokens=int_or_none(usage.get("total_tokens")),
        model_context_window=int_or_none(info.get("model_context_window")),
    )


def bucket_command(tool_name: str, command: str | None, invocation_label: str | None) -> str:
    text = (command or invocation_label or "").strip()
    lower = text.lower()
    name = tool_name.lower()

    if name == "write_stdin":
        return "interactive session output"
    if "git diff" in lower and not any(flag in lower for flag in ("--stat", "--name-only", "--numstat")):
        return "large git diff"
    if re.search(r"(^|[|;&]\s*)cat\s+", lower):
        return "unbounded cat"
    if re.search(r"\bls\b[^|;&\n]*\s-[a-z]*r[a-z]*\b", lower):
        return "recursive ls"
    if re.search(r"(^|[|;&]\s*)find\s+", lower):
        return "broad find"
    if re.search(r"(^|[|;&]\s*)rg\s+", lower):
        return "broad rg/search"
    if any(term in lower for term in ("cargo test", "pytest", "npm test", "pnpm test", "just test", "make test", "go test")):
        return "test/build log"
    if any(term in lower for term in ("cargo build", "npm run build", "pnpm build", "just build", "make ")):
        return "test/build log"
    if name.startswith("mcp") or lower.startswith("mcp:"):
        return "mcp tool output"
    if name in {"web_search", "web_search_call"}:
        return "web search output"
    if name in {"tool_search", "tool_search_call"}:
        return "tool search output"
    return "large tool output"


def find_rollout_paths(root: Path, since_days: int) -> list[Path]:
    if not root.exists():
        return []
    cutoff = None
    if since_days > 0:
        cutoff = dt.datetime.now().timestamp() - since_days * 24 * 60 * 60
    paths = []
    for path in root.rglob("rollout-*.jsonl"):
        try:
            stat = path.stat()
        except OSError:
            continue
        if cutoff is not None and stat.st_mtime < cutoff:
            continue
        paths.append(path)
    return sorted(paths, key=lambda p: p.stat().st_mtime, reverse=True)


def scan_rollout(path: Path, threshold_tokens: int, target_cwd: Path | None, include_global: bool) -> RolloutScan:
    session_id = None
    session_cwd = None
    current_turn_id = None
    current_cwd = None
    current_model = None
    cwds: list[str] = []
    calls: dict[str, ToolCall] = {}
    events: dict[str, ToolEvent] = {}
    outputs: list[OutputRecord] = []
    token_events: list[UsageRecord] = []
    compactions: list[CompactionRecord] = []
    parse_errors = 0

    try:
        lines = path.open(encoding="utf-8", errors="replace")
    except OSError:
        return RolloutScan(path, None, [], [], 1)

    with lines:
        for line_no, line in enumerate(lines, start=1):
            item = load_json_object(line)
            if item is None:
                parse_errors += 1
                continue
            item_type = item.get("type")
            timestamp = item.get("timestamp") if isinstance(item.get("timestamp"), str) else None
            payload = item.get("payload")
            payload = payload if isinstance(payload, dict) else {}

            if item_type == "session_meta":
                session_id = payload.get("id") if isinstance(payload.get("id"), str) else session_id
                cwd = payload.get("cwd")
                if isinstance(cwd, str):
                    session_cwd = cwd
                    current_cwd = current_cwd or cwd
                    cwds.append(cwd)
                continue

            if item_type == "turn_context":
                turn_id = payload.get("turn_id")
                cwd = payload.get("cwd")
                model = payload.get("model")
                current_turn_id = turn_id if isinstance(turn_id, str) else current_turn_id
                current_model = model if isinstance(model, str) else current_model
                if isinstance(cwd, str):
                    current_cwd = cwd
                    cwds.append(cwd)
                continue

            if item_type == "event_msg":
                if payload.get("type") == "token_count":
                    usage = token_usage_from_payload(payload)
                    if usage:
                        token_events.append(
                            UsageRecord(
                                rollout_path=path,
                                session_id=session_id,
                                timestamp=timestamp,
                                line_no=line_no,
                                usage=usage,
                                model=current_model,
                            )
                        )
                if payload.get("type") == "context_compacted":
                    compactions.append(
                        CompactionRecord(
                            rollout_path=path,
                            session_id=session_id,
                            timestamp=timestamp,
                            line_no=line_no,
                            marker="context_compacted",
                            model=current_model,
                        )
                    )
                call_id, event = event_from_payload(payload, line_no, timestamp)
                if call_id and event:
                    events[call_id] = event
                continue

            if item_type == "compacted":
                replacement_history = payload.get("replacement_history")
                compactions.append(
                    CompactionRecord(
                        rollout_path=path,
                        session_id=session_id,
                        timestamp=timestamp,
                        line_no=line_no,
                        marker="compacted",
                        model=current_model,
                        replacement_history_len=len(replacement_history)
                        if isinstance(replacement_history, list)
                        else None,
                    )
                )
                continue

            if item_type != "response_item":
                continue

            response_type = payload.get("type")
            call_id = payload.get("call_id")
            if not isinstance(call_id, str):
                continue

            if response_type in {"function_call", "custom_tool_call"}:
                name = payload.get("name")
                name = name if isinstance(name, str) else response_type
                args = parse_maybe_json(payload.get("arguments") if response_type == "function_call" else payload.get("input"))
                command, workdir, max_output_tokens, invocation_label = command_from_args(name, args)
                calls[call_id] = ToolCall(
                    call_id=call_id,
                    tool_type=response_type,
                    name=name,
                    timestamp=timestamp,
                    line_no=line_no,
                    turn_id=current_turn_id,
                    turn_cwd=current_cwd or session_cwd,
                    command=command,
                    workdir=workdir,
                    max_output_tokens=max_output_tokens,
                    invocation_label=invocation_label,
                )
                continue

            if response_type not in {"function_call_output", "custom_tool_call_output"}:
                continue

            output_text, retained_chars = output_text_and_size(payload.get("output"))
            retained_tokens_est = token_estimate_from_chars(retained_chars)
            original_tokens = extract_original_tokens(output_text)
            truncated = bool(TRUNCATION_RE.search(output_text)) or (
                original_tokens is not None and original_tokens > retained_tokens_est + 200
            )
            if (
                retained_tokens_est < threshold_tokens
                and (original_tokens is None or original_tokens < threshold_tokens)
                and not truncated
            ):
                continue

            call = calls.get(call_id)
            event = events.get(call_id)
            tool_name = call.name if call else response_type
            command = (event.command if event and event.command else None) or (call.command if call else None)
            workdir = (event.cwd if event and event.cwd else None) or (call.workdir if call else None)
            turn_cwd = call.turn_cwd if call else (current_cwd or session_cwd)
            invocation_label = (
                (event.invocation_label if event else None)
                or (call.invocation_label if call else None)
                or command
                or tool_name
            )

            if not include_global and not (
                path_matches(turn_cwd, target_cwd) or path_matches(workdir, target_cwd)
            ):
                continue

            outputs.append(
                OutputRecord(
                    rollout_path=path,
                    session_id=session_id,
                    timestamp=timestamp,
                    line_no=line_no,
                    call_id=call_id,
                    tool_name=tool_name,
                    tool_type=response_type,
                    command=command or invocation_label,
                    workdir=workdir,
                    turn_cwd=turn_cwd,
                    max_output_tokens=call.max_output_tokens if call else None,
                    retained_chars=retained_chars,
                    retained_tokens_est=retained_tokens_est,
                    original_tokens=original_tokens,
                    truncated=truncated,
                    pattern=bucket_command(tool_name, command, invocation_label),
                )
            )

    for output in outputs:
        output.next_usage = next((record.usage for record in token_events if record.line_no > output.line_no), None)

    for compaction in compactions:
        estimate = next(
            (
                record.usage.total_tokens
                for record in token_events
                if record.line_no > compaction.line_no and not record.is_real_turn
            ),
            None,
        )
        compaction.context_estimate_tokens = estimate

    return RolloutScan(path, session_id, sorted(set(cwds)), outputs, parse_errors, token_events, compactions)


def nearest_agents_md(cwd: Path | None) -> Path | None:
    if cwd is None:
        return None
    try:
        path = cwd.expanduser().resolve()
    except OSError:
        return None
    for candidate_dir in [path, *path.parents]:
        if candidate_dir == Path.home().resolve():
            break
        candidate = candidate_dir / "AGENTS.md"
        if candidate.exists():
            return candidate
    return path / "AGENTS.md"


def fmt_int(value: int | None) -> str:
    if value is None:
        return "-"
    return f"{value:,}"


def fmt_usage(usage: TokenUsage | None) -> str:
    if usage is None:
        return "next token_count: -"
    parts = [
        f"input={fmt_int(usage.input_tokens)}",
        f"cached={fmt_int(usage.cached_input_tokens)}",
        f"noncached={fmt_int(usage.noncached_input_tokens)}",
        f"output={fmt_int(usage.output_tokens)}",
        f"total={fmt_int(usage.total_tokens)}",
    ]
    return "next token_count: " + ", ".join(parts)


def fmt_float(value: float | None, *, precision: int = 4) -> str:
    if value is None:
        return "-"
    return f"{value:,.{precision}f}"


def normalize_model(model: str | None) -> str | None:
    if not model:
        return None
    normalized = model.lower().replace("_", "-")
    for known in sorted({*CODEX_CREDIT_RATES, *API_USD_RATES}, key=len, reverse=True):
        if normalized == known or normalized.startswith(f"{known}-"):
            return known
    return normalized


def rate_table(name: str) -> dict[str, RateEntry]:
    return API_USD_RATES if name == "api-usd" else CODEX_CREDIT_RATES


def resolve_rate(model: str | None, args: argparse.Namespace) -> tuple[RateEntry, bool]:
    table = rate_table(args.rate_card)
    requested_model = None if args.model == "auto" else args.model
    model_key = normalize_model(requested_model or model) or DEFAULT_MODEL
    assumed = model_key not in table
    base = table.get(model_key) or table[DEFAULT_MODEL]
    if any(
        override is not None
        for override in (args.input_rate, args.cached_input_rate, args.output_rate)
    ):
        return (
            RateEntry(
                model=base.model,
                input_rate=args.input_rate if args.input_rate is not None else base.input_rate,
                cached_input_rate=args.cached_input_rate
                if args.cached_input_rate is not None
                else base.cached_input_rate,
                output_rate=args.output_rate if args.output_rate is not None else base.output_rate,
                unit=base.unit,
                source=f"{base.source}; command-line overrides applied",
            ),
            assumed,
        )
    return base, assumed


def estimate_cost(usage: TokenUsage, rate: RateEntry) -> float:
    input_tokens = usage.input_tokens or 0
    cached_tokens = usage.cached_input_tokens or 0
    noncached_tokens = max(0, input_tokens - cached_tokens)
    output_tokens = usage.output_tokens or 0
    return (
        noncached_tokens * rate.input_rate
        + cached_tokens * rate.cached_input_rate
        + output_tokens * rate.output_rate
    ) / 1_000_000


def coalesced_compactions(scans: list[RolloutScan]) -> list[CompactionRecord]:
    compactions: list[CompactionRecord] = []
    for scan in scans:
        durable_lines = [item.line_no for item in scan.compactions if item.marker == "compacted"]
        for item in scan.compactions:
            if item.marker == "context_compacted" and any(
                0 < item.line_no - durable_line <= 5 for durable_line in durable_lines
            ):
                continue
            compactions.append(item)
    return sorted(compactions, key=lambda item: (str(item.rollout_path), item.line_no))


def real_usages_for_scan(scan: RolloutScan) -> list[UsageRecord]:
    return [record for record in scan.usages if record.is_real_turn]


def usage_cost_row(record: UsageRecord, rel: str, args: argparse.Namespace) -> dict[str, Any]:
    rate, assumed = resolve_rate(record.model, args)
    usage = record.usage
    return {
        "rel": rel,
        "line_no": record.line_no,
        "timestamp": record.timestamp,
        "model": normalize_model(record.model) or rate.model,
        "model_assumed": assumed or record.model is None,
        "input_tokens": usage.input_tokens,
        "cached_input_tokens": usage.cached_input_tokens,
        "noncached_input_tokens": usage.noncached_input_tokens,
        "output_tokens": usage.output_tokens,
        "total_tokens": usage.total_tokens,
        "cost": estimate_cost(usage, rate),
        "unit": rate.unit,
    }


def sum_sign(value: int) -> str:
    if value > 0:
        return f"+{value}"
    return str(value)


def average(values: list[float]) -> float | None:
    if not values:
        return None
    return sum(values) / len(values)


def scan_matches_scope(scan: RolloutScan, target_cwd: Path | None, include_global: bool) -> bool:
    if include_global:
        return True
    return any(path_matches(cwd, target_cwd) for cwd in scan.cwds)


def scoped_scans(scans: list[RolloutScan], args: argparse.Namespace, target_cwd: Path | None) -> list[RolloutScan]:
    include_global = args.scope == "global"
    return [scan for scan in scans if scan_matches_scope(scan, target_cwd, include_global)]


def compaction_windows(
    scan: RolloutScan,
    compaction: CompactionRecord,
    args: argparse.Namespace,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    usages = real_usages_for_scan(scan)
    before_records = [record for record in usages if record.line_no < compaction.line_no]
    after_records = [record for record in usages if record.line_no > compaction.line_no]
    before_window = before_records[-args.compaction_before:] if args.compaction_before > 0 else []
    after_window = after_records[: args.compaction_after] if args.compaction_after > 0 else []
    before = [
        usage_cost_row(record, f"-{len(before_window) - index}", args)
        for index, record in enumerate(before_window)
    ]
    after = [
        usage_cost_row(record, f"+{index}", args)
        for index, record in enumerate(after_window, start=1)
    ]
    return before, after


def cost_average(rows: list[dict[str, Any]]) -> float | None:
    return average([row["cost"] for row in rows])


def token_average(rows: list[dict[str, Any]], key: str) -> float | None:
    return average([(row.get(key) or 0) for row in rows])


def break_even_after_turn(before_avg_cost: float | None, after: list[dict[str, Any]]) -> int | None:
    if before_avg_cost is None:
        return None
    cumulative_delta = 0.0
    for index, row in enumerate(after, start=1):
        cumulative_delta += before_avg_cost - row["cost"]
        if cumulative_delta > 0:
            return index
    return None


def compaction_analysis_record(
    scan: RolloutScan,
    compaction: CompactionRecord,
    args: argparse.Namespace,
) -> dict[str, Any]:
    before, after = compaction_windows(scan, compaction, args)
    before_avg_cost = cost_average(before)
    after_avg_cost = cost_average(after)
    after_total_cost = sum(row["cost"] for row in after)
    projected_after_cost = before_avg_cost * len(after) if before_avg_cost is not None else None
    observed_savings = (
        projected_after_cost - after_total_cost
        if projected_after_cost is not None
        else None
    )
    rate, assumed = resolve_rate(compaction.model, args)
    return {
        "rollout_path": str(compaction.rollout_path),
        "session_id": compaction.session_id or scan.session_id,
        "timestamp": compaction.timestamp,
        "line_no": compaction.line_no,
        "marker": compaction.marker,
        "model": normalize_model(compaction.model) or rate.model,
        "model_assumed": assumed or compaction.model is None,
        "replacement_history_len": compaction.replacement_history_len,
        "context_estimate_tokens": compaction.context_estimate_tokens,
        "before": before,
        "after": after,
        "before_average_cost": before_avg_cost,
        "after_average_cost": after_avg_cost,
        "after_total_cost": after_total_cost,
        "projected_after_cost_without_compaction": projected_after_cost,
        "observed_savings": observed_savings,
        "break_even_after_turn": break_even_after_turn(before_avg_cost, after),
        "before_average_cached_tokens": token_average(before, "cached_input_tokens"),
        "after_average_cached_tokens": token_average(after, "cached_input_tokens"),
        "before_average_noncached_tokens": token_average(before, "noncached_input_tokens"),
        "after_average_noncached_tokens": token_average(after, "noncached_input_tokens"),
        "unit": rate.unit,
    }


def compaction_analysis_records(
    scans: list[RolloutScan],
    args: argparse.Namespace,
    target_cwd: Path | None,
) -> list[dict[str, Any]]:
    scans_by_path = {scan.path: scan for scan in scoped_scans(scans, args, target_cwd)}
    records = []
    for compaction in coalesced_compactions(list(scans_by_path.values())):
        scan = scans_by_path.get(compaction.rollout_path)
        if scan is None:
            continue
        records.append(compaction_analysis_record(scan, compaction, args))
    return records


def rate_entries_for_report(args: argparse.Namespace) -> list[RateEntry]:
    table = rate_table(args.rate_card)
    return [table[key] for key in sorted(table)]


def rate_card_markdown(args: argparse.Namespace) -> list[str]:
    lines = [
        "| model | input / 1M | cached input / 1M | output / 1M | unit |",
        "| --- | ---: | ---: | ---: | --- |",
    ]
    for entry in rate_entries_for_report(args):
        lines.append(
            f"| `{entry.model}` | {fmt_float(entry.input_rate, precision=3)} | "
            f"{fmt_float(entry.cached_input_rate, precision=3)} | "
            f"{fmt_float(entry.output_rate, precision=3)} | {entry.unit} |"
        )
    return lines


def rate_source_for_report(args: argparse.Namespace) -> str:
    if args.rate_card == "api-usd":
        return OPENAI_API_RATE_SOURCE
    return OPENAI_CODEX_RATE_SOURCE


def recommended_rules(pattern_counts: dict[str, int], high_max_output_count: int) -> list[str]:
    rules: list[str] = []
    patterns = set(pattern_counts)

    if "broad rg/search" in patterns:
        rules.append("For broad searches, scope `rg` to target paths and use `sed -n`/`head` or small `max_output_tokens` before expanding.")
    if "unbounded cat" in patterns:
        rules.append("Avoid whole-file `cat`; use `sed -n`, `head`, `tail`, or `wc -l` plus focused slices.")
    if "recursive ls" in patterns or "broad find" in patterns:
        rules.append("Avoid recursive `ls`/broad `find`; prefer `rg --files <scope> | head -200` or bounded `find -maxdepth` queries.")
    if "large git diff" in patterns:
        rules.append("Start diffs with `git diff --stat`, `--name-only`, or `--numstat`, then inspect specific files or hunks.")
    if "test/build log" in patterns:
        rules.append("For tests/builds, run scoped commands and capture failing cases plus exact errors instead of full verbose logs.")
    if "interactive session output" in patterns:
        rules.append("For interactive sessions, poll with small `max_output_tokens` and stop once the relevant status or error is captured.")
    if high_max_output_count:
        rules.append("Use exploratory `max_output_tokens` conservatively, then raise it only after locating the needed slice.")
    if not rules:
        rules.append("Keep exploratory tool output narrow; measure or locate first, then fetch focused slices.")
    return rules


def report_markdown(
    scans: list[RolloutScan],
    findings: list[OutputRecord],
    args: argparse.Namespace,
    target_cwd: Path | None,
) -> str:
    matched_rollouts = sum(1 for scan in scans if scan.outputs)
    parse_errors = sum(scan.parse_errors for scan in scans)
    pattern_counts: dict[str, int] = {}
    high_max_output_count = 0
    for finding in findings:
        pattern_counts[finding.pattern] = pattern_counts.get(finding.pattern, 0) + 1
        if finding.max_output_tokens and finding.max_output_tokens >= args.large_output_tokens:
            high_max_output_count += 1

    if args.scope == "global":
        target = Path.home() / ".codex" / "AGENTS.md"
    else:
        target = nearest_agents_md(target_cwd)

    lines = [
        "# Rollout Output Audit",
        "",
        f"- Scope: `{args.scope}`" + (f" (`{target_cwd}`)" if target_cwd else ""),
        f"- Sessions root: `{Path(args.sessions_root).expanduser()}`",
        f"- Since days: `{args.since_days}`",
        f"- Rollout files scanned: `{len(scans)}`",
        f"- Rollouts with findings: `{matched_rollouts}`",
        f"- Findings above threshold: `{len(findings)}`",
        f"- Large-output threshold: `{args.large_output_tokens:,}` approximate tokens",
    ]
    if parse_errors:
        lines.append(f"- JSONL parse errors skipped: `{parse_errors}`")
    if target:
        lines.append(f"- Recommended AGENTS.md target: `{target}`")

    lines.extend(["", "## Pattern Summary", ""])
    if pattern_counts:
        for pattern, count in sorted(pattern_counts.items(), key=lambda item: (-item[1], item[0])):
            lines.append(f"- `{pattern}`: {count}")
    else:
        lines.append("- No large retained outputs found for this scope/window.")

    if findings:
        lines.extend(["", f"## Top {min(args.top, len(findings))} Tool Outputs", ""])
        for index, finding in enumerate(findings[: args.top], start=1):
            command = compact_label(finding.command or finding.tool_name)
            cwd = compact_label(finding.workdir or finding.turn_cwd or "", limit=120)
            lines.extend(
                [
                    f"{index}. `{finding.pattern}` via `{finding.tool_name}`",
                    f"   - retained: ~{fmt_int(finding.retained_tokens_est)} tokens / {fmt_int(finding.retained_chars)} chars",
                    f"   - original: {fmt_int(finding.original_tokens)} tokens; requested max_output_tokens: {fmt_int(finding.max_output_tokens)}; truncated: `{str(finding.truncated).lower()}`",
                    f"   - {fmt_usage(finding.next_usage)}",
                    f"   - cwd/workdir: `{cwd or '-'}`",
                    f"   - command: `{command or '-'}`",
                    f"   - rollout: `{finding.rollout_path}:{finding.line_no}`",
                ]
            )

    lines.extend(["", "## Recommended Rules", ""])
    for rule in recommended_rules(pattern_counts, high_max_output_count):
        lines.append(f"- {rule}")

    if args.apply_suggestions == "print-patch" and target:
        lines.extend(["", "## Suggested AGENTS.md Patch Text", ""])
        lines.append("Add or merge these bullets into the relevant tool-output guidance:")
        for rule in recommended_rules(pattern_counts, high_max_output_count):
            lines.append(f"- {rule}")

    return "\n".join(lines) + "\n"


def report_json(
    scans: list[RolloutScan],
    findings: list[OutputRecord],
    args: argparse.Namespace,
    target_cwd: Path | None,
) -> str:
    pattern_counts: dict[str, int] = {}
    high_max_output_count = 0
    for finding in findings:
        pattern_counts[finding.pattern] = pattern_counts.get(finding.pattern, 0) + 1
        if finding.max_output_tokens and finding.max_output_tokens >= args.large_output_tokens:
            high_max_output_count += 1

    payload = {
        "scope": args.scope,
        "cwd": str(target_cwd) if target_cwd else None,
        "sessions_root": str(Path(args.sessions_root).expanduser()),
        "since_days": args.since_days,
        "rollout_files_scanned": len(scans),
        "rollouts_with_findings": sum(1 for scan in scans if scan.outputs),
        "findings_count": len(findings),
        "large_output_threshold_tokens": args.large_output_tokens,
        "pattern_summary": pattern_counts,
        "recommended_rules": recommended_rules(pattern_counts, high_max_output_count),
        "findings": [
            {
                "pattern": finding.pattern,
                "tool_name": finding.tool_name,
                "retained_tokens_est": finding.retained_tokens_est,
                "retained_chars": finding.retained_chars,
                "original_tokens": finding.original_tokens,
                "truncated": finding.truncated,
                "max_output_tokens": finding.max_output_tokens,
                "cwd_or_workdir": finding.workdir or finding.turn_cwd,
                "command_preview": compact_label(finding.command or finding.tool_name),
                "rollout_path": str(finding.rollout_path),
                "line_no": finding.line_no,
                "next_token_usage": dataclasses.asdict(finding.next_usage) if finding.next_usage else None,
            }
            for finding in findings[: args.top]
        ],
    }
    return json.dumps(payload, indent=2, sort_keys=True) + "\n"


def report_compactions_markdown(
    scans: list[RolloutScan],
    args: argparse.Namespace,
    target_cwd: Path | None,
) -> str:
    scoped = scoped_scans(scans, args, target_cwd)
    parse_errors = sum(scan.parse_errors for scan in scans)
    records = compaction_analysis_records(scans, args, target_cwd)
    shown = records[: args.top] if args.top > 0 else records
    default_rate, _ = resolve_rate(None, args)
    target = "all sessions" if args.scope == "global" else f"`{target_cwd}`"

    lines = [
        "# Rollout Compaction Cost Audit",
        "",
        f"- Scope: `{args.scope}` ({target})",
        f"- Sessions root: `{Path(args.sessions_root).expanduser()}`",
        f"- Since days: `{args.since_days}`",
        f"- Rollout files scanned: `{len(scans)}`",
        f"- Rollouts matching scope: `{len(scoped)}`",
        f"- Compactions found: `{len(records)}`",
        f"- Window: `{args.compaction_before}` real token_count turns before, `{args.compaction_after}` after",
        f"- Rate card: `{args.rate_card}`; model: `{args.model}`; default unknown model: `{DEFAULT_MODEL}`",
    ]
    if parse_errors:
        lines.append(f"- JSONL parse errors skipped: `{parse_errors}`")
    if any(record["model_assumed"] for record in records):
        lines.append("- Some rows use the default model because the rollout did not expose a model near that turn.")
    lines.extend(
        [
            "",
            "Cost formula: `(uncached_input * input_rate + cached_input * cached_rate + output * output_rate) / 1,000,000`.",
            "The zero-token `token_count` emitted immediately after compaction is reported as a context estimate, not billed as a turn.",
            f"Rates below are static references captured on 2026-06-03; refresh OpenAI pricing before treating this as billing truth. Unit: `{default_rate.unit}`.",
            f"Rate source: {rate_source_for_report(args)}",
            "",
            "## Reference Rate Card",
            "",
            *rate_card_markdown(args),
        ]
    )

    if not shown:
        lines.extend(["", "## Compactions", "", "No compaction events found for this scope/window."])
        return "\n".join(lines) + "\n"

    lines.extend(["", f"## Compactions ({len(shown)} shown)", ""])
    for index, record in enumerate(shown, start=1):
        before_avg = record["before_average_cost"]
        after_avg = record["after_average_cost"]
        projected = record["projected_after_cost_without_compaction"]
        observed_savings = record["observed_savings"]
        break_even = record["break_even_after_turn"]
        cached_delta = None
        if record["before_average_cached_tokens"] is not None and record["after_average_cached_tokens"] is not None:
            cached_delta = record["after_average_cached_tokens"] - record["before_average_cached_tokens"]
        noncached_delta = None
        if (
            record["before_average_noncached_tokens"] is not None
            and record["after_average_noncached_tokens"] is not None
        ):
            noncached_delta = (
                record["after_average_noncached_tokens"]
                - record["before_average_noncached_tokens"]
            )

        lines.extend(
            [
                f"### {index}. `{record['session_id'] or '-'}` line {record['line_no']}",
                "",
                f"- Marker: `{record['marker']}`; model: `{record['model']}`; rollout: `{record['rollout_path']}`",
                f"- Post-compaction context estimate: `{fmt_int(record['context_estimate_tokens'])}` tokens",
                f"- Before avg: `{fmt_float(before_avg)}` {record['unit']} / turn; after avg: `{fmt_float(after_avg)}` {record['unit']} / turn",
                f"- Observed after-window cost: `{fmt_float(record['after_total_cost'])}` {record['unit']}; projected at before avg: `{fmt_float(projected)}` {record['unit']}; delta: `{fmt_float(observed_savings)}` {record['unit']}",
                f"- Avg cached-token delta after compaction: `{fmt_float(cached_delta, precision=0)}`; avg noncached-token delta: `{fmt_float(noncached_delta, precision=0)}`",
                f"- Break-even within observed after-window: `{break_even if break_even is not None else '-'}` turns",
                "",
                "| rel | line | input | cached | noncached | output | cost |",
                "| ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
            ]
        )
        for row in [*record["before"], *record["after"]]:
            lines.append(
                f"| {row['rel']} | {row['line_no']} | {fmt_int(row['input_tokens'])} | "
                f"{fmt_int(row['cached_input_tokens'])} | {fmt_int(row['noncached_input_tokens'])} | "
                f"{fmt_int(row['output_tokens'])} | {fmt_float(row['cost'])} |"
            )
        lines.append("")

    return "\n".join(lines) + "\n"


def report_compactions_json(
    scans: list[RolloutScan],
    args: argparse.Namespace,
    target_cwd: Path | None,
) -> str:
    records = compaction_analysis_records(scans, args, target_cwd)
    payload = {
        "analysis": "compactions",
        "scope": args.scope,
        "cwd": str(target_cwd) if target_cwd else None,
        "sessions_root": str(Path(args.sessions_root).expanduser()),
        "since_days": args.since_days,
        "rollout_files_scanned": len(scans),
        "rollouts_matching_scope": len(scoped_scans(scans, args, target_cwd)),
        "compactions_count": len(records),
        "compaction_before": args.compaction_before,
        "compaction_after": args.compaction_after,
        "rate_card": args.rate_card,
        "model": args.model,
        "default_model": DEFAULT_MODEL,
        "rates_per_million": [dataclasses.asdict(entry) for entry in rate_entries_for_report(args)],
        "compactions": records[: args.top] if args.top > 0 else records,
    }
    return json.dumps(payload, indent=2, sort_keys=True) + "\n"


def main() -> int:
    args = parse_args()
    sessions_root = Path(args.sessions_root).expanduser()
    target_cwd = None
    include_global = args.scope == "global"
    if not include_global:
        target_cwd = Path(args.cwd).expanduser().resolve()

    paths = find_rollout_paths(sessions_root, args.since_days)
    scans = [
        scan_rollout(
            path,
            threshold_tokens=args.large_output_tokens,
            target_cwd=target_cwd,
            include_global=include_global,
        )
        for path in paths
    ]
    if args.analysis == "compactions":
        if args.format == "json":
            sys.stdout.write(report_compactions_json(scans, args, target_cwd))
        else:
            sys.stdout.write(report_compactions_markdown(scans, args, target_cwd))
        return 0

    findings = [output for scan in scans for output in scan.outputs]
    findings.sort(
        key=lambda item: (
            item.rank_tokens,
            item.original_tokens or 0,
            item.retained_chars,
        ),
        reverse=True,
    )

    if args.format == "json":
        sys.stdout.write(report_json(scans, findings, args, target_cwd))
    else:
        sys.stdout.write(report_markdown(scans, findings, args, target_cwd))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
