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


@dataclasses.dataclass
class TokenUsage:
    input_tokens: int | None = None
    cached_input_tokens: int | None = None
    output_tokens: int | None = None
    total_tokens: int | None = None
    model_context_window: int | None = None

    @property
    def noncached_input_tokens(self) -> int | None:
        if self.input_tokens is None or self.cached_input_tokens is None:
            return None
        return max(0, self.input_tokens - self.cached_input_tokens)


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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Audit Codex rollout JSONL files for large retained tool outputs."
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
    cwds: list[str] = []
    calls: dict[str, ToolCall] = {}
    events: dict[str, ToolEvent] = {}
    outputs: list[OutputRecord] = []
    token_events: list[tuple[int, TokenUsage]] = []
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
                current_turn_id = turn_id if isinstance(turn_id, str) else current_turn_id
                if isinstance(cwd, str):
                    current_cwd = cwd
                    cwds.append(cwd)
                continue

            if item_type == "event_msg":
                if payload.get("type") == "token_count":
                    usage = token_usage_from_payload(payload)
                    if usage:
                        token_events.append((line_no, usage))
                call_id, event = event_from_payload(payload, line_no, timestamp)
                if call_id and event:
                    events[call_id] = event
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
        output.next_usage = next((usage for line_no, usage in token_events if line_no > output.line_no), None)

    return RolloutScan(path, session_id, sorted(set(cwds)), outputs, parse_errors)


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
