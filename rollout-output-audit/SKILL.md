---
name: rollout-output-audit
description: Audit Codex session rollout JSONL files for large retained tool outputs that inflate model context, correlate them with token-count events, and recommend scoped AGENTS.md rules. Use when the user asks to analyze Codex session usage, large command output, tool-output bloat, context inflation, compaction cost, or wants rules to reduce future output consumption.
---

# Rollout Output Audit

Use this skill to find command/tool patterns in Codex rollouts that add large outputs to future model input, then recommend concise `AGENTS.md` rules.

## Scope Gate

Before auditing, determine scope.

- If the user names a cwd/path, audit that cwd.
- If the user explicitly asks for global or machine-wide analysis, audit all sessions.
- Otherwise ask: "Should I audit only sessions for the current cwd, or all Codex sessions globally?"
- Recommend current-cwd scope by default; global rollouts may include unrelated private commands, messages, and source snippets.

## Workflow

1. Run the bundled audit script.

```bash
python3 "$HOME/.agents/skills/rollout-output-audit/scripts/audit_rollout_outputs.py" \
  --scope current-cwd \
  --cwd "$PWD" \
  --top 20
```

Use `--scope cwd --cwd <path>` for a named path, or `--scope global` for all rollouts. Add `--since-days <N>` when the user requests a time window.

2. Review the compact report.
- Focus on retained output tokens, high original output counts, repeated command buckets, requested `max_output_tokens`, and the nearest following `token_count`.
- Do not paste large rollout excerpts into the response.

3. Recommend `AGENTS.md` rules.
- For cwd-scoped audits, prefer the nearest repo/local `AGENTS.md`.
- For global audits, prefer `~/.codex/AGENTS.md` and list repo-specific patterns separately.
- Recommend exact rule text, but edit files only when the user explicitly asks or approves.

## Privacy Rules

- Never replay large tool outputs, transcript text, secrets, or source snippets from rollouts.
- Report command patterns, counts, paths, token impact, and short redacted command previews only.
- The script is read-only: it does not import, execute, or replay commands found in rollouts.

## Resources

- `scripts/audit_rollout_outputs.py`: Parse rollout JSONL files and produce Markdown or JSON audit reports.
