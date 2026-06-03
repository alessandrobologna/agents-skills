---
name: rollout-output-audit
description: Audit Codex session rollout JSONL files for large retained tool outputs that inflate model context, correlate them with token-count events, and recommend scoped AGENTS.md rules. Use when the user asks to analyze Codex session usage, large command output, tool-output bloat, context inflation, compaction cost, or wants rules to reduce future output consumption.
---

# Rollout Output Audit

Use this skill to find command/tool patterns in Codex rollouts that add large outputs to future model input, estimate compaction cost effects, then recommend concise `AGENTS.md` rules.

## Scope Gate

Before auditing, determine scope.

- If the user names a session id, use `--session-id <id>` and do not ask for cwd/global scope.
- If the user names a rollout JSONL path, use `--rollout-path <path>` and do not ask for cwd/global scope.
- If the user names a cwd/path, audit that cwd.
- If the user explicitly asks for global or machine-wide analysis, audit all sessions.
- Otherwise ask: "Should I audit only sessions for the current cwd, or all Codex sessions globally?"
- Recommend current-cwd scope by default; global rollouts may include unrelated private commands, messages, and source snippets.

## Workflow

1. Choose an analysis mode.

- Use `--analysis tool-output` to find large retained tool outputs and recommend `AGENTS.md` rules. This is the default.
- Use `--analysis compactions` when the question is about whether compacting changed per-turn cost or token mix.
- Use `--analysis both` for broad prompts like "analyze this session" or when the user gives only a session id.

2. Run the bundled audit script.

For a single session id:

```bash
python3 "$HOME/.agents/skills/rollout-output-audit/scripts/audit_rollout_outputs.py" \
  --session-id 019e8631-70ce-7793-a225-f1390dc13085 \
  --analysis both \
  --top 20
```

For a single rollout file:

```bash
python3 "$HOME/.agents/skills/rollout-output-audit/scripts/audit_rollout_outputs.py" \
  --rollout-path "$HOME/.codex/sessions/YYYY/MM/DD/rollout-...jsonl" \
  --analysis both \
  --top 20
```

```bash
python3 "$HOME/.agents/skills/rollout-output-audit/scripts/audit_rollout_outputs.py" \
  --analysis tool-output \
  --scope current-cwd \
  --cwd "$PWD" \
  --top 20
```

Use `--scope cwd --cwd <path>` for a named path, or `--scope global` for all rollouts. Add `--since-days <N>` when the user requests a time window.
Exact `--session-id` and `--rollout-path` selectors ignore `--since-days`. Use `--top 0` to include all records.

For compaction cost analysis:

```bash
python3 "$HOME/.agents/skills/rollout-output-audit/scripts/audit_rollout_outputs.py" \
  --analysis compactions \
  --scope current-cwd \
  --cwd "$PWD" \
  --rate-card codex-credits \
  --model auto
```

3. Review the compact report.
- Focus on retained output tokens, high original output counts, repeated command buckets, requested `max_output_tokens`, and the nearest following `token_count`.
- For compactions, compare the before/after turn window, cached-token drop, noncached-token jump, observed after-window cost, and break-even turn.
- JSON compaction records include both descriptive fields (`observed_savings`, `before_average_cost`, `after_average_cost`) and short aliases (`cost_delta`, `before_avg_cost`, `after_avg_cost`). Positive `cost_delta` means the observed after-window was cheaper than projecting the before-window average.
- Do not paste large rollout excerpts into the response.

4. Recommend `AGENTS.md` rules.
- For cwd-scoped audits, prefer the nearest repo/local `AGENTS.md`.
- For global audits, prefer `~/.codex/AGENTS.md` and list repo-specific patterns separately.
- Recommend exact rule text, but edit files only when the user explicitly asks or approves.

## Cost Model

- Token counts come from rollout `event_msg` entries with payload type `token_count`, using `last_token_usage`.
- Compactions are detected from durable `type: "compacted"` rows and nearby UI `context_compacted` events; nearby duplicates are coalesced.
- The zero-token `token_count` emitted immediately after compaction can expose a new context estimate. Treat it as context size signal, not a billed model turn.
- Built-in rates are static OpenAI reference values captured on 2026-06-03. Use `--rate-card codex-credits` for Codex credits, `--rate-card api-usd` for API pricing reference, or pass `--input-rate`, `--cached-input-rate`, and `--output-rate` to override per-1M-token rates.
- Estimates are directional, not exact billing records: they compare observed after-compaction turns against the average cost of the turns immediately before compaction.

## Privacy Rules

- Never replay large tool outputs, transcript text, secrets, or source snippets from rollouts.
- Report command patterns, counts, paths, token impact, and short redacted command previews only.
- The script is read-only: it does not import, execute, or replay commands found in rollouts.

## Resources

- `scripts/audit_rollout_outputs.py`: Parse rollout JSONL files and produce Markdown or JSON audit reports.
