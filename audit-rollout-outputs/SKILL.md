---
name: audit-rollout-outputs
description: Alias for rollout-output-audit. Use when the user invokes $audit-rollout-outputs or asks to audit rollout outputs, calculate cumulative current token usage for this session, inspect token_count usage, compactions, context inflation, or large tool-output bloat in Codex rollout JSONL files.
---

# Audit Rollout Outputs

This is a naming alias for `rollout-output-audit`.

1. Read `../rollout-output-audit/SKILL.md` completely and follow that workflow.
2. Use `../rollout-output-audit/scripts/audit_rollout_outputs.py` for execution.
3. For "calculate the current token usage for this session", run:

```bash
python3 "$HOME/.agents/skills/rollout-output-audit/scripts/audit_rollout_outputs.py" \
  --analysis token-usage
```

The sibling skill owns the implementation and reporting behavior. Do not duplicate the audit script here.
For token-usage requests, report the cumulative session totals first.
