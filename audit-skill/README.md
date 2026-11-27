# Audit Skill

An agent skill for source-grounded audits: post-mortems, claim checks,
experiment reviews, fork reviews, provenance reviews, and
"does this work actually match what was asked?" investigations.

I use it when I need an agent to check whether a result is real: whether the
code that ran is the code the write-up describes, whether a benchmark leaked
the target, whether a branch inherited a problem from its parent, and what can
honestly be claimed afterwards.

The core rule: never audit from filenames, summaries, labels, or claimed
results alone. Audit from execution paths, logs, timestamps, data lineage, and
reproducible artifacts.

## What it does

- Freezes the audit question and the allowed actions (read-only by default)
- Builds an artifact inventory and a timestamp/provenance timeline
- Defines valid/invalid evidence gates *before* judging results
- Inspects mechanisms, not labels
- Classifies each component as `clean`, `valid-but-different`, `partial`,
  `contaminated`, `invalid`, or `unknown`
- Writes correction-ready claims ("Do not say / Say / Because")
- Produces and verifies audit artifacts (memo, CSV table, post-mortem)

## Files

- `SKILL.md` - the skill itself (Claude Code / Agent Skills format, with YAML frontmatter)
- `agents/openai.yaml` - display metadata for Codex-style agents

## Install

Copy the `audit-skill/` folder into your agent's skills directory, for example
`~/.claude/skills/audit-skill/` for Claude Code, then ask the agent to
"audit this".
