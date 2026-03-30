# FerroML Plugin Design — Claude Code Integration

**Status:** Approved
**Date:** 2026-03-29
**Location:** `ferroml-plugin/` (separate repo from `ferroml`)

## Purpose

Make FerroML the easiest way for anyone to do machine learning from Claude Code. A non-technical user installs the plugin, says "I have sales data, predict next month's revenue," and Claude handles everything — data profiling, model selection, training, diagnostics, and a plain-language report.

## Architecture Decision

**Plugin (Option A)** — no MCP server.

The plugin bundles a skill, 5 commands, 3 hooks, and 1 agent. Claude imports `ferroml` directly and runs Python code. No server process, no port management, no startup latency.

MCP was rejected because FerroML is a local Python library. Claude can call it directly — an MCP server would wrap a wrapper. MCP shines for external systems (databases, APIs, cloud services), not local libraries.

## Target Users

- **Non-technical:** "I have a CSV of sales data, help me predict next month's revenue"
- **Developers new to ML:** "Fit a model and tell me which features matter"
- **ML engineers:** "Compare gradient boosting variants with statistical significance"

The skill adapts language and depth based on detected technical level.

## Plugin Structure

```
ferroml-plugin/
├── .claude-plugin/
│   └── plugin.json
├── commands/
│   ├── ml.md                    # /ml [data_file]
│   ├── ml-eval.md               # /ml-eval [model]
│   ├── ml-report.md             # /ml-report [audience]
│   ├── ml-status.md             # /ml-status [prod_data]
│   └── ml-compare.md            # /ml-compare [data_file]
├── skills/
│   └── ferroml-ml/
│       ├── SKILL.md             # Core decision logic, 6 workflows
│       ├── references/          # 14 reference docs
│       ├── scripts/             # 30 template scripts
│       └── assets/              # 8 configs + 2 templates
├── hooks/
│   ├── hooks.json               # 3 hooks
│   └── scripts/
│       ├── session-start.sh     # Dependency check + data file scan
│       └── validate-ml-bash.py  # Script safety gate
└── agents/
    └── ml-engineer.md           # Subagent for heavy workflows
```

## Commands (5)

Commands are discoverable entry points for users who don't know what to ask or who want one-shot invocation. Everything else flows through the skill via natural conversation.

### `/ml [data_file]`

Start an end-to-end ML project. Profiles data, recommends models, trains, evaluates, and produces a plain-language report.

This is the zero-knowledge entry point. A non-technical user sees `/ml` in `/help` and knows what to type. Without it, they must already know to say "train a machine learning model on my data" to trigger the skill.

**Arguments:** Optional file path (`/ml sales_data.csv`). If omitted, Claude prompts.
**Workflow:** End-to-End (explore → audit → leakage check → engineer → select → pipeline → errors → report).

### `/ml-eval [model]`

Run full diagnostics on a trained model — residuals, calibration, learning curves, error breakdown, assumption tests.

A re-entrant operation. The user trained a model yesterday and wants to interrogate it now. The skill triggers on "how good is my model?" but someone who knows what they want types one command and gets the full diagnostic suite.

**Arguments:** Optional model path or name.
**Workflow:** Evaluation & Diagnostics (error analysis → learning curves → calibration → assumptions → report).

### `/ml-report [audience]`

Generate a plain-language summary of the current model's performance, targeted to a specific audience.

A handoff operation — the ML work is done, the user needs to communicate results. Surfacing this as a command makes it feel like a first-class deliverable: `/ml-report boss` or `/ml-report client`.

**Arguments:** Optional audience label: `boss`, `client`, `team`, `technical`. Defaults to non-technical.
**Workflow:** Explain to Stakeholders (explain model → cost analysis → report).

### `/ml-status [prod_data]`

Check if a deployed model is still healthy — detect data drift, compare current vs training distributions, flag degradation.

A monitoring operation with no natural conversational trigger. Nobody says "machine learning" when checking model health. `/ml-status` is something a user runs on Monday morning or puts in a scheduled task.

**Arguments:** Optional path to production data file.
**Workflow:** Production Readiness (drift detection → reproducibility snapshot).

### `/ml-compare [data_file]`

Run multiple model types on the same data and return a ranked leaderboard with statistical significance tests.

Model comparison serves users second-guessing a prior choice. The output — "Random Forest beats Linear Regression with 95% confidence" — is clear and decisive for non-technical users.

**Arguments:** Optional data file path.
**Workflow:** Model Selection (explore → recommend → compare models).

### What is NOT a command

- `/ml-deploy` — requires conversation context (model object, feature names, scaler)
- `/ml-tune` — never someone's starting point; they arrive here from training
- `/ml-explain` — too close to `/ml-report`; not discoverable as a distinct concept
- `/ml-abtest` — two-phase structure (design → collect → analyze) needs persistent conversation
- Per-script wrappers — 30 commands would be absurd; scripts are implementation details

## Hooks (3)

### Hook 1: SessionStart — Environment Bootstrap

**Type:** `command` (deterministic, no LLM needed)
**Script:** `hooks/scripts/session-start.sh`
**Timeout:** 15s

Checks whether `ferroml` is importable. If not, injects a message telling Claude to install it before any ML work. Scans the working directory for `.csv` and `.parquet` files and injects them as "available datasets" into context. Writes version and data files to `$CLAUDE_ENV_FILE` for session persistence.

**Why essential:** Without this, Claude writes ferroml code, the import fails, and the non-technical user sees a confusing traceback. Catching the missing dependency at session start — before any code runs — eliminates that failure class entirely.

### Hook 2: PreToolUse (Bash) — Script Safety Gate

**Type:** `command` (fast pattern match)
**Matcher:** `Bash`
**Script:** `hooks/scripts/validate-ml-bash.py`
**Timeout:** 5s

Detects when Claude runs an ML script on a data file. Checks if the referenced file exists. If missing, blocks with a message: "data file X not found — ask the user to confirm the filename." If Claude tries to `pip install sklearn`, warns that FerroML covers the use case natively.

**Why essential:** Wrong filename is the most common failure for non-technical users. Catching it before a Python traceback is the difference between a smooth experience and abandonment.

### Hook 3: Stop — Diagnostic Completeness Check

**Type:** `prompt` (requires semantic reasoning about the transcript)
**Matcher:** `*`
**Timeout:** 20s

Before Claude stops, reviews the transcript. If the user asked to train or evaluate an ML model, checks three things:

1. Was a model actually trained and fitted?
2. Were diagnostics shown — not just accuracy, but at minimum one of: residuals, confidence intervals, feature importance, or a summary table?
3. Were results explained in plain language?

If all three pass or the session was not about ML, approves. Otherwise blocks with what's missing.

**Why essential:** This enforces FerroML's core differentiator. Non-technical users don't know to ask for diagnostics. Claude sometimes reports bare accuracy. This hook ensures every ML session delivers statistically rigorous output.

### Hooks Rejected

- **UserPromptSubmit (ML intent detection)** — SKILL.md triggers already handle this. Would add latency to every message in every session.
- **PostToolUse (script output validation)** — Stop hook catches incomplete workflows more reliably by reviewing the full transcript.
- **SessionEnd (experiment logging)** — `reproducibility_snapshot.py` already handles this in-workflow. A hook would create a parallel logging mechanism.
- **PreCompact (preserve ML state)** — Good idea, flagged for v2 after core hooks are validated. Long ML sessions can lose context during compaction.

## Agent (1)

### ml-engineer

A subagent for heavy ML workflows. Claude dispatches it for:

- AutoML search (60+ seconds)
- Full end-to-end pipeline on large datasets
- Comparing 5+ models with cross-validation
- Any "go do this and come back with results" workflow

For quick single-model training or answering ML questions, Claude handles it directly — no agent dispatch needed.

The agent gets the full skill context and access to all 30 scripts. It runs autonomously and returns structured results.

## Skill (existing)

The skill is the brain of the plugin — already built at `ferroml-python/skills/ferroml-ml/`. It moves into the plugin as `skills/ferroml-ml/`.

**Contents:** 55 files, 10,878 lines.

| Component | Count | Purpose |
|-----------|-------|---------|
| SKILL.md | 1 | Decision logic, 6 workflows, model guide, error handling |
| Scripts | 30 | Full ML lifecycle templates |
| References | 14 | API cheatsheet, diagnostics, model picking, deployment, fairness |
| Configs | 8 | Pre-built pipeline configs per task type |
| Templates | 2 | Report template + experiment log |

The skill auto-activates when Claude detects ML intent (data files, training requests, model evaluation). Commands provide explicit entry points; the skill provides the intelligence.

## Session State

Scripts write a `.ferroml-session.json` file to the working directory during workflows. This file tracks:

- Models trained in this session (name, algorithm, metrics, timestamp)
- Data files profiled
- Current best model

The `/ml-status` command reads this file. The SessionStart hook checks for it to provide context on previous work.

## Implementation Order

1. **Plugin manifest + skeleton** — `.claude-plugin/plugin.json`, directory structure
2. **Move skill** — Copy `ferroml-ml/` from `ferroml-python/skills/` into plugin `skills/`
3. **Commands** — 5 markdown files
4. **Hooks** — `hooks.json` + `session-start.sh` + `validate-ml-bash.py`
5. **Agent** — `ml-engineer.md`
6. **Session state** — Update scripts to write `.ferroml-session.json`
7. **Test end-to-end** — Install plugin locally, verify all commands/hooks/skill work
8. **Publish** — Claude Code plugin marketplace

## Design Principles

1. **Zero friction** — Plugin install is the only setup. Hook handles pip install.
2. **Commands are entry points, not wrappers** — 5 commands, not 30. The skill handles the rest.
3. **Hooks enforce quality** — Non-technical users get diagnostics whether they ask or not.
4. **Skill is the brain** — All intelligence lives in SKILL.md and the 30 scripts. Commands route to it.
5. **No MCP** — Local library, direct Python access. No server process overhead.
