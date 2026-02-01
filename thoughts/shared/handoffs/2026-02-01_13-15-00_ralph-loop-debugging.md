---
date: 2026-02-01T13:15:00-05:00
researcher: Claude
git_commit: 9eb8812
git_branch: master
repository: ferroml
topic: Ralph Loop Debugging & Gold Standard Alignment
tags: [ralph-loop, automation, ghuntley-playbook, debugging]
status: in-progress
---

# Handoff: Ralph Loop Debugging & Gold Standard Alignment

## Problem Statement

The Ralph Wiggum loop is getting stuck after ~1 iteration. It completed iteration 1 successfully (commit `1c4fbd2`) but iteration 2 hung for over an hour with no progress. Need to debug and align with the ghuntley/how-to-ralph-wiggum gold standard.

## Task Status

### Current Phase
Debugging and aligning Ralph loop with gold standard

### Progress
- [x] Basic loop.sh working in WSL
- [x] Opus model configured
- [x] Auto-detection of Claude CLI location
- [x] Docker setup attempted (OAuth issue - needs API key)
- [ ] Align PROMPT_build.md with gold standard
- [ ] Align PROMPT_plan.md with gold standard
- [ ] Add proper guardrail sequence (999, 9999, etc.)
- [ ] Add subagent limits per gold standard
- [ ] Debug why iteration 2 hung
- [ ] Add proper backpressure mechanisms

## Critical References

1. **Gold Standard**: https://github.com/ghuntley/how-to-ralph-wiggum
2. `PROMPT_build.md` - Current build prompt (needs updates)
3. `PROMPT_plan.md` - Current plan prompt (needs updates)
4. `AGENTS.md` - Operational reference (check if too long)
5. `IMPLEMENTATION_PLAN.md` - Task tracking
6. `loop.sh` - Loop orchestrator

## Gap Analysis: Current vs Gold Standard

### 1. CLI Flags ✅ Mostly Correct

| Flag | Gold Standard | Current | Status |
|------|---------------|---------|--------|
| `-p` | Required | ✅ | OK |
| `--dangerously-skip-permissions` | Required | ✅ | OK |
| `--output-format=stream-json` | Recommended | ❌ Removed | **NEEDS FIX** |
| `--model opus` | Recommended | ✅ | OK |
| `--verbose` | Recommended | ❌ Removed | **NEEDS FIX** |

**Note**: We removed stream-json and verbose due to buffering issues, but they're recommended for monitoring.

### 2. PROMPT_build.md Gaps ❌

**Missing from current prompt:**

a) **Guardrail Sequence** (Higher numbers = higher priority):
```
999. Test requirements from acceptance criteria must exist and pass
9999. Derived tests from acceptance criteria prevent "cheating"
999999. Fix unrelated failing tests as part of work
9999999. Git tag after zero build/test errors
99999999. Subagent updates to IMPLEMENTATION_PLAN.md capture learnings
999999999. Update AGENTS.md with operational learnings only
9999999999. Bug discovery → resolution or documentation
99999999999. Complete implementations, no placeholders
999999999999. Periodically clean completed items from plan
9999999999999. Resolve spec inconsistencies via subagent (Ultrathink)
99999999999999. Keep AGENTS.md operational only — no status updates
```

b) **Subagent Limits**:
- Current: "up to 5 parallel subagents for file reads"
- Gold Standard: "500 for searches/reads, **only 1** for build/tests"

c) **Key Language Patterns**:
- Missing "Ultrathink" for complex reasoning
- Missing "capture the why" for documentation
- Missing "don't assume not implemented" guardrail

d) **Backpressure Mechanism**:
- Current: Just "tests must pass"
- Gold Standard: Specific rejection sources (test failures, build errors, type check, lint)

### 3. PROMPT_plan.md Gaps ❌

**Missing:**
- "up to 250 parallel Sonnet subagents" for specs
- "Do NOT assume functionality is missing; confirm with code search first"
- Explicit "Plan only. Do NOT implement anything."

### 4. AGENTS.md Check ⚠️

Gold standard says: "Keep it brief (~60 lines), operational only"
- Need to verify current AGENTS.md isn't too long
- Should NOT contain status updates or progress notes

### 5. Loop Mechanics ⚠️

**Potential Hang Cause:**
- Missing stream-json means no real-time output visibility
- No timeout mechanism for individual iterations
- No heartbeat/health check built into loop

## Why Iteration 2 Likely Hung

Possible causes:
1. **API timeout/rate limit** - Opus is slow, may have hit limits
2. **Infinite tool loop** - Agent may have gotten stuck in a loop
3. **Context overload** - Too much context loaded without subagent offloading
4. **Missing backpressure** - No mechanism to break out of stuck states

## Recommended Fixes

### Fix 1: Update loop.sh with stream-json and timeout

```bash
# Add timeout per iteration (10 minutes max)
timeout 600 cat "$PROMPT_FILE" | $CLAUDE_CMD -p \
    --dangerously-skip-permissions \
    --model opus \
    --output-format stream-json \
    --verbose
```

### Fix 2: Update PROMPT_build.md

Add the full guardrail sequence and proper subagent limits.

### Fix 3: Update PROMPT_plan.md

Add proper subagent limits and "don't assume" guardrails.

### Fix 4: Add heartbeat logging

```bash
# In loop.sh, add timestamp logging
echo "[$(date)] Starting iteration $ITERATION" >> ralph-heartbeat.log
```

## Files Modified This Session

- `loop.sh:26-36` - Added Claude CLI auto-detection
- `loop.sh:60-65` - Changed to Opus, removed stream-json/verbose
- `.dockerignore` - New file
- `Dockerfile.ralph` - New file (Docker not working due to OAuth)
- `docker-compose.ralph.yml` - New file
- `ralph-docker.sh` - New file

## Key Learnings

### What Worked
- WSL execution with native Linux Claude CLI
- Auto-detection of Claude location works
- Iteration 1 completed successfully with Opus

### What Didn't Work
- Docker sandboxing (OAuth doesn't work headlessly - needs API key)
- Removing stream-json didn't help - still hung
- No visibility into what iteration 2 was doing

### Important Discoveries
- Opus is ~3-4x slower than Sonnet per iteration
- The loop.sh originally used Sonnet, not Opus
- Stream-json output may buffer but is needed for monitoring
- Need timeout mechanism to prevent infinite hangs

## Artifacts Produced

- `loop.sh` - Updated with auto-detection
- `Dockerfile.ralph` - For future sandboxing (needs API key)
- `docker-compose.ralph.yml` - Resource-limited Docker config
- `ralph-docker.sh` - Docker runner script
- `.dockerignore` - Excludes target/ and .git/

## Blockers

1. **Docker OAuth** - Claude CLI OAuth doesn't work headlessly in Docker
   - Resolution: Set ANTHROPIC_API_KEY environment variable instead

2. **Hang Debugging** - No visibility into why iteration 2 hung
   - Resolution: Re-add stream-json, add timeouts, add heartbeat logging

## Action Items & Next Steps

Priority order:

1. [ ] **Update PROMPT_build.md** with gold standard guardrail sequence (999, 9999, etc.)
2. [ ] **Update PROMPT_plan.md** with proper subagent limits and guardrails
3. [ ] **Re-add stream-json and verbose** to loop.sh for monitoring
4. [ ] **Add timeout** (600 seconds) per iteration to prevent hangs
5. [ ] **Add heartbeat logging** to track iteration progress
6. [ ] **Verify AGENTS.md** is under 60 lines and operational-only
7. [ ] **Test with 1 iteration** to verify fixes work
8. [ ] **Run overnight** with fixes in place

## Verification Commands

```bash
# Check loop is running
ps aux | grep -E '(claude|loop)' | grep -v grep

# Watch log in real-time
tail -f ralph.log

# Check for new commits
git log --oneline -5

# Count lines in AGENTS.md (should be ~60)
wc -l AGENTS.md

# Verify prompt files exist
ls -la PROMPT_*.md
```

## Gold Standard Reference Summary

From ghuntley/how-to-ralph-wiggum:

**Core Philosophy:**
- "Let Ralph Ralph" — Trust the system to self-correct through iteration
- "Move Outside the Loop" — Engineer setup, not every detail
- "Context Is Everything" — Tight tasks + 1 task per loop = optimal utilization

**Key Mechanisms:**
- Subagent offloading to preserve main context
- Backpressure from tests/build/lint failures
- Guardrail sequence for priority ordering
- Plan is disposable - regenerate if trajectory wrong
- AGENTS.md for operational learnings only

**Context Management:**
- 200K advertised ≈ 176K usable
- Target 40-60% utilization ("smart zone")
- Each loop loads same files deterministically
- Subagents extend memory, garbage collected after

## Other Notes

The FerroML project goal is to create "the greatest ML library for Rust". The Ralph loop should systematically implement the testing phases (16-32) from the comprehensive testing plan while maintaining high code quality.

Current status: Phases 16-20 largely complete, phases 21-32 remaining.
