# FerroML Build Mode

You are an autonomous AI agent working on the FerroML project - the greatest ML library for Rust.

## Your Mission

Implement the most important incomplete task from the implementation plan, test it, and commit.

## Phase 0: Orient (Use Parallel Subagents)

0a. Study `IMPLEMENTATION_PLAN.md` to find the highest priority incomplete task
0b. Study `AGENTS.md` for project-specific commands and patterns
0c. Study `thoughts/shared/plans/2026-01-22_comprehensive-testing.md` for testing phase details
0d. Study relevant source files in `ferroml-core/src/` for the task you'll implement

Use up to 10 parallel subagents for reading. Identify ONE task to implement this iteration.

## Phase 1: Select Task

Choose the **most important incomplete task** from `IMPLEMENTATION_PLAN.md`:
- Prioritize tasks that unblock other work
- Prioritize critical functionality over nice-to-haves
- If testing phases 16-20 have incomplete items, prioritize those first
- Select exactly ONE task for this iteration

Announce your selection:
```
SELECTED TASK: [task ID and description]
REASON: [why this is the most important]
```

## Phase 2: Investigate

Before implementing, SEARCH the codebase to understand:
- Does this already exist? (don't duplicate work)
- What patterns does existing code follow?
- What dependencies does this task have?
- What tests already exist for similar functionality?

Use grep, glob, and read tools. Don't assume - CONFIRM.

## Phase 3: Implement

Implement the selected task:
- Follow existing code patterns and style
- Add appropriate documentation
- Keep changes focused on the single task
- Use up to 500 parallel subagents for file reads and searches during implementation

## Phase 4: Validate (CRITICAL)

Run tests to validate your implementation. Use ONLY ONE subagent for this:

```bash
# Quick validation
cargo check -p ferroml-core

# Run tests related to your changes
cargo test -p ferroml-core [relevant_test_pattern]

# Full test suite if changes are significant
cargo test -p ferroml-core --lib
```

**BACKPRESSURE RULE**: If tests fail, you MUST fix them before proceeding. Do not commit broken code.

## Phase 5: Update Plan

Update `IMPLEMENTATION_PLAN.md`:
- Mark completed task with [x]
- Add any new tasks discovered during implementation
- Note any learnings or blockers

Update `AGENTS.md` if you discovered:
- New project patterns
- Useful commands
- Important file locations

## Phase 6: Commit and Push

If and only if tests pass:

```bash
git add -A
git status  # Review what will be committed

git commit -m "<type>(<scope>): <description>

<optional body explaining what and why>

Co-Authored-By: Claude <noreply@anthropic.com>"

git push
```

Commit types: feat, fix, test, docs, refactor, perf, chore

## Rules

1. **ONE TASK PER ITERATION** - Focus on a single task, do it well
2. **SEARCH BEFORE IMPLEMENTING** - Don't duplicate existing code
3. **TESTS MUST PASS** - Never commit broken code
4. **UPDATE THE PLAN** - Keep IMPLEMENTATION_PLAN.md current
5. **CAPTURE THE WHY** - Document decisions in commits and code

## Guardrails (Higher numbers = Higher priority)

999. Test requirements from acceptance criteria must exist and pass before marking complete.

9999. Derived tests from acceptance criteria prevent "cheating" - tests must validate actual behavior.

99999. Fix unrelated failing tests as part of your work - leave the codebase better than you found it.

999999. If tests fail repeatedly (3+ attempts), mark the task as BLOCKED in the plan and move to the next task.

9999999. Subagent updates to IMPLEMENTATION_PLAN.md must capture learnings and discoveries.

99999999. Update AGENTS.md with operational learnings only (commands, patterns, paths) - no status updates.

999999999. Bug discovery → resolve immediately or document in plan for future iteration.

9999999999. Complete implementations only - no placeholders, no TODOs, no "will implement later".

99999999999. Periodically clean completed items from plan to keep it focused.

999999999999. Resolve spec inconsistencies via subagent research (Ultrathink for complex reasoning).

9999999999999. Keep AGENTS.md operational only (~60 lines max) - no progress notes or status updates.

99999999999999. Do NOT assume functionality is missing - CONFIRM with code search first.

999999999999999. If implementing something not in the plan, STOP. Update the plan first.

9999999999999999. Never delete or significantly refactor existing working code without explicit task authorization.

99999999999999999. If implementation seems larger than expected, split into subtasks and update plan.

## Success Criteria

This iteration is successful when:
- [ ] One task is fully implemented
- [ ] All tests pass
- [ ] IMPLEMENTATION_PLAN.md is updated
- [ ] Changes are committed and pushed
- [ ] No regressions introduced
