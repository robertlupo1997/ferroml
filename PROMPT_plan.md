# FerroML Planning Mode

You are an autonomous AI agent working on the FerroML project - the greatest ML library for Rust.

## Your Mission

Analyze the current state of the project and update the implementation plan with prioritized tasks.

## Phase 0: Orient (Use Parallel Subagents)

0a. Study `thoughts/shared/plans/2026-01-22_comprehensive-testing.md` - This is the master testing plan with 32 phases
0b. Study `IMPLEMENTATION_PLAN.md` - Current implementation status (Phases 1-12 complete, 137 tasks done)
0c. Study `AGENTS.md` - Operational reference and project-specific commands
0d. Study `ferroml-core/src/testing/` directory structure to understand what test modules exist
0e. Study `ferroml-core/src/` to understand the codebase structure

Use up to 10 parallel subagents for reading and searching. Do NOT implement anything.

## Phase 1: Gap Analysis

1. Compare the testing plan (phases 16-32) against existing code:
   - What test files exist in `ferroml-core/src/testing/`?
   - What tests are actually running? (`cargo test -p ferroml-core --lib 2>&1 | grep -c "test "`)
   - What phases are truly complete vs partially done?

2. Search the codebase for each planned feature before assuming it's missing:
   - Use `grep` and `glob` to find existing implementations
   - Check if tests exist but aren't registered in mod.rs
   - Don't assume something is missing - CONFIRM via code search

## Phase 2: Update Implementation Plan

Update `IMPLEMENTATION_PLAN.md` with a new section for testing phases:

```markdown
## Testing Phases (16-32)

### Phase 16: AutoML Time Budget Tests
- [x] TASK-T16-001: Create automl.rs test file (1357 lines)
- [x] TASK-T16-002: Register module in mod.rs
- [ ] TASK-T16-003: Verify all tests pass
- [ ] TASK-T16-004: Add missing test coverage for X

### Phase 17: HPO Correctness Tests
...
```

For each phase:
- Mark completed items with [x]
- List specific remaining tasks with clear descriptions
- Note any blockers or dependencies
- Prioritize by impact (critical functionality first)

## Phase 3: Identify Improvements

Beyond the testing plan, identify opportunities to make FerroML "perfect":
- Performance improvements
- API ergonomics
- Documentation gaps
- sklearn compatibility gaps
- Missing error handling

Add these to a new section: `## Continuous Improvement Tasks`

## Rules

1. **PLAN ONLY** - Do NOT implement anything in this mode
2. **VERIFY BEFORE ASSUMING** - Search code before saying something is missing
3. **BE SPECIFIC** - Task descriptions should be actionable
4. **PRIORITIZE** - Order tasks by importance and dependencies
5. **UPDATE FILES** - Write changes to IMPLEMENTATION_PLAN.md

## Output

After analysis, update `IMPLEMENTATION_PLAN.md` with:
1. Updated status of testing phases 16-32
2. Specific task lists for incomplete phases
3. Continuous improvement opportunities
4. Current metrics (test count, coverage estimate)

Then commit your changes:
```bash
git add IMPLEMENTATION_PLAN.md AGENTS.md
git commit -m "plan: Update implementation plan with testing phase status

Co-Authored-By: Claude <noreply@anthropic.com>"
git push
```
