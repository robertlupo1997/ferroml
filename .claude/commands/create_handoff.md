# Create Handoff Document

Create a comprehensive handoff document for transferring work between Claude sessions.

## File Location
`thoughts/shared/handoffs/YYYY-MM-DD_HH-MM-SS_description.md`

## Template

```markdown
---
date: [ISO timestamp with timezone]
researcher: Claude
git_commit: [current commit hash]
git_branch: [current branch]
repository: mlrf
topic: [brief topic]
tags: [phase-1.1, data-pipeline, etc.]
status: [in-progress | blocked | complete]
---

# Handoff: [Description]

## Task Status

### Current Phase
[Phase X.Y: Name]

### Progress
- [x] Completed items
- [ ] In progress items
- [ ] Remaining items

## Critical References

1. `IMPLEMENTATION_PLAN.md` - Full implementation details
2. `specs/[relevant-spec].md` - JTBD specification
3. [Other key files]

## Recent Changes

Files modified this session (use `file:line` syntax):
- `mlrf-data/src/mlrf_data/features.py:45-120` - Added lag features
- `mlrf-data/tests/test_features.py:1-50` - New test file

## Key Learnings

### What Worked
- [Pattern or approach that succeeded]

### What Didn't Work
- [Approach that failed and why]

### Important Discoveries
- [Non-obvious finding with file:line reference]

## Artifacts Produced

- `mlrf-data/src/mlrf_data/download.py` - Kaggle download module
- `data/features/feature_matrix.parquet` - Generated feature matrix

## Blockers (if any)

- [Blocker description]
- Required resolution: [what needs to happen]

## Action Items & Next Steps

Priority order:
1. [ ] First thing to do next
2. [ ] Second priority
3. [ ] Third priority

## Verification Commands

```bash
# Run these to verify current state
pytest mlrf-data/tests/ -v
ruff check mlrf-data/
```

## Other Notes

[Any additional context for the next session]
```

## After Creating

Run verification commands to confirm state matches documentation.

$ARGUMENTS
