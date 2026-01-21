# Resume Work from Handoff

Continue work from a handoff document with proper state verification.

## Input Types

1. **Direct path**: `thoughts/shared/handoffs/YYYY-MM-DD_*.md`
2. **Description search**: Find most recent matching handoff
3. **No parameters**: List available handoffs

## Process

### 1. Initial Analysis
- Read the complete handoff document
- Extract: tasks, changes, learnings, artifacts, action items
- Note any blockers or issues

### 2. State Verification

**Critical: Never assume handoff state matches current state.**

Spawn focused research tasks:
- Verify files exist as documented
- Check git state matches expected
- Run verification commands from handoff
- Confirm dependencies are installed

### 3. Present Findings

```markdown
## Handoff Resume: [Topic]

### Original State (from handoff)
- Phase: [X.Y]
- Last action: [description]
- Commit: [hash]

### Current State (verified now)
- Files present: [yes/no with details]
- Tests passing: [yes/no]
- Discrepancies: [list if any]

### Recommended Actions
1. [First step based on verification]
2. [Second step]

Proceed with resume?
```

### 4. Implementation

After confirmation:
- Convert action items to todo list
- Reference documented learnings throughout
- Validate file references remain applicable
- Build upon discovered solutions

## Common Scenarios

### Clean Continuation
Handoff state matches current - proceed with action items.

### Diverged Codebase
Files changed since handoff - identify what's different, adjust approach.

### Incomplete Prior Work
Partially completed items - verify what's done, continue from there.

### Stale Handoff
Too much has changed - recommend creating new plan instead.

## Key Principles

- **Verify before acting**: Check current state against handoff
- **Leverage learnings**: Don't repeat previous mistakes
- **Confirm at decision points**: Get approval before major actions
- **Update as you go**: Create new handoff if session runs long

$ARGUMENTS
