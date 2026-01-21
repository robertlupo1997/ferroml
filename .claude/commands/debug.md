# Debug Issues

Troubleshoot issues by investigating logs, data, and git history. **Read-only investigation only.**

## Initial Response

With context: "I'll help debug issues with [topic]"
Without context: "What issue are you experiencing?"

## Investigation Resources

### 1. Logs
- Application logs in `logs/` if present
- Docker container logs: `docker-compose logs [service]`
- Python tracebacks

### 2. Data State
- Feature matrix: `data/features/feature_matrix.parquet`
- Model artifacts: `models/`
- Cached predictions: Redis if running

### 3. Git State
- Current branch and commit
- Recent changes: `git log --oneline -10`
- Uncommitted changes: `git status`

## Investigation Process

### Step 1: Understand the Problem
- Read relevant context
- Assess current state
- Identify what should vs does happen

### Step 2: Investigate Systematically
Deploy parallel checks:
- Review recent git changes
- Check relevant logs
- Verify data state
- Test specific components

### Step 3: Present Findings

```markdown
## Debug Report: [Issue]

### Problem Statement
[What was reported]

### Evidence Found
- [Finding 1 with file:line or log excerpt]
- [Finding 2]

### Root Cause
[Most likely cause based on evidence]

### Recommended Next Steps
1. [First action]
2. [Second action]

### Additional Investigation (if needed)
- [What else to check]
```

## Constraints

- **Read-only**: Do not modify files during debug
- **Require problem description**: Can't debug without knowing what's wrong
- **Focus on evidence**: Base conclusions on actual findings

## Quick Reference Commands

```bash
# Check recent changes
git log --oneline -10
git diff HEAD~3

# Check test status
pytest -v --tb=short

# Check lint status
ruff check .

# Docker logs
docker-compose logs api --tail=100
docker-compose logs dashboard --tail=100

# Check running services
docker-compose ps
```

## Out of Scope

- Browser console errors (need screenshot or copy)
- Runtime memory issues (need profiler output)
- Network issues (need curl/httpie output)

$ARGUMENTS
