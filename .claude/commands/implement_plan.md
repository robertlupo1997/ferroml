# Implement Plan

Execute an implementation plan from `thoughts/shared/plans/` or `IMPLEMENTATION_PLAN.md`.

## Process

### 1. Load Context
- Read the plan file completely, noting any existing checkmarks
- Review all referenced files without truncation
- Understand how components interconnect
- Create a todo list for tracking

### 2. Implementation Approach

**Follow the plan's intent while adapting to what you find.**

Plans guide work, but use judgment when reality diverges from expectations.

When issues arise, stop and articulate:
- Expected vs actual state
- Why the discrepancy matters
- Request guidance on proceeding

### 3. Verification Process

After each phase:
1. Run automated checks (tests, lint, type check)
2. Pause and notify human with format:

```
## Phase [X.Y] Complete

### Automated Verification
- [x] `pytest tests/ -v` - All tests pass
- [x] `ruff check src/` - No lint errors

### Manual Verification Required
- [ ] Verify feature matrix has >2M rows
- [ ] Check no data leakage in lag features

Ready to proceed to Phase [X.Z]?
```

3. Only proceed after manual verification confirmed

### 4. Resuming Work
- Trust completed checkmarks
- Begin at first unchecked item
- Resume momentum without re-verifying prior work unless issues arise

## Commands by Phase

### Phase 1.1 (Data Pipeline)
```bash
cd mlrf-data && pip install -e ".[dev]"
pytest mlrf-data/tests/ -v
ruff check mlrf-data/
```

### Phase 1.2 (ML Pipeline)
```bash
cd mlrf-ml && pip install -e ".[dev]"
pytest mlrf-ml/tests/ -v
ruff check mlrf-ml/
```

### Phase 1.3 (Go API)
```bash
cd mlrf-api && go build ./cmd/server
go test ./... -v
```

### Phase 1.4 (Dashboard)
```bash
cd mlrf-dashboard && bun install
bun run typecheck
bun run lint
bun run build
```

### Phase 1.5 (Integration)
```bash
docker-compose up -d
# Run integration tests
docker-compose down
```

$ARGUMENTS
