# FerroML Build Mode

You are building FerroML, a statistically rigorous AutoML library in Rust.

## Your Task

Pick the highest-priority incomplete task from `IMPLEMENTATION_PLAN.md` and implement it.

## Instructions

1. **Read**: `AGENTS.md` for build commands and patterns
2. **Read**: `IMPLEMENTATION_PLAN.md` to find the next task
3. **Read**: Relevant `specs/*.md` for requirements
4. **Search**: Check if similar code exists (don't duplicate)
5. **Implement**: Write the code for ONE task
6. **Test**: Run `cargo check && cargo test`
7. **Update plan**: Mark task complete in `IMPLEMENTATION_PLAN.md`
8. **Exit**: So the next iteration can continue

## Rules

- **One task per iteration**: Don't try to do multiple tasks
- **Don't assume not implemented**: Always search first
- **Validate before marking complete**: Tests must pass
- **Follow existing patterns**: Match the codebase style
- **Statistical rigor**: Include CIs, effect sizes, assumption tests

## Validation Checklist

Before marking a task complete:
- [ ] `cargo check` passes
- [ ] `cargo test` passes (especially new tests)
- [ ] `cargo clippy` has no errors
- [ ] Code follows existing patterns in the codebase

## Output

1. Implement the task
2. Run validation
3. Update `IMPLEMENTATION_PLAN.md`
4. Exit
