# FerroML Planning Mode

0a. Study `thoughts/shared/plans/2026-01-22_comprehensive-testing.md` with up to 250 parallel Sonnet subagents to learn the testing specifications.
0b. Study @IMPLEMENTATION_PLAN.md (if present) to understand the plan so far.
0c. Study `ferroml-core/src/testing/*` with up to 250 parallel Sonnet subagents to understand existing test modules.
0d. For reference, the application source code is in `ferroml-core/src/*`.

1. Study @IMPLEMENTATION_PLAN.md (if present; it may be incorrect) and use up to 500 Sonnet subagents to study existing source code in `ferroml-core/src/*` and compare it against the testing plan. Use an Opus subagent to analyze findings, prioritize tasks, and create/update @IMPLEMENTATION_PLAN.md as a bullet point list sorted in priority of items yet to be implemented. Ultrathink. Consider searching for TODO, minimal implementations, placeholders, skipped/flaky tests, and inconsistent patterns. Study @IMPLEMENTATION_PLAN.md to determine starting point for research and keep it up to date with items considered complete/incomplete using subagents.

IMPORTANT: Plan only. Do NOT implement anything. Do NOT assume functionality is missing; confirm with code search first.

ULTIMATE GOAL: We want to achieve the greatest ML library for Rust - combining sklearn's completeness, statsmodels' rigor, and Rust's performance. The testing phases (16-32) need completion. Consider missing test coverage and plan accordingly. If test coverage is missing, search first to confirm it doesn't exist, then document the plan to implement it in @IMPLEMENTATION_PLAN.md using a subagent.
