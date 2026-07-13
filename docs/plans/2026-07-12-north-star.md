# FerroML North Star — Purpose, Endgame, and Execution Model

**Date:** 2026-07-12 · **Author:** Claude (Fable 5), under Trey's delegated judgment ("go nuts / get all the architecture down / know exactly what the purpose is and exactly how to achieve it") · **Status:** ACTIVE — this is the top-level document; read it first in every future session.

---

## 1. What FerroML is for (the honest version)

FerroML is **P3 of Trey's approved career-direction plan**
(`~/experiments/Ornith/docs/superpowers/specs/2026-07-04-career-direction-design.md`):
a portfolio project proving *"the stats degree made software."* It is **not** a market
competitor to scikit-learn, XGBoost, or Julia's MLJ.jl, and it never needs to be. No solo
project out-engineers libraries with decades of researcher-hours behind them — and no
employer expects one to.

What it must be instead:

1. **Shipped** — on PyPI, installable, versioned, green CI. (Already true: v1.0.1.)
2. **Correctness-certified** — every algorithm cross-validated against
   sklearn/scipy/statsmodels/linfa (~4,600 Rust + ~2,400 Python tests). This is the trust
   mechanism that makes AI-built ML code defensible: *nobody* — not Opus, not Codex, not
   Fable — is trusted; the reference implementations are the ground truth.
3. **Distinctive** — statistical diagnostics as first-class citizens, culminating in the
   one feature no mainstream AutoML has: **selection that is statistically honest** —
   "pick the simplest model not statistically distinguishable from the best, and show the
   evidence." That sentence is the interview.
4. **Explainable** — Trey's interview policy is "I orchestrate AI agents with verification
   gates." The fundamentals track (his own drills, outside this repo) exists so the
   explanation survives probing.

**Success criterion:** P3 done bar met by **Aug 29, 2026** (auto-select + honest benchmark
+ model card, live on PyPI as v1.1.0), then the repo enters low-touch maintenance. Adoption,
stars, and HN karma are non-goals.

## 2. Where the project stands (verified 2026-07-12)

- v1.0.0 + v1.0.1 on PyPI via trusted publishing; CI green since 2026-04-30.
- 55+ models, full CASH AutoML, CLI, AGENTS.md, llms.txt shipped (master-design Phase 5
  done except the MCP server).
- The Nadeau-Bengio corrected t-tests are computed and attached to AutoML results but
  **do not drive selection** — `best_model()` is still leaderboard top-1. Closing that
  loop is P3.
- Nothing from the P3 spec is implemented yet (verified by grep in the 2026-07-12 handoff).

## 3. The remaining roadmap — everything left, in order

### 3.1 P3: Intelligent AutoML v1 — THE priority (now → Aug 29, target mid-August)

Spec: `thoughts/2026-07-06-p3-intelligent-automl-v1-spec.md` — **approved 2026-07-12**
(Trey delegated the approval decision to Fable this session; approved as drafted).
Implementation plan: `docs/plans/2026-07-12-p3-intelligent-automl-v1-implementation.md`
(task-by-task, Codex-ready).

Five session-sized phases: P3.1 `SelectionRationale` + `statistically_best_model()` →
P3.2 `render_benchmark_report()` → P3.3 `ModelCardGenerator::from_automl_result()` →
P3.4 PyO3 exposure + README section → P3.5 release v1.1.0 (**Trey gate before PyPI
publish**). ~150 lines of core Rust plus bindings and tests.

### 3.2 The passion track — APPROVED by Trey 2026-07-12 ("attempt tier one and tier two")

Trey reframed the project: beyond the career done-bar, FerroML continues as a passion
project to build a genuinely good ML library. Approved scope, in build order (each item
gets its own implementation plan via superpowers:writing-plans, executed Codex-first,
one item at a time, sessions-sized):

**Tier 1 — make what exists excellent:**
1. Documentation site (mkdocs-material): user guide, API reference, correctness report
   published properly. The front door.
2. pyo3 0.24→0.29 migration — clears suppressed RUSTSEC-2026-0176/0177.
3. Robustness pass — `clippy::unwrap_used` triage in automl/cv/neural/hpo/metrics;
   no panics on adversarial input.
4. Report/leaderboard polish (v-next): by-unique-algorithm leaderboard view; CLI flags
   `--benchmark-report` / `--model-card` on `ferroml automl`.

**Tier 2 — capability gaps:**
5. Automated feature engineering in AutoML (polynomial/interactions, binning, selection).
6. Python `model_selection` parity: `cross_val_score`, `GridSearchCV` (Rust cv/ machinery
   already exists — binding gap only).
7. Hyperband/ASHA early stopping in AutoML.
8. sklearn interop (`__sklearn_tags__`) for pipeline/meta-estimator compatibility.
9. Meta-learning selection v2 — wire metafeatures.rs + warmstart.rs into recommend().

Pacing constraint (unchanged): Resume v3 and the P2 Ornith writeup outrank this track
until they ship; passion-track sessions get the hours those don't need.

### 3.2b Superseded value-ranking (kept for history; items merged above or dropped)

1. **Portfolio writeup** (the only Phase 6 item with career ROI before Sep 1): a blog
   post on treylupo.com — "An AutoML that refuses to overclaim" — using a real
   `render_benchmark_report()` output. Doubles as application/referral bait alongside the
   P2 Ornith writeup. ~1 session, mostly writing.
2. **MCP server** (`ferroml-mcp/`, FastMCP wrapping the CLI operations): on-narrative
   (Trey wired NetSuite→Claude MCP at work), ~1–2 sessions. Only if August has slack.
3. **v1.2 UX candidate** (observed in real P3 runs, 2026-07-12): short-budget AutoML
   leaderboards are dominated by duplicate trials of one algorithm, so the top-10 table
   repeats one row and the pairwise matrix compares an algorithm against itself. A
   "by unique algorithm" leaderboard view (best trial per algorithm) would fix the
   presentation without hiding trial counts. Also: `ferroml automl` CLI flags
   (`--benchmark-report`, `--model-card`) to expose the new outputs — found in the
   2026-07-12 docs audit. Out of P3 scope — do not build before Aug 29.
4. **Hygiene track** (background, ideal Codex/Ornith fodder, no deadline):
   **pyo3 0.24→0.29 migration is the top item** — RUSTSEC-2026-0176/0177 are suppressed
   in deny.toml + ci.yml until it lands (added 2026-07-12; quick-xml 0194/0195 wait on
   polars); `clippy::unwrap_used` triage in `automl`/`cv`/`neural`/`hpo`/`metrics`;
   review the other suppressed RustSec advisories; revisit the `ort` v2 RC pin when it
   goes stable; recreate `.venv` (stale shebangs from the repo move); gitignore the
   local cruft (`graphify-out/`, `*.abi3.so`, `memory_profile_results.json`).
5. **Explicitly dropped** (no career ROI before Sep 1, revisit only on real user demand):
   mkdocs documentation site, Hacker News launch, Kaggle notebook, conda-forge, JMLR
   paper, sklearn migration guide, meta-learning selection v2 (`recommend.rs` upgrade).

### 3.3 Maintenance mode (post-P3, indefinitely)

Monthly, ~15 minutes, delegable to a scheduled agent or done ad-hoc: check CI is green,
run `cargo audit`, bump flagged deps, merge nothing else. The repo's job from September on
is to *look maintained*, not to grow. One open loose end: the old `ci-monitor` cloud
routine (`trig_015xjmyh5N4pkVkkDgeuzZxb`) — Trey should check
claude.ai/code/scheduled and retire it; CI has been green for months and its log file was
never written.

## 4. Execution model — how all remaining work gets done

Trey's standing instruction (2026-07-12): **preserve Claude context/usage; use Codex
before Claude for implementation.** Division of labor:

| Role | Who | What |
|---|---|---|
| Architect / controller | Fable (or Opus) | Triage, plan authorship, dispatch, diff review, gates. Never writes implementation code first-hand while Codex is available. |
| Decomposition | Sonnet-tier subagents | Only when no task-level plan exists yet. (P3 already has one — skip this step for P3.) |
| Implementation | **Codex** via `codex:rescue` agent / codex-build skill | One task card at a time from the implementation plan: RED test → GREEN impl → fmt/clippy/test evidence. |
| Mechanical chores | Ornith local model (free) via `ornith-worker` | gitignore edits, doc formatting, unwrap-triage sweeps. |
| Verification | The test suite, not any model | `cargo fmt --check`, `clippy -D warnings`, `cargo test` (+ `--test correctness` when models are touched), `pytest ferroml-python/tests/` after binding work. Controller re-runs gates itself; implementer claims are never taken on faith (controller-discipline skill). |
| Human gates | Trey | PyPI publish (P3.5), anything public-facing (blog post, HN if ever), spending decisions. |

Rules that keep this safe: every task lands RED-first per the plan; one commit per task;
a task card must be self-contained (exact files, signatures, test code) so Codex needs no
repo-wide context; any deviation Codex proposes gets reviewed by the controller against
the spec before acceptance.

## 5. Decision log (2026-07-12, this session)

1. **P3 over Phase 6** — P3 has the deadline and the career payoff; Phase 6 had neither.
2. **P3 spec approved as drafted** — approval was the spec's only blocker; Trey delegated
   the call to Fable.
3. **Phase 6 shrunk to one item** (the writeup); docs site / HN / conda-forge / JMLR
   dropped per §3.2.
4. **Codex-first execution model adopted** (§4) — also recorded in Fable's persistent
   memory so future sessions inherit it.
5. **The "are we building this correctly?" question, answered:** yes, *given the actual
   goal.* Julia/sklearn comparisons measure market fitness, which was never the bar. The
   bar is the career-spec done bar, and the project is one small, well-specified slice
   away from it.

## 6. Session protocol (every future session)

1. Read `HANDOFF.md` → latest handoff, this document, then the P3 implementation plan.
2. Verify state (git log, CI, `cargo test` if code changed since last session).
3. Execute the next unchecked task in the implementation plan via the §4 model.
4. Commit per task; update the plan's checkboxes; write/refresh a handoff before ending.
