# P3 Intelligent AutoML v1 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking. Per the north-star execution model, tasks are dispatched to **Codex** one task at a time; the controller re-runs all verification gates itself.

**Goal:** Close the loop between FerroML's computed significance tests and its AutoML selection: pick the simplest model not statistically distinguishable from the best, render the evidence honestly, and generate a model card — shipped as v1.1.0.

**Architecture:** Three additive features on existing types — `statistically_best_model()` + `SelectionRationale` on `AutoMLResult` (fit.rs), `render_benchmark_report()` on `AutoMLResult` (fit.rs), `ModelCardGenerator::from_automl_result()` (model_card.rs) — then PyO3 exposure per the 3-step binding rule. No new statistical machinery; everything consumes structs that already exist.

**Tech Stack:** Rust (ndarray, serde/serde_json), PyO3/maturin, pytest.

**Spec:** `thoughts/2026-07-06-p3-intelligent-automl-v1-spec.md` (approved 2026-07-12). **Verified API reference:** all struct definitions cited below were extracted verbatim from the codebase on 2026-07-12.

## Global Constraints

- `cargo fmt --all` before every commit (pre-commit hook enforces it); `cargo clippy -- -D warnings` must be clean.
- `cargo test` alone only tests ferroml-core (workspace default-members). Use `cargo test -p ferroml-core` for core work, and `maturin develop --release -m ferroml-python/Cargo.toml && pytest ferroml-python/tests/` for binding work (activate `.venv` first).
- **No new integration test binaries** in `ferroml-core/tests/` — all new Rust tests go in the existing `#[cfg(test)] mod tests` blocks inside the source files (AutoML tests live at the bottom of `fit.rs`, starting line ~1554).
- Linear algebra is `nalgebra`, never `ndarray-linalg` (not needed here anyway).
- 3-step binding rule for anything Python-visible: Rust impl → PyO3 wrapper in `ferroml-python/src/` → re-export in `ferroml-python/python/ferroml/<submodule>/__init__.py`.
- RED first: write the failing test, watch it fail (in Rust this is usually a compile error — that counts), then implement.
- One commit per task, message style `feat(automl): ...` / `feat(python): ...` matching the repo's conventional-commit history.

## Verified API facts every task relies on

- `AutoMLResult` (fit.rs:80–110) fields used here: `leaderboard: Vec<LeaderboardEntry>`, `model_comparisons: Option<ModelComparisonResults>`, `data_characteristics: DataCharacteristics`, `cv_folds: usize`, `task: Task`, `metric_name: String`, `maximize: bool`, `total_time_seconds: f64`, `n_successful_trials: usize`, `n_failed_trials: usize`, `trials`, `ensemble`, `best_model_stats`, `aggregated_importance`.
- `LeaderboardEntry` (fit.rs:378–398): `rank: usize`, `trial_id: usize`, `algorithm: AlgorithmType`, `cv_score: f64`, `cv_std: f64`, `ci_lower: f64`, `ci_upper: f64`, `training_time_seconds: f64`, `params: HashMap<String, ParamValue>`.
- `ModelComparisonResults` (fit.rs:547–558): `pairwise_comparisons: Vec<PairwiseComparison>`, `best_is_significantly_better: bool`, `n_significantly_worse: usize`, `corrected_alpha: f64`, `correction_method: String` (the run pipeline sets it to `"Holm-Bonferroni"` — the spec's "BH" was inaccurate; use the field, never hardcode the method name).
- `PairwiseComparison` (fit.rs:652–682): `trial_id_1/2`, `model1_name/model2_name: String`, `mean_difference`, `p_value`, `p_value_corrected`, `ci_lower/ci_upper`, `significant`, `significant_corrected`.
- `competitive_models()` (fit.rs:214–231) returns `Vec<&LeaderboardEntry>`: the top model plus every model whose corrected pairwise test vs top is non-significant; falls back to just the top when `model_comparisons` is `None`. `models_competitive_with_best()` (fit.rs:584) keys everything on `pairwise_comparisons[0].trial_id_1` as "best" — fixtures MUST put the best model as `trial_id_1` in every pair.
- `AlgorithmComplexity` (portfolio.rs:612–620): `Low | Medium | High`, derives `Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize` — **no ordering derives yet** (Task 1 adds them). Per-algorithm complexity comes from `AlgorithmConfig::default_config(&algorithm)` returning `(SearchSpace, Vec<PreprocessingRequirement>, AlgorithmComplexity, bool)`.
- `AlgorithmType` (portfolio.rs:488+) has `pub fn name(&self) -> &'static str` (portfolio.rs:581). `LogisticRegression` is complexity `Low` (portfolio.rs:75–85, verified).
- `ModelCard` (model_card.rs:9–33): `name`, `task: Vec<String>`, `complexity: String`, `interpretability: String`, `supports_*: bool`, `strengths/limitations/references: Vec<String>`. `HasModelCard` (model_card.rs:36–39) is static: `fn model_card() -> ModelCard`. `ModelCardGenerator` does NOT exist yet. Every AutoML portfolio algorithm has a `HasModelCard` impl in model_card.rs (lines 45–1719).
- `serde_json` is already a ferroml-core dependency (Cargo.toml:52).
- Error idiom: `FerroError::not_fitted("...")` (used at fit.rs:162). Copy `fit.rs`'s existing `use` block paths for `FerroError`/`Result`/`Task`.
- Existing test fixtures: hand-built struct literals (see fit.rs:1846–1901 for a `ModelComparisonResults` literal); no builder helpers exist.

## File Structure

- Modify `ferroml-core/src/automl/portfolio.rs` — ordering derives + `AlgorithmType::complexity()` (Task 1)
- Modify `ferroml-core/src/automl/fit.rs` — `SelectionRationale`, `statistically_best_model()`, `ReportFormat`, `render_benchmark_report()` + tests (Tasks 2–3)
- Modify `ferroml-core/src/model_card.rs` — `model_card_for_algorithm()`, `ModelCardGenerator` + tests (Tasks 4–5)
- Modify `ferroml-python/src/automl.rs` — `PySelectionRationale`, three new methods on `PyAutoMLResult` (Tasks 6–7)
- Modify `ferroml-python/python/ferroml/automl/__init__.py` — re-exports (Task 8)
- Modify `ferroml-python/tests/test_automl_system.py` — end-to-end tests (Task 8)
- Modify `README.md` — "Intelligent model selection" section (Task 9)
- Modify `CHANGELOG.md`, version bump — release prep (Task 10, **Trey gate**)

---

### Task 1: Complexity ordering + lookup (P3.1a)

**Files:**
- Modify: `ferroml-core/src/automl/portfolio.rs` (enum at :612, `impl AlgorithmType` near :581)

**Interfaces:**
- Produces: `AlgorithmComplexity: Ord` with `Low < Medium < High`; `AlgorithmType::complexity(&self) -> AlgorithmComplexity`. Task 2 consumes both.

- [ ] **Step 1: Write the failing tests** — append inside the existing `#[cfg(test)] mod tests` block in portfolio.rs:

```rust
#[test]
fn test_complexity_is_ordered() {
    assert!(AlgorithmComplexity::Low < AlgorithmComplexity::Medium);
    assert!(AlgorithmComplexity::Medium < AlgorithmComplexity::High);
}

#[test]
fn test_algorithm_type_complexity_lookup() {
    // LogisticRegression is Low per default_config (portfolio.rs:75-85)
    assert_eq!(
        AlgorithmType::LogisticRegression.complexity(),
        AlgorithmComplexity::Low
    );
    // The lookup must agree with default_config for every algorithm
    let alg = AlgorithmType::RandomForestClassifier;
    assert_eq!(alg.complexity(), AlgorithmConfig::default_config(&alg).2);
}
```

- [ ] **Step 2: Run to verify failure**

Run: `cargo test -p ferroml-core complexity -- --nocapture`
Expected: compile error — `<` not implemented / no method `complexity`.

- [ ] **Step 3: Implement** — add `PartialOrd, Ord` to the enum's derive list (declaration order Low → Medium → High already matches the intended ordering):

```rust
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub enum AlgorithmComplexity {
```

and add to the existing `impl AlgorithmType` block (the one containing `name()`):

```rust
/// Complexity class of this algorithm, as used for time-budget allocation.
#[must_use]
pub fn complexity(&self) -> AlgorithmComplexity {
    AlgorithmConfig::default_config(self).2
}
```

(If `default_config` takes `AlgorithmType` by value or is named differently, match the actual signature at portfolio.rs:65 — do not restructure it.)

- [ ] **Step 4: Run to verify pass**

Run: `cargo test -p ferroml-core complexity`
Expected: both tests PASS; zero failures elsewhere.

- [ ] **Step 5: Gates + commit**

```bash
cargo fmt --all && cargo clippy -p ferroml-core -- -D warnings
git add ferroml-core/src/automl/portfolio.rs
git commit -m "feat(automl): ordered AlgorithmComplexity + AlgorithmType::complexity() lookup"
```

---

### Task 2: `SelectionRationale` + `statistically_best_model()` (P3.1b)

**Files:**
- Modify: `ferroml-core/src/automl/fit.rs` (impl block after `competitive_models()` ~:231; struct near `ModelStatistics` ~:401; tests in `mod tests` ~:1554)

**Interfaces:**
- Consumes: `AlgorithmType::complexity()` and `Ord` on `AlgorithmComplexity` (Task 1).
- Produces: `pub struct SelectionRationale` (fields below, `Serialize`d — Task 3's JSON and Task 5's card embed it) and `pub fn statistically_best_model(&self) -> Option<(&LeaderboardEntry, SelectionRationale)>`.

- [ ] **Step 1: Write the failing tests** — append to `mod tests` in fit.rs. First the shared fixture helpers (test-local; no production builder exists, per repo convention of literal fixtures):

```rust
fn lb_entry(
    rank: usize,
    trial_id: usize,
    algorithm: AlgorithmType,
    cv_score: f64,
    training_time_seconds: f64,
) -> LeaderboardEntry {
    LeaderboardEntry {
        rank,
        trial_id,
        algorithm,
        cv_score,
        cv_std: 0.01,
        ci_lower: cv_score - 0.02,
        ci_upper: cv_score + 0.02,
        training_time_seconds,
        params: HashMap::new(),
    }
}

fn pairwise_vs_best(
    best_id: usize,
    other_id: usize,
    best_name: &str,
    other_name: &str,
    p_corrected: f64,
    significant_corrected: bool,
) -> PairwiseComparison {
    PairwiseComparison {
        trial_id_1: best_id, // models_competitive_with_best() keys on trial_id_1
        trial_id_2: other_id,
        model1_name: best_name.to_string(),
        model2_name: other_name.to_string(),
        test_name: "Corrected Resampled t-test".to_string(),
        mean_difference: 0.01,
        std_error: 0.01,
        statistic: 1.0,
        p_value: p_corrected / 2.0,
        p_value_corrected: p_corrected,
        ci_lower: -0.01,
        ci_upper: 0.03,
        significant: significant_corrected,
        significant_corrected,
    }
}

fn synthetic_result(
    leaderboard: Vec<LeaderboardEntry>,
    model_comparisons: Option<ModelComparisonResults>,
) -> AutoMLResult {
    let n = leaderboard.len();
    AutoMLResult {
        trials: vec![],
        leaderboard,
        ensemble: None,
        total_time_seconds: 1.0,
        n_successful_trials: n,
        n_failed_trials: 0,
        task: Task::Classification,
        metric_name: "accuracy".to_string(),
        maximize: true,
        data_characteristics: DataCharacteristics {
            n_samples: 100,
            n_features: 2,
            n_classes: 2,
            has_missing_values: false,
            has_categorical: false,
            feature_variance_ratio: 1.0,
            class_imbalance_ratio: 1.0,
            is_regression: false,
        },
        cv_folds: 5,
        best_model_stats: None,
        aggregated_importance: None,
        model_comparisons,
    }
}

fn comparisons(pairs: Vec<PairwiseComparison>) -> ModelComparisonResults {
    let n_worse = pairs.iter().filter(|p| p.significant_corrected).count();
    ModelComparisonResults {
        best_is_significantly_better: pairs.iter().all(|p| p.significant_corrected),
        n_significantly_worse: n_worse,
        corrected_alpha: 0.05,
        correction_method: "Holm-Bonferroni".to_string(),
        pairwise_comparisons: pairs,
    }
}
```

(If `DataCharacteristics`, `AlgorithmType`, or `Task` are not already in scope in `mod tests` via `use super::*;`, add explicit `use crate::automl::portfolio::{AlgorithmType, DataCharacteristics};` — check the module's existing imports first.)

Then the five behavior tests:

```rust
#[test]
fn test_selects_simpler_noninferior_model() {
    // SVC tops the leaderboard, LogisticRegression (Low complexity) is
    // statistically indistinguishable -> selection must prefer the simpler model.
    let result = synthetic_result(
        vec![
            lb_entry(1, 0, AlgorithmType::SVC, 0.95, 10.0),
            lb_entry(2, 1, AlgorithmType::LogisticRegression, 0.94, 0.5),
        ],
        Some(comparisons(vec![pairwise_vs_best(
            0, 1, "SVC", "LogisticRegression", 0.40, false,
        )])),
    );
    let (selected, rationale) = result.statistically_best_model().unwrap();
    assert_eq!(selected.trial_id, 1, "must pick the simpler non-inferior model");
    assert_eq!(rationale.selected_algorithm, "LogisticRegression");
    assert_eq!(rationale.top_algorithm, "SVC");
    assert_eq!(rationale.p_value_vs_top, Some(0.40));
    assert!(rationale.explanation.contains("not statistically distinguishable"));
}

#[test]
fn test_keeps_top_when_simpler_is_significantly_worse() {
    let result = synthetic_result(
        vec![
            lb_entry(1, 0, AlgorithmType::SVC, 0.95, 10.0),
            lb_entry(2, 1, AlgorithmType::LogisticRegression, 0.80, 0.5),
        ],
        Some(comparisons(vec![pairwise_vs_best(
            0, 1, "SVC", "LogisticRegression", 0.001, true,
        )])),
    );
    let (selected, rationale) = result.statistically_best_model().unwrap();
    assert_eq!(selected.trial_id, 0, "significantly-worse model must NOT be chosen");
    assert!(rationale.p_value_vs_top.is_none()); // selected == top: no test vs itself
}

#[test]
fn test_single_model_degenerate_case() {
    let result = synthetic_result(
        vec![lb_entry(1, 0, AlgorithmType::LogisticRegression, 0.9, 0.5)],
        None,
    );
    let (selected, rationale) = result.statistically_best_model().unwrap();
    assert_eq!(selected.trial_id, 0);
    assert_eq!(rationale.correction_method, "none");
    assert!(result.statistically_best_model().is_some());

    let empty = synthetic_result(vec![], None);
    assert!(empty.statistically_best_model().is_none());
}

#[test]
fn test_no_comparisons_falls_back_to_top() {
    // Two models but statistical tests were disabled -> only top is competitive.
    let result = synthetic_result(
        vec![
            lb_entry(1, 0, AlgorithmType::SVC, 0.95, 10.0),
            lb_entry(2, 1, AlgorithmType::LogisticRegression, 0.94, 0.5),
        ],
        None,
    );
    let (selected, _) = result.statistically_best_model().unwrap();
    assert_eq!(selected.trial_id, 0, "without evidence, do not downgrade from top-1");
}

#[test]
fn test_selection_invariant_to_leaderboard_permutation() {
    // Selection is keyed on trial IDs + (complexity, time), so permuting the
    // leaderboard vec must not change WHICH model is selected.
    let pairs = vec![
        pairwise_vs_best(0, 1, "SVC", "LogisticRegression", 0.40, false),
        pairwise_vs_best(0, 2, "SVC", "GaussianNB", 0.60, false),
    ];
    let entries = vec![
        lb_entry(1, 0, AlgorithmType::SVC, 0.95, 10.0),
        lb_entry(2, 1, AlgorithmType::LogisticRegression, 0.94, 0.5),
        lb_entry(3, 2, AlgorithmType::GaussianNB, 0.93, 0.2),
    ];
    let forward = synthetic_result(entries.clone(), Some(comparisons(pairs.clone())));
    let mut reversed_entries = entries;
    reversed_entries.reverse();
    let reversed = synthetic_result(reversed_entries, Some(comparisons(pairs)));

    let (a, _) = forward.statistically_best_model().unwrap();
    let (b, _) = reversed.statistically_best_model().unwrap();
    assert_eq!(a.trial_id, b.trial_id);
}
```

(Note: `test_selection_invariant_to_leaderboard_permutation` relies on distinct `training_time_seconds` per entry so ties are impossible; both `LogisticRegression` and `GaussianNB` are `Low`, so the tie-break by time picks `GaussianNB` at 0.2s in both orders.)

- [ ] **Step 2: Run to verify failure**

Run: `cargo test -p ferroml-core statistically_best -- --nocapture`
Expected: compile error — `statistically_best_model` not found.

- [ ] **Step 3: Implement.** Add the struct next to `ModelStatistics` (~:415):

```rust
/// Evidence for why `statistically_best_model()` chose a model.
///
/// The rationale is the product: "the simplest model not statistically
/// distinguishable from the best" plus the numbers that prove it.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SelectionRationale {
    /// Trial ID of the selected model
    pub selected_trial_id: usize,
    /// Algorithm name of the selected model
    pub selected_algorithm: String,
    /// CV score (mean) of the selected model
    pub selected_score: f64,
    /// CV score standard deviation of the selected model
    pub selected_std: f64,
    /// 95% CI of the selected model's CV score
    pub selected_ci: (f64, f64),
    /// Complexity class of the selected model
    pub selected_complexity: AlgorithmComplexity,
    /// Trial ID of the leaderboard top-1
    pub top_trial_id: usize,
    /// Algorithm name of the leaderboard top-1
    pub top_algorithm: String,
    /// CV score (mean) of the top-1
    pub top_score: f64,
    /// CV score standard deviation of the top-1
    pub top_std: f64,
    /// Complexity class of the top-1
    pub top_complexity: AlgorithmComplexity,
    /// Corrected p-value of the pairwise test vs the top-1
    /// (`None` when selected == top-1, or no comparison exists)
    pub p_value_vs_top: Option<f64>,
    /// Number of models statistically competitive with the top-1
    pub n_competitive: usize,
    /// Multiple-testing correction method used ("none" when unavailable)
    pub correction_method: String,
    /// One-sentence human-readable explanation
    pub explanation: String,
}
```

(`AlgorithmComplexity` must be imported in fit.rs — check the existing `use crate::automl::portfolio::...` line and extend it.)

Add the method to `impl AutoMLResult`, directly after `competitive_models()`:

```rust
/// Select the simplest model that is not statistically distinguishable
/// from the leaderboard top-1, together with the evidence for that choice.
///
/// Selection rule: among [`Self::competitive_models`] (the top-1 plus every
/// model whose corrected pairwise test vs the top-1 is non-significant),
/// pick the lowest [`AlgorithmComplexity`]; break ties by shorter training
/// time. Falls back to the top-1 when no comparisons were computed.
#[must_use]
pub fn statistically_best_model(&self) -> Option<(&LeaderboardEntry, SelectionRationale)> {
    let top = self.best_model()?;
    let competitive = self.competitive_models();
    let selected = competitive.iter().copied().min_by(|a, b| {
        a.algorithm
            .complexity()
            .cmp(&b.algorithm.complexity())
            .then(
                a.training_time_seconds
                    .partial_cmp(&b.training_time_seconds)
                    .unwrap_or(std::cmp::Ordering::Equal),
            )
    })?;

    let p_value_vs_top = if selected.trial_id == top.trial_id {
        None
    } else {
        self.model_comparisons.as_ref().and_then(|c| {
            c.pairwise_comparisons
                .iter()
                .find(|p| {
                    (p.trial_id_1 == top.trial_id && p.trial_id_2 == selected.trial_id)
                        || (p.trial_id_2 == top.trial_id && p.trial_id_1 == selected.trial_id)
                })
                .map(|p| p.p_value_corrected)
        })
    };

    let correction_method = self
        .model_comparisons
        .as_ref()
        .map_or_else(|| "none".to_string(), |c| c.correction_method.clone());

    let explanation = if selected.trial_id == top.trial_id {
        format!(
            "{} is the top-ranked model and no simpler statistically \
             competitive alternative exists.",
            selected.algorithm.name()
        )
    } else {
        let p_text = p_value_vs_top
            .map_or_else(|| "n/a".to_string(), |p| format!("{p:.4}"));
        format!(
            "{} was selected over top-ranked {} because it is not statistically \
             distinguishable from it (corrected p = {p_text}) and is a simpler \
             model class.",
            selected.algorithm.name(),
            top.algorithm.name(),
        )
    };

    let rationale = SelectionRationale {
        selected_trial_id: selected.trial_id,
        selected_algorithm: selected.algorithm.name().to_string(),
        selected_score: selected.cv_score,
        selected_std: selected.cv_std,
        selected_ci: (selected.ci_lower, selected.ci_upper),
        selected_complexity: selected.algorithm.complexity(),
        top_trial_id: top.trial_id,
        top_algorithm: top.algorithm.name().to_string(),
        top_score: top.cv_score,
        top_std: top.cv_std,
        top_complexity: top.algorithm.complexity(),
        p_value_vs_top,
        n_competitive: competitive.len(),
        correction_method,
        explanation,
    };
    Some((selected, rationale))
}
```

- [ ] **Step 4: Run to verify pass**

Run: `cargo test -p ferroml-core statistically_best && cargo test -p ferroml-core selection_invariant && cargo test -p ferroml-core -- automl`
Expected: all new tests PASS, no existing automl test regresses.

- [ ] **Step 5: Gates + commit**

```bash
cargo fmt --all && cargo clippy -p ferroml-core -- -D warnings
git add ferroml-core/src/automl/fit.rs
git commit -m "feat(automl): statistically_best_model() — simplest non-inferior selection with SelectionRationale"
```

---

### Task 3: `render_benchmark_report()` — Markdown + JSON (P3.2)

**Files:**
- Modify: `ferroml-core/src/automl/fit.rs` (same impl block; tests in `mod tests`)

**Interfaces:**
- Consumes: `statistically_best_model()` (Task 2); reuses Task 2's test fixtures (`synthetic_result`, `lb_entry`, `pairwise_vs_best`, `comparisons`).
- Produces: `pub enum ReportFormat { Markdown, Json }` and `pub fn render_benchmark_report(&self, format: ReportFormat) -> String`. Task 7 binds both.

- [ ] **Step 1: Write the failing tests** — append to `mod tests`:

```rust
#[test]
fn test_benchmark_report_markdown_sections() {
    let result = synthetic_result(
        vec![
            lb_entry(1, 0, AlgorithmType::SVC, 0.95, 10.0),
            lb_entry(2, 1, AlgorithmType::LogisticRegression, 0.94, 0.5),
        ],
        Some(comparisons(vec![pairwise_vs_best(
            0, 1, "SVC", "LogisticRegression", 0.40, false,
        )])),
    );
    let md = result.render_benchmark_report(ReportFormat::Markdown);
    assert!(md.contains("# AutoML Benchmark Report"));
    assert!(md.contains("## Leaderboard"));
    assert!(md.contains("## Pairwise significance"));
    assert!(md.contains("## Selection"));
    assert!(md.contains("SVC"));
    assert!(md.contains("LogisticRegression"));
    assert!(md.contains("Holm-Bonferroni"));
}

#[test]
fn test_benchmark_report_states_indistinguishable_honestly() {
    let result = synthetic_result(
        vec![
            lb_entry(1, 0, AlgorithmType::SVC, 0.95, 10.0),
            lb_entry(2, 1, AlgorithmType::LogisticRegression, 0.94, 0.5),
        ],
        Some(comparisons(vec![pairwise_vs_best(
            0, 1, "SVC", "LogisticRegression", 0.40, false,
        )])),
    );
    let md = result.render_benchmark_report(ReportFormat::Markdown);
    assert!(
        md.contains("statistically indistinguishable"),
        "the report must say plainly when models cannot be told apart"
    );
}

#[test]
fn test_benchmark_report_states_significance() {
    let result = synthetic_result(
        vec![
            lb_entry(1, 0, AlgorithmType::SVC, 0.95, 10.0),
            lb_entry(2, 1, AlgorithmType::LogisticRegression, 0.80, 0.5),
        ],
        Some(comparisons(vec![pairwise_vs_best(
            0, 1, "SVC", "LogisticRegression", 0.001, true,
        )])),
    );
    let md = result.render_benchmark_report(ReportFormat::Markdown);
    assert!(md.contains("statistically distinguishable"));
}

#[test]
fn test_benchmark_report_no_comparisons() {
    let result = synthetic_result(
        vec![lb_entry(1, 0, AlgorithmType::LogisticRegression, 0.9, 0.5)],
        None,
    );
    let md = result.render_benchmark_report(ReportFormat::Markdown);
    assert!(md.contains("No statistical comparisons were computed"));
}

#[test]
fn test_benchmark_report_json_schema_stable() {
    let result = synthetic_result(
        vec![
            lb_entry(1, 0, AlgorithmType::SVC, 0.95, 10.0),
            lb_entry(2, 1, AlgorithmType::LogisticRegression, 0.94, 0.5),
        ],
        Some(comparisons(vec![pairwise_vs_best(
            0, 1, "SVC", "LogisticRegression", 0.40, false,
        )])),
    );
    let json = result.render_benchmark_report(ReportFormat::Json);
    let v: serde_json::Value = serde_json::from_str(&json).expect("report must be valid JSON");
    assert_eq!(v["schema_version"], 1);
    assert_eq!(v["leaderboard"].as_array().unwrap().len(), 2);
    assert_eq!(v["leaderboard"][0]["model"], "SVC");
    assert_eq!(v["metric"]["name"], "accuracy");
    assert_eq!(v["comparisons"]["correction_method"], "Holm-Bonferroni");
    assert_eq!(v["selection"]["selected_algorithm"], "LogisticRegression");
    // schema stability: exactly these top-level keys
    let keys: Vec<&str> = v.as_object().unwrap().keys().map(String::as_str).collect();
    assert_eq!(
        keys,
        vec![
            "comparisons", "cv_folds", "dataset", "leaderboard", "metric",
            "schema_version", "selection", "task", "total_time_seconds",
        ]
    );
}
```

(serde_json object keys iterate in sorted order by default — the key list above is alphabetical for that reason. If the crate enables `preserve_order`, sort before comparing.)

- [ ] **Step 2: Run to verify failure**

Run: `cargo test -p ferroml-core benchmark_report`
Expected: compile error — `ReportFormat` / `render_benchmark_report` not found.

- [ ] **Step 3: Implement.** Add near `SelectionRationale`:

```rust
/// Output format for [`AutoMLResult::render_benchmark_report`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ReportFormat {
    /// Human-readable Markdown
    Markdown,
    /// Machine-readable JSON (schema_version 1)
    Json,
}
```

Add to `impl AutoMLResult`:

```rust
/// Render an honest benchmark report of this AutoML run.
///
/// The report states plainly when models are statistically
/// indistinguishable — absence of significance is reported as a result,
/// not hidden.
#[must_use]
pub fn render_benchmark_report(&self, format: ReportFormat) -> String {
    match format {
        ReportFormat::Markdown => self.benchmark_report_markdown(),
        ReportFormat::Json => self.benchmark_report_json(),
    }
}

fn benchmark_report_markdown(&self) -> String {
    let mut md = String::new();
    md.push_str("# AutoML Benchmark Report\n\n");
    md.push_str(&format!(
        "- **Task:** {:?} · **Metric:** {} ({}) · **CV folds:** {}\n",
        self.task,
        self.metric_name,
        if self.maximize { "higher is better" } else { "lower is better" },
        self.cv_folds
    ));
    md.push_str(&format!(
        "- **Dataset:** {} samples × {} features\n",
        self.data_characteristics.n_samples, self.data_characteristics.n_features
    ));
    md.push_str(&format!(
        "- **Trials:** {} succeeded, {} failed · **Total time:** {:.1}s\n\n",
        self.n_successful_trials, self.n_failed_trials, self.total_time_seconds
    ));

    md.push_str("## Leaderboard\n\n");
    md.push_str("| Rank | Model | CV mean | Std | 95% CI | Train time (s) |\n");
    md.push_str("|---|---|---|---|---|---|\n");
    for e in &self.leaderboard {
        md.push_str(&format!(
            "| {} | {} | {:.4} | {:.4} | [{:.4}, {:.4}] | {:.2} |\n",
            e.rank,
            e.algorithm.name(),
            e.cv_score,
            e.cv_std,
            e.ci_lower,
            e.ci_upper,
            e.training_time_seconds
        ));
    }
    md.push('\n');

    md.push_str("## Pairwise significance\n\n");
    match &self.model_comparisons {
        None => md.push_str(
            "No statistical comparisons were computed for this run \
             (statistical tests disabled or fewer than two successful models).\n\n",
        ),
        Some(c) => {
            md.push_str(&format!(
                "Corrected resampled t-tests with {} correction (corrected α = {:.4}).\n\n",
                c.correction_method, c.corrected_alpha
            ));
            md.push_str("| Model A | Model B | Δ mean | p (corrected) | Significant? |\n");
            md.push_str("|---|---|---|---|---|\n");
            for p in &c.pairwise_comparisons {
                md.push_str(&format!(
                    "| {} | {} | {:+.4} | {:.4} | {} |\n",
                    p.model1_name,
                    p.model2_name,
                    p.mean_difference,
                    p.p_value_corrected,
                    if p.significant_corrected { "yes" } else { "no" }
                ));
            }
            md.push('\n');
            let n_indistinguishable = c
                .pairwise_comparisons
                .iter()
                .filter(|p| !p.significant_corrected)
                .count();
            if n_indistinguishable > 0 {
                md.push_str(&format!(
                    "**Conclusion:** no statistically significant difference was found \
                     between the top-ranked model and {n_indistinguishable} \
                     competitor(s); these models are statistically indistinguishable \
                     on this dataset.\n\n"
                ));
            } else {
                md.push_str(
                    "**Conclusion:** the top-ranked model is statistically \
                     distinguishable from every competitor after correction.\n\n",
                );
            }
        }
    }

    md.push_str("## Selection\n\n");
    match self.statistically_best_model() {
        Some((_, rationale)) => {
            md.push_str(&rationale.explanation);
            md.push('\n');
        }
        None => md.push_str("No successful models — nothing to select.\n"),
    }
    md
}

fn benchmark_report_json(&self) -> String {
    let selection = self
        .statistically_best_model()
        .map(|(_, r)| serde_json::to_value(&r).unwrap_or(serde_json::Value::Null));
    let report = serde_json::json!({
        "schema_version": 1,
        "task": format!("{:?}", self.task),
        "metric": { "name": self.metric_name, "maximize": self.maximize },
        "cv_folds": self.cv_folds,
        "dataset": {
            "n_samples": self.data_characteristics.n_samples,
            "n_features": self.data_characteristics.n_features,
        },
        "total_time_seconds": self.total_time_seconds,
        "leaderboard": self.leaderboard.iter().map(|e| serde_json::json!({
            "rank": e.rank,
            "model": e.algorithm.name(),
            "cv_mean": e.cv_score,
            "cv_std": e.cv_std,
            "ci_95": [e.ci_lower, e.ci_upper],
            "training_time_seconds": e.training_time_seconds,
        })).collect::<Vec<_>>(),
        "comparisons": self.model_comparisons.as_ref().map(|c| serde_json::json!({
            "correction_method": c.correction_method,
            "corrected_alpha": c.corrected_alpha,
            "pairwise": c.pairwise_comparisons.iter().map(|p| serde_json::json!({
                "model_a": p.model1_name,
                "model_b": p.model2_name,
                "mean_difference": p.mean_difference,
                "p_value_corrected": p.p_value_corrected,
                "significant_corrected": p.significant_corrected,
            })).collect::<Vec<_>>(),
        })),
        "selection": selection,
    });
    serde_json::to_string_pretty(&report).unwrap_or_else(|_| "{}".to_string())
}
```

- [ ] **Step 4: Run to verify pass**

Run: `cargo test -p ferroml-core benchmark_report && cargo test -p ferroml-core -- automl`
Expected: all 5 new tests PASS; no automl regressions.

- [ ] **Step 5: Gates + commit**

```bash
cargo fmt --all && cargo clippy -p ferroml-core -- -D warnings
git add ferroml-core/src/automl/fit.rs
git commit -m "feat(automl): render_benchmark_report() — honest Markdown/JSON benchmark reporting"
```

---

### Task 4: `model_card_for_algorithm()` (P3.3a)

**Files:**
- Modify: `ferroml-core/src/model_card.rs`

**Interfaces:**
- Produces: `pub fn model_card_for_algorithm(algorithm: &AlgorithmType) -> ModelCard`. Task 5 consumes it.

- [ ] **Step 1: Write the failing test** — model_card.rs has no `mod tests` for this yet; append one at the bottom (source-file unit tests are the repo convention; do NOT create a new file in `ferroml-core/tests/`):

```rust
#[cfg(test)]
mod automl_card_tests {
    use super::*;
    use crate::automl::portfolio::AlgorithmType;

    #[test]
    fn test_model_card_for_algorithm_returns_populated_cards() {
        for alg in [
            AlgorithmType::LogisticRegression,
            AlgorithmType::GaussianNB,
            AlgorithmType::RandomForestClassifier,
            AlgorithmType::SVC,
            AlgorithmType::LinearRegression,
            AlgorithmType::HistGradientBoostingRegressor,
        ] {
            let card = model_card_for_algorithm(&alg);
            assert!(!card.name.is_empty(), "{alg:?} card has empty name");
            assert!(!card.strengths.is_empty(), "{alg:?} card has no strengths");
            assert!(!card.references.is_empty(), "{alg:?} card has no references");
        }
    }
}
```

- [ ] **Step 2: Run to verify failure**

Run: `cargo test -p ferroml-core model_card_for_algorithm`
Expected: compile error — function not found.

- [ ] **Step 3: Implement.** Add to model_card.rs (below the `HasModelCard` trait). The match must be **exhaustive with no wildcard arm** — the compiler then guarantees every portfolio algorithm gets a card. Map each `AlgorithmType` variant to the `model_card()` of its model type using the impl paths already present in this file (lines 45–1719), e.g.:

```rust
use crate::automl::portfolio::AlgorithmType;

/// Static model card for an AutoML portfolio algorithm.
///
/// Exhaustive by construction: adding an `AlgorithmType` variant without a
/// card is a compile error.
#[must_use]
pub fn model_card_for_algorithm(algorithm: &AlgorithmType) -> ModelCard {
    match algorithm {
        AlgorithmType::LogisticRegression => {
            crate::models::LogisticRegression::model_card()
        }
        AlgorithmType::GaussianNB => crate::models::naive_bayes::GaussianNB::model_card(),
        AlgorithmType::MultinomialNB => {
            crate::models::naive_bayes::MultinomialNB::model_card()
        }
        AlgorithmType::CategoricalNB => {
            crate::models::naive_bayes::CategoricalNB::model_card()
        }
        AlgorithmType::KNeighborsClassifier => {
            crate::models::knn::KNeighborsClassifier::model_card()
        }
        AlgorithmType::SVC => crate::models::svm::SVC::model_card(),
        AlgorithmType::LinearSVC => crate::models::svm::LinearSVC::model_card(),
        AlgorithmType::DecisionTreeClassifier => {
            crate::models::tree::DecisionTreeClassifier::model_card()
        }
        AlgorithmType::RandomForestClassifier => {
            crate::models::forest::RandomForestClassifier::model_card()
        }
        AlgorithmType::GradientBoostingClassifier => {
            crate::models::boosting::GradientBoostingClassifier::model_card()
        }
        AlgorithmType::HistGradientBoostingClassifier => {
            crate::models::hist_boosting::HistGradientBoostingClassifier::model_card()
        }
        AlgorithmType::LinearRegression => crate::models::LinearRegression::model_card(),
        AlgorithmType::Ridge => crate::models::RidgeRegression::model_card(),
        AlgorithmType::Lasso => crate::models::LassoRegression::model_card(),
        AlgorithmType::ElasticNet => crate::models::ElasticNet::model_card(),
        AlgorithmType::QuantileRegression => {
            crate::models::quantile::QuantileRegression::model_card()
        }
        AlgorithmType::RobustRegression => {
            crate::models::robust::RobustRegression::model_card()
        }
        AlgorithmType::KNeighborsRegressor => {
            crate::models::knn::KNeighborsRegressor::model_card()
        }
        AlgorithmType::SVR => crate::models::svm::SVR::model_card(),
        AlgorithmType::LinearSVR => crate::models::svm::LinearSVR::model_card(),
        AlgorithmType::DecisionTreeRegressor => {
            crate::models::tree::DecisionTreeRegressor::model_card()
        }
        AlgorithmType::RandomForestRegressor => {
            crate::models::forest::RandomForestRegressor::model_card()
        }
        AlgorithmType::GradientBoostingRegressor => {
            crate::models::boosting::GradientBoostingRegressor::model_card()
        }
        AlgorithmType::HistGradientBoostingRegressor => {
            crate::models::hist_boosting::HistGradientBoostingRegressor::model_card()
        }
    }
}
```

The variant list above covers portfolio.rs:488–553. **If the enum has additional variants past line 553, the compiler will name them — add an arm for each, using the matching `impl HasModelCard for <path>` already present in this file.** Every portfolio algorithm has one. Do not add a `_ =>` arm under any circumstances.

- [ ] **Step 4: Run to verify pass**

Run: `cargo test -p ferroml-core model_card_for_algorithm`
Expected: PASS.

- [ ] **Step 5: Gates + commit**

```bash
cargo fmt --all && cargo clippy -p ferroml-core -- -D warnings
git add ferroml-core/src/model_card.rs
git commit -m "feat(model-card): model_card_for_algorithm() — exhaustive AlgorithmType→ModelCard mapping"
```

---

### Task 5: `ModelCardGenerator::from_automl_result()` (P3.3b)

**Files:**
- Modify: `ferroml-core/src/model_card.rs`

**Interfaces:**
- Consumes: `model_card_for_algorithm()` (Task 4), `AutoMLResult::statistically_best_model()` (Task 2).
- Produces: `pub struct ModelCardGenerator` with `pub fn from_automl_result(result: &AutoMLResult) -> Result<String>` (Markdown). Task 7 binds it.

- [ ] **Step 1: Write the failing tests** — append inside `mod automl_card_tests`. The fixture helpers are duplicated from Task 2's tests because fit.rs test helpers are private to its module (repeat-don't-import is fine in test code):

```rust
use crate::automl::fit::{
    AutoMLResult, LeaderboardEntry, ModelComparisonResults, PairwiseComparison,
};
use crate::automl::portfolio::DataCharacteristics;
use std::collections::HashMap;

fn card_test_result() -> AutoMLResult {
    let entries = vec![
        LeaderboardEntry {
            rank: 1,
            trial_id: 0,
            algorithm: AlgorithmType::SVC,
            cv_score: 0.95,
            cv_std: 0.01,
            ci_lower: 0.93,
            ci_upper: 0.97,
            training_time_seconds: 10.0,
            params: HashMap::new(),
        },
        LeaderboardEntry {
            rank: 2,
            trial_id: 1,
            algorithm: AlgorithmType::LogisticRegression,
            cv_score: 0.94,
            cv_std: 0.01,
            ci_lower: 0.92,
            ci_upper: 0.96,
            training_time_seconds: 0.5,
            params: HashMap::new(),
        },
    ];
    let comparisons = ModelComparisonResults {
        pairwise_comparisons: vec![PairwiseComparison {
            trial_id_1: 0,
            trial_id_2: 1,
            model1_name: "SVC".to_string(),
            model2_name: "LogisticRegression".to_string(),
            test_name: "Corrected Resampled t-test".to_string(),
            mean_difference: 0.01,
            std_error: 0.01,
            statistic: 1.0,
            p_value: 0.20,
            p_value_corrected: 0.40,
            ci_lower: -0.01,
            ci_upper: 0.03,
            significant: false,
            significant_corrected: false,
        }],
        best_is_significantly_better: false,
        n_significantly_worse: 0,
        corrected_alpha: 0.05,
        correction_method: "Holm-Bonferroni".to_string(),
    };
    AutoMLResult {
        trials: vec![],
        leaderboard: entries,
        ensemble: None,
        total_time_seconds: 12.0,
        n_successful_trials: 2,
        n_failed_trials: 0,
        task: crate::automl::config::Task::Classification, // match fit.rs's Task path
        metric_name: "accuracy".to_string(),
        maximize: true,
        data_characteristics: DataCharacteristics {
            n_samples: 100,
            n_features: 2,
            n_classes: 2,
            has_missing_values: false,
            has_categorical: false,
            feature_variance_ratio: 1.0,
            class_imbalance_ratio: 1.0,
            is_regression: false,
        },
        cv_folds: 5,
        best_model_stats: None,
        aggregated_importance: None,
        model_comparisons: Some(comparisons),
    }
}

#[test]
fn test_from_automl_result_renders_selected_model_card() {
    let result = card_test_result();
    let card = ModelCardGenerator::from_automl_result(&result).unwrap();
    // Card is for the SELECTED model (LogisticRegression), not the top-1 (SVC)
    assert!(card.starts_with("# Model Card: LogisticRegression"));
    assert!(card.contains("## Why this model"));
    assert!(card.contains("## Performance (this run)"));
    assert!(card.contains("0.9400")); // selected CV score
    assert!(card.contains("CV folds: 5"));
    assert!(card.contains("### Strengths"));
    assert!(card.contains("### References"));
}

#[test]
fn test_from_automl_result_errors_on_empty_result() {
    let mut result = card_test_result();
    result.leaderboard.clear();
    result.n_successful_trials = 0;
    assert!(ModelCardGenerator::from_automl_result(&result).is_err());
}
```

(If `Task` lives at a different path than `crate::automl::config::Task`, use whatever path fit.rs imports — check its `use` block. Same for `Result`/`FerroError` below.)

- [ ] **Step 2: Run to verify failure**

Run: `cargo test -p ferroml-core from_automl_result`
Expected: compile error — `ModelCardGenerator` not found.

- [ ] **Step 3: Implement** — add to model_card.rs:

```rust
use crate::automl::fit::AutoMLResult;
use crate::error::{FerroError, Result}; // match the error-module path used in fit.rs

/// Generates run-specific model cards from AutoML results.
pub struct ModelCardGenerator;

impl ModelCardGenerator {
    /// Render a Markdown model card for the model selected by
    /// [`AutoMLResult::statistically_best_model`], merging the algorithm's
    /// static card with this run's evidence.
    ///
    /// # Errors
    /// Returns [`FerroError::NotFitted`] when the result has no successful models.
    pub fn from_automl_result(result: &AutoMLResult) -> Result<String> {
        let (entry, rationale) = result.statistically_best_model().ok_or_else(|| {
            FerroError::not_fitted(
                "No successful models in AutoML result — cannot generate a model card",
            )
        })?;
        let card = model_card_for_algorithm(&entry.algorithm);
        let d = &result.data_characteristics;

        let mut md = String::new();
        md.push_str(&format!("# Model Card: {}\n\n", card.name));
        md.push_str(&format!(
            "*Generated by FerroML AutoML from a {} × {} dataset ({}).*\n\n",
            d.n_samples,
            d.n_features,
            if d.is_regression { "regression" } else { "classification" }
        ));

        md.push_str("## Why this model\n\n");
        md.push_str(&rationale.explanation);
        md.push_str("\n\n");
        md.push_str(&format!(
            "- Correction method: {} · statistically competitive models: {}\n",
            rationale.correction_method, rationale.n_competitive
        ));
        if let Some(p) = rationale.p_value_vs_top {
            md.push_str(&format!(
                "- Corrected p-value vs top-ranked {}: {p:.4}\n",
                rationale.top_algorithm
            ));
        }
        md.push('\n');

        md.push_str("## Performance (this run)\n\n");
        md.push_str(&format!(
            "- CV {}: {:.4} ± {:.4}\n- 95% CI: [{:.4}, {:.4}]\n- CV folds: {}\n- Training time: {:.2}s\n\n",
            result.metric_name,
            entry.cv_score,
            entry.cv_std,
            entry.ci_lower,
            entry.ci_upper,
            result.cv_folds,
            entry.training_time_seconds
        ));

        md.push_str("## Algorithm profile\n\n");
        md.push_str(&format!(
            "- Task: {}\n- Time complexity: {}\n- Interpretability: {}\n\n",
            card.task.join(", "),
            card.complexity,
            card.interpretability
        ));

        md.push_str("### Strengths\n\n");
        for s in &card.strengths {
            md.push_str(&format!("- {s}\n"));
        }
        md.push_str("\n### Limitations\n\n");
        for l in &card.limitations {
            md.push_str(&format!("- {l}\n"));
        }
        md.push_str("\n### References\n\n");
        for r in &card.references {
            md.push_str(&format!("- {r}\n"));
        }

        Ok(md)
    }
}
```

- [ ] **Step 4: Run to verify pass**

Run: `cargo test -p ferroml-core from_automl_result && cargo test -p ferroml-core`
Expected: new tests PASS; full core suite green.

- [ ] **Step 5: Gates + commit**

```bash
cargo fmt --all && cargo clippy -p ferroml-core -- -D warnings
git add ferroml-core/src/model_card.rs
git commit -m "feat(model-card): ModelCardGenerator::from_automl_result() — run-specific Markdown model cards"
```

---

### Task 6: PyO3 — `PySelectionRationale` + `statistically_best_model()` (P3.4a)

**Files:**
- Modify: `ferroml-python/src/automl.rs` (PyAutoMLResult `#[pymethods]` near :634; registration fn at :1034–1047)

**Interfaces:**
- Consumes: `SelectionRationale`, `statistically_best_model()` (Task 2); existing `PyLeaderboardEntry { inner }` idiom (automl.rs:634–638).
- Produces: Python-visible `SelectionRationale` class; `AutoMLResult.statistically_best_model() -> Optional[tuple[LeaderboardEntry, SelectionRationale]]`. Task 8's tests consume.

- [ ] **Step 1: Implement wrapper** (Rust-side binding; the RED test for this task is Task 8's pytest suite — bindings can't be tested without building the extension, so Tasks 6–8 share one RED/GREEN cycle at the Python level). Add to automl.rs, extending the existing `use` of ferroml-core automl types to also import `SelectionRationale` and `ReportFormat`:

```rust
/// Evidence for why AutoML selected a model.
#[pyclass(name = "SelectionRationale", module = "ferroml.automl")]
pub struct PySelectionRationale {
    inner: SelectionRationale,
}

#[pymethods]
impl PySelectionRationale {
    #[getter]
    fn selected_algorithm(&self) -> &str {
        &self.inner.selected_algorithm
    }
    #[getter]
    fn selected_score(&self) -> f64 {
        self.inner.selected_score
    }
    #[getter]
    fn selected_std(&self) -> f64 {
        self.inner.selected_std
    }
    #[getter]
    fn selected_ci(&self) -> (f64, f64) {
        self.inner.selected_ci
    }
    #[getter]
    fn selected_complexity(&self) -> String {
        format!("{:?}", self.inner.selected_complexity)
    }
    #[getter]
    fn top_algorithm(&self) -> &str {
        &self.inner.top_algorithm
    }
    #[getter]
    fn top_score(&self) -> f64 {
        self.inner.top_score
    }
    #[getter]
    fn p_value_vs_top(&self) -> Option<f64> {
        self.inner.p_value_vs_top
    }
    #[getter]
    fn n_competitive(&self) -> usize {
        self.inner.n_competitive
    }
    #[getter]
    fn correction_method(&self) -> &str {
        &self.inner.correction_method
    }
    #[getter]
    fn explanation(&self) -> &str {
        &self.inner.explanation
    }
    fn __repr__(&self) -> String {
        format!("SelectionRationale({})", self.inner.explanation)
    }
}
```

And on `PyAutoMLResult`'s `#[pymethods]` block (next to `best_model`, :634):

```rust
/// Select the simplest model not statistically distinguishable from the
/// best, with the evidence for that choice.
fn statistically_best_model(&self) -> Option<(PyLeaderboardEntry, PySelectionRationale)> {
    self.inner.statistically_best_model().map(|(entry, rationale)| {
        (
            PyLeaderboardEntry {
                inner: entry.clone(),
            },
            PySelectionRationale { inner: rationale },
        )
    })
}
```

Register the class in `register_automl_module` (:1034), after the existing `add_class` lines:

```rust
automl_module.add_class::<PySelectionRationale>()?;
```

- [ ] **Step 2: Verify it compiles**

Run: `cargo check -p ferroml-python`
Expected: clean. (If `PyLeaderboardEntry`'s `inner` field is private to another module, construct it however `best_model()` at :634 does — same file, so `inner` is accessible.)

- [ ] **Step 3: Gates + commit**

```bash
cargo fmt --all && cargo clippy -p ferroml-python -- -D warnings
git add ferroml-python/src/automl.rs
git commit -m "feat(python): bind statistically_best_model() + SelectionRationale"
```

---

### Task 7: PyO3 — `benchmark_report()` + `model_card()` (P3.4b)

**Files:**
- Modify: `ferroml-python/src/automl.rs`

**Interfaces:**
- Consumes: `ReportFormat`, `render_benchmark_report()` (Task 3), `ModelCardGenerator` (Task 5), `crate::errors::ferro_to_pyerr` (errors.rs:43).
- Produces: `AutoMLResult.benchmark_report(format="markdown"|"json") -> str`, `AutoMLResult.model_card() -> str`.

- [ ] **Step 1: Implement** — add to `PyAutoMLResult`'s `#[pymethods]`:

```rust
/// Render an honest benchmark report ("markdown" or "json").
#[pyo3(signature = (format = "markdown"))]
fn benchmark_report(&self, format: &str) -> PyResult<String> {
    let fmt = match format {
        "markdown" => ReportFormat::Markdown,
        "json" => ReportFormat::Json,
        other => {
            return Err(pyo3::exceptions::PyValueError::new_err(format!(
                "format must be 'markdown' or 'json', got '{other}'"
            )))
        }
    };
    Ok(self.inner.render_benchmark_report(fmt))
}

/// Markdown model card for the statistically selected model.
fn model_card(&self) -> PyResult<String> {
    ModelCardGenerator::from_automl_result(&self.inner).map_err(crate::errors::ferro_to_pyerr)
}
```

(Import `ModelCardGenerator` from ferroml-core using the same crate path style the file's existing `use` block uses for `AutoMLResult`.)

- [ ] **Step 2: Verify it compiles**

Run: `cargo check -p ferroml-python && cargo clippy -p ferroml-python -- -D warnings`
Expected: clean.

- [ ] **Step 3: Commit**

```bash
cargo fmt --all
git add ferroml-python/src/automl.rs
git commit -m "feat(python): bind benchmark_report() and model_card() on AutoMLResult"
```

---

### Task 8: Python re-exports + end-to-end tests (P3.4c)

**Files:**
- Modify: `ferroml-python/python/ferroml/automl/__init__.py` (:53–70)
- Modify: `ferroml-python/tests/test_automl_system.py`

**Interfaces:**
- Consumes: everything from Tasks 6–7.
- Produces: `from ferroml.automl import SelectionRationale`; green pytest suite.

- [ ] **Step 1: Re-export.** In `__init__.py`, after `LeaderboardEntry = _native.automl.LeaderboardEntry` add:

```python
SelectionRationale = _native.automl.SelectionRationale
```

and add `"SelectionRationale"` to `__all__`.

- [ ] **Step 2: Write the failing tests** — append to `test_automl_system.py`, following the file's existing conventions (`@pytest.mark.slow` classes, `make_classification_data()` / `_clf_config()` helpers). Check `_clf_config()`: if it disables statistical tests, enable them for this class (mirror whatever config field controls it; the Rust config field is `statistical_tests: bool`):

```python
@pytest.mark.slow
class TestIntelligentSelection:
    """P3: statistically-grounded selection, honest reporting, model cards."""

    def _fit(self):
        X, y = make_classification_data()
        config = _clf_config()
        automl = AutoML(config)
        result = automl.fit(X, y)
        assert result.is_successful()
        return result

    def test_statistically_best_model_returns_entry_and_rationale(self):
        result = self._fit()
        selection = result.statistically_best_model()
        assert selection is not None
        entry, rationale = selection
        assert rationale.explanation
        assert rationale.selected_algorithm
        assert rationale.n_competitive >= 1
        assert rationale.correction_method
        assert 0.0 <= rationale.selected_score <= 1.0

    def test_benchmark_report_markdown_and_json(self):
        result = self._fit()
        md = result.benchmark_report()
        assert "# AutoML Benchmark Report" in md
        assert "## Leaderboard" in md

        import json
        report = json.loads(result.benchmark_report(format="json"))
        assert report["schema_version"] == 1
        assert len(report["leaderboard"]) >= 1
        assert report["selection"] is not None

        with pytest.raises(ValueError):
            result.benchmark_report(format="xml")

    def test_model_card_renders_markdown(self):
        result = self._fit()
        card = result.model_card()
        assert card.startswith("# Model Card:")
        assert "## Why this model" in card
        assert "## Performance (this run)" in card
```

- [ ] **Step 3: Run to verify failure, then build, then pass**

```bash
source .venv/bin/activate
pytest ferroml-python/tests/test_automl_system.py -k IntelligentSelection -x -q  # RED: AttributeError (old .so)
maturin develop --release -m ferroml-python/Cargo.toml
pytest ferroml-python/tests/test_automl_system.py -q                              # GREEN: all pass
```

Expected: before rebuild, `AttributeError: ... has no attribute 'statistically_best_model'`; after rebuild, whole file passes.

- [ ] **Step 4: Full binding-side gate**

Run: `cargo test -p ferroml-python && pytest ferroml-python/tests/ -q`
Expected: green (this catches binding-side Rust tests `cargo test` alone skips).

- [ ] **Step 5: Commit**

```bash
git add ferroml-python/python/ferroml/automl/__init__.py ferroml-python/tests/test_automl_system.py
git commit -m "feat(python): re-export SelectionRationale + end-to-end intelligent-selection tests"
```

---

### Task 9: README "Intelligent model selection" section (P3.4d)

**Files:**
- Modify: `README.md` (add section after the existing AutoML section — locate the `automl` example and insert below it)

- [ ] **Step 1: Generate real output.** Run this snippet in the venv (with the Task 8 build) and capture actual output — the README must show a real run, not invented numbers:

```python
import numpy as np
from ferroml.automl import AutoML, AutoMLConfig
# build a small classification dataset the same way test_automl_system.py does,
# fit, then print:
entry, rationale = result.statistically_best_model()
print(rationale.explanation)
print(result.benchmark_report()[:2000])
```

- [ ] **Step 2: Write the section** (replace the output block with the real captured output):

````markdown
## Intelligent model selection

Most AutoML picks the leaderboard top-1 and calls it a day. FerroML computes
corrected resampled t-tests (Nadeau-Bengio) across models, applies
multiple-testing correction, and then selects **the simplest model that is not
statistically distinguishable from the best** — and tells you why:

```python
result = automl.fit(X, y)

entry, rationale = result.statistically_best_model()
print(rationale.explanation)
# <real output here>

print(result.benchmark_report())        # honest Markdown benchmark report
report = result.benchmark_report(format="json")  # machine-readable, schema_version 1
card = result.model_card()               # Markdown model card for the selected model
```

The benchmark report says "statistically indistinguishable" out loud when
that's the truth — absence of significance is a result, not an embarrassment.
````

- [ ] **Step 3: Commit**

```bash
git add README.md
git commit -m "docs: README section for intelligent model selection"
```

---

### Task 10: Release v1.1.0 (P3.5) — **STOP FOR TREY BEFORE PUBLISH**

**Files:**
- Modify: `CHANGELOG.md`, workspace `Cargo.toml` (version), `ferroml-python/Cargo.toml` / `pyproject.toml` if versions are declared there (check how v1.0.1 bumped these: `git show v1.0.1 --stat`)

- [ ] **Step 1: CHANGELOG entry** summarizing the three features + Python API additions.
- [ ] **Step 2: Version bump** to 1.1.0 exactly the way the v1.0.0→v1.0.1 bump did it (inspect that commit; the CLI version test is dynamic, ccdc617).
- [ ] **Step 3: Full local gate**: `cargo test --all && pytest ferroml-python/tests/ -q` after a fresh `maturin develop --release`.
- [ ] **Step 4: Push and verify CI green** on all matrices.
- [ ] **Step 5: HARD STOP — Trey gate.** Present the diff summary and CI link; only after Trey approves: tag `v1.1.0`, push tag, confirm publish-pypi.yml succeeds, verify `pip install ferroml==1.1.0` in a clean venv.

---

## Self-review notes (done at authoring time)

- **Spec coverage:** spec features 1/2/3 → Tasks 2/3/4+5; Python surface → 6/7/8; README example → 9; release → 10. Testing bullets (a)–(d) → Task 2 tests a–d + Task 3 phrasing tests; property test → `test_selection_invariant_to_leaderboard_permutation`. ✔
- **Known deviations from spec text:** correction method is Holm-Bonferroni (not BH as the spec said) — reports read the `correction_method` field, so no behavior difference. `statistically_best_model()` returns `Option<(&LeaderboardEntry, SelectionRationale)>` rather than a bespoke struct — simpler, reuses existing types.
- **Type consistency:** `SelectionRationale` field names match across Task 2 (definition), Task 3 (JSON embeds via serde), Task 5 (card), Task 6 (getters). `ReportFormat` names match Tasks 3/7. ✔
- **Deliberate flexibility points** (single-line adjustments the implementer resolves against the actual code, all flagged inline): exact `use` paths for `Task`/`FerroError`/`Result`, `default_config` call shape, enum variants past portfolio.rs:553 (compiler-enforced), `_clf_config()` statistical-tests flag.
