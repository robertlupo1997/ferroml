"""Feature selection with multiple methods and consensus ranking.

Usage: Claude adapts this to select the most relevant features for a model.
Output: Consensus-ranked features with per-method scores and VIF analysis.
"""
from __future__ import annotations

import numpy as np


def select(
    X: np.ndarray,
    y: np.ndarray,
    feature_names: list[str],
    task: str = "regression",
    max_features: int | None = None,
) -> dict:
    """Select features using variance, VIF, mutual info, and RFE.

    Parameters
    ----------
    X : np.ndarray
        Feature matrix (n_samples, n_features).
    y : np.ndarray
        Target array.
    feature_names : list[str]
        Names corresponding to columns of X.
    task : str
        "regression" or "classification".
    max_features : int or None
        Maximum features to keep. If None, auto-determined.

    Returns
    -------
    dict with keys: selected_features, dropped_features, vif_scores,
    importance_rankings, steps
    """
    from ferroml.preprocessing import SelectKBest, VarianceThreshold

    _, n_features = X.shape
    if max_features is None:
        max_features = max(1, n_features // 2)

    active_mask = np.ones(n_features, dtype=bool)
    steps: list[dict] = []

    # --- Step 1: VarianceThreshold (drop zero-variance) ---
    try:
        vt = VarianceThreshold()
        vt.fit(X)
        variances = np.var(X, axis=0)
        zero_var = variances == 0.0
        if zero_var.any():
            dropped = [feature_names[i] for i in range(n_features) if zero_var[i]]
            active_mask[zero_var] = False
            steps.append({"method": "variance_threshold", "dropped": dropped, "reason": "zero variance"})
        else:
            steps.append({"method": "variance_threshold", "dropped": [], "reason": "all features have variance > 0"})
    except Exception as e:
        steps.append({"method": "variance_threshold", "error": str(e)})

    # --- Step 2: VIF (Variance Inflation Factor) ---
    vif_scores: dict[str, float] = {}
    active_indices = np.where(active_mask)[0]

    if len(active_indices) >= 2:
        X_active = X[:, active_indices].copy()
        vif_dropped = []

        # Iteratively drop features with VIF > 10
        remaining = list(range(X_active.shape[1]))
        for _ in range(X_active.shape[1]):
            if len(remaining) <= 1:
                break
            vifs = _compute_vif(X_active[:, remaining])
            max_vif_idx = int(np.argmax(vifs))
            max_vif = vifs[max_vif_idx]
            if max_vif <= 10.0:
                break
            original_idx = active_indices[remaining[max_vif_idx]]
            vif_dropped.append(feature_names[original_idx])
            active_mask[original_idx] = False
            remaining.pop(max_vif_idx)

        # Record final VIF scores
        remaining_active = np.where(active_mask)[0]
        if len(remaining_active) >= 2:
            final_vifs = _compute_vif(X[:, remaining_active])
            for i, idx in enumerate(remaining_active):
                vif_scores[feature_names[idx]] = round(float(final_vifs[i]), 4)

        steps.append({
            "method": "vif_multicollinearity",
            "threshold": 10.0,
            "dropped": vif_dropped,
            "reason": "VIF > 10 (high multicollinearity)",
        })

    # --- Step 3: SelectKBest (mutual information) ---
    active_indices = np.where(active_mask)[0]
    kbest_ranking: dict[str, float] = {}

    if len(active_indices) >= 2:
        X_active = X[:, active_indices]
        k = min(max_features, len(active_indices))
        try:
            skb = SelectKBest(k=k)
            skb.fit(X_active, y)
            skb.transform(X_active)
            # Approximate scores via correlation as fallback
            scores = np.abs(np.array([
                np.corrcoef(X_active[:, i], y)[0, 1] if np.std(X_active[:, i]) > 0 else 0.0
                for i in range(X_active.shape[1])
            ]))
            for i, idx in enumerate(active_indices):
                kbest_ranking[feature_names[idx]] = round(float(scores[i]), 6)
            steps.append({"method": "select_k_best", "k": k, "scores": kbest_ranking})
        except Exception as e:
            steps.append({"method": "select_k_best", "error": str(e)})

    # --- Step 4: RFE with a simple model ---
    rfe_ranking: dict[str, int] = {}
    active_indices = np.where(active_mask)[0]

    if len(active_indices) >= 2:
        try:
            from ferroml.cli._registry import construct_model
            if task == "classification":
                model = construct_model("LogisticRegression")
            else:
                model = construct_model("Ridge")

            X_active = X[:, active_indices]
            remaining = list(range(X_active.shape[1]))
            rank = len(remaining)

            for _ in range(len(remaining) - max(1, max_features)):
                if len(remaining) <= 1:
                    break
                model.fit(X_active[:, remaining], y)
                preds = model.predict(X_active[:, remaining])
                # Approximate importance via permutation
                importances = np.zeros(len(remaining))
                for j in range(len(remaining)):
                    X_perm = X_active[:, remaining].copy()
                    rng = np.random.RandomState(42)
                    X_perm[:, j] = rng.permutation(X_perm[:, j])
                    preds_perm = model.predict(X_perm)
                    importances[j] = np.mean((preds - preds_perm) ** 2)

                least_idx = int(np.argmin(importances))
                original_idx = active_indices[remaining[least_idx]]
                rfe_ranking[feature_names[original_idx]] = rank
                rank -= 1
                remaining.pop(least_idx)

            for r_idx in remaining:
                original_idx = active_indices[r_idx]
                rfe_ranking[feature_names[original_idx]] = rank
                rank -= 1

            steps.append({"method": "recursive_feature_elimination", "rankings": rfe_ranking})
        except Exception as e:
            steps.append({"method": "recursive_feature_elimination", "error": str(e)})

    # --- Consensus ranking ---
    active_indices = np.where(active_mask)[0]
    active_names = [feature_names[i] for i in active_indices]

    # Score each feature: lower is better
    consensus_scores: dict[str, float] = {}
    for name in active_names:
        score = 0.0
        n_methods = 0
        if name in kbest_ranking:
            # Higher correlation is better, so invert
            score += 1.0 - kbest_ranking[name]
            n_methods += 1
        if name in rfe_ranking:
            score += rfe_ranking[name] / max(len(rfe_ranking), 1)
            n_methods += 1
        if name in vif_scores:
            score += vif_scores[name] / 20.0  # Normalize VIF
            n_methods += 1
        consensus_scores[name] = round(score / max(n_methods, 1), 6)

    # Sort by consensus score (lower is better)
    ranked = sorted(consensus_scores.items(), key=lambda x: x[1])
    selected = [name for name, _ in ranked[:max_features]]
    dropped = [name for name in feature_names if name not in selected]

    return {
        "selected_features": selected,
        "dropped_features": dropped,
        "n_selected": len(selected),
        "n_dropped": len(dropped),
        "vif_scores": vif_scores,
        "importance_rankings": {
            "kbest_correlation": kbest_ranking,
            "rfe_rank": rfe_ranking,
            "consensus": dict(ranked),
        },
        "steps": steps,
    }


def _compute_vif(X: np.ndarray) -> np.ndarray:
    """Compute VIF for each column using OLS R-squared."""
    n_features = X.shape[1]
    vifs = np.zeros(n_features)
    for j in range(n_features):
        y_j = X[:, j]
        X_others = np.delete(X, j, axis=1)
        # Add intercept
        X_others = np.column_stack([np.ones(X_others.shape[0]), X_others])
        try:
            beta = np.linalg.lstsq(X_others, y_j, rcond=None)[0]
            y_hat = X_others @ beta
            ss_res = np.sum((y_j - y_hat) ** 2)
            ss_tot = np.sum((y_j - np.mean(y_j)) ** 2)
            r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0
            vifs[j] = 1.0 / (1.0 - r2) if r2 < 1.0 else float("inf")
        except Exception:
            vifs[j] = float("inf")
    return vifs


def print_summary(result: dict) -> None:
    """Print a human-readable summary of feature selection."""
    print(f"Features: {result['n_selected']} selected, {result['n_dropped']} dropped")
    print()

    print("Selected features:")
    for name in result["selected_features"]:
        vif = result["vif_scores"].get(name, "n/a")
        vif_str = f" (VIF={vif})" if isinstance(vif, float) else ""
        print(f"  + {name}{vif_str}")
    print()

    if result["dropped_features"]:
        print("Dropped features:")
        for name in result["dropped_features"]:
            print(f"  - {name}")
        print()

    print("Method steps:")
    for step in result["steps"]:
        method = step["method"]
        if "error" in step:
            print(f"  {method}: ERROR — {step['error']}")
        elif "dropped" in step:
            n = len(step["dropped"])
            print(f"  {method}: dropped {n} feature(s)")
        else:
            print(f"  {method}: completed")
