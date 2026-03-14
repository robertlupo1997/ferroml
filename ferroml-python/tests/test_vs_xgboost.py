"""Cross-library validation: FerroML vs XGBoost

Tests gradient boosting and histogram boosting models against XGBoost
for correctness and competitive performance.
"""

import time
import numpy as np
import pytest
from sklearn.datasets import make_classification, make_regression
from sklearn.model_selection import train_test_split
from scipy.stats import pearsonr

# ─── helpers ────────────────────────────────────────────────────────


def make_cls_data(n=1000, p=20, seed=42):
    X, y = make_classification(
        n_samples=n, n_features=p, n_informative=p // 2,
        n_redundant=p // 4, random_state=seed,
    )
    return train_test_split(X, y.astype(np.float64), test_size=0.3, random_state=seed)


def make_reg_data(n=1000, p=20, seed=42):
    X, y = make_regression(
        n_samples=n, n_features=p, n_informative=p // 2,
        noise=10.0, random_state=seed,
    )
    return train_test_split(X, y, test_size=0.3, random_state=seed)


def timed(fn, *args, n_runs=3, **kwargs):
    times = []
    result = None
    for _ in range(n_runs):
        t0 = time.perf_counter()
        result = fn(*args, **kwargs)
        times.append(time.perf_counter() - t0)
    return result, np.median(times)


def r2_score(y_true, y_pred):
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    return 1.0 - ss_res / ss_tot if ss_tot > 0 else 1.0


# ─── GradientBoosting Regressor vs XGBRegressor ─────────────────────


class TestGBRegressorVsXGB:
    @pytest.fixture(autouse=True)
    def setup(self):
        from ferroml.trees import GradientBoostingRegressor
        import xgboost as xgb
        self.FerroGBR = GradientBoostingRegressor
        self.XGBRegressor = xgb.XGBRegressor

    @pytest.mark.parametrize("n", [1000, 5000])
    def test_r2_competitive(self, n):
        X_train, X_test, y_train, y_test = make_reg_data(n=n)

        ferro = self.FerroGBR(n_estimators=100, max_depth=5, learning_rate=0.1)
        ferro.fit(X_train, y_train)
        ferro_pred = ferro.predict(X_test)
        ferro_r2 = r2_score(y_test, ferro_pred)

        xgb_model = self.XGBRegressor(
            n_estimators=100, max_depth=5, learning_rate=0.1,
            random_state=42, verbosity=0,
        )
        xgb_model.fit(X_train, y_train)
        xgb_pred = xgb_model.predict(X_test)
        xgb_r2 = r2_score(y_test, xgb_pred)

        assert ferro_r2 > 0.5, f"FerroML GBR R² too low: {ferro_r2:.4f}"
        assert xgb_r2 > 0.5, f"XGBoost GBR R² too low: {xgb_r2:.4f}"

        # Prediction correlation should be high
        corr, _ = pearsonr(ferro_pred, xgb_pred)
        assert corr > 0.85, f"GBR prediction correlation: {corr:.4f}"

    def test_prediction_agreement(self):
        X_train, X_test, y_train, y_test = make_reg_data(n=2000)

        ferro = self.FerroGBR(n_estimators=50, max_depth=3, learning_rate=0.1)
        ferro.fit(X_train, y_train)
        ferro_pred = ferro.predict(X_test)

        xgb_model = self.XGBRegressor(
            n_estimators=50, max_depth=3, learning_rate=0.1,
            random_state=42, verbosity=0,
        )
        xgb_model.fit(X_train, y_train)
        xgb_pred = xgb_model.predict(X_test)

        # Relative error should be bounded
        relative_rmse = np.sqrt(np.mean((ferro_pred - xgb_pred) ** 2)) / np.std(y_test)
        assert relative_rmse < 0.5, f"GBR relative RMSE vs XGB: {relative_rmse:.4f}"


# ─── GradientBoosting Classifier vs XGBClassifier ───────────────────


class TestGBClassifierVsXGB:
    @pytest.fixture(autouse=True)
    def setup(self):
        from ferroml.trees import GradientBoostingClassifier
        import xgboost as xgb
        self.FerroGBC = GradientBoostingClassifier
        self.XGBClassifier = xgb.XGBClassifier

    @pytest.mark.parametrize("n", [1000, 5000])
    def test_accuracy_competitive(self, n):
        X_train, X_test, y_train, y_test = make_cls_data(n=n)

        ferro = self.FerroGBC(n_estimators=100, max_depth=5, learning_rate=0.1)
        ferro.fit(X_train, y_train)
        ferro_pred = ferro.predict(X_test)
        ferro_acc = np.mean(ferro_pred == y_test)

        xgb_model = self.XGBClassifier(
            n_estimators=100, max_depth=5, learning_rate=0.1,
            random_state=42, verbosity=0, eval_metric="logloss",
        )
        xgb_model.fit(X_train, y_train)
        xgb_pred = xgb_model.predict(X_test)
        xgb_acc = np.mean(xgb_pred == y_test)

        assert ferro_acc > 0.70, f"FerroML GBC acc: {ferro_acc:.3f}"
        assert xgb_acc > 0.70, f"XGBoost GBC acc: {xgb_acc:.3f}"
        assert abs(ferro_acc - xgb_acc) < 0.15, \
            f"GBC accuracy gap: ferro={ferro_acc:.3f}, xgb={xgb_acc:.3f}"

    def test_prediction_agreement(self):
        X_train, X_test, y_train, y_test = make_cls_data(n=2000)

        ferro = self.FerroGBC(n_estimators=50, max_depth=3, learning_rate=0.1)
        ferro.fit(X_train, y_train)
        ferro_pred = ferro.predict(X_test)

        xgb_model = self.XGBClassifier(
            n_estimators=50, max_depth=3, learning_rate=0.1,
            random_state=42, verbosity=0, eval_metric="logloss",
        )
        xgb_model.fit(X_train, y_train)
        xgb_pred = xgb_model.predict(X_test)

        agreement = np.mean(ferro_pred == xgb_pred)
        assert agreement > 0.80, f"GBC prediction agreement: {agreement:.3f}"


# ─── HistGradientBoosting vs XGBoost ────────────────────────────────


class TestHistGBVsXGB:
    @pytest.fixture(autouse=True)
    def setup(self):
        from ferroml.trees import HistGradientBoostingRegressor, HistGradientBoostingClassifier
        import xgboost as xgb
        self.FerroHGBR = HistGradientBoostingRegressor
        self.FerroHGBC = HistGradientBoostingClassifier
        self.XGBRegressor = xgb.XGBRegressor
        self.XGBClassifier = xgb.XGBClassifier

    def test_regressor_competitive(self):
        X_train, X_test, y_train, y_test = make_reg_data(n=5000, p=30)

        ferro = self.FerroHGBR(max_iter=100, max_depth=5, learning_rate=0.1)
        ferro.fit(X_train, y_train)
        ferro_pred = ferro.predict(X_test)
        ferro_r2 = r2_score(y_test, ferro_pred)

        xgb_model = self.XGBRegressor(
            n_estimators=100, max_depth=5, learning_rate=0.1,
            tree_method="hist", random_state=42, verbosity=0,
        )
        xgb_model.fit(X_train, y_train)
        xgb_pred = xgb_model.predict(X_test)
        xgb_r2 = r2_score(y_test, xgb_pred)

        assert ferro_r2 > 0.5, f"FerroML HGB R²: {ferro_r2:.4f}"
        corr, _ = pearsonr(ferro_pred, xgb_pred)
        assert corr > 0.80, f"HGB vs XGB correlation: {corr:.4f}"

    def test_classifier_competitive(self):
        X_train, X_test, y_train, y_test = make_cls_data(n=5000, p=30)

        ferro = self.FerroHGBC(max_iter=100, max_depth=5, learning_rate=0.1)
        ferro.fit(X_train, y_train)
        ferro_pred = ferro.predict(X_test)
        ferro_acc = np.mean(ferro_pred == y_test)

        xgb_model = self.XGBClassifier(
            n_estimators=100, max_depth=5, learning_rate=0.1,
            tree_method="hist", random_state=42, verbosity=0, eval_metric="logloss",
        )
        xgb_model.fit(X_train, y_train)
        xgb_pred = xgb_model.predict(X_test)
        xgb_acc = np.mean(xgb_pred == y_test)

        assert ferro_acc > 0.70, f"FerroML HGB acc: {ferro_acc:.3f}"
        assert abs(ferro_acc - xgb_acc) < 0.15, \
            f"HGB accuracy gap: ferro={ferro_acc:.3f}, xgb={xgb_acc:.3f}"


# ─── Performance timing ─────────────────────────────────────────────


class TestPerformanceVsXGB:
    @pytest.fixture(autouse=True)
    def setup(self):
        from ferroml.trees import GradientBoostingRegressor
        import xgboost as xgb
        self.FerroGBR = GradientBoostingRegressor
        self.XGBRegressor = xgb.XGBRegressor

    @pytest.mark.parametrize("n", [1000, 5000])
    def test_timing_recorded(self, n):
        """Record timing — not a pass/fail, just documentation."""
        X_train, X_test, y_train, y_test = make_reg_data(n=n)

        ferro = self.FerroGBR(n_estimators=50, max_depth=5, learning_rate=0.1)
        _, ferro_fit_t = timed(ferro.fit, X_train, y_train)
        _, ferro_pred_t = timed(ferro.predict, X_test, n_runs=5)

        xgb_model = self.XGBRegressor(
            n_estimators=50, max_depth=5, learning_rate=0.1,
            random_state=42, verbosity=0,
        )
        _, xgb_fit_t = timed(xgb_model.fit, X_train, y_train)
        _, xgb_pred_t = timed(xgb_model.predict, X_test, n_runs=5)

        print(f"\n[n={n}] FerroML fit: {ferro_fit_t*1000:.0f}ms, predict: {ferro_pred_t*1000:.1f}ms")
        print(f"[n={n}] XGBoost fit: {xgb_fit_t*1000:.0f}ms, predict: {xgb_pred_t*1000:.1f}ms")
        print(f"[n={n}] Fit speedup: {xgb_fit_t/ferro_fit_t:.2f}x, Pred speedup: {xgb_pred_t/ferro_pred_t:.2f}x")
