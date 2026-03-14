"""Cross-library validation: FerroML vs LightGBM

Tests histogram gradient boosting against LightGBM for correctness
and competitive performance.
"""

import numpy as np
import pytest
from sklearn.datasets import make_classification, make_regression
from sklearn.model_selection import train_test_split
from scipy.stats import pearsonr


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


def r2_score(y_true, y_pred):
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    return 1.0 - ss_res / ss_tot if ss_tot > 0 else 1.0


# ─── HistGradientBoosting Regressor vs LightGBM ─────────────────────


class TestHGBRegressorVsLGBM:
    @pytest.fixture(autouse=True)
    def setup(self):
        from ferroml.trees import HistGradientBoostingRegressor
        import lightgbm as lgb
        self.FerroHGBR = HistGradientBoostingRegressor
        self.LGBMRegressor = lgb.LGBMRegressor

    @pytest.mark.parametrize("n", [1000, 5000, 10000])
    def test_r2_competitive(self, n):
        X_train, X_test, y_train, y_test = make_reg_data(n=n)

        ferro = self.FerroHGBR(max_iter=100, max_depth=5, learning_rate=0.1)
        ferro.fit(X_train, y_train)
        ferro_pred = ferro.predict(X_test)
        ferro_r2 = r2_score(y_test, ferro_pred)

        lgbm = self.LGBMRegressor(
            n_estimators=100, max_depth=5, learning_rate=0.1,
            random_state=42, verbosity=-1,
        )
        lgbm.fit(X_train, y_train)
        lgbm_pred = lgbm.predict(X_test)
        lgbm_r2 = r2_score(y_test, lgbm_pred)

        assert ferro_r2 > 0.5, f"FerroML HGB R²: {ferro_r2:.4f}"
        assert lgbm_r2 > 0.5, f"LightGBM R²: {lgbm_r2:.4f}"

        corr, _ = pearsonr(ferro_pred, lgbm_pred)
        assert corr > 0.80, f"HGB vs LGBM correlation: {corr:.4f}"

    def test_feature_importance_correlated(self):
        X_train, X_test, y_train, y_test = make_reg_data(n=5000, p=10)

        ferro = self.FerroHGBR(max_iter=100, max_depth=5, learning_rate=0.1)
        ferro.fit(X_train, y_train)
        ferro_imp = np.array(ferro.feature_importances_)

        lgbm = self.LGBMRegressor(
            n_estimators=100, max_depth=5, learning_rate=0.1,
            random_state=42, verbosity=-1,
        )
        lgbm.fit(X_train, y_train)
        lgbm_imp = lgbm.feature_importances_.astype(float)

        # Normalize
        ferro_imp = ferro_imp / ferro_imp.sum() if ferro_imp.sum() > 0 else ferro_imp
        lgbm_imp = lgbm_imp / lgbm_imp.sum() if lgbm_imp.sum() > 0 else lgbm_imp

        # Top-3 important features should overlap
        ferro_top3 = set(np.argsort(ferro_imp)[-3:])
        lgbm_top3 = set(np.argsort(lgbm_imp)[-3:])
        overlap = len(ferro_top3 & lgbm_top3)
        assert overlap >= 2, f"Top-3 feature overlap: {overlap}/3 (ferro={ferro_top3}, lgbm={lgbm_top3})"


# ─── HistGradientBoosting Classifier vs LightGBM ────────────────────


class TestHGBClassifierVsLGBM:
    @pytest.fixture(autouse=True)
    def setup(self):
        from ferroml.trees import HistGradientBoostingClassifier
        import lightgbm as lgb
        self.FerroHGBC = HistGradientBoostingClassifier
        self.LGBMClassifier = lgb.LGBMClassifier

    @pytest.mark.parametrize("n", [1000, 5000, 10000])
    def test_accuracy_competitive(self, n):
        X_train, X_test, y_train, y_test = make_cls_data(n=n)

        ferro = self.FerroHGBC(max_iter=100, max_depth=5, learning_rate=0.1)
        ferro.fit(X_train, y_train)
        ferro_pred = ferro.predict(X_test)
        ferro_acc = np.mean(ferro_pred == y_test)

        lgbm = self.LGBMClassifier(
            n_estimators=100, max_depth=5, learning_rate=0.1,
            random_state=42, verbosity=-1,
        )
        lgbm.fit(X_train, y_train)
        lgbm_pred = lgbm.predict(X_test)
        lgbm_acc = np.mean(lgbm_pred == y_test)

        assert ferro_acc > 0.70, f"FerroML HGB acc: {ferro_acc:.3f}"
        assert lgbm_acc > 0.70, f"LightGBM acc: {lgbm_acc:.3f}"
        assert abs(ferro_acc - lgbm_acc) < 0.15, \
            f"HGB accuracy gap: ferro={ferro_acc:.3f}, lgbm={lgbm_acc:.3f}"

    def test_prediction_agreement(self):
        X_train, X_test, y_train, y_test = make_cls_data(n=5000)

        ferro = self.FerroHGBC(max_iter=50, max_depth=3, learning_rate=0.1)
        ferro.fit(X_train, y_train)
        ferro_pred = ferro.predict(X_test)

        lgbm = self.LGBMClassifier(
            n_estimators=50, max_depth=3, learning_rate=0.1,
            random_state=42, verbosity=-1,
        )
        lgbm.fit(X_train, y_train)
        lgbm_pred = lgbm.predict(X_test)

        agreement = np.mean(ferro_pred == lgbm_pred)
        assert agreement > 0.80, f"HGB prediction agreement with LGBM: {agreement:.3f}"


# ─── GradientBoosting Regressor vs LightGBM (exact mode) ────────────


class TestGBRegressorVsLGBM:
    @pytest.fixture(autouse=True)
    def setup(self):
        from ferroml.trees import GradientBoostingRegressor
        import lightgbm as lgb
        self.FerroGBR = GradientBoostingRegressor
        self.LGBMRegressor = lgb.LGBMRegressor

    def test_r2_competitive(self):
        X_train, X_test, y_train, y_test = make_reg_data(n=5000)

        ferro = self.FerroGBR(n_estimators=100, max_depth=5, learning_rate=0.1)
        ferro.fit(X_train, y_train)
        ferro_pred = ferro.predict(X_test)
        ferro_r2 = r2_score(y_test, ferro_pred)

        lgbm = self.LGBMRegressor(
            n_estimators=100, max_depth=5, learning_rate=0.1,
            random_state=42, verbosity=-1,
        )
        lgbm.fit(X_train, y_train)
        lgbm_pred = lgbm.predict(X_test)
        lgbm_r2 = r2_score(y_test, lgbm_pred)

        assert ferro_r2 > 0.5, f"FerroML GBR R²: {ferro_r2:.4f}"
        corr, _ = pearsonr(ferro_pred, lgbm_pred)
        assert corr > 0.80, f"GBR vs LGBM correlation: {corr:.4f}"
