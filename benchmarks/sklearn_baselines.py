#!/usr/bin/env python3
"""Generate sklearn baseline fixtures for FerroML accuracy tests.

Outputs JSON fixture files to benchmarks/fixtures/ containing training data,
predictions, scores, and model parameters for comparison against FerroML.
"""

import json
import os
import numpy as np
from pathlib import Path

FIXTURES_DIR = Path(__file__).parent / "fixtures"
FIXTURES_DIR.mkdir(exist_ok=True)

def to_list(arr):
    """Convert numpy array to nested list for JSON serialization."""
    if hasattr(arr, 'tolist'):
        return arr.tolist()
    return arr

def save_fixture(filename, data):
    """Save fixture data as JSON."""
    path = FIXTURES_DIR / filename
    with open(path, 'w') as f:
        json.dump(data, f, indent=2)
    print(f"  Saved {path}")

# =============================================================================
# Datasets
# =============================================================================

def make_iris():
    from sklearn.datasets import load_iris
    data = load_iris()
    return data.data, data.target.astype(float)

def make_diabetes():
    from sklearn.datasets import load_diabetes
    data = load_diabetes()
    return data.data, data.target

def make_wine():
    from sklearn.datasets import load_wine
    data = load_wine()
    return data.data, data.target.astype(float)

def make_binary_classification(n=100, seed=42):
    from sklearn.datasets import make_classification
    X, y = make_classification(n_samples=n, n_features=5, n_informative=3,
                                n_redundant=1, random_state=seed)
    return X, y.astype(float)

# =============================================================================
# Model Fixtures
# =============================================================================

def generate_logistic_regression():
    print("Generating logistic_regression.json...")
    from sklearn.linear_model import LogisticRegression
    X, y = make_iris()
    X_train, X_test = X[:120], X[120:]
    y_train, y_test = y[:120], y[120:]

    model = LogisticRegression(random_state=42, max_iter=200)
    model.fit(X_train, y_train)

    save_fixture("logistic_regression.json", {
        "model": "LogisticRegression",
        "params": {"random_state": 42, "max_iter": 200},
        "dataset": "iris",
        "X_train": to_list(X_train),
        "y_train": to_list(y_train),
        "X_test": to_list(X_test),
        "y_test": to_list(y_test),
        "predictions": to_list(model.predict(X_test)),
        "predict_proba": to_list(model.predict_proba(X_test)),
        "accuracy": float(model.score(X_test, y_test)),
        "coef": to_list(model.coef_),
        "intercept": to_list(model.intercept_),
    })

def generate_random_forest():
    print("Generating random_forest.json...")
    from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

    # Classifier on iris
    X_iris, y_iris = make_iris()
    X_train_c, X_test_c = X_iris[:120], X_iris[120:]
    y_train_c, y_test_c = y_iris[:120], y_iris[120:]

    rfc = RandomForestClassifier(n_estimators=10, random_state=42, n_jobs=1)
    rfc.fit(X_train_c, y_train_c)

    # Regressor on diabetes
    X_diab, y_diab = make_diabetes()
    X_train_r, X_test_r = X_diab[:350], X_diab[350:]
    y_train_r, y_test_r = y_diab[:350], y_diab[350:]

    rfr = RandomForestRegressor(n_estimators=10, random_state=42, n_jobs=1)
    rfr.fit(X_train_r, y_train_r)

    save_fixture("random_forest.json", {
        "classifier": {
            "model": "RandomForestClassifier",
            "params": {"n_estimators": 10, "random_state": 42},
            "dataset": "iris",
            "X_train": to_list(X_train_c),
            "y_train": to_list(y_train_c),
            "X_test": to_list(X_test_c),
            "y_test": to_list(y_test_c),
            "predictions": to_list(rfc.predict(X_test_c)),
            "accuracy": float(rfc.score(X_test_c, y_test_c)),
            "feature_importances": to_list(rfc.feature_importances_),
        },
        "regressor": {
            "model": "RandomForestRegressor",
            "params": {"n_estimators": 10, "random_state": 42},
            "dataset": "diabetes",
            "X_train": to_list(X_train_r),
            "y_train": to_list(y_train_r),
            "X_test": to_list(X_test_r),
            "y_test": to_list(y_test_r),
            "predictions": to_list(rfr.predict(X_test_r)),
            "r2_score": float(rfr.score(X_test_r, y_test_r)),
            "feature_importances": to_list(rfr.feature_importances_),
        },
    })

def generate_gradient_boosting():
    print("Generating gradient_boosting.json...")
    from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor

    X_iris, y_iris = make_iris()
    X_train_c, X_test_c = X_iris[:120], X_iris[120:]
    y_train_c, y_test_c = y_iris[:120], y_iris[120:]

    gbc = GradientBoostingClassifier(n_estimators=10, random_state=42, learning_rate=0.1, max_depth=3)
    gbc.fit(X_train_c, y_train_c)

    X_diab, y_diab = make_diabetes()
    X_train_r, X_test_r = X_diab[:350], X_diab[350:]
    y_train_r, y_test_r = y_diab[:350], y_diab[350:]

    gbr = GradientBoostingRegressor(n_estimators=10, random_state=42, learning_rate=0.1, max_depth=3)
    gbr.fit(X_train_r, y_train_r)

    save_fixture("gradient_boosting.json", {
        "classifier": {
            "model": "GradientBoostingClassifier",
            "params": {"n_estimators": 10, "random_state": 42, "learning_rate": 0.1, "max_depth": 3},
            "dataset": "iris",
            "X_train": to_list(X_train_c),
            "y_train": to_list(y_train_c),
            "X_test": to_list(X_test_c),
            "y_test": to_list(y_test_c),
            "predictions": to_list(gbc.predict(X_test_c)),
            "accuracy": float(gbc.score(X_test_c, y_test_c)),
            "feature_importances": to_list(gbc.feature_importances_),
        },
        "regressor": {
            "model": "GradientBoostingRegressor",
            "params": {"n_estimators": 10, "random_state": 42, "learning_rate": 0.1, "max_depth": 3},
            "dataset": "diabetes",
            "X_train": to_list(X_train_r),
            "y_train": to_list(y_train_r),
            "X_test": to_list(X_test_r),
            "y_test": to_list(y_test_r),
            "predictions": to_list(gbr.predict(X_test_r)),
            "r2_score": float(gbr.score(X_test_r, y_test_r)),
            "feature_importances": to_list(gbr.feature_importances_),
        },
    })

def generate_knn():
    print("Generating knn.json...")
    from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor

    X_iris, y_iris = make_iris()
    X_train_c, X_test_c = X_iris[:120], X_iris[120:]
    y_train_c, y_test_c = y_iris[:120], y_iris[120:]

    knnc = KNeighborsClassifier(n_neighbors=5)
    knnc.fit(X_train_c, y_train_c)

    X_diab, y_diab = make_diabetes()
    X_train_r, X_test_r = X_diab[:350], X_diab[350:]
    y_train_r, y_test_r = y_diab[:350], y_diab[350:]

    knnr = KNeighborsRegressor(n_neighbors=5)
    knnr.fit(X_train_r, y_train_r)

    save_fixture("knn.json", {
        "classifier": {
            "model": "KNeighborsClassifier",
            "params": {"n_neighbors": 5},
            "dataset": "iris",
            "X_train": to_list(X_train_c),
            "y_train": to_list(y_train_c),
            "X_test": to_list(X_test_c),
            "y_test": to_list(y_test_c),
            "predictions": to_list(knnc.predict(X_test_c)),
            "accuracy": float(knnc.score(X_test_c, y_test_c)),
        },
        "regressor": {
            "model": "KNeighborsRegressor",
            "params": {"n_neighbors": 5},
            "dataset": "diabetes",
            "X_train": to_list(X_train_r),
            "y_train": to_list(y_train_r),
            "X_test": to_list(X_test_r),
            "y_test": to_list(y_test_r),
            "predictions": to_list(knnr.predict(X_test_r)),
            "r2_score": float(knnr.score(X_test_r, y_test_r)),
        },
    })

def generate_naive_bayes():
    print("Generating naive_bayes.json...")
    from sklearn.naive_bayes import GaussianNB, MultinomialNB, BernoulliNB

    X_iris, y_iris = make_iris()
    X_train, X_test = X_iris[:120], X_iris[120:]
    y_train, y_test = y_iris[:120], y_iris[120:]

    gnb = GaussianNB()
    gnb.fit(X_train, y_train)

    # MultinomialNB needs non-negative features
    X_mn = X_iris - X_iris.min(axis=0)
    X_train_mn, X_test_mn = X_mn[:120], X_mn[120:]

    mnb = MultinomialNB(alpha=1.0)
    mnb.fit(X_train_mn, y_train)

    # BernoulliNB with binarization
    bnb = BernoulliNB(alpha=1.0, binarize=3.0)
    bnb.fit(X_train, y_train)

    save_fixture("naive_bayes.json", {
        "gaussian": {
            "model": "GaussianNB",
            "params": {},
            "dataset": "iris",
            "X_train": to_list(X_train),
            "y_train": to_list(y_train),
            "X_test": to_list(X_test),
            "y_test": to_list(y_test),
            "predictions": to_list(gnb.predict(X_test)),
            "predict_proba": to_list(gnb.predict_proba(X_test)),
            "accuracy": float(gnb.score(X_test, y_test)),
        },
        "multinomial": {
            "model": "MultinomialNB",
            "params": {"alpha": 1.0},
            "dataset": "iris_nonneg",
            "X_train": to_list(X_train_mn),
            "y_train": to_list(y_train),
            "X_test": to_list(X_test_mn),
            "y_test": to_list(y_test),
            "predictions": to_list(mnb.predict(X_test_mn)),
            "predict_proba": to_list(mnb.predict_proba(X_test_mn)),
            "accuracy": float(mnb.score(X_test_mn, y_test)),
        },
        "bernoulli": {
            "model": "BernoulliNB",
            "params": {"alpha": 1.0, "binarize": 3.0},
            "dataset": "iris",
            "X_train": to_list(X_train),
            "y_train": to_list(y_train),
            "X_test": to_list(X_test),
            "y_test": to_list(y_test),
            "predictions": to_list(bnb.predict(X_test)),
            "predict_proba": to_list(bnb.predict_proba(X_test)),
            "accuracy": float(bnb.score(X_test, y_test)),
        },
    })

def generate_svm():
    print("Generating svm.json...")
    from sklearn.svm import SVC, SVR
    from sklearn.preprocessing import StandardScaler

    # SVC on iris (scaled)
    X_iris, y_iris = make_iris()
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_iris)
    X_train_c, X_test_c = X_scaled[:120], X_scaled[120:]
    y_train_c, y_test_c = y_iris[:120], y_iris[120:]

    svc = SVC(kernel='rbf', C=1.0, random_state=42, probability=True)
    svc.fit(X_train_c, y_train_c)

    # SVR on diabetes (scaled)
    X_diab, y_diab = make_diabetes()
    scaler_r = StandardScaler()
    X_diab_scaled = scaler_r.fit_transform(X_diab)
    X_train_r, X_test_r = X_diab_scaled[:350], X_diab_scaled[350:]
    y_train_r, y_test_r = y_diab[:350], y_diab[350:]

    svr = SVR(kernel='rbf', C=1.0, epsilon=0.1)
    svr.fit(X_train_r, y_train_r)

    save_fixture("svm.json", {
        "classifier": {
            "model": "SVC",
            "params": {"kernel": "rbf", "C": 1.0, "random_state": 42},
            "dataset": "iris_scaled",
            "X_train": to_list(X_train_c),
            "y_train": to_list(y_train_c),
            "X_test": to_list(X_test_c),
            "y_test": to_list(y_test_c),
            "predictions": to_list(svc.predict(X_test_c)),
            "accuracy": float(svc.score(X_test_c, y_test_c)),
        },
        "regressor": {
            "model": "SVR",
            "params": {"kernel": "rbf", "C": 1.0, "epsilon": 0.1},
            "dataset": "diabetes_scaled",
            "X_train": to_list(X_train_r),
            "y_train": to_list(y_train_r),
            "X_test": to_list(X_test_r),
            "y_test": to_list(y_test_r),
            "predictions": to_list(svr.predict(X_test_r)),
            "r2_score": float(svr.score(X_test_r, y_test_r)),
        },
    })

def generate_regularized():
    print("Generating regularized.json...")
    from sklearn.linear_model import ElasticNet, RidgeCV, LassoCV

    X_diab, y_diab = make_diabetes()
    X_train, X_test = X_diab[:350], X_diab[350:]
    y_train, y_test = y_diab[:350], y_diab[350:]

    en = ElasticNet(alpha=0.1, l1_ratio=0.5, random_state=42, max_iter=10000)
    en.fit(X_train, y_train)

    rcv = RidgeCV(alphas=[0.1, 1.0, 10.0], cv=5)
    rcv.fit(X_train, y_train)

    lcv = LassoCV(n_alphas=20, cv=5, random_state=42, max_iter=10000)
    lcv.fit(X_train, y_train)

    save_fixture("regularized.json", {
        "elasticnet": {
            "model": "ElasticNet",
            "params": {"alpha": 0.1, "l1_ratio": 0.5, "random_state": 42},
            "dataset": "diabetes",
            "X_train": to_list(X_train),
            "y_train": to_list(y_train),
            "X_test": to_list(X_test),
            "y_test": to_list(y_test),
            "predictions": to_list(en.predict(X_test)),
            "r2_score": float(en.score(X_test, y_test)),
            "coef": to_list(en.coef_),
            "intercept": float(en.intercept_),
        },
        "ridgecv": {
            "model": "RidgeCV",
            "params": {"alphas": [0.1, 1.0, 10.0], "cv": 5},
            "dataset": "diabetes",
            "X_train": to_list(X_train),
            "y_train": to_list(y_train),
            "X_test": to_list(X_test),
            "y_test": to_list(y_test),
            "predictions": to_list(rcv.predict(X_test)),
            "r2_score": float(rcv.score(X_test, y_test)),
            "best_alpha": float(rcv.alpha_),
            "coef": to_list(rcv.coef_),
            "intercept": float(rcv.intercept_),
        },
        "lassocv": {
            "model": "LassoCV",
            "params": {"n_alphas": 20, "cv": 5, "random_state": 42},
            "dataset": "diabetes",
            "X_train": to_list(X_train),
            "y_train": to_list(y_train),
            "X_test": to_list(X_test),
            "y_test": to_list(y_test),
            "predictions": to_list(lcv.predict(X_test)),
            "r2_score": float(lcv.score(X_test, y_test)),
            "best_alpha": float(lcv.alpha_),
            "coef": to_list(lcv.coef_),
            "intercept": float(lcv.intercept_),
        },
    })

def generate_adaboost():
    print("Generating adaboost.json...")
    from sklearn.ensemble import AdaBoostClassifier, AdaBoostRegressor

    X_iris, y_iris = make_iris()
    X_train_c, X_test_c = X_iris[:120], X_iris[120:]
    y_train_c, y_test_c = y_iris[:120], y_iris[120:]

    abc = AdaBoostClassifier(n_estimators=10, random_state=42)
    abc.fit(X_train_c, y_train_c)

    X_diab, y_diab = make_diabetes()
    X_train_r, X_test_r = X_diab[:350], X_diab[350:]
    y_train_r, y_test_r = y_diab[:350], y_diab[350:]

    abr = AdaBoostRegressor(n_estimators=10, random_state=42)
    abr.fit(X_train_r, y_train_r)

    save_fixture("adaboost.json", {
        "classifier": {
            "model": "AdaBoostClassifier",
            "params": {"n_estimators": 10, "random_state": 42},
            "dataset": "iris",
            "X_train": to_list(X_train_c),
            "y_train": to_list(y_train_c),
            "X_test": to_list(X_test_c),
            "y_test": to_list(y_test_c),
            "predictions": to_list(abc.predict(X_test_c)),
            "accuracy": float(abc.score(X_test_c, y_test_c)),
            "feature_importances": to_list(abc.feature_importances_),
        },
        "regressor": {
            "model": "AdaBoostRegressor",
            "params": {"n_estimators": 10, "random_state": 42},
            "dataset": "diabetes",
            "X_train": to_list(X_train_r),
            "y_train": to_list(y_train_r),
            "X_test": to_list(X_test_r),
            "y_test": to_list(y_test_r),
            "predictions": to_list(abr.predict(X_test_r)),
            "r2_score": float(abr.score(X_test_r, y_test_r)),
        },
    })

# =============================================================================
# Preprocessing Fixtures
# =============================================================================

def generate_preprocessing():
    print("Generating preprocessing.json...")
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import (
        PolynomialFeatures, OneHotEncoder, LabelEncoder, OrdinalEncoder,
        PowerTransformer, QuantileTransformer, StandardScaler
    )
    from sklearn.impute import SimpleImputer, KNNImputer
    from sklearn.feature_selection import VarianceThreshold, SelectKBest, f_classif

    X_iris, y_iris = make_iris()

    # PCA
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_iris)

    # PolynomialFeatures
    X_small = X_iris[:10, :2]
    poly = PolynomialFeatures(degree=2, include_bias=True)
    X_poly = poly.fit_transform(X_small)

    # OneHotEncoder
    X_cat = np.array([[0], [1], [2], [0], [1]]).astype(float)
    ohe = OneHotEncoder(sparse_output=False)
    X_ohe = ohe.fit_transform(X_cat)

    # LabelEncoder
    y_labels = np.array([2.0, 0.0, 1.0, 2.0, 1.0, 0.0])
    le = LabelEncoder()
    y_encoded = le.fit_transform(y_labels)

    # OrdinalEncoder
    X_ord = np.array([[1.0], [3.0], [2.0], [1.0], [3.0]])
    oe = OrdinalEncoder()
    X_ord_encoded = oe.fit_transform(X_ord)

    # SimpleImputer
    X_missing = np.array([[1.0, 2.0], [np.nan, 3.0], [7.0, np.nan], [4.0, 5.0]])
    imp = SimpleImputer(strategy='mean')
    X_imputed = imp.fit_transform(X_missing)

    # KNNImputer
    knn_imp = KNNImputer(n_neighbors=2)
    X_knn_imputed = knn_imp.fit_transform(X_missing)

    # PowerTransformer (Yeo-Johnson)
    pt = PowerTransformer(method='yeo-johnson', standardize=True)
    X_pt = pt.fit_transform(X_iris)

    # QuantileTransformer
    qt = QuantileTransformer(output_distribution='uniform', n_quantiles=50, random_state=42)
    X_qt = qt.fit_transform(X_iris)

    # VarianceThreshold
    X_var = np.array([[0, 2, 0], [0, 1, 4], [0, 1, 1], [0, 3, 2]])
    vt = VarianceThreshold(threshold=0.0)
    X_vt = vt.fit_transform(X_var.astype(float))

    # SelectKBest
    skb = SelectKBest(f_classif, k=2)
    X_skb = skb.fit_transform(X_iris, y_iris)

    save_fixture("preprocessing.json", {
        "pca": {
            "transformer": "PCA",
            "params": {"n_components": 2},
            "X_input": to_list(X_iris),
            "X_transformed": to_list(X_pca),
            "explained_variance_ratio": to_list(pca.explained_variance_ratio_),
            "components": to_list(pca.components_),
        },
        "polynomial_features": {
            "transformer": "PolynomialFeatures",
            "params": {"degree": 2, "include_bias": True},
            "X_input": to_list(X_small),
            "X_transformed": to_list(X_poly),
            "n_output_features": int(X_poly.shape[1]),
        },
        "one_hot_encoder": {
            "transformer": "OneHotEncoder",
            "params": {},
            "X_input": to_list(X_cat),
            "X_transformed": to_list(X_ohe),
        },
        "label_encoder": {
            "transformer": "LabelEncoder",
            "params": {},
            "y_input": to_list(y_labels),
            "y_encoded": to_list(y_encoded.astype(float)),
        },
        "ordinal_encoder": {
            "transformer": "OrdinalEncoder",
            "params": {},
            "X_input": to_list(X_ord),
            "X_transformed": to_list(X_ord_encoded),
        },
        "simple_imputer": {
            "transformer": "SimpleImputer",
            "params": {"strategy": "mean"},
            "X_input": to_list(np.where(np.isnan(X_missing), None, X_missing)),
            "X_transformed": to_list(X_imputed),
        },
        "knn_imputer": {
            "transformer": "KNNImputer",
            "params": {"n_neighbors": 2},
            "X_input": to_list(np.where(np.isnan(X_missing), None, X_missing)),
            "X_transformed": to_list(X_knn_imputed),
        },
        "power_transformer": {
            "transformer": "PowerTransformer",
            "params": {"method": "yeo-johnson", "standardize": True},
            "X_input": to_list(X_iris),
            "X_transformed": to_list(X_pt),
            "lambdas": to_list(pt.lambdas_),
        },
        "quantile_transformer": {
            "transformer": "QuantileTransformer",
            "params": {"output_distribution": "uniform", "n_quantiles": 50, "random_state": 42},
            "X_input": to_list(X_iris),
            "X_transformed": to_list(X_qt),
        },
        "variance_threshold": {
            "transformer": "VarianceThreshold",
            "params": {"threshold": 0.0},
            "X_input": to_list(X_var.astype(float)),
            "X_transformed": to_list(X_vt.astype(float)),
            "variances": to_list(vt.variances_),
            "selected_features": to_list(vt.get_support().astype(int)),
        },
        "select_k_best": {
            "transformer": "SelectKBest",
            "params": {"k": 2, "score_func": "f_classif"},
            "X_input": to_list(X_iris),
            "y_input": to_list(y_iris),
            "X_transformed": to_list(X_skb),
            "scores": to_list(skb.scores_),
            "pvalues": to_list(skb.pvalues_),
            "selected_features": to_list(skb.get_support().astype(int)),
        },
    })


if __name__ == "__main__":
    print("Generating sklearn baseline fixtures...")
    generate_logistic_regression()
    generate_random_forest()
    generate_gradient_boosting()
    generate_knn()
    generate_naive_bayes()
    generate_svm()
    generate_regularized()
    generate_adaboost()
    generate_preprocessing()
    print("\nDone! All fixtures saved to benchmarks/fixtures/")
