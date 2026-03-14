#!/usr/bin/env python3
"""Generate sklearn reference fixtures for FerroML text pipeline correctness tests."""

import json
import os
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer, TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

FIXTURE_DIR = os.path.join(os.path.dirname(__file__), "..", "ferroml-core", "fixtures", "text_pipeline")


class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)


def save_fixture(name, data):
    path = os.path.join(FIXTURE_DIR, name)
    with open(path, "w") as f:
        json.dump(data, f, indent=2, cls=NumpyEncoder)
    print(f"  Saved {path} ({os.path.getsize(path)} bytes)")


def generate_tfidf_default():
    print("Fixture 1: tfidf_default.json")
    corpus = [
        "the cat sat on the mat",
        "the dog sat on the log",
        "the cat and the dog",
        "a bird flew over the house",
        "the bird sat on the cat",
    ]
    vec = TfidfVectorizer()
    X = vec.fit_transform(corpus)
    save_fixture("tfidf_default.json", {
        "corpus": corpus,
        "matrix": X.toarray().tolist(),
        "vocabulary": vec.vocabulary_,
        "idf": vec.idf_.tolist(),
        "feature_names": vec.get_feature_names_out().tolist(),
    })


def generate_tfidf_nondefault():
    print("Fixture 2: tfidf_nondefault.json")
    corpus = [
        "the cat sat on the mat",
        "the dog sat on the log",
        "the cat and the dog",
        "a bird flew over the house",
        "the bird sat on the cat",
    ]
    vec = TfidfVectorizer(sublinear_tf=True, norm="l1", max_features=10, ngram_range=(1, 2))
    X = vec.fit_transform(corpus)
    save_fixture("tfidf_nondefault.json", {
        "corpus": corpus,
        "params": {
            "sublinear_tf": True,
            "norm": "l1",
            "max_features": 10,
            "ngram_range": [1, 2],
        },
        "matrix": X.toarray().tolist(),
        "vocabulary": vec.vocabulary_,
        "idf": vec.idf_.tolist(),
        "feature_names": vec.get_feature_names_out().tolist(),
    })


def generate_pipeline_multinomialnb():
    print("Fixture 3: pipeline_multinomialnb.json")
    corpus_train = [
        "football game score touchdown",
        "basketball court player dribble",
        "soccer goal penalty kick",
        "tennis match serve rally",
        "baseball pitch homerun bat",
        "hockey puck ice goal",
        "swimming pool race medal",
        "track sprint hurdle race",
        "boxing ring punch knockout",
        "volleyball spike serve block",
        "computer software program code",
        "database server query index",
        "algorithm machine learning data",
        "network router protocol packet",
        "encryption security firewall password",
        "processor chip memory cache",
        "operating system kernel boot",
        "compiler syntax parsing token",
        "browser website html javascript",
        "cloud storage backup sync",
    ]
    y_train = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]

    corpus_test = [
        "quarterback throws football field",
        "python programming function variable",
        "wrestling pin mat takedown",
        "api endpoint server response",
    ]
    y_test = [0, 1, 0, 1]

    pipe = Pipeline([
        ("tfidf", TfidfVectorizer()),
        ("nb", MultinomialNB()),
    ])
    pipe.fit(corpus_train, y_train)
    train_preds = pipe.predict(corpus_train).tolist()
    test_preds = pipe.predict(corpus_test).tolist()
    vocab_size = len(pipe.named_steps["tfidf"].vocabulary_)

    save_fixture("pipeline_multinomialnb.json", {
        "corpus_train": corpus_train,
        "y_train": y_train,
        "corpus_test": corpus_test,
        "y_test": y_test,
        "train_predictions": train_preds,
        "test_predictions": test_preds,
        "vocabulary_size": vocab_size,
    })
    print(f"    train accuracy: {sum(a == b for a, b in zip(y_train, train_preds))}/{len(y_train)}")
    print(f"    test predictions: {test_preds} (expected {y_test})")


def generate_pipeline_logistic():
    print("Fixture 4: pipeline_logistic.json")
    corpus_train = [
        "football game score touchdown",
        "basketball court player dribble",
        "soccer goal penalty kick",
        "tennis match serve rally",
        "baseball pitch homerun bat",
        "hockey puck ice goal",
        "swimming pool race medal",
        "track sprint hurdle race",
        "boxing ring punch knockout",
        "volleyball spike serve block",
        "computer software program code",
        "database server query index",
        "algorithm machine learning data",
        "network router protocol packet",
        "encryption security firewall password",
        "processor chip memory cache",
        "operating system kernel boot",
        "compiler syntax parsing token",
        "browser website html javascript",
        "cloud storage backup sync",
    ]
    y_train = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]

    pipe = Pipeline([
        ("count", CountVectorizer()),
        ("tfidf", TfidfTransformer()),
        ("lr", LogisticRegression()),
    ])
    pipe.fit(corpus_train, y_train)
    train_preds = pipe.predict(corpus_train).tolist()

    save_fixture("pipeline_logistic.json", {
        "corpus_train": corpus_train,
        "y_train": y_train,
        "train_predictions": train_preds,
    })
    print(f"    train accuracy: {sum(a == b for a, b in zip(y_train, train_preds))}/{len(y_train)}")


if __name__ == "__main__":
    print(f"Generating text pipeline fixtures in {FIXTURE_DIR}\n")
    generate_tfidf_default()
    generate_tfidf_nondefault()
    generate_pipeline_multinomialnb()
    generate_pipeline_logistic()
    print("\nDone. All fixtures generated.")
