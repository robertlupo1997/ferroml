"""Tests for TextPipeline — text document processing pipeline."""

import pytest
import numpy as np
from scipy import sparse

from ferroml.pipeline import TextPipeline
from ferroml.preprocessing import TfidfVectorizer, CountVectorizer, TfidfTransformer
from ferroml.naive_bayes import MultinomialNB
from ferroml.linear import LogisticRegression


def sample_corpus():
    return [
        "the cat sat on the mat",
        "the dog sat on the log",
        "the cat and the dog",
        "a bird flew over the house",
    ]


def binary_labels():
    return np.array([0.0, 1.0, 0.0, 1.0])


def separable_corpus():
    """Corpus with clearly separable classes for accuracy checks."""
    docs = [
        "python programming code software developer",
        "java coding algorithm data structure",
        "programming language compiler debug",
        "software engineering code review",
        "code optimization algorithm complexity",
        "football soccer goal stadium match",
        "basketball player score court game",
        "tennis match serve volley racket",
        "soccer team championship league win",
        "baseball pitcher home run bat",
    ]
    labels = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0])
    return docs, labels


class TestTextPipelineTfidfMultinomialNB:
    """TfidfVectorizer + MultinomialNB pipeline."""

    def test_fit_predict_basic(self):
        pipe = TextPipeline([
            ("tfidf", TfidfVectorizer()),
            ("model", MultinomialNB()),
        ])
        docs = sample_corpus()
        y = binary_labels()
        pipe.fit(docs, y)
        preds = pipe.predict(docs)
        assert preds.shape == (4,)

    def test_predictions_are_binary(self):
        pipe = TextPipeline([
            ("tfidf", TfidfVectorizer()),
            ("model", MultinomialNB()),
        ])
        docs = sample_corpus()
        y = binary_labels()
        pipe.fit(docs, y)
        preds = pipe.predict(docs)
        assert set(preds).issubset({0.0, 1.0})


class TestTextPipelineTfidfLogistic:
    """TfidfVectorizer + LogisticRegression pipeline."""

    def test_fit_predict(self):
        pipe = TextPipeline([
            ("tfidf", TfidfVectorizer()),
            ("model", LogisticRegression()),
        ])
        docs = sample_corpus()
        y = binary_labels()
        pipe.fit(docs, y)
        preds = pipe.predict(docs)
        assert preds.shape == (4,)

    def test_predict_new_docs(self):
        pipe = TextPipeline([
            ("tfidf", TfidfVectorizer()),
            ("model", LogisticRegression()),
        ])
        docs = sample_corpus()
        y = binary_labels()
        pipe.fit(docs, y)
        new_docs = ["a new cat story", "the dog runs fast"]
        preds = pipe.predict(new_docs)
        assert preds.shape == (2,)


class TestTextPipelineCountVectorizerNB:
    """CountVectorizer + MultinomialNB pipeline."""

    def test_fit_predict(self):
        pipe = TextPipeline([
            ("cv", CountVectorizer()),
            ("model", MultinomialNB()),
        ])
        docs = sample_corpus()
        y = binary_labels()
        pipe.fit(docs, y)
        preds = pipe.predict(docs)
        assert preds.shape == (4,)


class TestTextPipelineMultiStep:
    """CountVectorizer + TfidfTransformer + model (multi-step)."""

    def test_count_tfidf_nb(self):
        pipe = TextPipeline([
            ("cv", CountVectorizer()),
            ("tfidf", TfidfTransformer()),
            ("model", MultinomialNB()),
        ])
        docs = sample_corpus()
        y = binary_labels()
        pipe.fit(docs, y)
        preds = pipe.predict(docs)
        assert preds.shape == (4,)

    def test_count_tfidf_logistic(self):
        pipe = TextPipeline([
            ("cv", CountVectorizer()),
            ("tfidf", TfidfTransformer()),
            ("model", LogisticRegression()),
        ])
        docs = sample_corpus()
        y = binary_labels()
        pipe.fit(docs, y)
        preds = pipe.predict(docs)
        assert preds.shape == (4,)


class TestTransformReturnsSparse:
    """transform() returns sparse matrix from transformer steps."""

    def test_tfidf_transform_sparse(self):
        pipe = TextPipeline([
            ("tfidf", TfidfVectorizer()),
            ("model", MultinomialNB()),
        ])
        docs = sample_corpus()
        y = binary_labels()
        pipe.fit(docs, y)
        X = pipe.transform(docs)
        assert sparse.issparse(X)
        assert X.shape[0] == 4

    def test_count_transform_returns_array(self):
        pipe = TextPipeline([
            ("cv", CountVectorizer()),
            ("model", MultinomialNB()),
        ])
        docs = sample_corpus()
        y = binary_labels()
        pipe.fit(docs, y)
        X = pipe.transform(docs)
        # CountVectorizer returns dense ndarray
        assert X.shape[0] == 4


class TestNotFittedError:
    """Calling predict/transform before fit raises error."""

    def test_predict_not_fitted(self):
        pipe = TextPipeline([
            ("tfidf", TfidfVectorizer()),
            ("model", MultinomialNB()),
        ])
        with pytest.raises(ValueError, match="not fitted"):
            pipe.predict(["hello world"])

    def test_transform_not_fitted(self):
        pipe = TextPipeline([
            ("tfidf", TfidfVectorizer()),
            ("model", MultinomialNB()),
        ])
        with pytest.raises(ValueError, match="not fitted"):
            pipe.transform(["hello world"])


class TestSingleDocPredict:
    """Predict on a single document."""

    def test_single_doc(self):
        pipe = TextPipeline([
            ("tfidf", TfidfVectorizer()),
            ("model", MultinomialNB()),
        ])
        docs = sample_corpus()
        y = binary_labels()
        pipe.fit(docs, y)
        preds = pipe.predict(["the cat sat"])
        assert preds.shape == (1,)


class TestLargeCorpus:
    """Pipeline with 100+ documents."""

    def test_large_corpus(self):
        np.random.seed(42)
        words_a = ["alpha", "beta", "gamma", "delta", "epsilon"]
        words_b = ["zeta", "eta", "theta", "iota", "kappa"]
        docs = []
        labels = []
        for _ in range(60):
            doc = " ".join(np.random.choice(words_a, size=5))
            docs.append(doc)
            labels.append(0.0)
        for _ in range(60):
            doc = " ".join(np.random.choice(words_b, size=5))
            docs.append(doc)
            labels.append(1.0)
        y = np.array(labels)

        pipe = TextPipeline([
            ("tfidf", TfidfVectorizer()),
            ("model", MultinomialNB()),
        ])
        pipe.fit(docs, y)
        preds = pipe.predict(docs)
        assert preds.shape == (120,)
        # With perfectly separable vocab, accuracy should be very high
        acc = np.mean(preds == y)
        assert acc > 0.9


class TestNamedSteps:
    """Access pipeline steps by name."""

    def test_named_steps(self):
        tfidf = TfidfVectorizer()
        model = MultinomialNB()
        pipe = TextPipeline([
            ("tfidf", tfidf),
            ("model", model),
        ])
        ns = pipe.named_steps
        assert "tfidf" in ns
        assert "model" in ns


class TestGetStepNames:
    """get_step_names() returns step names."""

    def test_step_names(self):
        pipe = TextPipeline([
            ("tfidf", TfidfVectorizer()),
            ("model", MultinomialNB()),
        ])
        names = pipe.get_step_names()
        assert names == ["tfidf", "model"]

    def test_three_steps(self):
        pipe = TextPipeline([
            ("cv", CountVectorizer()),
            ("tfidf", TfidfTransformer()),
            ("model", LogisticRegression()),
        ])
        names = pipe.get_step_names()
        assert names == ["cv", "tfidf", "model"]


class TestFitPredict:
    """fit_predict combines fit and predict."""

    def test_fit_predict(self):
        pipe = TextPipeline([
            ("tfidf", TfidfVectorizer()),
            ("model", MultinomialNB()),
        ])
        docs = sample_corpus()
        y = binary_labels()
        preds = pipe.fit_predict(docs, y)
        assert preds.shape == (4,)
        assert pipe.is_fitted


class TestIsFitted:
    """is_fitted property."""

    def test_not_fitted_initially(self):
        pipe = TextPipeline([
            ("tfidf", TfidfVectorizer()),
            ("model", MultinomialNB()),
        ])
        assert not pipe.is_fitted

    def test_fitted_after_fit(self):
        pipe = TextPipeline([
            ("tfidf", TfidfVectorizer()),
            ("model", MultinomialNB()),
        ])
        pipe.fit(sample_corpus(), binary_labels())
        assert pipe.is_fitted


class TestAccuracySanityCheck:
    """Accuracy on clearly separable text data."""

    def test_separable_accuracy(self):
        docs, y = separable_corpus()
        pipe = TextPipeline([
            ("tfidf", TfidfVectorizer()),
            ("model", MultinomialNB()),
        ])
        pipe.fit(docs, y)
        preds = pipe.predict(docs)
        acc = np.mean(preds == y)
        # Separable data: training accuracy should be 1.0 or near it
        assert acc >= 0.9


class TestRepr:
    """__repr__ returns a human-readable string."""

    def test_repr(self):
        pipe = TextPipeline([
            ("tfidf", TfidfVectorizer()),
            ("model", MultinomialNB()),
        ])
        r = repr(pipe)
        assert "TextPipeline" in r
        assert "tfidf" in r
        assert "model" in r


class TestLen:
    """__len__ returns number of steps."""

    def test_len_two(self):
        pipe = TextPipeline([
            ("tfidf", TfidfVectorizer()),
            ("model", MultinomialNB()),
        ])
        assert len(pipe) == 2

    def test_len_three(self):
        pipe = TextPipeline([
            ("cv", CountVectorizer()),
            ("tfidf", TfidfTransformer()),
            ("model", MultinomialNB()),
        ])
        assert len(pipe) == 3


class TestShapeMismatch:
    """documents and y must have same length."""

    def test_shape_mismatch(self):
        pipe = TextPipeline([
            ("tfidf", TfidfVectorizer()),
            ("model", MultinomialNB()),
        ])
        with pytest.raises(ValueError, match="samples"):
            pipe.fit(["a", "b"], np.array([1.0, 2.0, 3.0]))


class TestSklearnCrossValidation:
    """Cross-validate FerroML TfidfVectorizer output against sklearn."""

    def test_tfidf_vectorizer_matches_sklearn(self):
        """FerroML TfidfVectorizer should produce same output as sklearn's."""
        from sklearn.feature_extraction.text import TfidfVectorizer as SkTfidfVectorizer

        corpus = [
            "the cat sat on the mat",
            "the dog sat on the log",
            "the cat and the dog",
            "a bird flew over the house",
            "the bird sat on the cat",
        ]

        # sklearn reference
        sk_tv = SkTfidfVectorizer()
        sk_result = sk_tv.fit_transform(corpus).toarray()

        # FerroML
        ferro_tv = TfidfVectorizer()
        ferro_tv.fit(corpus)
        ferro_result = ferro_tv.transform(corpus)
        if sparse.issparse(ferro_result):
            ferro_result = ferro_result.toarray()

        # Both should have same shape
        assert sk_result.shape == ferro_result.shape, (
            f"Shape mismatch: sklearn={sk_result.shape}, ferroml={ferro_result.shape}"
        )

        # Vocabulary should match
        sk_vocab = sk_tv.vocabulary_
        ferro_vocab = ferro_tv.vocabulary_
        assert set(sk_vocab.keys()) == set(ferro_vocab.keys()), (
            f"Vocab mismatch: sklearn={sorted(sk_vocab.keys())}, ferroml={sorted(ferro_vocab.keys())}"
        )

        # Reorder FerroML columns to match sklearn's vocabulary order
        ferro_reordered = np.zeros_like(sk_result)
        for term, sk_idx in sk_vocab.items():
            ferro_idx = ferro_vocab[term]
            ferro_reordered[:, sk_idx] = ferro_result[:, ferro_idx]

        np.testing.assert_allclose(
            ferro_reordered, sk_result, rtol=1e-6, atol=1e-10,
            err_msg="TfidfVectorizer values differ from sklearn",
        )

    def test_pipeline_predictions_match_sklearn(self):
        """FerroML TextPipeline predictions should match sklearn Pipeline."""
        from sklearn.feature_extraction.text import TfidfVectorizer as SkTfidfVectorizer
        from sklearn.naive_bayes import MultinomialNB as SkMultinomialNB
        from sklearn.pipeline import Pipeline as SkPipeline

        train_docs = [
            "football game score touchdown",
            "basketball court player dribble",
            "soccer goal penalty kick",
            "tennis match serve rally",
            "baseball pitch homerun bat",
            "computer software program code",
            "database server query index",
            "algorithm machine learning data",
            "network router protocol packet",
            "encryption security firewall password",
        ]
        y_train = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0])

        test_docs = [
            "quarterback throws football field",
            "python programming function variable",
        ]

        # sklearn
        sk_pipe = SkPipeline([
            ("tfidf", SkTfidfVectorizer()),
            ("model", SkMultinomialNB()),
        ])
        sk_pipe.fit(train_docs, y_train)
        sk_preds = sk_pipe.predict(test_docs)

        # FerroML
        ferro_pipe = TextPipeline([
            ("tfidf", TfidfVectorizer()),
            ("model", MultinomialNB()),
        ])
        ferro_pipe.fit(train_docs, y_train)
        ferro_preds = ferro_pipe.predict(test_docs)

        np.testing.assert_array_equal(
            ferro_preds, sk_preds,
            err_msg="FerroML TextPipeline predictions differ from sklearn Pipeline",
        )
