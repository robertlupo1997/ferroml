"""Tests for CountVectorizer Python bindings."""

import numpy as np
import pytest
from ferroml.preprocessing import CountVectorizer, TfidfTransformer


class TestCountVectorizerBasic:
    """Basic CountVectorizer functionality."""

    def test_basic_fit_transform(self):
        corpus = ["the cat sat on the mat", "the dog sat on the log"]
        cv = CountVectorizer()
        X = cv.fit_transform(corpus)
        assert X.shape[0] == 2
        assert X.shape[1] > 0
        assert X.dtype == np.float64

    def test_vocabulary_property(self):
        corpus = ["cat dog", "cat fish"]
        cv = CountVectorizer()
        cv.fit(corpus)
        vocab = cv.vocabulary_
        assert isinstance(vocab, dict)
        assert "cat" in vocab
        assert "dog" in vocab
        assert "fish" in vocab
        assert len(vocab) == 3

    def test_get_feature_names_out(self):
        corpus = ["zebra apple mango"]
        cv = CountVectorizer()
        cv.fit(corpus)
        names = cv.get_feature_names_out()
        # Should be sorted alphabetically
        assert names == ["apple", "mango", "zebra"]

    def test_max_features(self):
        corpus = [
            "alpha bravo charlie",
            "alpha bravo delta",
            "alpha echo foxtrot",
        ]
        cv = CountVectorizer(max_features=2)
        cv.fit(corpus)
        vocab = cv.vocabulary_
        assert len(vocab) == 2
        # "alpha" (df=3) and "bravo" (df=2) are top 2
        assert "alpha" in vocab
        assert "bravo" in vocab

    def test_binary_mode(self):
        corpus = ["hello hello hello world"]
        cv = CountVectorizer(binary=True)
        X = cv.fit_transform(corpus)
        # All non-zero values should be 1.0
        assert np.all((X == 0.0) | (X == 1.0))
        vocab = cv.vocabulary_
        hello_idx = vocab["hello"]
        assert X[0, hello_idx] == 1.0

    def test_ngram_range(self):
        corpus = ["the cat sat", "the dog sat"]
        cv = CountVectorizer(ngram_range=(1, 2))
        cv.fit(corpus)
        vocab = cv.vocabulary_
        # Should have unigrams
        assert "the" in vocab
        assert "cat" in vocab
        # Should have bigrams
        assert "the cat" in vocab
        assert "cat sat" in vocab
        assert "the dog" in vocab

    def test_stop_words(self):
        corpus = ["the cat sat on the mat"]
        cv = CountVectorizer(stop_words=["the", "on"])
        cv.fit(corpus)
        vocab = cv.vocabulary_
        assert "the" not in vocab
        assert "on" not in vocab
        assert "cat" in vocab
        assert "sat" in vocab

    def test_empty_documents_in_corpus(self):
        corpus = ["hello world", "", "hello again"]
        cv = CountVectorizer()
        X = cv.fit_transform(corpus)
        assert X.shape[0] == 3
        # Empty doc row should be all zeros
        assert np.all(X[1] == 0.0)

    def test_transform_unseen_terms(self):
        train = ["cat dog fish"]
        test = ["cat elephant giraffe"]
        cv = CountVectorizer()
        cv.fit(train)
        X = cv.transform(test)
        vocab = cv.vocabulary_
        # "cat" is in vocab and in test doc
        assert X[0, vocab["cat"]] == 1.0
        # "dog" and "fish" are in vocab but not in test doc
        assert X[0, vocab["dog"]] == 0.0
        assert X[0, vocab["fish"]] == 0.0
        # "elephant" and "giraffe" are not in vocab, should be ignored
        assert "elephant" not in vocab


class TestCountVectorizerPipeline:
    """Test CountVectorizer in a pipeline with TfidfTransformer."""

    def test_cv_to_tfidf(self):
        corpus = [
            "the cat sat on the mat",
            "the dog sat on the log",
            "the cat and the dog",
        ]
        cv = CountVectorizer()
        counts = cv.fit_transform(corpus)

        tfidf = TfidfTransformer()
        X_tfidf = tfidf.fit_transform(counts)

        assert X_tfidf.shape == counts.shape
        # Default L2 norm: each row should have unit L2 norm
        for i in range(X_tfidf.shape[0]):
            row_norm = np.linalg.norm(X_tfidf[i])
            if row_norm > 0:
                assert abs(row_norm - 1.0) < 1e-10

    def test_cv_to_tfidf_to_classification(self):
        """End-to-end: CountVectorizer -> TfidfTransformer -> MultinomialNB."""
        try:
            from ferroml.naive_bayes import MultinomialNB
        except ImportError:
            pytest.skip("MultinomialNB not available")

        # Simple binary classification corpus
        train_docs = [
            "good great excellent awesome",
            "wonderful fantastic amazing",
            "bad terrible awful horrible",
            "poor dreadful miserable",
        ]
        train_labels = np.array([1.0, 1.0, 0.0, 0.0])

        cv = CountVectorizer()
        counts = cv.fit_transform(train_docs)

        tfidf = TfidfTransformer()
        X_train = tfidf.fit_transform(counts)

        nb = MultinomialNB()
        nb.fit(X_train, train_labels)

        # Test prediction
        test_docs = ["great awesome wonderful"]
        test_counts = cv.transform(test_docs)
        X_test = tfidf.transform(test_counts)
        preds = nb.predict(X_test)
        # Should predict positive class
        assert preds[0] == 1.0


class TestCountVectorizerEdgeCases:
    """Edge cases and parameter combinations."""

    def test_lowercase_default(self):
        corpus = ["Hello HELLO hello"]
        cv = CountVectorizer()
        X = cv.fit_transform(corpus)
        vocab = cv.vocabulary_
        assert len(vocab) == 1
        assert "hello" in vocab
        assert X[0, vocab["hello"]] == 3.0

    def test_case_sensitive(self):
        corpus = ["Hello HELLO hello"]
        cv = CountVectorizer(lowercase=False)
        X = cv.fit_transform(corpus)
        vocab = cv.vocabulary_
        assert len(vocab) == 3  # "Hello", "HELLO", "hello"

    def test_min_df_count(self):
        corpus = [
            "apple banana",
            "apple cherry",
            "apple date",
        ]
        cv = CountVectorizer(min_df=2)
        cv.fit(corpus)
        vocab = cv.vocabulary_
        assert "apple" in vocab
        assert "banana" not in vocab

    def test_not_fitted_vocabulary(self):
        cv = CountVectorizer()
        with pytest.raises(RuntimeError):
            _ = cv.vocabulary_

    def test_not_fitted_transform(self):
        cv = CountVectorizer()
        with pytest.raises(RuntimeError):
            cv.transform(["hello"])

    def test_not_fitted_feature_names(self):
        cv = CountVectorizer()
        with pytest.raises(RuntimeError):
            cv.get_feature_names_out()

    def test_single_document(self):
        corpus = ["one two three"]
        cv = CountVectorizer()
        X = cv.fit_transform(corpus)
        assert X.shape == (1, 3)

    def test_repeated_fit_overwrites(self):
        cv = CountVectorizer()
        cv.fit(["cat dog"])
        vocab1 = cv.vocabulary_
        assert "cat" in vocab1

        cv.fit(["fish bird"])
        vocab2 = cv.vocabulary_
        assert "cat" not in vocab2
        assert "fish" in vocab2
