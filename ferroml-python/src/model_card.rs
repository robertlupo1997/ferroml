//! Python binding for ModelCard structured metadata.

use ferroml_core::model_card::ModelCard;
use pyo3::prelude::*;

/// Structured metadata about a FerroML model.
///
/// Every model exposes a static ``model_card()`` method returning a ``ModelCard``
/// that describes the algorithm's task, complexity, interpretability, and more.
///
/// Attributes
/// ----------
/// name : str
///     Python class name (e.g. ``"LinearRegression"``).
/// task : list[str]
///     Task types (e.g. ``["classification"]``, ``["regression"]``).
/// complexity : str
///     Algorithmic time complexity (e.g. ``"O(n*p)"``).
/// interpretability : str
///     ``"high"``, ``"medium"``, or ``"low"``.
/// supports_sparse : bool
///     Whether the model accepts sparse input.
/// supports_incremental : bool
///     Whether the model supports ``partial_fit``.
/// supports_sample_weight : bool
///     Whether the model supports ``fit_weighted``.
/// strengths : list[str]
///     Key strengths of the algorithm.
/// limitations : list[str]
///     Key limitations of the algorithm.
/// references : list[str]
///     Academic references.
///
/// Examples
/// --------
/// >>> from ferroml.linear import LinearRegression
/// >>> card = LinearRegression.model_card()
/// >>> card.name
/// 'LinearRegression'
/// >>> card.task
/// ['regression']
/// >>> card.interpretability
/// 'high'
#[pyclass(name = "ModelCard", module = "ferroml")]
pub struct PyModelCard {
    pub(crate) inner: ModelCard,
}

impl PyModelCard {
    pub fn new(inner: ModelCard) -> Self {
        Self { inner }
    }
}

#[pymethods]
impl PyModelCard {
    #[getter]
    fn name(&self) -> &str {
        &self.inner.name
    }

    #[getter]
    fn task(&self) -> Vec<String> {
        self.inner.task.clone()
    }

    #[getter]
    fn complexity(&self) -> &str {
        &self.inner.complexity
    }

    #[getter]
    fn interpretability(&self) -> &str {
        &self.inner.interpretability
    }

    #[getter]
    fn supports_sparse(&self) -> bool {
        self.inner.supports_sparse
    }

    #[getter]
    fn supports_incremental(&self) -> bool {
        self.inner.supports_incremental
    }

    #[getter]
    fn supports_sample_weight(&self) -> bool {
        self.inner.supports_sample_weight
    }

    #[getter]
    fn strengths(&self) -> Vec<String> {
        self.inner.strengths.clone()
    }

    #[getter]
    fn limitations(&self) -> Vec<String> {
        self.inner.limitations.clone()
    }

    #[getter]
    fn references(&self) -> Vec<String> {
        self.inner.references.clone()
    }

    fn __repr__(&self) -> String {
        format!(
            "ModelCard(name='{}', task={:?})",
            self.inner.name, self.inner.task
        )
    }
}
