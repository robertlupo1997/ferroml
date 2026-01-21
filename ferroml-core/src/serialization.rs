//! Model Serialization and Persistence
//!
//! This module provides functionality to save and load FerroML models
//! in multiple formats: JSON, MessagePack, and Bincode.
//!
//! ## Features
//!
//! - **Multiple formats**: JSON (human-readable), MessagePack (compact binary), Bincode (fast binary)
//! - **Versioning**: Metadata includes library version for forward compatibility
//! - **File and memory serialization**: Save to files or serialize to byte vectors
//! - **Error handling**: Detailed error messages for debugging
//!
//! ## Format Comparison
//!
//! | Format | Size | Speed | Human-Readable | Best For |
//! |--------|------|-------|----------------|----------|
//! | JSON | Large | Slow | Yes | Debugging, config files |
//! | MessagePack | Small | Fast | No | Production, network transfer |
//! | Bincode | Smallest | Fastest | No | High-performance, local storage |
//!
//! ## Example
//!
//! ```ignore
//! use ferroml_core::models::linear::LinearRegression;
//! use ferroml_core::serialization::{save_model, load_model, Format};
//!
//! // Train a model
//! let mut model = LinearRegression::new();
//! model.fit(&x, &y)?;
//!
//! // Save in different formats
//! save_model(&model, "model.json", Format::Json)?;
//! save_model(&model, "model.msgpack", Format::MessagePack)?;
//! save_model(&model, "model.bin", Format::Bincode)?;
//!
//! // Load the model
//! let loaded: LinearRegression = load_model("model.json", Format::Json)?;
//! ```

use crate::{FerroError, Result};
use serde::{de::DeserializeOwned, Deserialize, Serialize};
use std::fs::File;
use std::io::{BufReader, BufWriter, Read, Write};
use std::path::Path;

/// Current library version for serialization metadata
pub const FERROML_VERSION: &str = env!("CARGO_PKG_VERSION");

/// Magic bytes for FerroML binary formats
const MAGIC_BYTES: [u8; 4] = *b"FRML";

/// Serialization format
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum Format {
    /// JSON format (human-readable, larger file size)
    Json,
    /// JSON format with pretty printing
    JsonPretty,
    /// MessagePack format (compact binary, good balance)
    MessagePack,
    /// Bincode format (fastest, smallest, Rust-specific)
    Bincode,
}

impl Format {
    /// Get the typical file extension for this format
    #[must_use]
    pub fn extension(&self) -> &'static str {
        match self {
            Format::Json | Format::JsonPretty => "json",
            Format::MessagePack => "msgpack",
            Format::Bincode => "bin",
        }
    }

    /// Infer format from file extension
    #[must_use]
    pub fn from_extension(path: &Path) -> Option<Self> {
        path.extension()
            .and_then(|ext| ext.to_str())
            .and_then(|ext| match ext.to_lowercase().as_str() {
                "json" => Some(Format::Json),
                "msgpack" | "mp" => Some(Format::MessagePack),
                "bin" | "bincode" => Some(Format::Bincode),
                _ => None,
            })
    }
}

/// Metadata stored with serialized models for versioning and compatibility
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SerializationMetadata {
    /// FerroML library version that created this file
    pub ferroml_version: String,
    /// Type name of the serialized model
    pub model_type: String,
    /// Serialization format used
    pub format: Format,
    /// Timestamp when the model was serialized (Unix epoch seconds)
    pub timestamp: u64,
    /// Optional user-provided description
    pub description: Option<String>,
}

impl SerializationMetadata {
    /// Create new metadata for a model
    pub fn new<T>(format: Format) -> Self {
        Self {
            ferroml_version: FERROML_VERSION.to_string(),
            model_type: std::any::type_name::<T>().to_string(),
            format,
            timestamp: std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map(|d| d.as_secs())
                .unwrap_or(0),
            description: None,
        }
    }

    /// Add a description to the metadata
    #[must_use]
    pub fn with_description(mut self, description: impl Into<String>) -> Self {
        self.description = Some(description.into());
        self
    }
}

/// Wrapper for serialized model with metadata
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelContainer<T> {
    /// Metadata about the serialization
    pub metadata: SerializationMetadata,
    /// The actual model data
    pub model: T,
}

impl<T> ModelContainer<T> {
    /// Create a new container with the given model and format
    pub fn new(model: T, format: Format) -> Self
    where
        T: Serialize,
    {
        Self {
            metadata: SerializationMetadata::new::<T>(format),
            model,
        }
    }

    /// Create a container with custom metadata
    #[must_use]
    pub fn with_metadata(model: T, metadata: SerializationMetadata) -> Self {
        Self { metadata, model }
    }
}

// =============================================================================
// File I/O Functions
// =============================================================================

/// Save a model to a file in the specified format
///
/// # Arguments
/// * `model` - The model to serialize
/// * `path` - Output file path
/// * `format` - Serialization format to use
///
/// # Errors
/// Returns an error if serialization or file writing fails
///
/// # Example
///
/// ```ignore
/// use ferroml_core::serialization::{save_model, Format};
///
/// save_model(&model, "model.json", Format::Json)?;
/// ```
pub fn save_model<T, P>(model: &T, path: P, format: Format) -> Result<()>
where
    T: Serialize,
    P: AsRef<Path>,
{
    let container = ModelContainer::new(model, format);
    save_container(&container, path, format)
}

/// Save a model with custom description
///
/// # Arguments
/// * `model` - The model to serialize
/// * `path` - Output file path
/// * `format` - Serialization format to use
/// * `description` - User-provided description
pub fn save_model_with_description<T, P>(
    model: &T,
    path: P,
    format: Format,
    description: impl Into<String>,
) -> Result<()>
where
    T: Serialize,
    P: AsRef<Path>,
{
    let metadata = SerializationMetadata::new::<T>(format).with_description(description);
    let container = ModelContainer::with_metadata(model, metadata);
    save_container(&container, path, format)
}

/// Save a model container to a file
fn save_container<T, P>(container: &ModelContainer<&T>, path: P, format: Format) -> Result<()>
where
    T: Serialize,
    P: AsRef<Path>,
{
    let file = File::create(path.as_ref()).map_err(FerroError::IoError)?;
    let mut writer = BufWriter::new(file);

    match format {
        Format::Json => {
            serde_json::to_writer(&mut writer, container).map_err(|e| {
                FerroError::SerializationError(format!("JSON serialization failed: {e}"))
            })?;
        }
        Format::JsonPretty => {
            serde_json::to_writer_pretty(&mut writer, container).map_err(|e| {
                FerroError::SerializationError(format!("JSON serialization failed: {e}"))
            })?;
        }
        Format::MessagePack => {
            rmp_serde::encode::write(&mut writer, container).map_err(|e| {
                FerroError::SerializationError(format!("MessagePack serialization failed: {e}"))
            })?;
        }
        Format::Bincode => {
            // Write magic bytes for identification
            writer
                .write_all(&MAGIC_BYTES)
                .map_err(FerroError::IoError)?;
            bincode::serialize_into(&mut writer, container).map_err(|e| {
                FerroError::SerializationError(format!("Bincode serialization failed: {e}"))
            })?;
        }
    }

    writer.flush().map_err(FerroError::IoError)?;
    Ok(())
}

/// Load a model from a file
///
/// # Arguments
/// * `path` - Input file path
/// * `format` - Serialization format to use
///
/// # Returns
/// The deserialized model
///
/// # Errors
/// Returns an error if deserialization or file reading fails
///
/// # Example
///
/// ```ignore
/// use ferroml_core::serialization::{load_model, Format};
/// use ferroml_core::models::linear::LinearRegression;
///
/// let model: LinearRegression = load_model("model.json", Format::Json)?;
/// ```
pub fn load_model<T, P>(path: P, format: Format) -> Result<T>
where
    T: DeserializeOwned,
    P: AsRef<Path>,
{
    let container: ModelContainer<T> = load_container(path, format)?;
    Ok(container.model)
}

/// Load a model with its metadata
///
/// # Arguments
/// * `path` - Input file path
/// * `format` - Serialization format to use
///
/// # Returns
/// A tuple of (model, metadata)
pub fn load_model_with_metadata<T, P>(path: P, format: Format) -> Result<(T, SerializationMetadata)>
where
    T: DeserializeOwned,
    P: AsRef<Path>,
{
    let container: ModelContainer<T> = load_container(path, format)?;
    Ok((container.model, container.metadata))
}

/// Load a model container from a file
fn load_container<T, P>(path: P, format: Format) -> Result<ModelContainer<T>>
where
    T: DeserializeOwned,
    P: AsRef<Path>,
{
    let file = File::open(path.as_ref()).map_err(FerroError::IoError)?;
    let mut reader = BufReader::new(file);

    let container: ModelContainer<T> = match format {
        Format::Json | Format::JsonPretty => serde_json::from_reader(&mut reader).map_err(|e| {
            FerroError::SerializationError(format!("JSON deserialization failed: {e}"))
        })?,
        Format::MessagePack => rmp_serde::decode::from_read(&mut reader).map_err(|e| {
            FerroError::SerializationError(format!("MessagePack deserialization failed: {e}"))
        })?,
        Format::Bincode => {
            // Verify magic bytes
            let mut magic = [0u8; 4];
            reader.read_exact(&mut magic).map_err(FerroError::IoError)?;
            if magic != MAGIC_BYTES {
                return Err(FerroError::SerializationError(
                    "Invalid file format: not a FerroML bincode file".to_string(),
                ));
            }
            bincode::deserialize_from(&mut reader).map_err(|e| {
                FerroError::SerializationError(format!("Bincode deserialization failed: {e}"))
            })?
        }
    };

    Ok(container)
}

/// Load a model with auto-detected format based on file extension
///
/// # Arguments
/// * `path` - Input file path
///
/// # Returns
/// The deserialized model
///
/// # Errors
/// Returns an error if the format cannot be inferred or deserialization fails
pub fn load_model_auto<T, P>(path: P) -> Result<T>
where
    T: DeserializeOwned,
    P: AsRef<Path>,
{
    let path = path.as_ref();
    let format = Format::from_extension(path).ok_or_else(|| {
        FerroError::SerializationError(format!(
            "Cannot infer format from file extension: {}",
            path.display()
        ))
    })?;
    load_model(path, format)
}

// =============================================================================
// In-Memory Serialization
// =============================================================================

/// Serialize a model to bytes
///
/// # Arguments
/// * `model` - The model to serialize
/// * `format` - Serialization format to use
///
/// # Returns
/// A byte vector containing the serialized model
pub fn to_bytes<T>(model: &T, format: Format) -> Result<Vec<u8>>
where
    T: Serialize,
{
    let container = ModelContainer::new(model, format);

    let bytes = match format {
        Format::Json => serde_json::to_vec(&container).map_err(|e| {
            FerroError::SerializationError(format!("JSON serialization failed: {e}"))
        })?,
        Format::JsonPretty => serde_json::to_vec_pretty(&container).map_err(|e| {
            FerroError::SerializationError(format!("JSON serialization failed: {e}"))
        })?,
        Format::MessagePack => rmp_serde::to_vec(&container).map_err(|e| {
            FerroError::SerializationError(format!("MessagePack serialization failed: {e}"))
        })?,
        Format::Bincode => {
            let mut bytes = MAGIC_BYTES.to_vec();
            let model_bytes = bincode::serialize(&container).map_err(|e| {
                FerroError::SerializationError(format!("Bincode serialization failed: {e}"))
            })?;
            bytes.extend(model_bytes);
            bytes
        }
    };

    Ok(bytes)
}

/// Deserialize a model from bytes
///
/// # Arguments
/// * `bytes` - Byte slice containing the serialized model
/// * `format` - Serialization format used
///
/// # Returns
/// The deserialized model
pub fn from_bytes<T>(bytes: &[u8], format: Format) -> Result<T>
where
    T: DeserializeOwned,
{
    let container: ModelContainer<T> = match format {
        Format::Json | Format::JsonPretty => serde_json::from_slice(bytes).map_err(|e| {
            FerroError::SerializationError(format!("JSON deserialization failed: {e}"))
        })?,
        Format::MessagePack => rmp_serde::from_slice(bytes).map_err(|e| {
            FerroError::SerializationError(format!("MessagePack deserialization failed: {e}"))
        })?,
        Format::Bincode => {
            if bytes.len() < 4 || bytes[..4] != MAGIC_BYTES {
                return Err(FerroError::SerializationError(
                    "Invalid data format: not a FerroML bincode serialization".to_string(),
                ));
            }
            bincode::deserialize(&bytes[4..]).map_err(|e| {
                FerroError::SerializationError(format!("Bincode deserialization failed: {e}"))
            })?
        }
    };

    Ok(container.model)
}

/// Deserialize a model from bytes with metadata
///
/// # Arguments
/// * `bytes` - Byte slice containing the serialized model
/// * `format` - Serialization format used
///
/// # Returns
/// A tuple of (model, metadata)
pub fn from_bytes_with_metadata<T>(
    bytes: &[u8],
    format: Format,
) -> Result<(T, SerializationMetadata)>
where
    T: DeserializeOwned,
{
    let container: ModelContainer<T> = match format {
        Format::Json | Format::JsonPretty => serde_json::from_slice(bytes).map_err(|e| {
            FerroError::SerializationError(format!("JSON deserialization failed: {e}"))
        })?,
        Format::MessagePack => rmp_serde::from_slice(bytes).map_err(|e| {
            FerroError::SerializationError(format!("MessagePack deserialization failed: {e}"))
        })?,
        Format::Bincode => {
            if bytes.len() < 4 || bytes[..4] != MAGIC_BYTES {
                return Err(FerroError::SerializationError(
                    "Invalid data format: not a FerroML bincode serialization".to_string(),
                ));
            }
            bincode::deserialize(&bytes[4..]).map_err(|e| {
                FerroError::SerializationError(format!("Bincode deserialization failed: {e}"))
            })?
        }
    };

    Ok((container.model, container.metadata))
}

// =============================================================================
// Convenience Trait
// =============================================================================

/// Extension trait for easy model serialization
///
/// This trait is automatically implemented for any type that implements
/// `Serialize` and `DeserializeOwned`.
pub trait ModelSerialize: Serialize + DeserializeOwned + Sized {
    /// Save the model to a file
    fn save(&self, path: impl AsRef<Path>, format: Format) -> Result<()> {
        save_model(self, path, format)
    }

    /// Save the model to a file with a description
    fn save_with_description(
        &self,
        path: impl AsRef<Path>,
        format: Format,
        description: impl Into<String>,
    ) -> Result<()> {
        save_model_with_description(self, path, format, description)
    }

    /// Load a model from a file
    fn load(path: impl AsRef<Path>, format: Format) -> Result<Self> {
        load_model(path, format)
    }

    /// Load a model with auto-detected format
    fn load_auto(path: impl AsRef<Path>) -> Result<Self> {
        load_model_auto(path)
    }

    /// Serialize to bytes
    fn to_bytes(&self, format: Format) -> Result<Vec<u8>> {
        to_bytes(self, format)
    }

    /// Deserialize from bytes
    fn from_bytes(bytes: &[u8], format: Format) -> Result<Self> {
        from_bytes(bytes, format)
    }

    /// Save to JSON file (convenience method)
    fn save_json(&self, path: impl AsRef<Path>) -> Result<()> {
        self.save(path, Format::Json)
    }

    /// Save to pretty JSON file (convenience method)
    fn save_json_pretty(&self, path: impl AsRef<Path>) -> Result<()> {
        self.save(path, Format::JsonPretty)
    }

    /// Save to MessagePack file (convenience method)
    fn save_msgpack(&self, path: impl AsRef<Path>) -> Result<()> {
        self.save(path, Format::MessagePack)
    }

    /// Save to Bincode file (convenience method)
    fn save_bincode(&self, path: impl AsRef<Path>) -> Result<()> {
        self.save(path, Format::Bincode)
    }

    /// Load from JSON file (convenience method)
    fn load_json(path: impl AsRef<Path>) -> Result<Self> {
        Self::load(path, Format::Json)
    }

    /// Load from MessagePack file (convenience method)
    fn load_msgpack(path: impl AsRef<Path>) -> Result<Self> {
        Self::load(path, Format::MessagePack)
    }

    /// Load from Bincode file (convenience method)
    fn load_bincode(path: impl AsRef<Path>) -> Result<Self> {
        Self::load(path, Format::Bincode)
    }
}

// Implement for all serializable types
impl<T> ModelSerialize for T where T: Serialize + DeserializeOwned {}

// =============================================================================
// Utilities
// =============================================================================

/// Get metadata from a serialized file without loading the full model
///
/// This is useful for inspecting model files before loading them.
///
/// # Arguments
/// * `path` - Input file path
/// * `format` - Serialization format used
///
/// # Returns
/// The metadata from the file
///
/// # Note
/// This currently loads the full container. A more efficient implementation
/// would parse only the metadata portion.
pub fn peek_metadata<P>(path: P, format: Format) -> Result<SerializationMetadata>
where
    P: AsRef<Path>,
{
    // For JSON, we could potentially parse just the metadata field
    // For binary formats, we need to load the full container
    // This is a simplified implementation that loads everything
    #[derive(Deserialize)]
    struct MetadataOnly {
        metadata: SerializationMetadata,
    }

    let file = File::open(path.as_ref()).map_err(FerroError::IoError)?;
    let mut reader = BufReader::new(file);

    let meta: MetadataOnly = match format {
        Format::Json | Format::JsonPretty => serde_json::from_reader(&mut reader).map_err(|e| {
            FerroError::SerializationError(format!("JSON deserialization failed: {e}"))
        })?,
        Format::MessagePack => rmp_serde::decode::from_read(&mut reader).map_err(|e| {
            FerroError::SerializationError(format!("MessagePack deserialization failed: {e}"))
        })?,
        Format::Bincode => {
            let mut magic = [0u8; 4];
            reader.read_exact(&mut magic).map_err(FerroError::IoError)?;
            if magic != MAGIC_BYTES {
                return Err(FerroError::SerializationError(
                    "Invalid file format: not a FerroML bincode file".to_string(),
                ));
            }
            bincode::deserialize_from(&mut reader).map_err(|e| {
                FerroError::SerializationError(format!("Bincode deserialization failed: {e}"))
            })?
        }
    };

    Ok(meta.metadata)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::models::linear::LinearRegression;
    use crate::models::Model;
    use ndarray::{Array1, Array2};
    use tempfile::tempdir;

    fn create_test_model() -> LinearRegression {
        let x = Array2::from_shape_vec((5, 1), vec![1.0, 2.0, 3.0, 4.0, 5.0]).unwrap();
        let y = Array1::from_vec(vec![3.0, 5.0, 7.0, 9.0, 11.0]);

        let mut model = LinearRegression::new();
        model.fit(&x, &y).unwrap();
        model
    }

    #[test]
    fn test_json_roundtrip() {
        let model = create_test_model();
        let bytes = to_bytes(&model, Format::Json).unwrap();
        let loaded: LinearRegression = from_bytes(&bytes, Format::Json).unwrap();

        assert!(loaded.is_fitted());
        assert!((loaded.intercept().unwrap() - model.intercept().unwrap()).abs() < 1e-10);
        assert!(
            (loaded.coefficients().unwrap()[0] - model.coefficients().unwrap()[0]).abs() < 1e-10
        );
    }

    #[test]
    fn test_messagepack_roundtrip() {
        let model = create_test_model();
        let bytes = to_bytes(&model, Format::MessagePack).unwrap();
        let loaded: LinearRegression = from_bytes(&bytes, Format::MessagePack).unwrap();

        assert!(loaded.is_fitted());
        assert!((loaded.intercept().unwrap() - model.intercept().unwrap()).abs() < 1e-10);
    }

    #[test]
    fn test_bincode_roundtrip() {
        let model = create_test_model();
        let bytes = to_bytes(&model, Format::Bincode).unwrap();
        let loaded: LinearRegression = from_bytes(&bytes, Format::Bincode).unwrap();

        assert!(loaded.is_fitted());
        assert!((loaded.intercept().unwrap() - model.intercept().unwrap()).abs() < 1e-10);
    }

    #[test]
    fn test_file_roundtrip_json() {
        let model = create_test_model();
        let dir = tempdir().unwrap();
        let path = dir.path().join("model.json");

        save_model(&model, &path, Format::Json).unwrap();
        let loaded: LinearRegression = load_model(&path, Format::Json).unwrap();

        assert!(loaded.is_fitted());
        assert!((loaded.intercept().unwrap() - model.intercept().unwrap()).abs() < 1e-10);
    }

    #[test]
    fn test_file_roundtrip_msgpack() {
        let model = create_test_model();
        let dir = tempdir().unwrap();
        let path = dir.path().join("model.msgpack");

        save_model(&model, &path, Format::MessagePack).unwrap();
        let loaded: LinearRegression = load_model(&path, Format::MessagePack).unwrap();

        assert!(loaded.is_fitted());
    }

    #[test]
    fn test_file_roundtrip_bincode() {
        let model = create_test_model();
        let dir = tempdir().unwrap();
        let path = dir.path().join("model.bin");

        save_model(&model, &path, Format::Bincode).unwrap();
        let loaded: LinearRegression = load_model(&path, Format::Bincode).unwrap();

        assert!(loaded.is_fitted());
    }

    #[test]
    fn test_metadata() {
        let model = create_test_model();
        let bytes = to_bytes(&model, Format::Json).unwrap();
        let (loaded, metadata): (LinearRegression, _) =
            from_bytes_with_metadata(&bytes, Format::Json).unwrap();

        assert_eq!(metadata.ferroml_version, FERROML_VERSION);
        assert!(metadata.model_type.contains("LinearRegression"));
        assert_eq!(metadata.format, Format::Json);
        assert!(loaded.is_fitted());
    }

    #[test]
    fn test_save_with_description() {
        let model = create_test_model();
        let dir = tempdir().unwrap();
        let path = dir.path().join("model.json");

        save_model_with_description(&model, &path, Format::Json, "My test model").unwrap();

        let (loaded, metadata): (LinearRegression, _) =
            load_model_with_metadata(&path, Format::Json).unwrap();

        assert_eq!(metadata.description, Some("My test model".to_string()));
        assert!(loaded.is_fitted());
    }

    #[test]
    fn test_auto_format_detection() {
        let model = create_test_model();
        let dir = tempdir().unwrap();

        // Test JSON
        let json_path = dir.path().join("model.json");
        save_model(&model, &json_path, Format::Json).unwrap();
        let loaded: LinearRegression = load_model_auto(&json_path).unwrap();
        assert!(loaded.is_fitted());

        // Test MessagePack
        let mp_path = dir.path().join("model.msgpack");
        save_model(&model, &mp_path, Format::MessagePack).unwrap();
        let loaded: LinearRegression = load_model_auto(&mp_path).unwrap();
        assert!(loaded.is_fitted());

        // Test Bincode
        let bin_path = dir.path().join("model.bin");
        save_model(&model, &bin_path, Format::Bincode).unwrap();
        let loaded: LinearRegression = load_model_auto(&bin_path).unwrap();
        assert!(loaded.is_fitted());
    }

    #[test]
    fn test_format_extension() {
        assert_eq!(Format::Json.extension(), "json");
        assert_eq!(Format::JsonPretty.extension(), "json");
        assert_eq!(Format::MessagePack.extension(), "msgpack");
        assert_eq!(Format::Bincode.extension(), "bin");
    }

    #[test]
    fn test_format_from_extension() {
        use std::path::PathBuf;

        assert_eq!(
            Format::from_extension(&PathBuf::from("model.json")),
            Some(Format::Json)
        );
        assert_eq!(
            Format::from_extension(&PathBuf::from("model.msgpack")),
            Some(Format::MessagePack)
        );
        assert_eq!(
            Format::from_extension(&PathBuf::from("model.mp")),
            Some(Format::MessagePack)
        );
        assert_eq!(
            Format::from_extension(&PathBuf::from("model.bin")),
            Some(Format::Bincode)
        );
        assert_eq!(Format::from_extension(&PathBuf::from("model.xyz")), None);
    }

    #[test]
    fn test_trait_methods() {
        let model = create_test_model();
        let dir = tempdir().unwrap();

        // Test trait method save_json
        let path = dir.path().join("model.json");
        model.save_json(&path).unwrap();
        let loaded = LinearRegression::load_json(&path).unwrap();
        assert!(loaded.is_fitted());

        // Test trait method save_msgpack
        let path = dir.path().join("model.msgpack");
        model.save_msgpack(&path).unwrap();
        let loaded = LinearRegression::load_msgpack(&path).unwrap();
        assert!(loaded.is_fitted());

        // Test trait method save_bincode
        let path = dir.path().join("model.bin");
        model.save_bincode(&path).unwrap();
        let loaded = LinearRegression::load_bincode(&path).unwrap();
        assert!(loaded.is_fitted());
    }

    #[test]
    fn test_peek_metadata() {
        let model = create_test_model();
        let dir = tempdir().unwrap();
        let path = dir.path().join("model.json");

        save_model_with_description(&model, &path, Format::Json, "Peek test").unwrap();

        let metadata = peek_metadata(&path, Format::Json).unwrap();
        assert_eq!(metadata.description, Some("Peek test".to_string()));
        assert_eq!(metadata.format, Format::Json);
    }

    #[test]
    fn test_bincode_magic_bytes() {
        let model = create_test_model();
        let bytes = to_bytes(&model, Format::Bincode).unwrap();

        // Verify magic bytes at start
        assert_eq!(&bytes[..4], &MAGIC_BYTES);

        // Verify loading works
        let loaded: LinearRegression = from_bytes(&bytes, Format::Bincode).unwrap();
        assert!(loaded.is_fitted());
    }

    #[test]
    fn test_bincode_invalid_magic_bytes() {
        let invalid_bytes = vec![0, 1, 2, 3, 4, 5];
        let result: Result<LinearRegression> = from_bytes(&invalid_bytes, Format::Bincode);
        assert!(result.is_err());
    }

    #[test]
    fn test_format_sizes() {
        let model = create_test_model();

        let json_bytes = to_bytes(&model, Format::Json).unwrap();
        let msgpack_bytes = to_bytes(&model, Format::MessagePack).unwrap();
        let bincode_bytes = to_bytes(&model, Format::Bincode).unwrap();

        // All formats should produce non-empty output
        assert!(!json_bytes.is_empty());
        assert!(!msgpack_bytes.is_empty());
        assert!(!bincode_bytes.is_empty());

        // Binary formats should generally be smaller than JSON
        // (not a strict requirement, but a sanity check)
        assert!(
            msgpack_bytes.len() < json_bytes.len() || bincode_bytes.len() < json_bytes.len(),
            "At least one binary format should be smaller than JSON"
        );
    }

    #[test]
    fn test_predictions_after_load() {
        let model = create_test_model();
        let x_test = Array2::from_shape_vec((3, 1), vec![6.0, 7.0, 8.0]).unwrap();
        let original_preds = model.predict(&x_test).unwrap();

        // Test all formats
        for format in [Format::Json, Format::MessagePack, Format::Bincode] {
            let bytes = to_bytes(&model, format).unwrap();
            let loaded: LinearRegression = from_bytes(&bytes, format).unwrap();
            let loaded_preds = loaded.predict(&x_test).unwrap();

            for (orig, load) in original_preds.iter().zip(loaded_preds.iter()) {
                assert!(
                    (orig - load).abs() < 1e-10,
                    "Predictions differ for format {:?}",
                    format
                );
            }
        }
    }
}
