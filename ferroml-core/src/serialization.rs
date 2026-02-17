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
//! ```
//! # fn main() -> ferroml_core::Result<()> {
//! use ferroml_core::models::linear::LinearRegression;
//! use ferroml_core::models::Model;
//! use ferroml_core::serialization::{save_model, load_model, Format};
//! # use ndarray::{Array1, Array2};
//! # let dir = tempfile::tempdir()?;
//! # let x = Array2::from_shape_vec((5, 1), vec![1.0, 2.0, 3.0, 4.0, 5.0]).unwrap();
//! # let y = Array1::from_vec(vec![3.0, 5.0, 7.0, 9.0, 11.0]);
//!
//! // Train a model
//! let mut model = LinearRegression::new();
//! model.fit(&x, &y)?;
//!
//! // Save in different formats
//! save_model(&model, dir.path().join("model.json"), Format::Json)?;
//! save_model(&model, dir.path().join("model.msgpack"), Format::MessagePack)?;
//! save_model(&model, dir.path().join("model.bin"), Format::Bincode)?;
//!
//! // Load the model
//! let loaded: LinearRegression = load_model(dir.path().join("model.json"), Format::Json)?;
//! # Ok(())
//! # }
//! ```

use crate::{FerroError, Result};
use serde::{de::DeserializeOwned, Deserialize, Serialize};
use std::fmt;
use std::fs::File;
use std::io::{BufReader, BufWriter, Read, Write};
use std::path::Path;
use std::str::FromStr;

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

/// Options for saving models
#[derive(Debug, Clone)]
pub struct SaveOptions {
    /// Serialization format
    pub format: Format,
    /// Optional description to include in metadata
    pub description: Option<String>,
    /// Whether to include metadata (default: true)
    pub include_metadata: bool,
    /// Progress callback: called with (bytes_written, total_bytes_estimate)
    /// The total_bytes_estimate may be 0 if unknown.
    pub progress_callback: Option<fn(u64, u64)>,
}

impl SaveOptions {
    /// Create new save options with the given format
    pub fn new(format: Format) -> Self {
        Self {
            format,
            description: None,
            include_metadata: true,
            progress_callback: None,
        }
    }

    /// Set the description
    #[must_use]
    pub fn with_description(mut self, description: impl Into<String>) -> Self {
        self.description = Some(description.into());
        self
    }

    /// Set whether to include metadata
    #[must_use]
    pub fn without_metadata(mut self) -> Self {
        self.include_metadata = false;
        self
    }

    /// Set progress callback
    #[must_use]
    pub fn with_progress(mut self, callback: fn(u64, u64)) -> Self {
        self.progress_callback = Some(callback);
        self
    }
}

/// Options for loading models
#[derive(Debug, Clone)]
pub struct LoadOptions {
    /// Serialization format
    pub format: Format,
    /// Whether to verify CRC32 checksum for bincode format (default: true)
    pub verify_checksum: bool,
    /// Whether to allow loading models from a different major version (default: false)
    pub allow_version_mismatch: bool,
    /// Progress callback: called with (bytes_read, total_bytes)
    pub progress_callback: Option<fn(u64, u64)>,
}

impl LoadOptions {
    /// Create new load options with the given format
    pub fn new(format: Format) -> Self {
        Self {
            format,
            verify_checksum: true,
            allow_version_mismatch: false,
            progress_callback: None,
        }
    }

    /// Skip CRC32 checksum verification (bincode only)
    #[must_use]
    pub fn skip_checksum(mut self) -> Self {
        self.verify_checksum = false;
        self
    }

    /// Allow loading models from incompatible versions
    #[must_use]
    pub fn allow_version_mismatch(mut self) -> Self {
        self.allow_version_mismatch = true;
        self
    }

    /// Set progress callback
    #[must_use]
    pub fn with_progress(mut self, callback: fn(u64, u64)) -> Self {
        self.progress_callback = Some(callback);
        self
    }
}

/// A semantic version with proper ordering (major.minor.patch)
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct SemanticVersion {
    pub major: u32,
    pub minor: u32,
    pub patch: u32,
}

impl SemanticVersion {
    pub fn new(major: u32, minor: u32, patch: u32) -> Self {
        Self {
            major,
            minor,
            patch,
        }
    }

    /// Parse from the compile-time crate version
    pub fn current() -> Self {
        FERROML_VERSION.parse().expect("valid CARGO_PKG_VERSION")
    }

    /// Check if this version is compatible with another (same major version)
    pub fn is_compatible_with(&self, other: &Self) -> bool {
        self.major == other.major
    }
}

impl fmt::Display for SemanticVersion {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}.{}.{}", self.major, self.minor, self.patch)
    }
}

impl FromStr for SemanticVersion {
    type Err = String;
    fn from_str(s: &str) -> std::result::Result<Self, Self::Err> {
        let parts: Vec<&str> = s.split('.').collect();
        if parts.len() != 3 {
            return Err(format!("expected major.minor.patch, got '{s}'"));
        }
        Ok(Self {
            major: parts[0].parse().map_err(|e| format!("bad major: {e}"))?,
            minor: parts[1].parse().map_err(|e| format!("bad minor: {e}"))?,
            patch: parts[2].parse().map_err(|e| format!("bad patch: {e}"))?,
        })
    }
}

impl PartialOrd for SemanticVersion {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for SemanticVersion {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.major
            .cmp(&other.major)
            .then(self.minor.cmp(&other.minor))
            .then(self.patch.cmp(&other.patch))
    }
}

// Serialize/Deserialize as string for backward compatibility
impl Serialize for SemanticVersion {
    fn serialize<S: serde::Serializer>(
        &self,
        serializer: S,
    ) -> std::result::Result<S::Ok, S::Error> {
        serializer.serialize_str(&self.to_string())
    }
}

impl<'de> Deserialize<'de> for SemanticVersion {
    fn deserialize<D: serde::Deserializer<'de>>(
        deserializer: D,
    ) -> std::result::Result<Self, D::Error> {
        let s = String::deserialize(deserializer)?;
        s.parse().map_err(serde::de::Error::custom)
    }
}

/// Metadata stored with serialized models for versioning and compatibility
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SerializationMetadata {
    /// FerroML library version that created this file
    pub ferroml_version: SemanticVersion,
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
            ferroml_version: SemanticVersion::current(),
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
/// ```
/// # fn main() -> ferroml_core::Result<()> {
/// use ferroml_core::serialization::{save_model, Format};
/// # use ferroml_core::models::linear::LinearRegression;
/// # let model = LinearRegression::new();
/// # let dir = tempfile::tempdir()?;
/// save_model(&model, dir.path().join("model.json"), Format::Json)?;
/// # Ok(())
/// # }
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

/// Save a model with options
///
/// # Arguments
/// * `model` - The model to serialize
/// * `path` - Output file path
/// * `options` - Save options controlling format, metadata, etc.
pub fn save_model_with_options<T, P>(model: &T, path: P, options: &SaveOptions) -> Result<()>
where
    T: Serialize,
    P: AsRef<Path>,
{
    let format = options.format;

    if options.include_metadata {
        let mut metadata = SerializationMetadata::new::<T>(format);
        if let Some(ref desc) = options.description {
            metadata.description = Some(desc.clone());
        }
        let container = ModelContainer::with_metadata(model, metadata);
        save_container(&container, path.as_ref(), format)?;
    } else {
        // Save without metadata wrapper — serialize model directly
        let file = File::create(path.as_ref()).map_err(FerroError::IoError)?;
        let mut writer = BufWriter::new(file);
        match format {
            Format::Json => {
                serde_json::to_writer(&mut writer, model).map_err(|e| {
                    FerroError::SerializationError(format!("JSON serialization failed: {e}"))
                })?;
            }
            Format::JsonPretty => {
                serde_json::to_writer_pretty(&mut writer, model).map_err(|e| {
                    FerroError::SerializationError(format!("JSON serialization failed: {e}"))
                })?;
            }
            Format::MessagePack => {
                rmp_serde::encode::write(&mut writer, model).map_err(|e| {
                    FerroError::SerializationError(format!("MessagePack serialization failed: {e}"))
                })?;
            }
            Format::Bincode => {
                let serialized = bincode::serialize(model).map_err(|e| {
                    FerroError::SerializationError(format!("Bincode serialization failed: {e}"))
                })?;
                writer.write_all(&serialized).map_err(FerroError::IoError)?;
            }
        }
        writer.flush().map_err(FerroError::IoError)?;
    }

    if let Some(callback) = options.progress_callback {
        // Report completion — we don't track incremental progress for non-streaming
        let file_size = std::fs::metadata(path.as_ref())
            .map(|m| m.len())
            .unwrap_or(0);
        callback(file_size, file_size);
    }

    Ok(())
}

/// Load a model with options
///
/// # Arguments
/// * `path` - Input file path
/// * `options` - Load options controlling format, checksum verification, etc.
///
/// # Returns
/// The deserialized model
pub fn load_model_with_options<T, P>(path: P, options: &LoadOptions) -> Result<T>
where
    T: DeserializeOwned,
    P: AsRef<Path>,
{
    let file_size = std::fs::metadata(path.as_ref())
        .map(|m| m.len())
        .unwrap_or(0);

    if let Some(callback) = options.progress_callback {
        callback(0, file_size);
    }

    let container: ModelContainer<T> = load_container_with_options(path.as_ref(), options)?;

    // Version compatibility check
    if !options.allow_version_mismatch {
        let current = SemanticVersion::current();
        if !current.is_compatible_with(&container.metadata.ferroml_version) {
            return Err(FerroError::SerializationError(format!(
                "Version mismatch: model was saved with v{}, current is v{}. \
                 Use LoadOptions::allow_version_mismatch() to override.",
                container.metadata.ferroml_version, current
            )));
        }
    }

    if let Some(callback) = options.progress_callback {
        callback(file_size, file_size);
    }

    Ok(container.model)
}

/// Load a model container with options
fn load_container_with_options<T>(path: &Path, options: &LoadOptions) -> Result<ModelContainer<T>>
where
    T: DeserializeOwned,
{
    let file = File::open(path).map_err(FerroError::IoError)?;
    let mut reader = BufReader::new(file);

    let container: ModelContainer<T> = match options.format {
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
            let mut payload = Vec::new();
            reader
                .read_to_end(&mut payload)
                .map_err(FerroError::IoError)?;
            if payload.len() < 4 {
                return Err(FerroError::SerializationError(
                    "Bincode file too short for CRC32".to_string(),
                ));
            }
            let (data, crc_bytes) = payload.split_at(payload.len() - 4);

            if options.verify_checksum {
                let stored_crc =
                    u32::from_le_bytes([crc_bytes[0], crc_bytes[1], crc_bytes[2], crc_bytes[3]]);
                let computed_crc = crc32fast::hash(data);
                if stored_crc != computed_crc {
                    return Err(FerroError::SerializationError(format!(
                        "CRC32 mismatch: file corrupted (expected {stored_crc:#x}, got {computed_crc:#x})"
                    )));
                }
            }

            bincode::deserialize(data).map_err(|e| {
                FerroError::SerializationError(format!("Bincode deserialization failed: {e}"))
            })?
        }
    };

    Ok(container)
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
            let serialized = bincode::serialize(container).map_err(|e| {
                FerroError::SerializationError(format!("Bincode serialization failed: {e}"))
            })?;
            // Write serialized data + CRC32 checksum
            let crc = crc32fast::hash(&serialized);
            writer.write_all(&serialized).map_err(FerroError::IoError)?;
            writer
                .write_all(&crc.to_le_bytes())
                .map_err(FerroError::IoError)?;
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
/// ```
/// # fn main() -> ferroml_core::Result<()> {
/// use ferroml_core::serialization::{save_model, load_model, Format};
/// use ferroml_core::models::linear::LinearRegression;
/// # let model = LinearRegression::new();
/// # let dir = tempfile::tempdir()?;
/// # save_model(&model, dir.path().join("model.json"), Format::Json)?;
///
/// let model: LinearRegression = load_model(dir.path().join("model.json"), Format::Json)?;
/// # Ok(())
/// # }
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
            // Read remaining bytes, verify CRC32
            let mut payload = Vec::new();
            reader
                .read_to_end(&mut payload)
                .map_err(FerroError::IoError)?;
            if payload.len() < 4 {
                return Err(FerroError::SerializationError(
                    "Bincode file too short for CRC32".to_string(),
                ));
            }
            let (data, crc_bytes) = payload.split_at(payload.len() - 4);
            let stored_crc =
                u32::from_le_bytes([crc_bytes[0], crc_bytes[1], crc_bytes[2], crc_bytes[3]]);
            let computed_crc = crc32fast::hash(data);
            if stored_crc != computed_crc {
                return Err(FerroError::SerializationError(
                    format!("CRC32 mismatch: file corrupted (expected {stored_crc:#x}, got {computed_crc:#x})")
                ));
            }
            bincode::deserialize(data).map_err(|e| {
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
            let crc = crc32fast::hash(&model_bytes);
            bytes.extend(&model_bytes);
            bytes.extend(crc.to_le_bytes());
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
            let data = verify_bincode_bytes(bytes)?;
            bincode::deserialize(data).map_err(|e| {
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
            let data = verify_bincode_bytes(bytes)?;
            bincode::deserialize(data).map_err(|e| {
                FerroError::SerializationError(format!("Bincode deserialization failed: {e}"))
            })?
        }
    };

    Ok((container.model, container.metadata))
}

/// Verify magic bytes and CRC32 on in-memory bincode data.
/// Returns the payload slice (between magic bytes and CRC32).
fn verify_bincode_bytes(bytes: &[u8]) -> Result<&[u8]> {
    // MAGIC(4) + at least 1 byte payload + CRC32(4)
    if bytes.len() < 9 || bytes[..4] != MAGIC_BYTES {
        return Err(FerroError::SerializationError(
            "Invalid data format: not a FerroML bincode serialization".to_string(),
        ));
    }
    let payload = &bytes[4..];
    let (data, crc_bytes) = payload.split_at(payload.len() - 4);
    let stored_crc = u32::from_le_bytes([crc_bytes[0], crc_bytes[1], crc_bytes[2], crc_bytes[3]]);
    let computed_crc = crc32fast::hash(data);
    if stored_crc != computed_crc {
        return Err(FerroError::SerializationError(format!(
            "CRC32 mismatch: data corrupted (expected {stored_crc:#x}, got {computed_crc:#x})"
        )));
    }
    Ok(data)
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
// Streaming Serialization
// =============================================================================

/// Default chunk size for streaming: 1 MB
const DEFAULT_CHUNK_SIZE: usize = 1024 * 1024;

/// Streaming writer for serializing large models in chunks.
///
/// Serializes to bincode format with chunked I/O and optional progress reporting.
/// Each chunk is written as: [4-byte length][chunk data], followed by a final
/// [4-byte zero] sentinel, then a [4-byte CRC32] of all chunk data.
pub struct StreamingWriter<W: Write> {
    writer: W,
    chunk_size: usize,
    progress_callback: Option<fn(u64, u64)>,
    bytes_written: u64,
}

impl<W: Write> StreamingWriter<W> {
    /// Create a new streaming writer with default chunk size (1 MB)
    pub fn new(writer: W) -> Self {
        Self {
            writer,
            chunk_size: DEFAULT_CHUNK_SIZE,
            progress_callback: None,
            bytes_written: 0,
        }
    }

    /// Set the chunk size in bytes
    #[must_use]
    pub fn with_chunk_size(mut self, chunk_size: usize) -> Self {
        self.chunk_size = chunk_size.max(1024); // minimum 1 KB
        self
    }

    /// Set progress callback: called with (bytes_written, total_estimate)
    #[must_use]
    pub fn with_progress(mut self, callback: fn(u64, u64)) -> Self {
        self.progress_callback = Some(callback);
        self
    }

    /// Write a model in chunks
    pub fn write_model<T: Serialize>(&mut self, model: &T) -> Result<()> {
        // Serialize to memory first, then write in chunks
        let data = bincode::serialize(model).map_err(|e| {
            FerroError::SerializationError(format!("Bincode serialization failed: {e}"))
        })?;

        let total = data.len() as u64;

        // Write magic bytes
        self.writer
            .write_all(&MAGIC_BYTES)
            .map_err(FerroError::IoError)?;
        self.bytes_written += 4;

        // Write version byte (format version 1 = streaming)
        self.writer.write_all(&[1u8]).map_err(FerroError::IoError)?;
        self.bytes_written += 1;

        // CRC32 of all chunk data
        let mut hasher = crc32fast::Hasher::new();

        // Write data in chunks: [4-byte len][chunk data]
        for chunk in data.chunks(self.chunk_size) {
            let len = chunk.len() as u32;
            self.writer
                .write_all(&len.to_le_bytes())
                .map_err(FerroError::IoError)?;
            self.writer.write_all(chunk).map_err(FerroError::IoError)?;
            hasher.update(chunk);
            self.bytes_written += 4 + chunk.len() as u64;

            if let Some(callback) = self.progress_callback {
                callback(self.bytes_written, total + 13); // 4 magic + 1 version + 4 sentinel + 4 crc
            }
        }

        // Write sentinel (zero-length chunk)
        self.writer
            .write_all(&0u32.to_le_bytes())
            .map_err(FerroError::IoError)?;
        self.bytes_written += 4;

        // Write CRC32
        let crc = hasher.finalize();
        self.writer
            .write_all(&crc.to_le_bytes())
            .map_err(FerroError::IoError)?;
        self.bytes_written += 4;

        self.writer.flush().map_err(FerroError::IoError)?;

        if let Some(callback) = self.progress_callback {
            callback(self.bytes_written, self.bytes_written);
        }

        Ok(())
    }

    /// Get total bytes written
    pub fn bytes_written(&self) -> u64 {
        self.bytes_written
    }
}

/// Streaming reader for deserializing large models in chunks.
///
/// Reads the chunked format written by `StreamingWriter`.
pub struct StreamingReader<R: Read> {
    reader: R,
    progress_callback: Option<fn(u64, u64)>,
    bytes_read: u64,
}

impl<R: Read> StreamingReader<R> {
    /// Create a new streaming reader
    pub fn new(reader: R) -> Self {
        Self {
            reader,
            progress_callback: None,
            bytes_read: 0,
        }
    }

    /// Set progress callback: called with (bytes_read, total_bytes)
    #[must_use]
    pub fn with_progress(mut self, callback: fn(u64, u64)) -> Self {
        self.progress_callback = Some(callback);
        self
    }

    /// Read a model from the chunked stream
    pub fn read_model<T: DeserializeOwned>(&mut self) -> Result<T> {
        // Read and verify magic bytes
        let mut magic = [0u8; 4];
        self.reader
            .read_exact(&mut magic)
            .map_err(FerroError::IoError)?;
        if magic != MAGIC_BYTES {
            return Err(FerroError::SerializationError(
                "Invalid streaming format: not a FerroML file".to_string(),
            ));
        }
        self.bytes_read += 4;

        // Read version byte
        let mut version = [0u8; 1];
        self.reader
            .read_exact(&mut version)
            .map_err(FerroError::IoError)?;
        if version[0] != 1 {
            return Err(FerroError::SerializationError(format!(
                "Unsupported streaming format version: {}",
                version[0]
            )));
        }
        self.bytes_read += 1;

        // Read chunks
        let mut data = Vec::new();
        let mut hasher = crc32fast::Hasher::new();

        loop {
            let mut len_bytes = [0u8; 4];
            self.reader
                .read_exact(&mut len_bytes)
                .map_err(FerroError::IoError)?;
            let chunk_len = u32::from_le_bytes(len_bytes) as usize;
            self.bytes_read += 4;

            if chunk_len == 0 {
                break; // Sentinel reached
            }

            let mut chunk = vec![0u8; chunk_len];
            self.reader
                .read_exact(&mut chunk)
                .map_err(FerroError::IoError)?;
            hasher.update(&chunk);
            data.extend_from_slice(&chunk);
            self.bytes_read += chunk_len as u64;

            if let Some(callback) = self.progress_callback {
                callback(self.bytes_read, 0); // total unknown during streaming read
            }
        }

        // Read and verify CRC32
        let mut crc_bytes = [0u8; 4];
        self.reader
            .read_exact(&mut crc_bytes)
            .map_err(FerroError::IoError)?;
        let stored_crc = u32::from_le_bytes(crc_bytes);
        let computed_crc = hasher.finalize();
        self.bytes_read += 4;

        if stored_crc != computed_crc {
            return Err(FerroError::SerializationError(format!(
                "CRC32 mismatch in streaming data (expected {stored_crc:#x}, got {computed_crc:#x})"
            )));
        }

        if let Some(callback) = self.progress_callback {
            callback(self.bytes_read, self.bytes_read);
        }

        // Deserialize the reassembled data
        bincode::deserialize(&data).map_err(|e| {
            FerroError::SerializationError(format!("Bincode deserialization failed: {e}"))
        })
    }

    /// Get total bytes read
    pub fn bytes_read(&self) -> u64 {
        self.bytes_read
    }
}

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
            let mut payload = Vec::new();
            reader
                .read_to_end(&mut payload)
                .map_err(FerroError::IoError)?;
            if payload.len() < 4 {
                return Err(FerroError::SerializationError(
                    "Bincode file too short for CRC32".to_string(),
                ));
            }
            let (data, crc_bytes) = payload.split_at(payload.len() - 4);
            let stored_crc =
                u32::from_le_bytes([crc_bytes[0], crc_bytes[1], crc_bytes[2], crc_bytes[3]]);
            let computed_crc = crc32fast::hash(data);
            if stored_crc != computed_crc {
                return Err(FerroError::SerializationError(
                    format!("CRC32 mismatch: file corrupted (expected {stored_crc:#x}, got {computed_crc:#x})")
                ));
            }
            bincode::deserialize(data).map_err(|e| {
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

        assert_eq!(metadata.ferroml_version, SemanticVersion::current());
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

    #[test]
    fn test_semantic_version_parse_display() {
        let v: SemanticVersion = "1.2.3".parse().unwrap();
        assert_eq!(v, SemanticVersion::new(1, 2, 3));
        assert_eq!(v.to_string(), "1.2.3");
    }

    #[test]
    fn test_semantic_version_ordering() {
        let v010 = SemanticVersion::new(0, 1, 0);
        let v020 = SemanticVersion::new(0, 2, 0);
        let v0100 = SemanticVersion::new(0, 10, 0);
        // String comparison would get "0.10.0" < "0.2.0" wrong
        assert!(v010 < v020);
        assert!(v020 < v0100);
    }

    #[test]
    fn test_semantic_version_compatibility() {
        let v1 = SemanticVersion::new(1, 0, 0);
        let v1_1 = SemanticVersion::new(1, 1, 0);
        let v2 = SemanticVersion::new(2, 0, 0);
        assert!(v1.is_compatible_with(&v1_1));
        assert!(!v1.is_compatible_with(&v2));
    }

    #[test]
    fn test_semantic_version_serde_roundtrip() {
        let v = SemanticVersion::new(0, 1, 0);
        let json = serde_json::to_string(&v).unwrap();
        assert_eq!(json, "\"0.1.0\"");
        let v2: SemanticVersion = serde_json::from_str(&json).unwrap();
        assert_eq!(v, v2);
    }

    #[test]
    fn test_bincode_crc32_corruption_detected() {
        let model = create_test_model();
        let mut bytes = to_bytes(&model, Format::Bincode).unwrap();
        // Flip a byte in the middle of the serialized data
        let mid = bytes.len() / 2;
        bytes[mid] ^= 0xFF;
        let result: Result<LinearRegression> = from_bytes(&bytes, Format::Bincode);
        assert!(result.is_err());
        let err_msg = format!("{}", result.unwrap_err());
        assert!(
            err_msg.contains("CRC32")
                || err_msg.contains("corrupted")
                || err_msg.contains("deserialization")
        );
    }

    #[test]
    fn test_bincode_truncation_detected() {
        let model = create_test_model();
        let bytes = to_bytes(&model, Format::Bincode).unwrap();
        // Truncate — removes CRC32
        let truncated = &bytes[..bytes.len() - 2];
        let result: Result<LinearRegression> = from_bytes(truncated, Format::Bincode);
        assert!(result.is_err());
    }

    #[test]
    fn test_save_load_with_options_json() {
        let model = create_test_model();
        let dir = tempdir().unwrap();
        let path = dir.path().join("model_opts.json");

        let save_opts = SaveOptions::new(Format::Json).with_description("options test");
        save_model_with_options(&model, &path, &save_opts).unwrap();

        let load_opts = LoadOptions::new(Format::Json);
        let loaded: LinearRegression = load_model_with_options(&path, &load_opts).unwrap();
        assert!(loaded.is_fitted());

        // Verify description was saved
        let meta = peek_metadata(&path, Format::Json).unwrap();
        assert_eq!(meta.description, Some("options test".to_string()));
    }

    #[test]
    fn test_save_load_with_options_bincode() {
        let model = create_test_model();
        let dir = tempdir().unwrap();
        let path = dir.path().join("model_opts.bin");

        let save_opts = SaveOptions::new(Format::Bincode);
        save_model_with_options(&model, &path, &save_opts).unwrap();

        // Load with checksum verification
        let load_opts = LoadOptions::new(Format::Bincode);
        let loaded: LinearRegression = load_model_with_options(&path, &load_opts).unwrap();
        assert!(loaded.is_fitted());

        // Load with skipped checksum
        let load_opts_skip = LoadOptions::new(Format::Bincode).skip_checksum();
        let loaded2: LinearRegression = load_model_with_options(&path, &load_opts_skip).unwrap();
        assert!(loaded2.is_fitted());
    }

    #[test]
    fn test_save_without_metadata() {
        let model = create_test_model();
        let dir = tempdir().unwrap();
        let path = dir.path().join("raw_model.json");

        let save_opts = SaveOptions::new(Format::Json).without_metadata();
        save_model_with_options(&model, &path, &save_opts).unwrap();

        // Should be loadable as raw model (not wrapped in ModelContainer)
        let file = File::open(&path).unwrap();
        let loaded: LinearRegression = serde_json::from_reader(file).unwrap();
        assert!(loaded.is_fitted());
    }

    #[test]
    fn test_load_options_progress_callback() {
        use std::sync::atomic::{AtomicU64, Ordering};

        static LAST_BYTES: AtomicU64 = AtomicU64::new(0);
        static CALL_COUNT: AtomicU64 = AtomicU64::new(0);

        fn progress(bytes_done: u64, _total: u64) {
            LAST_BYTES.store(bytes_done, Ordering::SeqCst);
            CALL_COUNT.fetch_add(1, Ordering::SeqCst);
        }

        LAST_BYTES.store(0, Ordering::SeqCst);
        CALL_COUNT.store(0, Ordering::SeqCst);

        let model = create_test_model();
        let dir = tempdir().unwrap();
        let path = dir.path().join("model_progress.json");

        save_model(&model, &path, Format::Json).unwrap();

        let load_opts = LoadOptions::new(Format::Json).with_progress(progress);
        let _loaded: LinearRegression = load_model_with_options(&path, &load_opts).unwrap();

        assert!(
            CALL_COUNT.load(Ordering::SeqCst) >= 2,
            "progress should be called at least twice (start + end)"
        );
        assert!(
            LAST_BYTES.load(Ordering::SeqCst) > 0,
            "should report nonzero bytes at completion"
        );
    }

    #[test]
    fn test_streaming_roundtrip() {
        let model = create_test_model();

        // Write using streaming writer
        let mut buf = Vec::new();
        let mut writer = StreamingWriter::new(&mut buf).with_chunk_size(64);
        writer.write_model(&model).unwrap();
        assert!(writer.bytes_written() > 0);

        // Read using streaming reader
        let mut reader = StreamingReader::new(buf.as_slice());
        let loaded: LinearRegression = reader.read_model().unwrap();
        assert!(loaded.is_fitted());
        assert!((loaded.intercept().unwrap() - model.intercept().unwrap()).abs() < 1e-10);
    }

    #[test]
    fn test_streaming_predictions_match() {
        let model = create_test_model();
        let x_test = Array2::from_shape_vec((3, 1), vec![6.0, 7.0, 8.0]).unwrap();
        let original_preds = model.predict(&x_test).unwrap();

        let mut buf = Vec::new();
        StreamingWriter::new(&mut buf)
            .with_chunk_size(128)
            .write_model(&model)
            .unwrap();

        let loaded: LinearRegression = StreamingReader::new(buf.as_slice()).read_model().unwrap();
        let loaded_preds = loaded.predict(&x_test).unwrap();

        for (orig, load) in original_preds.iter().zip(loaded_preds.iter()) {
            assert!((orig - load).abs() < 1e-10, "Streaming predictions differ");
        }
    }

    #[test]
    fn test_streaming_crc_corruption_detected() {
        let model = create_test_model();

        let mut buf = Vec::new();
        StreamingWriter::new(&mut buf).write_model(&model).unwrap();

        // Corrupt a byte in the middle
        let mid = buf.len() / 2;
        buf[mid] ^= 0xFF;

        let result: Result<LinearRegression> = StreamingReader::new(buf.as_slice()).read_model();
        assert!(result.is_err());
        let err_msg = format!("{}", result.unwrap_err());
        assert!(err_msg.contains("CRC32") || err_msg.contains("deserialization"));
    }

    #[test]
    fn test_streaming_large_model_multiple_chunks() {
        // Create a model, write with very small chunk size to force multiple chunks
        let model = create_test_model();

        let mut buf = Vec::new();
        let mut writer = StreamingWriter::new(&mut buf).with_chunk_size(1024); // 1 KB chunks
        writer.write_model(&model).unwrap();

        // Verify bytes_written is reasonable
        assert!(writer.bytes_written() > 20);

        let loaded: LinearRegression = StreamingReader::new(buf.as_slice()).read_model().unwrap();
        assert!(loaded.is_fitted());
    }
}
