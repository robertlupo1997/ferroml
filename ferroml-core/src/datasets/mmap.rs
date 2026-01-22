//! Memory-mapped datasets for large data.
//!
//! This module provides memory-mapped file support for datasets that don't fit in RAM.
//! It uses the `memmap2` crate for zero-copy file access, allowing efficient processing
//! of large datasets by mapping them directly into virtual memory.
//!
//! # Overview
//!
//! Memory-mapped datasets store data in a binary format with a header containing
//! metadata (shape, dtype) followed by raw data. This allows:
//!
//! - **Zero-copy access**: Data is read directly from disk without copying to RAM
//! - **Lazy loading**: Only accessed pages are loaded into memory
//! - **Shared memory**: Multiple processes can read the same file
//! - **Large datasets**: Handle datasets larger than available RAM
//!
//! # File Format
//!
//! The file format uses a simple structure:
//! ```text
//! [Magic: 4 bytes "FRMD"]
//! [Version: 1 byte]
//! [n_samples: 8 bytes (u64 LE)]
//! [n_features: 8 bytes (u64 LE)]
//! [has_targets: 1 byte (bool)]
//! [Reserved: 10 bytes]
//! [Feature data: n_samples * n_features * 8 bytes (f64 LE)]
//! [Target data (optional): n_samples * 8 bytes (f64 LE)]
//! ```
//!
//! # Example
//!
//! ```ignore
//! use ferroml_core::datasets::mmap::{MemmappedDataset, MemmappedDatasetBuilder};
//! use ferroml_core::datasets::Dataset;
//!
//! // Create a memory-mapped dataset from an existing dataset
//! let dataset = Dataset::new(x, y);
//! MemmappedDatasetBuilder::new("large_dataset.fmm")
//!     .from_dataset(&dataset)?
//!     .build()?;
//!
//! // Load it back for processing
//! let mmap_dataset = MemmappedDataset::open("large_dataset.fmm")?;
//!
//! // Access data as array views (zero-copy)
//! let x_view = mmap_dataset.x_view();
//! let y_view = mmap_dataset.y_view();
//!
//! println!("Samples: {}, Features: {}", mmap_dataset.n_samples(), mmap_dataset.n_features());
//! ```
//!
//! # Performance Considerations
//!
//! - Memory-mapped I/O is most beneficial for large datasets (> 1GB)
//! - Random access patterns may cause excessive page faults
//! - Sequential access is generally optimal
//! - Consider using SSD storage for best performance

// Allow clippy lints that are acceptable for this low-level memory-mapping code
#![allow(clippy::doc_markdown)]
#![allow(clippy::must_use_candidate)]
#![allow(clippy::ptr_as_ptr)]
#![allow(clippy::cast_ptr_alignment)]
#![allow(clippy::cast_possible_truncation)]
#![allow(clippy::similar_names)]

use std::fs::{File, OpenOptions};
use std::io::{Read, Seek, SeekFrom, Write};
use std::path::{Path, PathBuf};

use memmap2::{Mmap, MmapMut, MmapOptions};
use ndarray::{Array1, Array2, ArrayView1, ArrayView2, ArrayViewMut2};

use crate::{FerroError, Result};
use super::Dataset;

/// Magic bytes identifying FerroML memory-mapped dataset files.
const MAGIC: &[u8; 4] = b"FRMD";

/// Current file format version.
const VERSION: u8 = 1;

/// Size of the header in bytes.
const HEADER_SIZE: usize = 32;

/// A memory-mapped 2D array for efficient access to large feature matrices.
///
/// This struct provides read-only access to a memory-mapped array stored in
/// a binary file. Data is accessed directly from disk via virtual memory
/// mapping, avoiding the need to load the entire array into RAM.
#[derive(Debug)]
pub struct MemmappedArray2 {
    mmap: Mmap,
    n_rows: usize,
    n_cols: usize,
}

impl MemmappedArray2 {
    /// Create a new memory-mapped 2D array from a file.
    ///
    /// # Arguments
    ///
    /// * `path` - Path to the binary file
    /// * `n_rows` - Number of rows in the array
    /// * `n_cols` - Number of columns in the array
    /// * `offset` - Byte offset where data starts in the file
    ///
    /// # Safety
    ///
    /// The file must contain at least `offset + n_rows * n_cols * 8` bytes.
    pub fn open<P: AsRef<Path>>(
        path: P,
        n_rows: usize,
        n_cols: usize,
        offset: usize,
    ) -> Result<Self> {
        let file = File::open(path.as_ref())?;
        let expected_size = offset + n_rows * n_cols * std::mem::size_of::<f64>();

        let file_size = file.metadata()?.len() as usize;
        if file_size < expected_size {
            return Err(FerroError::invalid_input(format!(
                "File too small: expected at least {} bytes, got {}",
                expected_size, file_size
            )));
        }

        // SAFETY: We've verified the file is large enough for the mapping
        let mmap = unsafe {
            MmapOptions::new()
                .offset(offset as u64)
                .len(n_rows * n_cols * std::mem::size_of::<f64>())
                .map(&file)?
        };

        Ok(Self { mmap, n_rows, n_cols })
    }

    /// Get the shape of the array as (rows, cols).
    pub fn shape(&self) -> (usize, usize) {
        (self.n_rows, self.n_cols)
    }

    /// Get the number of rows.
    pub fn n_rows(&self) -> usize {
        self.n_rows
    }

    /// Get the number of columns.
    pub fn n_cols(&self) -> usize {
        self.n_cols
    }

    /// Get a view of the array as an ndarray ArrayView2.
    ///
    /// This provides zero-copy access to the underlying data.
    pub fn view(&self) -> ArrayView2<'_, f64> {
        let data_ptr = self.mmap.as_ptr() as *const f64;
        // SAFETY: We've verified the size during construction
        unsafe {
            ArrayView2::from_shape_ptr((self.n_rows, self.n_cols), data_ptr)
        }
    }

    /// Get a specific row as an array slice.
    pub fn row(&self, idx: usize) -> Option<&[f64]> {
        if idx >= self.n_rows {
            return None;
        }
        let start = idx * self.n_cols;
        let data_ptr = self.mmap.as_ptr() as *const f64;
        // SAFETY: We've bounds checked and verified size during construction
        Some(unsafe {
            std::slice::from_raw_parts(data_ptr.add(start), self.n_cols)
        })
    }

    /// Get a specific element.
    pub fn get(&self, row: usize, col: usize) -> Option<f64> {
        if row >= self.n_rows || col >= self.n_cols {
            return None;
        }
        let idx = row * self.n_cols + col;
        let data_ptr = self.mmap.as_ptr() as *const f64;
        // SAFETY: We've bounds checked
        Some(unsafe { *data_ptr.add(idx) })
    }

    /// Copy a slice of rows to a dense array.
    ///
    /// Useful for batch processing where you need to work with a subset.
    pub fn rows_to_array(&self, start: usize, end: usize) -> Result<Array2<f64>> {
        if start >= self.n_rows || end > self.n_rows || start >= end {
            return Err(FerroError::invalid_input(format!(
                "Invalid row range: {}..{} for array with {} rows",
                start, end, self.n_rows
            )));
        }
        let view = self.view();
        let slice = view.slice(ndarray::s![start..end, ..]);
        Ok(slice.to_owned())
    }
}

/// A mutable memory-mapped 2D array.
#[derive(Debug)]
pub struct MemmappedArray2Mut {
    mmap: MmapMut,
    n_rows: usize,
    n_cols: usize,
}

impl MemmappedArray2Mut {
    /// Create a new mutable memory-mapped array.
    pub fn create<P: AsRef<Path>>(
        path: P,
        n_rows: usize,
        n_cols: usize,
        offset: usize,
    ) -> Result<Self> {
        let file = OpenOptions::new()
            .read(true)
            .write(true)
            .open(path.as_ref())?;

        let total_size = offset + n_rows * n_cols * std::mem::size_of::<f64>();
        let file_size = file.metadata()?.len() as usize;
        if file_size < total_size {
            return Err(FerroError::invalid_input(format!(
                "File too small: expected at least {} bytes, got {}",
                total_size, file_size
            )));
        }

        // SAFETY: File is opened for writing and has correct size
        let mmap = unsafe {
            MmapOptions::new()
                .offset(offset as u64)
                .len(n_rows * n_cols * std::mem::size_of::<f64>())
                .map_mut(&file)?
        };

        Ok(Self { mmap, n_rows, n_cols })
    }

    /// Get a mutable view of the array.
    pub fn view_mut(&mut self) -> ArrayViewMut2<'_, f64> {
        let data_ptr = self.mmap.as_mut_ptr() as *mut f64;
        // SAFETY: We have exclusive mutable access
        unsafe {
            ArrayViewMut2::from_shape_ptr((self.n_rows, self.n_cols), data_ptr)
        }
    }

    /// Get an immutable view of the array.
    pub fn view(&self) -> ArrayView2<'_, f64> {
        let data_ptr = self.mmap.as_ptr() as *const f64;
        // SAFETY: Size verified during construction
        unsafe {
            ArrayView2::from_shape_ptr((self.n_rows, self.n_cols), data_ptr)
        }
    }

    /// Set a specific element.
    pub fn set(&mut self, row: usize, col: usize, value: f64) -> Result<()> {
        if row >= self.n_rows || col >= self.n_cols {
            return Err(FerroError::invalid_input(format!(
                "Index out of bounds: ({}, {}) for shape ({}, {})",
                row, col, self.n_rows, self.n_cols
            )));
        }
        let idx = row * self.n_cols + col;
        let data_ptr = self.mmap.as_mut_ptr() as *mut f64;
        // SAFETY: Bounds checked above
        unsafe { *data_ptr.add(idx) = value };
        Ok(())
    }

    /// Flush changes to disk.
    pub fn flush(&self) -> Result<()> {
        self.mmap.flush()?;
        Ok(())
    }
}

/// A memory-mapped 1D array for targets.
#[derive(Debug)]
pub struct MemmappedArray1 {
    mmap: Mmap,
    len: usize,
}

impl MemmappedArray1 {
    /// Create a new memory-mapped 1D array.
    pub fn open<P: AsRef<Path>>(path: P, len: usize, offset: usize) -> Result<Self> {
        let file = File::open(path.as_ref())?;
        let expected_size = offset + len * std::mem::size_of::<f64>();

        let file_size = file.metadata()?.len() as usize;
        if file_size < expected_size {
            return Err(FerroError::invalid_input(format!(
                "File too small: expected at least {} bytes, got {}",
                expected_size, file_size
            )));
        }

        // SAFETY: Size verified above
        let mmap = unsafe {
            MmapOptions::new()
                .offset(offset as u64)
                .len(len * std::mem::size_of::<f64>())
                .map(&file)?
        };

        Ok(Self { mmap, len })
    }

    /// Get the length of the array.
    pub fn len(&self) -> usize {
        self.len
    }

    /// Check if the array is empty.
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// Get a view of the array.
    pub fn view(&self) -> ArrayView1<'_, f64> {
        let data_ptr = self.mmap.as_ptr() as *const f64;
        // SAFETY: Size verified during construction
        unsafe {
            ArrayView1::from_shape_ptr(self.len, data_ptr)
        }
    }

    /// Get a specific element.
    pub fn get(&self, idx: usize) -> Option<f64> {
        if idx >= self.len {
            return None;
        }
        let data_ptr = self.mmap.as_ptr() as *const f64;
        // SAFETY: Bounds checked
        Some(unsafe { *data_ptr.add(idx) })
    }

    /// Copy a slice to a dense array.
    pub fn slice_to_array(&self, start: usize, end: usize) -> Result<Array1<f64>> {
        if start >= self.len || end > self.len || start >= end {
            return Err(FerroError::invalid_input(format!(
                "Invalid range: {}..{} for array with {} elements",
                start, end, self.len
            )));
        }
        let view = self.view();
        let slice = view.slice(ndarray::s![start..end]);
        Ok(slice.to_owned())
    }
}

/// A memory-mapped dataset for large data that doesn't fit in RAM.
///
/// This struct provides efficient access to large datasets by memory-mapping
/// the underlying file. Data is accessed on-demand through virtual memory,
/// avoiding the need to load the entire dataset into RAM.
///
/// # Example
///
/// ```ignore
/// use ferroml_core::datasets::mmap::MemmappedDataset;
///
/// // Open an existing memory-mapped dataset
/// let dataset = MemmappedDataset::open("large_data.fmm")?;
///
/// // Access data as array views (zero-copy)
/// let x = dataset.x_view();
/// let y = dataset.y_view();
///
/// // Process in batches
/// for batch_start in (0..dataset.n_samples()).step_by(1000) {
///     let batch_end = (batch_start + 1000).min(dataset.n_samples());
///     let x_batch = dataset.x_rows(batch_start, batch_end)?;
///     // Process batch...
/// }
/// ```
#[derive(Debug)]
pub struct MemmappedDataset {
    path: PathBuf,
    mmap: Mmap,
    n_samples: usize,
    n_features: usize,
    has_targets: bool,
}

impl MemmappedDataset {
    /// Open an existing memory-mapped dataset file.
    ///
    /// # Arguments
    ///
    /// * `path` - Path to the `.fmm` file
    ///
    /// # Returns
    ///
    /// A `MemmappedDataset` providing zero-copy access to the data.
    pub fn open<P: AsRef<Path>>(path: P) -> Result<Self> {
        let path = path.as_ref().to_path_buf();
        let file = File::open(&path)?;

        // Read and validate header
        let mut header = [0u8; HEADER_SIZE];
        {
            let mut reader = std::io::BufReader::new(&file);
            reader.read_exact(&mut header)?;
        }

        // Validate magic
        if &header[0..4] != MAGIC {
            return Err(FerroError::invalid_input(
                "Invalid file format: magic bytes don't match FerroML memory-mapped dataset"
            ));
        }

        // Validate version
        let version = header[4];
        if version != VERSION {
            return Err(FerroError::invalid_input(format!(
                "Unsupported file version: {} (expected {})",
                version, VERSION
            )));
        }

        // Parse dimensions
        let n_samples = u64::from_le_bytes(header[5..13].try_into().unwrap()) as usize;
        let n_features = u64::from_le_bytes(header[13..21].try_into().unwrap()) as usize;
        let has_targets = header[21] != 0;

        // Memory map the entire file
        // SAFETY: File is read-only and we've validated the header
        let mmap = unsafe { Mmap::map(&file)? };

        // Validate file size
        let expected_size = Self::calculate_file_size(n_samples, n_features, has_targets);
        if mmap.len() < expected_size {
            return Err(FerroError::invalid_input(format!(
                "File corrupted: expected {} bytes, got {}",
                expected_size, mmap.len()
            )));
        }

        Ok(Self {
            path,
            mmap,
            n_samples,
            n_features,
            has_targets,
        })
    }

    /// Create a new memory-mapped dataset from arrays.
    ///
    /// # Arguments
    ///
    /// * `path` - Path for the new `.fmm` file
    /// * `x` - Feature matrix (n_samples, n_features)
    /// * `y` - Optional target vector (n_samples,)
    pub fn create<P: AsRef<Path>>(
        path: P,
        x: &Array2<f64>,
        y: Option<&Array1<f64>>,
    ) -> Result<Self> {
        let path = path.as_ref().to_path_buf();
        let n_samples = x.nrows();
        let n_features = x.ncols();
        let has_targets = y.is_some();

        // Validate shapes
        if let Some(targets) = y {
            if targets.len() != n_samples {
                return Err(FerroError::shape_mismatch(
                    format!("({}, {})", n_samples, n_features),
                    format!("target length {}", targets.len()),
                ));
            }
        }

        // Calculate file size
        let file_size = Self::calculate_file_size(n_samples, n_features, has_targets);

        // Create and initialize file
        let mut file = OpenOptions::new()
            .read(true)
            .write(true)
            .create(true)
            .truncate(true)
            .open(&path)?;

        // Pre-allocate file
        file.set_len(file_size as u64)?;

        // Write header
        let mut header = [0u8; HEADER_SIZE];
        header[0..4].copy_from_slice(MAGIC);
        header[4] = VERSION;
        header[5..13].copy_from_slice(&(n_samples as u64).to_le_bytes());
        header[13..21].copy_from_slice(&(n_features as u64).to_le_bytes());
        header[21] = if has_targets { 1 } else { 0 };
        // Reserved bytes 22..32 are already zero

        file.seek(SeekFrom::Start(0))?;
        file.write_all(&header)?;

        // Write feature data
        // Convert to row-major contiguous if needed and write raw bytes
        let x_contiguous = x.as_standard_layout();
        let x_slice = x_contiguous.as_slice().ok_or_else(|| {
            FerroError::invalid_input("Failed to get contiguous slice from feature array")
        })?;

        file.seek(SeekFrom::Start(HEADER_SIZE as u64))?;
        // SAFETY: f64 is POD and we're writing the correct number of bytes
        let x_bytes = unsafe {
            std::slice::from_raw_parts(
                x_slice.as_ptr() as *const u8,
                x_slice.len() * std::mem::size_of::<f64>(),
            )
        };
        file.write_all(x_bytes)?;

        // Write target data if present
        if let Some(targets) = y {
            let y_contiguous = targets.as_standard_layout();
            let y_slice = y_contiguous.as_slice().ok_or_else(|| {
                FerroError::invalid_input("Failed to get contiguous slice from target array")
            })?;

            // SAFETY: f64 is POD and we're writing the correct number of bytes
            let y_bytes = unsafe {
                std::slice::from_raw_parts(
                    y_slice.as_ptr() as *const u8,
                    y_slice.len() * std::mem::size_of::<f64>(),
                )
            };
            file.write_all(y_bytes)?;
        }

        file.sync_all()?;
        drop(file);

        // Open the created file as a memory-mapped dataset
        Self::open(path)
    }

    /// Create a memory-mapped dataset from an existing Dataset.
    pub fn from_dataset<P: AsRef<Path>>(path: P, dataset: &Dataset) -> Result<Self> {
        Self::create(path, dataset.x(), Some(dataset.y()))
    }

    /// Calculate the expected file size.
    fn calculate_file_size(n_samples: usize, n_features: usize, has_targets: bool) -> usize {
        let mut size = HEADER_SIZE;
        size += n_samples * n_features * std::mem::size_of::<f64>(); // Features
        if has_targets {
            size += n_samples * std::mem::size_of::<f64>(); // Targets
        }
        size
    }

    /// Get the path to the underlying file.
    pub fn path(&self) -> &Path {
        &self.path
    }

    /// Get the number of samples.
    pub fn n_samples(&self) -> usize {
        self.n_samples
    }

    /// Get the number of features.
    pub fn n_features(&self) -> usize {
        self.n_features
    }

    /// Get the shape as (n_samples, n_features).
    pub fn shape(&self) -> (usize, usize) {
        (self.n_samples, self.n_features)
    }

    /// Check if the dataset has targets.
    pub fn has_targets(&self) -> bool {
        self.has_targets
    }

    /// Get a view of the feature matrix (zero-copy).
    ///
    /// This returns an `ArrayView2` that directly references the memory-mapped
    /// data without copying.
    pub fn x_view(&self) -> ArrayView2<'_, f64> {
        let data_ptr = unsafe { self.mmap.as_ptr().add(HEADER_SIZE) as *const f64 };
        // SAFETY: Size verified during open()
        unsafe {
            ArrayView2::from_shape_ptr((self.n_samples, self.n_features), data_ptr)
        }
    }

    /// Get a view of the target vector (zero-copy).
    ///
    /// Returns `None` if the dataset doesn't have targets.
    pub fn y_view(&self) -> Option<ArrayView1<'_, f64>> {
        if !self.has_targets {
            return None;
        }
        let offset = HEADER_SIZE + self.n_samples * self.n_features * std::mem::size_of::<f64>();
        let data_ptr = unsafe { self.mmap.as_ptr().add(offset) as *const f64 };
        // SAFETY: Size verified during open()
        Some(unsafe {
            ArrayView1::from_shape_ptr(self.n_samples, data_ptr)
        })
    }

    /// Get a specific row of features.
    pub fn row(&self, idx: usize) -> Option<ArrayView1<'_, f64>> {
        if idx >= self.n_samples {
            return None;
        }
        let row_offset = HEADER_SIZE + idx * self.n_features * std::mem::size_of::<f64>();
        let data_ptr = unsafe { self.mmap.as_ptr().add(row_offset) as *const f64 };
        // SAFETY: Bounds checked
        Some(unsafe {
            ArrayView1::from_shape_ptr(self.n_features, data_ptr)
        })
    }

    /// Get a specific element from the feature matrix.
    pub fn get(&self, row: usize, col: usize) -> Option<f64> {
        if row >= self.n_samples || col >= self.n_features {
            return None;
        }
        let offset = HEADER_SIZE + (row * self.n_features + col) * std::mem::size_of::<f64>();
        let data_ptr = unsafe { self.mmap.as_ptr().add(offset) as *const f64 };
        // SAFETY: Bounds checked
        Some(unsafe { *data_ptr })
    }

    /// Get the target value for a specific sample.
    pub fn get_target(&self, idx: usize) -> Option<f64> {
        if !self.has_targets || idx >= self.n_samples {
            return None;
        }
        let offset = HEADER_SIZE
            + self.n_samples * self.n_features * std::mem::size_of::<f64>()
            + idx * std::mem::size_of::<f64>();
        let data_ptr = unsafe { self.mmap.as_ptr().add(offset) as *const f64 };
        // SAFETY: Bounds checked
        Some(unsafe { *data_ptr })
    }

    /// Copy a range of rows to a dense array.
    ///
    /// Use this for batch processing where you need to work with subsets.
    pub fn x_rows(&self, start: usize, end: usize) -> Result<Array2<f64>> {
        if start >= self.n_samples || end > self.n_samples || start >= end {
            return Err(FerroError::invalid_input(format!(
                "Invalid row range: {}..{} for {} samples",
                start, end, self.n_samples
            )));
        }
        let view = self.x_view();
        let slice = view.slice(ndarray::s![start..end, ..]);
        Ok(slice.to_owned())
    }

    /// Copy a range of targets to a dense array.
    pub fn y_slice(&self, start: usize, end: usize) -> Result<Array1<f64>> {
        if !self.has_targets {
            return Err(FerroError::invalid_input("Dataset has no targets"));
        }
        if start >= self.n_samples || end > self.n_samples || start >= end {
            return Err(FerroError::invalid_input(format!(
                "Invalid range: {}..{} for {} samples",
                start, end, self.n_samples
            )));
        }
        let view = self.y_view().unwrap();
        let slice = view.slice(ndarray::s![start..end]);
        Ok(slice.to_owned())
    }

    /// Convert to a regular Dataset by loading all data into memory.
    ///
    /// Use with caution for large datasets - this will allocate memory
    /// for the entire dataset.
    pub fn to_dataset(&self) -> Dataset {
        let x = self.x_view().to_owned();
        let y = self.y_view()
            .map(|v| v.to_owned())
            .unwrap_or_else(|| Array1::zeros(self.n_samples));
        Dataset::new(x, y)
    }

    /// Get an iterator over batches of samples.
    ///
    /// Returns an iterator that yields (x_batch, y_batch) tuples.
    pub fn batches(&self, batch_size: usize) -> BatchIterator<'_> {
        BatchIterator {
            dataset: self,
            batch_size,
            current: 0,
        }
    }

    /// Get a sample iterator for streaming access.
    pub fn samples(&self) -> SampleIterator<'_> {
        SampleIterator {
            dataset: self,
            current: 0,
        }
    }
}

/// Iterator over batches of samples from a memory-mapped dataset.
pub struct BatchIterator<'a> {
    dataset: &'a MemmappedDataset,
    batch_size: usize,
    current: usize,
}

impl<'a> Iterator for BatchIterator<'a> {
    type Item = (Array2<f64>, Option<Array1<f64>>);

    fn next(&mut self) -> Option<Self::Item> {
        if self.current >= self.dataset.n_samples {
            return None;
        }

        let start = self.current;
        let end = (start + self.batch_size).min(self.dataset.n_samples);
        self.current = end;

        let x_batch = self.dataset.x_rows(start, end).ok()?;
        let y_batch = if self.dataset.has_targets {
            self.dataset.y_slice(start, end).ok()
        } else {
            None
        };

        Some((x_batch, y_batch))
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        let remaining = self.dataset.n_samples.saturating_sub(self.current);
        let batches = (remaining + self.batch_size - 1) / self.batch_size;
        (batches, Some(batches))
    }
}

impl ExactSizeIterator for BatchIterator<'_> {}

/// Iterator over individual samples from a memory-mapped dataset.
pub struct SampleIterator<'a> {
    dataset: &'a MemmappedDataset,
    current: usize,
}

impl<'a> Iterator for SampleIterator<'a> {
    type Item = (ArrayView1<'a, f64>, Option<f64>);

    fn next(&mut self) -> Option<Self::Item> {
        if self.current >= self.dataset.n_samples {
            return None;
        }

        let idx = self.current;
        self.current += 1;

        let x = self.dataset.row(idx)?;
        let y = self.dataset.get_target(idx);

        Some((x, y))
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        let remaining = self.dataset.n_samples.saturating_sub(self.current);
        (remaining, Some(remaining))
    }
}

impl ExactSizeIterator for SampleIterator<'_> {}

/// Builder for creating memory-mapped datasets.
///
/// Provides a fluent interface for creating memory-mapped datasets from
/// various sources.
///
/// # Example
///
/// ```ignore
/// use ferroml_core::datasets::mmap::MemmappedDatasetBuilder;
///
/// // Create from arrays
/// let dataset = MemmappedDatasetBuilder::new("data.fmm")
///     .with_features(x)
///     .with_targets(y)
///     .build()?;
///
/// // Create from existing Dataset
/// let dataset = MemmappedDatasetBuilder::new("data.fmm")
///     .from_dataset(&existing_dataset)?
///     .build()?;
/// ```
pub struct MemmappedDatasetBuilder {
    path: PathBuf,
    x: Option<Array2<f64>>,
    y: Option<Array1<f64>>,
}

impl MemmappedDatasetBuilder {
    /// Create a new builder with the target file path.
    pub fn new<P: AsRef<Path>>(path: P) -> Self {
        Self {
            path: path.as_ref().to_path_buf(),
            x: None,
            y: None,
        }
    }

    /// Set the feature matrix.
    pub fn with_features(mut self, x: Array2<f64>) -> Self {
        self.x = Some(x);
        self
    }

    /// Set the target vector.
    pub fn with_targets(mut self, y: Array1<f64>) -> Self {
        self.y = Some(y);
        self
    }

    /// Initialize from an existing Dataset.
    pub fn from_dataset(mut self, dataset: &Dataset) -> Self {
        self.x = Some(dataset.x().clone());
        self.y = Some(dataset.y().clone());
        self
    }

    /// Build the memory-mapped dataset.
    pub fn build(self) -> Result<MemmappedDataset> {
        let x = self.x.ok_or_else(|| {
            FerroError::invalid_input("Feature matrix (x) is required")
        })?;

        MemmappedDataset::create(&self.path, &x, self.y.as_ref())
    }
}

/// Get information about a memory-mapped dataset file without fully loading it.
///
/// # Arguments
///
/// * `path` - Path to the `.fmm` file
///
/// # Returns
///
/// A tuple of (n_samples, n_features, has_targets)
pub fn peek_mmap_info<P: AsRef<Path>>(path: P) -> Result<(usize, usize, bool)> {
    let mut file = File::open(path.as_ref())?;

    let mut header = [0u8; HEADER_SIZE];
    file.read_exact(&mut header)?;

    // Validate magic
    if &header[0..4] != MAGIC {
        return Err(FerroError::invalid_input(
            "Invalid file format: not a FerroML memory-mapped dataset"
        ));
    }

    let n_samples = u64::from_le_bytes(header[5..13].try_into().unwrap()) as usize;
    let n_features = u64::from_le_bytes(header[13..21].try_into().unwrap()) as usize;
    let has_targets = header[21] != 0;

    Ok((n_samples, n_features, has_targets))
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;
    use tempfile::NamedTempFile;

    #[test]
    fn test_create_and_open() {
        let file = NamedTempFile::with_suffix(".fmm").unwrap();
        let path = file.path();

        let x = array![[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]];
        let y = array![0.0, 1.0, 0.0];

        // Create
        let dataset = MemmappedDataset::create(path, &x, Some(&y)).unwrap();
        assert_eq!(dataset.n_samples(), 3);
        assert_eq!(dataset.n_features(), 2);
        assert!(dataset.has_targets());

        // Close and reopen
        drop(dataset);
        let dataset = MemmappedDataset::open(path).unwrap();
        assert_eq!(dataset.n_samples(), 3);
        assert_eq!(dataset.n_features(), 2);
    }

    #[test]
    fn test_x_view() {
        let file = NamedTempFile::with_suffix(".fmm").unwrap();
        let path = file.path();

        let x = array![[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]];
        let y = array![0.0, 1.0, 0.0];

        let dataset = MemmappedDataset::create(path, &x, Some(&y)).unwrap();
        let x_view = dataset.x_view();

        assert_eq!(x_view.shape(), &[3, 2]);
        assert!((x_view[[0, 0]] - 1.0).abs() < 1e-10);
        assert!((x_view[[0, 1]] - 2.0).abs() < 1e-10);
        assert!((x_view[[1, 0]] - 3.0).abs() < 1e-10);
        assert!((x_view[[2, 1]] - 6.0).abs() < 1e-10);
    }

    #[test]
    fn test_y_view() {
        let file = NamedTempFile::with_suffix(".fmm").unwrap();
        let path = file.path();

        let x = array![[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]];
        let y = array![10.0, 20.0, 30.0];

        let dataset = MemmappedDataset::create(path, &x, Some(&y)).unwrap();
        let y_view = dataset.y_view().unwrap();

        assert_eq!(y_view.len(), 3);
        assert!((y_view[0] - 10.0).abs() < 1e-10);
        assert!((y_view[1] - 20.0).abs() < 1e-10);
        assert!((y_view[2] - 30.0).abs() < 1e-10);
    }

    #[test]
    fn test_no_targets() {
        let file = NamedTempFile::with_suffix(".fmm").unwrap();
        let path = file.path();

        let x = array![[1.0, 2.0], [3.0, 4.0]];

        let dataset = MemmappedDataset::create(path, &x, None).unwrap();
        assert!(!dataset.has_targets());
        assert!(dataset.y_view().is_none());
    }

    #[test]
    fn test_row_access() {
        let file = NamedTempFile::with_suffix(".fmm").unwrap();
        let path = file.path();

        let x = array![[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]];
        let y = array![0.0, 1.0, 2.0];

        let dataset = MemmappedDataset::create(path, &x, Some(&y)).unwrap();

        let row0 = dataset.row(0).unwrap();
        assert_eq!(row0.len(), 3);
        assert!((row0[0] - 1.0).abs() < 1e-10);
        assert!((row0[1] - 2.0).abs() < 1e-10);
        assert!((row0[2] - 3.0).abs() < 1e-10);

        let row2 = dataset.row(2).unwrap();
        assert!((row2[0] - 7.0).abs() < 1e-10);

        assert!(dataset.row(3).is_none());
    }

    #[test]
    fn test_element_access() {
        let file = NamedTempFile::with_suffix(".fmm").unwrap();
        let path = file.path();

        let x = array![[1.0, 2.0], [3.0, 4.0]];
        let y = array![10.0, 20.0];

        let dataset = MemmappedDataset::create(path, &x, Some(&y)).unwrap();

        assert_eq!(dataset.get(0, 0), Some(1.0));
        assert_eq!(dataset.get(0, 1), Some(2.0));
        assert_eq!(dataset.get(1, 0), Some(3.0));
        assert_eq!(dataset.get(1, 1), Some(4.0));
        assert_eq!(dataset.get(2, 0), None);
        assert_eq!(dataset.get(0, 2), None);

        assert_eq!(dataset.get_target(0), Some(10.0));
        assert_eq!(dataset.get_target(1), Some(20.0));
        assert_eq!(dataset.get_target(2), None);
    }

    #[test]
    fn test_x_rows() {
        let file = NamedTempFile::with_suffix(".fmm").unwrap();
        let path = file.path();

        let x = array![[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0]];
        let y = array![0.0, 1.0, 2.0, 3.0];

        let dataset = MemmappedDataset::create(path, &x, Some(&y)).unwrap();

        let batch = dataset.x_rows(1, 3).unwrap();
        assert_eq!(batch.shape(), &[2, 2]);
        assert!((batch[[0, 0]] - 3.0).abs() < 1e-10);
        assert!((batch[[1, 1]] - 6.0).abs() < 1e-10);
    }

    #[test]
    fn test_y_slice() {
        let file = NamedTempFile::with_suffix(".fmm").unwrap();
        let path = file.path();

        let x = array![[1.0], [2.0], [3.0], [4.0]];
        let y = array![10.0, 20.0, 30.0, 40.0];

        let dataset = MemmappedDataset::create(path, &x, Some(&y)).unwrap();

        let y_batch = dataset.y_slice(1, 3).unwrap();
        assert_eq!(y_batch.len(), 2);
        assert!((y_batch[0] - 20.0).abs() < 1e-10);
        assert!((y_batch[1] - 30.0).abs() < 1e-10);
    }

    #[test]
    fn test_to_dataset() {
        let file = NamedTempFile::with_suffix(".fmm").unwrap();
        let path = file.path();

        let x = array![[1.0, 2.0], [3.0, 4.0]];
        let y = array![0.0, 1.0];

        let mmap_dataset = MemmappedDataset::create(path, &x, Some(&y)).unwrap();
        let regular_dataset = mmap_dataset.to_dataset();

        assert_eq!(regular_dataset.n_samples(), 2);
        assert_eq!(regular_dataset.n_features(), 2);
        assert_eq!(regular_dataset.x(), &x);
        assert_eq!(regular_dataset.y(), &y);
    }

    #[test]
    fn test_batch_iterator() {
        let file = NamedTempFile::with_suffix(".fmm").unwrap();
        let path = file.path();

        let x = array![[1.0], [2.0], [3.0], [4.0], [5.0]];
        let y = array![10.0, 20.0, 30.0, 40.0, 50.0];

        let dataset = MemmappedDataset::create(path, &x, Some(&y)).unwrap();

        let batches: Vec<_> = dataset.batches(2).collect();
        assert_eq!(batches.len(), 3); // 5 samples with batch_size 2 = 3 batches

        assert_eq!(batches[0].0.nrows(), 2);
        assert_eq!(batches[1].0.nrows(), 2);
        assert_eq!(batches[2].0.nrows(), 1); // Last batch has 1 sample
    }

    #[test]
    fn test_sample_iterator() {
        let file = NamedTempFile::with_suffix(".fmm").unwrap();
        let path = file.path();

        let x = array![[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]];
        let y = array![10.0, 20.0, 30.0];

        let dataset = MemmappedDataset::create(path, &x, Some(&y)).unwrap();

        let samples: Vec<_> = dataset.samples().collect();
        assert_eq!(samples.len(), 3);

        assert!((samples[0].0[0] - 1.0).abs() < 1e-10);
        assert_eq!(samples[0].1, Some(10.0));
        assert!((samples[2].0[1] - 6.0).abs() < 1e-10);
        assert_eq!(samples[2].1, Some(30.0));
    }

    #[test]
    fn test_builder() {
        let file = NamedTempFile::with_suffix(".fmm").unwrap();
        let path = file.path();

        let x = array![[1.0, 2.0], [3.0, 4.0]];
        let y = array![0.0, 1.0];

        let dataset = MemmappedDatasetBuilder::new(path)
            .with_features(x.clone())
            .with_targets(y.clone())
            .build()
            .unwrap();

        assert_eq!(dataset.n_samples(), 2);
        assert_eq!(dataset.n_features(), 2);
        assert_eq!(dataset.x_view(), x.view());
    }

    #[test]
    fn test_builder_from_dataset() {
        let file = NamedTempFile::with_suffix(".fmm").unwrap();
        let path = file.path();

        let x = array![[1.0, 2.0], [3.0, 4.0]];
        let y = array![0.0, 1.0];
        let source = Dataset::new(x.clone(), y.clone());

        let dataset = MemmappedDatasetBuilder::new(path)
            .from_dataset(&source)
            .build()
            .unwrap();

        assert_eq!(dataset.x_view(), x.view());
        assert_eq!(dataset.y_view().unwrap(), y.view());
    }

    #[test]
    fn test_peek_info() {
        let file = NamedTempFile::with_suffix(".fmm").unwrap();
        let path = file.path();

        let x = array![[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]];
        let y = array![0.0, 1.0];

        MemmappedDataset::create(path, &x, Some(&y)).unwrap();

        let (n_samples, n_features, has_targets) = peek_mmap_info(path).unwrap();
        assert_eq!(n_samples, 2);
        assert_eq!(n_features, 3);
        assert!(has_targets);
    }

    #[test]
    fn test_from_dataset() {
        let file = NamedTempFile::with_suffix(".fmm").unwrap();
        let path = file.path();

        let x = array![[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]];
        let y = array![0.0, 1.0, 0.0];
        let source = Dataset::new(x.clone(), y.clone());

        let mmap_dataset = MemmappedDataset::from_dataset(path, &source).unwrap();

        assert_eq!(mmap_dataset.n_samples(), 3);
        assert_eq!(mmap_dataset.n_features(), 2);
        assert_eq!(mmap_dataset.x_view(), x.view());
        assert_eq!(mmap_dataset.y_view().unwrap(), y.view());
    }

    #[test]
    fn test_invalid_file() {
        let file = NamedTempFile::with_suffix(".fmm").unwrap();
        let path = file.path();

        // Write invalid data
        std::fs::write(path, b"invalid data").unwrap();

        let result = MemmappedDataset::open(path);
        assert!(result.is_err());
    }

    #[test]
    fn test_shape_mismatch() {
        let file = NamedTempFile::with_suffix(".fmm").unwrap();
        let path = file.path();

        let x = array![[1.0, 2.0], [3.0, 4.0]];
        let y = array![0.0, 1.0, 2.0]; // Wrong length

        let result = MemmappedDataset::create(path, &x, Some(&y));
        assert!(result.is_err());
    }

    #[test]
    fn test_large_dataset() {
        let file = NamedTempFile::with_suffix(".fmm").unwrap();
        let path = file.path();

        // Create a larger dataset
        let n_samples = 10_000;
        let n_features = 100;
        let x = Array2::from_shape_fn((n_samples, n_features), |(i, j)| (i * n_features + j) as f64);
        let y = Array1::from_shape_fn(n_samples, |i| (i % 10) as f64);

        let dataset = MemmappedDataset::create(path, &x, Some(&y)).unwrap();

        assert_eq!(dataset.n_samples(), n_samples);
        assert_eq!(dataset.n_features(), n_features);

        // Check some values
        assert!((dataset.get(0, 0).unwrap() - 0.0).abs() < 1e-10);
        assert!((dataset.get(5000, 50).unwrap() - (5000.0 * 100.0 + 50.0)).abs() < 1e-10);
        assert_eq!(dataset.get_target(9999), Some(9.0));
    }

    #[test]
    fn test_memmapped_array2() {
        let file = NamedTempFile::with_suffix(".fmm").unwrap();
        let path = file.path();

        let x = array![[1.0, 2.0], [3.0, 4.0]];
        let y = array![0.0, 1.0];

        // Create dataset to get proper file
        MemmappedDataset::create(path, &x, Some(&y)).unwrap();

        // Open just the array portion
        let arr = MemmappedArray2::open(path, 2, 2, HEADER_SIZE).unwrap();
        assert_eq!(arr.shape(), (2, 2));
        assert_eq!(arr.get(0, 0), Some(1.0));
        assert_eq!(arr.get(1, 1), Some(4.0));

        let row = arr.row(1).unwrap();
        assert_eq!(row, &[3.0, 4.0]);

        let slice = arr.rows_to_array(0, 2).unwrap();
        assert_eq!(slice, x);
    }

    #[test]
    fn test_memmapped_array1() {
        let file = NamedTempFile::with_suffix(".fmm").unwrap();
        let path = file.path();

        let x = array![[1.0, 2.0], [3.0, 4.0]];
        let y = array![10.0, 20.0];

        // Create dataset to get proper file
        MemmappedDataset::create(path, &x, Some(&y)).unwrap();

        // Open just the target portion
        let offset = HEADER_SIZE + 2 * 2 * std::mem::size_of::<f64>();
        let arr = MemmappedArray1::open(path, 2, offset).unwrap();

        assert_eq!(arr.len(), 2);
        assert!(!arr.is_empty());
        assert_eq!(arr.get(0), Some(10.0));
        assert_eq!(arr.get(1), Some(20.0));
        assert_eq!(arr.get(2), None);

        let slice = arr.slice_to_array(0, 2).unwrap();
        assert_eq!(slice, y);
    }

    #[test]
    fn test_persistence_across_open() {
        let file = NamedTempFile::with_suffix(".fmm").unwrap();
        let path = file.path().to_path_buf();

        let x = array![[1.5, 2.5], [3.5, 4.5]];
        let y = array![100.0, 200.0];

        // Create and close
        {
            let _dataset = MemmappedDataset::create(&path, &x, Some(&y)).unwrap();
        }

        // Reopen and verify
        let dataset = MemmappedDataset::open(&path).unwrap();
        assert!((dataset.get(0, 0).unwrap() - 1.5).abs() < 1e-10);
        assert!((dataset.get(1, 1).unwrap() - 4.5).abs() < 1e-10);
        assert!((dataset.get_target(0).unwrap() - 100.0).abs() < 1e-10);
        assert!((dataset.get_target(1).unwrap() - 200.0).abs() < 1e-10);
    }
}
