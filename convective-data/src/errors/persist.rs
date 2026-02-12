//! # convective-data :: errors :: persist

/// Errors from persistence operations
#[derive(Debug)]
pub enum PersistError {
    Io(std::io::Error),
    Json(serde_json::Error),
    #[cfg(feature = "parquet")]
    Parquet(parquet::errors::ParquetError),
    Parse(String),
    #[cfg(feature = "parquet")]
    Arrow(arrow::error::ArrowError),
    UnsupportedFormat(String),
}

impl std::fmt::Display for PersistError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Io(e) => write!(f, "IO error: {}", e),
            Self::Json(e) => write!(f, "JSON error: {}", e),
            #[cfg(feature = "parquet")]
            Self::Parquet(e) => write!(f, "Parquet error: {}", e),
            #[cfg(feature = "parquet")]
            Self::Arrow(e) => write!(f, "Arrow error: {}", e),
            Self::UnsupportedFormat(s) => write!(f, "Unsupported format: {}", s),
            PersistError::Parse(_) => todo!(),
        }
    }
}

impl std::error::Error for PersistError {}

impl From<std::io::Error> for PersistError {
    fn from(e: std::io::Error) -> Self {
        Self::Io(e)
    }
}

impl From<serde_json::Error> for PersistError {
    fn from(e: serde_json::Error) -> Self {
        Self::Json(e)
    }
}

#[cfg(feature = "parquet")]
impl From<parquet::errors::ParquetError> for PersistError {
    fn from(e: parquet::errors::ParquetError) -> Self {
        Self::Parquet(e)
    }
}

#[cfg(feature = "parquet")]
impl From<arrow::error::ArrowError> for PersistError {
    fn from(e: arrow::error::ArrowError) -> Self {
        Self::Arrow(e)
    }
}
