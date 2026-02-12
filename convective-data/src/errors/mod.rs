//! # convective-data :: errors
pub mod persist;
pub use persist::PersistError;
pub mod datasets;
pub use datasets::DatasetError;
pub mod configs;
pub use configs::ConfigError;
