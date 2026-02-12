#[derive(Debug)]
pub enum ConfigError {
    /// I/o kind of error
    Io(std::io::Error),
    /// toml related
    Toml(toml::de::Error),
    /// file was not found
    FileNotFound,
    /// Failed to parse the file
    ParseError(String),
    /// Format not supported
    UnsupportedFormat(String),
}

impl std::fmt::Display for ConfigError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Io(e) => write!(f, "IO error: {}", e),
            Self::Toml(e) => write!(f, "Toml error: {}", e),
            Self::FileNotFound => write!(f, "Config file not found"),
            Self::ParseError(file) => write!(f, "Parse error for file: {}", file),
            Self::UnsupportedFormat(s) => write!(f, "Unsupported format: {}", s),
        }
    }
}

impl std::error::Error for ConfigError {}

impl From<std::io::Error> for ConfigError {
    fn from(e: std::io::Error) -> Self {
        Self::Io(e)
    }
}

impl From<toml::de::Error> for ConfigError {
    fn from(e: toml::de::Error) -> Self {
        Self::Toml(e)
    }
}
