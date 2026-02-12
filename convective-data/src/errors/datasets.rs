use thiserror::Error;
#[derive(Error, Debug)]
pub enum DatasetError {
    #[error("The Datset Generator function failed")]
    DatasetFailure,
}
