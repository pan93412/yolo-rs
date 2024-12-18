//! The errors of the Yolo crate.

#[derive(thiserror::Error, Debug)]
pub enum YoloError {
    #[error("build session: {0}")]
    OrtSessionBuildError(ort::Error),
    #[error("load session: {0}")]
    OrtSessionLoadError(ort::Error),
    #[error("load model: {0}")]
    OrtInputError(ort::Error),
    #[error("run inference: {0}")]
    OrtInferenceError(ort::Error),
    #[error("extract sensor: {0}")]
    OrtExtractSensorError(ort::Error),
}
