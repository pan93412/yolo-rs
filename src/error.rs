//! The errors of the Yolo crate.

/// An ONNX Runtime error, labelled with the step that failed.
#[derive(thiserror::Error, Debug)]
pub enum YoloError {
    /// Creating the session builder failed.
    #[error("build session: {0}")]
    OrtSessionBuildError(ort::Error),
    /// Loading the model into the session failed.
    #[error("load session: {0}")]
    OrtSessionLoadError(ort::Error),
    /// The input tensor could not be passed to ONNX Runtime.
    #[error("load model: {0}")]
    OrtInputError(ort::Error),
    /// Running the model failed.
    #[error("run inference: {0}")]
    OrtInferenceError(ort::Error),
    /// The output tensor could not be read.
    #[error("extract sensor: {0}")]
    OrtExtractSensorError(ort::Error),
}
