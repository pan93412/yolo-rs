//! The errors of the Yolo crate.

pub enum YoloError {
    OrtSessionBuildError(ort::Error),
    OrtSessionLoadError(ort::Error),
    OrtInputError(ort::Error),
    OrtInferenceError(ort::Error),
    OrtExtractSensorError(ort::Error),
}
