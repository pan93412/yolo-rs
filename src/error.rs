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
    #[error("extract tensor: {0}")]
    OrtExtractTensorError(ort::Error),
    #[error("model has no inputs")]
    MissingModelInput,
    #[error("model has no outputs")]
    MissingModelOutput,
    #[error("model has no segmentation mask output")]
    MissingMaskOutput,
    #[error("unsupported model output shape {0:?}; expected a 3D detection tensor")]
    InvalidOutputShape(Vec<usize>),
    #[error("unsupported mask output shape {0:?}; expected a 4D mask prototype tensor")]
    InvalidMaskShape(Vec<usize>),
    #[error(
        "model output has {available} detection channels, but {required} are required for 4 box coordinates plus {label_count} labels"
    )]
    InsufficientDetectionChannels {
        available: usize,
        required: usize,
        label_count: usize,
    },
    #[error(
        "model output has {available} mask coefficients, but {required} are required for {prototype_count} mask prototypes"
    )]
    InsufficientMaskCoefficients {
        available: usize,
        required: usize,
        prototype_count: usize,
    },
    #[error("detected class index {class_id} is out of range for {label_count} labels")]
    UnknownClassId { class_id: usize, label_count: usize },
}
