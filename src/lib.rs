//! A Rust library for YOLO-style ONNX detection and segmentation models.
//!
//! This library provides a high-level API for running YOLO-style ONNX models,
//! including exported closed-set YOLOE segmentation models.

pub mod error;
pub mod model;
pub mod prompt;

use arcstr::ArcStr;
use error::YoloError;
use image::{DynamicImage, GenericImageView, GrayImage, Luma, Rgba, imageops::FilterType};
use model::YoloModelSession;
use ndarray::{Array4, ArrayBase, ArrayView3, ArrayView4};
use ort::{inputs, value::TensorRef};
use prompt::YoloPromptEmbeddingsView;

const DEFAULT_MODEL_SIZE: u32 = 640;

#[derive(Debug, Clone, Copy)]
pub struct BoundingBox {
    pub x1: f32,
    pub y1: f32,
    pub x2: f32,
    pub y2: f32,
}

#[derive(Debug, Clone)]
pub struct YoloInput {
    pub tensor: Array4<f32>, // 640x640
    pub raw_width: u32,
    pub raw_height: u32,
    pub model_width: u32,
    pub model_height: u32,
    pub resize_ratio: f32,
    pub pad_w: f32,
    pub pad_h: f32,
}

impl YoloInput {
    pub fn view(&self) -> YoloInputView<'_> {
        YoloInputView {
            tensor_view: self.tensor.view(),
            raw_width: self.raw_width,
            raw_height: self.raw_height,
            model_width: self.model_width,
            model_height: self.model_height,
            resize_ratio: self.resize_ratio,
            pad_w: self.pad_w,
            pad_h: self.pad_h,
        }
    }
}

#[derive(Debug, Clone, Copy)]
pub struct YoloInputView<'a> {
    pub tensor_view: ArrayView4<'a, f32>,
    pub raw_width: u32,
    pub raw_height: u32,
    pub model_width: u32,
    pub model_height: u32,
    pub resize_ratio: f32,
    pub pad_w: f32,
    pub pad_h: f32,
}

#[derive(Debug, Clone)]
pub struct YoloEntityOutput {
    pub bounding_box: BoundingBox,
    /// The label of the detected entity.
    ///
    /// You can check the metadata of the model with
    /// [Netron](https://netron.app) to get the labels.
    pub label: ArcStr,
    /// The confidence of the detected entity.
    pub confidence: f32,
}

#[derive(Debug, Clone)]
pub struct YoloSegmentationOutput {
    pub entity: YoloEntityOutput,
    pub mask: GrayImage,
}

fn validate_prompt_embeddings(
    prompts: YoloPromptEmbeddingsView<'_>,
) -> Result<YoloPromptEmbeddingsView<'_>, YoloError> {
    if prompts.embeddings.shape().len() != 3 || prompts.embeddings.shape()[0] != 1 {
        return Err(YoloError::InvalidPromptEmbeddingShape(
            prompts.embeddings.shape().to_vec(),
        ));
    }
    if prompts.labels.len() != prompts.embeddings.shape()[1] {
        return Err(YoloError::PromptLabelEmbeddingMismatch {
            labels: prompts.labels.len(),
            embeddings: prompts.embeddings.shape()[1],
        });
    }

    Ok(prompts)
}

#[derive(Debug, Clone)]
struct DecodedDetection {
    entity: YoloEntityOutput,
    mask_coefficients: Vec<f32>,
}

#[derive(Debug, Clone, Copy)]
struct LetterboxInfo {
    raw_width: u32,
    raw_height: u32,
    model_width: u32,
    model_height: u32,
    resize_ratio: f32,
    pad_w: f32,
    pad_h: f32,
}

fn output_layout(output: ArrayView3<'_, f32>, label_count: usize) -> Result<(usize, usize), YoloError> {
    if output.shape()[0] != 1 {
        return Err(YoloError::InvalidOutputShape(output.shape().to_vec()));
    }

    let channels_axis = if output.shape()[1] <= output.shape()[2] { 1 } else { 2 };
    let detection_axis = if channels_axis == 1 { 2 } else { 1 };

    let channel_count = output.shape()[channels_axis];
    let required_channels = 4 + label_count;

    if channel_count < required_channels {
        return Err(YoloError::InsufficientDetectionChannels {
            available: channel_count,
            required: required_channels,
            label_count,
        });
    }

    Ok((channels_axis, output.shape()[detection_axis]))
}

fn intersection(box1: &BoundingBox, box2: &BoundingBox) -> f32 {
    (box1.x2.min(box2.x2) - box1.x1.max(box2.x1)) * (box1.y2.min(box2.y2) - box1.y1.max(box2.y1))
}

fn union(box1: &BoundingBox, box2: &BoundingBox) -> f32 {
    ((box1.x2 - box1.x1) * (box1.y2 - box1.y1)) + ((box2.x2 - box2.x1) * (box2.y2 - box2.y1))
        - intersection(box1, box2)
}

fn non_maximum_suppression(
    mut boxes: Vec<DecodedDetection>,
    iou_threshold: f32,
) -> Vec<DecodedDetection> {
    if boxes.is_empty() {
        return Vec::new();
    }

    boxes.sort_unstable_by(|a, b| b.entity.confidence.total_cmp(&a.entity.confidence));

    let mut result = Vec::with_capacity(boxes.len());
    for current in boxes {
        if result.iter().all(|selected: &DecodedDetection| {
            let iou = intersection(&selected.entity.bounding_box, &current.entity.bounding_box)
                / union(&selected.entity.bounding_box, &current.entity.bounding_box);
            iou < iou_threshold
        }) {
            result.push(current);
        }
    }

    result.shrink_to_fit();
    result
}

fn decode_detections(
    output: ArrayView3<'_, f32>,
    labels: &[ArcStr],
    probability_threshold: f32,
    letterbox: LetterboxInfo,
    expected_mask_coefficients: Option<usize>,
) -> Result<Vec<DecodedDetection>, YoloError> {
    let (channels_axis, detection_count) = output_layout(output, labels.len())?;

    Ok((0..detection_count)
        .filter_map(|row| {
            let row = if channels_axis == 1 {
                output.slice(ndarray::s![0, .., row])
            } else {
                output.slice(ndarray::s![0, row, ..])
            };

            let (class_id, prob) = row
                .iter()
                .skip(4)
                .take(labels.len())
                .enumerate()
                .map(|(index, value)| (index, *value))
                .reduce(|accum, row| if row.1 > accum.1 { row } else { accum })
                .filter(|(_, prob)| *prob >= probability_threshold)?;

            let label = labels
                .get(class_id)
                .cloned()
                .ok_or(YoloError::UnknownClassId {
                    class_id,
                    label_count: labels.len(),
                })
                .ok()?;

            let mask_coefficients = row
                .iter()
                .skip(4 + labels.len())
                .copied()
                .collect::<Vec<_>>();

            if let Some(required) = expected_mask_coefficients
                && mask_coefficients.len() < required
            {
                return Some(Err(YoloError::InsufficientMaskCoefficients {
                    available: mask_coefficients.len(),
                    required,
                    prototype_count: required,
                }));
            }

            let x1 = ((row[0_usize] - row[2_usize] / 2.0) - letterbox.pad_w) / letterbox.resize_ratio;
            let y1 = ((row[1_usize] - row[3_usize] / 2.0) - letterbox.pad_h) / letterbox.resize_ratio;
            let x2 = ((row[0_usize] + row[2_usize] / 2.0) - letterbox.pad_w) / letterbox.resize_ratio;
            let y2 = ((row[1_usize] + row[3_usize] / 2.0) - letterbox.pad_h) / letterbox.resize_ratio;

            Some(Ok(DecodedDetection {
                entity: YoloEntityOutput {
                    bounding_box: BoundingBox {
                        x1: x1.clamp(0.0, letterbox.raw_width as f32),
                        y1: y1.clamp(0.0, letterbox.raw_height as f32),
                        x2: x2.clamp(0.0, letterbox.raw_width as f32),
                        y2: y2.clamp(0.0, letterbox.raw_height as f32),
                    },
                    label,
                    confidence: prob,
                },
                mask_coefficients,
            }))
        })
        .collect::<Result<Vec<_>, _>>()?)
}

fn decode_segmentation_masks(
    detections: Vec<DecodedDetection>,
    prototypes: ArrayView3<'_, f32>,
    letterbox: LetterboxInfo,
) -> Result<Vec<YoloSegmentationOutput>, YoloError> {
    let prototype_count = prototypes.shape()[0];
    let mask_height = prototypes.shape()[1];
    let mask_width = prototypes.shape()[2];

    let mask_pad_w = letterbox.pad_w * (mask_width as f32 / letterbox.model_width as f32);
    let mask_pad_h = letterbox.pad_h * (mask_height as f32 / letterbox.model_height as f32);
    let mask_content_width = (mask_width as f32 - (mask_pad_w * 2.0)).max(1.0);
    let mask_content_height = (mask_height as f32 - (mask_pad_h * 2.0)).max(1.0);

    let mut results = Vec::with_capacity(detections.len());
    for detection in detections {
        let mut mask = GrayImage::new(letterbox.raw_width, letterbox.raw_height);
        let bbox = detection.entity.bounding_box;

        let start_x = bbox.x1.floor().max(0.0) as u32;
        let end_x = bbox.x2.ceil().min(letterbox.raw_width as f32) as u32;
        let start_y = bbox.y1.floor().max(0.0) as u32;
        let end_y = bbox.y2.ceil().min(letterbox.raw_height as f32) as u32;

        let mut proto_mask = vec![0.0f32; mask_width * mask_height];
        for proto_y in 0..mask_height {
            for proto_x in 0..mask_width {
                let value = detection
                    .mask_coefficients
                    .iter()
                    .take(prototype_count)
                    .enumerate()
                    .map(|(index, coefficient)| coefficient * prototypes[[index, proto_y, proto_x]])
                    .sum::<f32>();
                proto_mask[proto_y * mask_width + proto_x] = value;
            }
        }

        for y in start_y..end_y {
            for x in start_x..end_x {
                let scaled_x = ((x as f32 + 0.5) / letterbox.raw_width as f32) * mask_content_width + mask_pad_w;
                let scaled_y = ((y as f32 + 0.5) / letterbox.raw_height as f32) * mask_content_height + mask_pad_h;
                let value = bilinear_sample(&proto_mask, mask_width, mask_height, scaled_x, scaled_y);

                if value > 0.5 {
                    mask.put_pixel(x, y, Luma([255]));
                }
            }
        }

        results.push(YoloSegmentationOutput {
            entity: detection.entity,
            mask,
        });
    }

    Ok(results)
}

fn bilinear_sample(buffer: &[f32], width: usize, height: usize, x: f32, y: f32) -> f32 {
    let x = x.clamp(0.0, (width.saturating_sub(1)) as f32);
    let y = y.clamp(0.0, (height.saturating_sub(1)) as f32);

    let x0 = x.floor() as usize;
    let y0 = y.floor() as usize;
    let x1 = (x0 + 1).min(width.saturating_sub(1));
    let y1 = (y0 + 1).min(height.saturating_sub(1));

    let dx = x - x0 as f32;
    let dy = y - y0 as f32;

    let top_left = buffer[y0 * width + x0];
    let top_right = buffer[y0 * width + x1];
    let bottom_left = buffer[y1 * width + x0];
    let bottom_right = buffer[y1 * width + x1];

    let top = top_left + dx * (top_right - top_left);
    let bottom = bottom_left + dx * (bottom_right - bottom_left);
    top + dy * (bottom - top)
}

/// Convert an image to a YOLO input tensor.
///
/// The input image is resized to 640x640 and normalized to the range [0, 1].
/// The tensor has the shape (1, 3, 640, 640) and the layout is (R, G, B).
///
/// You can pass the resulting tensor to the [`inference`] function.
/// Note that you might need to call [`YoloInput::view`] to get a view of the tensor.
pub fn image_to_yolo_input_tensor(original_image: &DynamicImage) -> YoloInput {
    let model_width = DEFAULT_MODEL_SIZE;
    let model_height = DEFAULT_MODEL_SIZE;
    let mut input = ArrayBase::from_elem(
        (1, 3, model_height as usize, model_width as usize),
        114.0f32 / 255.0,
    );

    let raw_width = original_image.width();
    let raw_height = original_image.height();
    let resize_ratio = (model_width as f32 / raw_width as f32)
        .min(model_height as f32 / raw_height as f32);
    let resized_width = ((raw_width as f32) * resize_ratio).round() as u32;
    let resized_height = ((raw_height as f32) * resize_ratio).round() as u32;
    let pad_w = (model_width as f32 - resized_width as f32) / 2.0;
    let pad_h = (model_height as f32 - resized_height as f32) / 2.0;
    let left = (pad_w - 0.1).round().max(0.0) as usize;
    let top = (pad_h - 0.1).round().max(0.0) as usize;

    let image = original_image.resize_exact(resized_width, resized_height, FilterType::CatmullRom);
    for (x, y, Rgba([r, g, b, _])) in image.pixels() {
        let x = left + x as usize;
        let y = top + y as usize;

        input[[0, 0, y, x]] = (r as f32) / 255.;
        input[[0, 1, y, x]] = (g as f32) / 255.;
        input[[0, 2, y, x]] = (b as f32) / 255.;
    }

    YoloInput {
        tensor: input,
        raw_width,
        raw_height,
        model_width,
        model_height,
        resize_ratio,
        pad_w,
        pad_h,
    }
}

/// Inference on the YOLO model, returning the detected entities.
///
/// The input tensor should be obtained from the [`image_to_yolo_input_tensor`] function.
/// The [`YoloModelSession`] can be obtained from the [`YoloModelSession::from_filename_v8`]
/// or [`YoloModelSession::from_filename_with_labels`] methods.
pub fn inference(
    model: &mut YoloModelSession,
    YoloInputView {
        tensor_view,
        raw_width,
        raw_height,
        model_width,
        model_height,
        resize_ratio,
        pad_w,
        pad_h,
    }: YoloInputView,
) -> Result<Vec<YoloEntityOutput>, YoloError> {
    // Due to the lifetime of the model, we need to clone the
    // labels and thresholds early.
    let iou_threshold = model.get_iou_threshold();
    let probability_threshold = model.get_probability_threshold();
    let labels = model.get_labels().to_vec();
    let input_name = model.get_input_name()?.to_owned();
    let output_name = model.get_output_name()?.to_owned();

    // Run YOLO-style inference using the session's configured input/output names.
    let inputs = inputs![input_name.as_str() => TensorRef::from_array_view(tensor_view).map_err(YoloError::OrtInputError)?];
    let outputs = model
        .as_mut()
        .run(inputs)
        .map_err(YoloError::OrtInferenceError)?;
    let output = outputs[output_name.as_str()]
        .try_extract_array::<f32>()
        .map_err(YoloError::OrtExtractTensorError)?;
    let output = output
        .view()
        .into_dimensionality::<ndarray::Ix3>()
        .map_err(|_| YoloError::InvalidOutputShape(output.shape().to_vec()))?;
    let letterbox = LetterboxInfo {
        raw_width,
        raw_height,
        model_width,
        model_height,
        resize_ratio,
        pad_w,
        pad_h,
    };
    let boxes = decode_detections(
        output,
        &labels,
        probability_threshold,
        letterbox,
        None,
    )?;

    Ok(non_maximum_suppression(boxes, iou_threshold)
        .into_iter()
        .map(|decoded| decoded.entity)
        .collect())
}

pub fn inference_with_prompts(
    model: &mut YoloModelSession,
    YoloInputView {
        tensor_view,
        raw_width,
        raw_height,
        model_width,
        model_height,
        resize_ratio,
        pad_w,
        pad_h,
    }: YoloInputView,
    prompts: YoloPromptEmbeddingsView<'_>,
) -> Result<Vec<YoloEntityOutput>, YoloError> {
    let prompts = validate_prompt_embeddings(prompts)?;
    let iou_threshold = model.get_iou_threshold();
    let probability_threshold = model.get_probability_threshold();
    let input_name = model.get_input_name()?.to_owned();
    let prompt_input_name = model.get_prompt_input_name()?.to_owned();
    let output_name = model.get_output_name()?.to_owned();

    let inputs = inputs![
        input_name.as_str() => TensorRef::from_array_view(tensor_view).map_err(YoloError::OrtInputError)?,
        prompt_input_name.as_str() => TensorRef::from_array_view(prompts.embeddings).map_err(YoloError::OrtInputError)?,
    ];
    let outputs = model
        .as_mut()
        .run(inputs)
        .map_err(YoloError::OrtInferenceError)?;
    let output = outputs[output_name.as_str()]
        .try_extract_array::<f32>()
        .map_err(YoloError::OrtExtractTensorError)?;
    let output = output
        .view()
        .into_dimensionality::<ndarray::Ix3>()
        .map_err(|_| YoloError::InvalidOutputShape(output.shape().to_vec()))?;
    let letterbox = LetterboxInfo {
        raw_width,
        raw_height,
        model_width,
        model_height,
        resize_ratio,
        pad_w,
        pad_h,
    };
    let boxes = decode_detections(
        output,
        prompts.labels,
        probability_threshold,
        letterbox,
        None,
    )?;

    Ok(non_maximum_suppression(boxes, iou_threshold)
        .into_iter()
        .map(|decoded| decoded.entity)
        .collect())
}

/// Inference on a segmentation-style YOLO model, returning the detected entities and decoded masks.
///
/// This is intended for YOLO segmentation exports, including exported closed-set YOLOE segmentation models.
pub fn inference_segment(
    model: &mut YoloModelSession,
    YoloInputView {
        tensor_view,
        raw_width,
        raw_height,
        model_width,
        model_height,
        resize_ratio,
        pad_w,
        pad_h,
    }: YoloInputView,
) -> Result<Vec<YoloSegmentationOutput>, YoloError> {
    let iou_threshold = model.get_iou_threshold();
    let probability_threshold = model.get_probability_threshold();
    let labels = model.get_labels().to_vec();
    let input_name = model.get_input_name()?.to_owned();
    let output_name = model.get_output_name()?.to_owned();
    let mask_output_name = model.get_mask_output_name()?.to_owned();

    let inputs = inputs![input_name.as_str() => TensorRef::from_array_view(tensor_view).map_err(YoloError::OrtInputError)?];
    let outputs = model
        .as_mut()
        .run(inputs)
        .map_err(YoloError::OrtInferenceError)?;

    let detection_output = outputs[output_name.as_str()]
        .try_extract_array::<f32>()
        .map_err(YoloError::OrtExtractTensorError)?;
    let detection_output = detection_output
        .view()
        .into_dimensionality::<ndarray::Ix3>()
        .map_err(|_| YoloError::InvalidOutputShape(detection_output.shape().to_vec()))?;

    let mask_output = outputs[mask_output_name.as_str()]
        .try_extract_array::<f32>()
        .map_err(YoloError::OrtExtractTensorError)?;
    let mask_output = mask_output
        .view()
        .into_dimensionality::<ndarray::Ix4>()
        .map_err(|_| YoloError::InvalidMaskShape(mask_output.shape().to_vec()))?;
    if mask_output.shape()[0] != 1 {
        return Err(YoloError::InvalidMaskShape(mask_output.shape().to_vec()));
    }

    let prototype_count = mask_output.shape()[1];
    let letterbox = LetterboxInfo {
        raw_width,
        raw_height,
        model_width,
        model_height,
        resize_ratio,
        pad_w,
        pad_h,
    };
    let boxes = decode_detections(
        detection_output,
        &labels,
        probability_threshold,
        letterbox,
        Some(prototype_count),
    )?;
    let boxes = non_maximum_suppression(boxes, iou_threshold);

    decode_segmentation_masks(
        boxes,
        mask_output.slice(ndarray::s![0, .., .., ..]),
        letterbox,
    )
}

pub fn inference_segment_with_prompts(
    model: &mut YoloModelSession,
    YoloInputView {
        tensor_view,
        raw_width,
        raw_height,
        model_width,
        model_height,
        resize_ratio,
        pad_w,
        pad_h,
    }: YoloInputView,
    prompts: YoloPromptEmbeddingsView<'_>,
) -> Result<Vec<YoloSegmentationOutput>, YoloError> {
    let prompts = validate_prompt_embeddings(prompts)?;
    let iou_threshold = model.get_iou_threshold();
    let probability_threshold = model.get_probability_threshold();
    let input_name = model.get_input_name()?.to_owned();
    let prompt_input_name = model.get_prompt_input_name()?.to_owned();
    let output_name = model.get_output_name()?.to_owned();
    let mask_output_name = model.get_mask_output_name()?.to_owned();

    let inputs = inputs![
        input_name.as_str() => TensorRef::from_array_view(tensor_view).map_err(YoloError::OrtInputError)?,
        prompt_input_name.as_str() => TensorRef::from_array_view(prompts.embeddings).map_err(YoloError::OrtInputError)?,
    ];
    let outputs = model
        .as_mut()
        .run(inputs)
        .map_err(YoloError::OrtInferenceError)?;

    let detection_output = outputs[output_name.as_str()]
        .try_extract_array::<f32>()
        .map_err(YoloError::OrtExtractTensorError)?;
    let detection_output = detection_output
        .view()
        .into_dimensionality::<ndarray::Ix3>()
        .map_err(|_| YoloError::InvalidOutputShape(detection_output.shape().to_vec()))?;

    let mask_output = outputs[mask_output_name.as_str()]
        .try_extract_array::<f32>()
        .map_err(YoloError::OrtExtractTensorError)?;
    let mask_output = mask_output
        .view()
        .into_dimensionality::<ndarray::Ix4>()
        .map_err(|_| YoloError::InvalidMaskShape(mask_output.shape().to_vec()))?;
    if mask_output.shape()[0] != 1 {
        return Err(YoloError::InvalidMaskShape(mask_output.shape().to_vec()));
    }

    let letterbox = LetterboxInfo {
        raw_width,
        raw_height,
        model_width,
        model_height,
        resize_ratio,
        pad_w,
        pad_h,
    };
    let prototype_count = mask_output.shape()[1];
    let boxes = decode_detections(
        detection_output,
        prompts.labels,
        probability_threshold,
        letterbox,
        Some(prototype_count),
    )?;
    let boxes = non_maximum_suppression(boxes, iou_threshold);

    decode_segmentation_masks(
        boxes,
        mask_output.slice(ndarray::s![0, .., .., ..]),
        letterbox,
    )
}
