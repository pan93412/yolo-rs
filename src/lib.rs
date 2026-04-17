//! A Rust library for the YOLO v11 object detection model.
//!
//! This library provides a high-level API for running the YOLO v11 object detection model.
//! Currently, it supports only the inference.

pub mod error;
pub mod model;

use arcstr::ArcStr;
use error::YoloError;
use image::{DynamicImage, GenericImageView, GrayImage, Luma, Rgba, imageops::FilterType};
use model::YoloModelSession;
use ndarray::{Array4, ArrayBase, ArrayView3, ArrayView4};
use ort::{inputs, value::TensorRef};

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
}

impl YoloInput {
    pub fn view(&self) -> YoloInputView<'_> {
        YoloInputView {
            tensor_view: self.tensor.view(),
            raw_width: self.raw_width,
            raw_height: self.raw_height,
        }
    }
}

#[derive(Debug, Clone, Copy)]
pub struct YoloInputView<'a> {
    pub tensor_view: ArrayView4<'a, f32>,
    pub raw_width: u32,
    pub raw_height: u32,
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

#[derive(Debug, Clone)]
struct DecodedDetection {
    entity: YoloEntityOutput,
    mask_coefficients: Vec<f32>,
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
    raw_width: u32,
    raw_height: u32,
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

            let xc = row[0_usize] / 640. * (raw_width as f32);
            let yc = row[1_usize] / 640. * (raw_height as f32);
            let w = row[2_usize] / 640. * (raw_width as f32);
            let h = row[3_usize] / 640. * (raw_height as f32);

            Some(Ok(DecodedDetection {
                entity: YoloEntityOutput {
                    bounding_box: BoundingBox {
                        x1: xc - w / 2.,
                        y1: yc - h / 2.,
                        x2: xc + w / 2.,
                        y2: yc + h / 2.,
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
    raw_width: u32,
    raw_height: u32,
) -> Result<Vec<YoloSegmentationOutput>, YoloError> {
    let prototype_count = prototypes.shape()[0];
    let mask_height = prototypes.shape()[1];
    let mask_width = prototypes.shape()[2];

    let mut results = Vec::with_capacity(detections.len());
    for detection in detections {
        let mut mask = GrayImage::new(raw_width, raw_height);
        let bbox = detection.entity.bounding_box;

        let start_x = bbox.x1.floor().max(0.0) as u32;
        let end_x = bbox.x2.ceil().min(raw_width as f32) as u32;
        let start_y = bbox.y1.floor().max(0.0) as u32;
        let end_y = bbox.y2.ceil().min(raw_height as f32) as u32;

        for y in start_y..end_y {
            let proto_y = ((y as usize) * mask_height) / (raw_height as usize);
            for x in start_x..end_x {
                let proto_x = ((x as usize) * mask_width) / (raw_width as usize);

                let value = detection
                    .mask_coefficients
                    .iter()
                    .take(prototype_count)
                    .enumerate()
                    .map(|(index, coefficient)| coefficient * prototypes[[index, proto_y, proto_x]])
                    .sum::<f32>();

                if value > 0.0 {
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

/// Convert an image to a YOLO input tensor.
///
/// The input image is resized to 640x640 and normalized to the range [0, 1].
/// The tensor has the shape (1, 3, 640, 640) and the layout is (R, G, B).
///
/// You can pass the resulting tensor to the [`inference`] function.
/// Note that you might need to call [`YoloInput::view`] to get a view of the tensor.
pub fn image_to_yolo_input_tensor(original_image: &DynamicImage) -> YoloInput {
    let mut input = ArrayBase::zeros((1, 3, 640, 640));

    let image = original_image.resize_exact(640, 640, FilterType::CatmullRom);
    for (x, y, Rgba([r, g, b, _])) in image.pixels() {
        let x = x as usize;
        let y = y as usize;

        input[[0, 0, y, x]] = (r as f32) / 255.;
        input[[0, 1, y, x]] = (g as f32) / 255.;
        input[[0, 2, y, x]] = (b as f32) / 255.;
    }

    YoloInput {
        tensor: input,
        raw_width: original_image.width(),
        raw_height: original_image.height(),
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
    let boxes = decode_detections(
        output,
        &labels,
        probability_threshold,
        raw_width,
        raw_height,
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
    let boxes = decode_detections(
        detection_output,
        &labels,
        probability_threshold,
        raw_width,
        raw_height,
        Some(prototype_count),
    )?;
    let boxes = non_maximum_suppression(boxes, iou_threshold);

    decode_segmentation_masks(
        boxes,
        mask_output.slice(ndarray::s![0, .., .., ..]),
        raw_width,
        raw_height,
    )
}
