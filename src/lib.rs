//! A Rust library for the YOLO v11 object detection model.
//!
//! This library provides a high-level API for running the YOLO v11 object detection model.
//! Currently, it supports only the inference.

pub mod error;
pub mod model;

use arcstr::ArcStr;
use error::YoloError;
use image::{DynamicImage, GenericImageView, Rgba, imageops::FilterType};
use model::YoloModelSession;
use ndarray::{Array4, ArrayBase, ArrayView4, Axis, s};
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
    pub fn view(&self) -> YoloInputView {
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
/// The [`YoloModelSession`] can be obtained from the [`YoloModelSession::from_filename_v8`] method.
pub fn inference(
    model: &mut YoloModelSession,
    YoloInputView {
        tensor_view,
        raw_width,
        raw_height,
    }: YoloInputView,
) -> Result<Vec<YoloEntityOutput>, YoloError> {
    fn intersection(box1: &BoundingBox, box2: &BoundingBox) -> f32 {
        (box1.x2.min(box2.x2) - box1.x1.max(box2.x1))
            * (box1.y2.min(box2.y2) - box1.y1.max(box2.y1))
    }

    fn union(box1: &BoundingBox, box2: &BoundingBox) -> f32 {
        ((box1.x2 - box1.x1) * (box1.y2 - box1.y1)) + ((box2.x2 - box2.x1) * (box2.y2 - box2.y1))
            - intersection(box1, box2)
    }

    fn non_maximum_suppression(
        mut boxes: Vec<YoloEntityOutput>,
        iou_threshold: f32,
    ) -> Vec<YoloEntityOutput> {
        // Early return if no boxes are provided
        if boxes.is_empty() {
            return Vec::new();
        }

        // Sort boxes by confidence descending using sort_unstable_by for better performance
        boxes.sort_unstable_by(|a, b| b.confidence.total_cmp(&a.confidence));

        let mut result = Vec::with_capacity(boxes.len());

        // Iterate through each box and select it if it doesn't overlap significantly with already selected boxes
        for current in boxes.into_iter() {
            // Check if the current box has a high IoU with any box in the result
            // Using `iter().all()` ensures we short-circuit on the first overlap found
            if result.iter().all(|selected: &YoloEntityOutput| {
                let iou = intersection(&selected.bounding_box, &current.bounding_box)
                    / union(&selected.bounding_box, &current.bounding_box);
                iou < iou_threshold
            }) {
                result.push(current);
            }
        }

        result.shrink_to_fit();

        result
    }

    // Due to the lifetime of the model, we need to clone the
    // labels and thresholds early.
    let iou_threshold = model.get_iou_threshold();
    let probability_threshold = model.get_probability_threshold();
    let labels = model.get_labels().to_vec();

    // Run YOLOv8 inference
    let inputs = inputs!["images" => TensorRef::from_array_view(tensor_view).map_err(YoloError::OrtInputError)?];
    let outputs = model
        .as_mut()
        .run(inputs)
        .map_err(YoloError::OrtInferenceError)?;
    let output = outputs["output0"]
        .try_extract_array::<f32>()
        .map_err(YoloError::OrtExtractSensorError)?
        .reversed_axes();
    let output = output.slice(s![.., .., 0]);

    // Turn the output tensor into bounding boxes
    let boxes = output
        .axis_iter(Axis(0))
        .filter_map(|row| {
            let (class_id, prob) = row
                .iter()
                .skip(4) // skip bounding box coordinates
                .enumerate()
                .map(|(index, value)| (index, *value))
                .reduce(|accum, row| if row.1 > accum.1 { row } else { accum })
                .filter(|(_, prob)| *prob >= probability_threshold)?;

            let label = labels[class_id].clone();

            let xc = row[0_usize] / 640. * (raw_width as f32);
            let yc = row[1_usize] / 640. * (raw_height as f32);
            let w = row[2_usize] / 640. * (raw_width as f32);
            let h = row[3_usize] / 640. * (raw_height as f32);

            Some(YoloEntityOutput {
                bounding_box: BoundingBox {
                    x1: xc - w / 2.,
                    y1: yc - h / 2.,
                    x2: xc + w / 2.,
                    y2: yc + h / 2.,
                },
                label,
                confidence: prob,
            })
        })
        .collect::<Vec<YoloEntityOutput>>();

    // Perform non-maximum suppression (NMS)
    Ok(non_maximum_suppression(boxes, iou_threshold))
}
