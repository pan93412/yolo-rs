//! YOLO object detection in Rust with ONNX Runtime.
//!
//! Runs ONNX exports of Ultralytics YOLOv8, YOLO11, YOLO12 and YOLO26 (exported with
//! `end2end=False`) and returns each detected object's bounding box, label and confidence.
//! Currently, it supports only the inference.

pub mod error;
pub mod model;

use arcstr::ArcStr;
use error::YoloError;
use image::DynamicImage;
use image::imageops::FilterType;
use model::YoloModelSession;
use ndarray::{Array4, ArrayView4, Axis, s};
use ort::inputs;
use ort::value::TensorRef;

/// A box in the coordinates of the original image, in pixels.
#[derive(Debug, Clone, Copy)]
pub struct BoundingBox {
    /// The left edge.
    pub x1: f32,
    /// The top edge.
    pub y1: f32,
    /// The right edge.
    pub x2: f32,
    /// The bottom edge.
    pub y2: f32,
}

/// An image prepared for [`inference`], made by [`image_to_yolo_input_tensor`].
#[derive(Debug, Clone)]
pub struct YoloInput {
    /// The image as a (1, 3, 640, 640) RGB tensor in the range [0, 1].
    pub tensor: Array4<f32>,
    /// The width of the original image, to scale the boxes back.
    pub raw_width: u32,
    /// The height of the original image, to scale the boxes back.
    pub raw_height: u32,
}

impl YoloInput {
    /// Borrow the input as the [`YoloInputView`] that [`inference`] takes.
    #[must_use]
    pub fn view(&self) -> YoloInputView<'_> {
        YoloInputView {
            tensor_view: self.tensor.view(),
            raw_width: self.raw_width,
            raw_height: self.raw_height,
        }
    }
}

/// A borrowed [`YoloInput`].
#[derive(Debug, Clone, Copy)]
pub struct YoloInputView<'a> {
    /// The image as a (1, 3, 640, 640) RGB tensor in the range [0, 1].
    pub tensor_view: ArrayView4<'a, f32>,
    /// The width of the original image.
    pub raw_width: u32,
    /// The height of the original image.
    pub raw_height: u32,
}

/// An object [`inference`] detected.
#[derive(Debug, Clone)]
pub struct YoloEntityOutput {
    /// Where the object is in the original image.
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
#[must_use]
pub fn image_to_yolo_input_tensor(original_image: &DynamicImage) -> YoloInput {
    let image = original_image
        .resize_exact(640, 640, FilterType::CatmullRom)
        .to_rgb8();
    // The pixels are laid out (y, x, channel); fill a tensor in that order, then move the channel
    // axis in front of the rows as the model expects.
    let mut input = Array4::<f32>::zeros((1, 640, 640, 3));
    for (value, &channel) in input.iter_mut().zip(image.as_raw()) {
        *value = f32::from(channel) / 255.;
    }
    let input = input
        .permuted_axes([0, 3, 1, 2])
        .as_standard_layout()
        .into_owned();

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
///
/// # Errors
///
/// Fails when ONNX Runtime cannot take the input or run the model, or when the model has no
/// `output0` output to read.
pub fn inference(
    model: &mut YoloModelSession,
    YoloInputView {
        tensor_view,
        raw_width,
        raw_height,
    }: YoloInputView<'_>,
) -> Result<Vec<YoloEntityOutput>, YoloError> {
    fn intersection(box1: &BoundingBox, box2: &BoundingBox) -> f32 {
        (box1.x2.min(box2.x2) - box1.x1.max(box2.x1))
            * (box1.y2.min(box2.y2) - box1.y1.max(box2.y1))
    }

    fn area(bounding_box: &BoundingBox) -> f32 {
        (bounding_box.x2 - bounding_box.x1) * (bounding_box.y2 - bounding_box.y1)
    }

    fn union(box1: &BoundingBox, box2: &BoundingBox) -> f32 {
        area(box1) + area(box2) - intersection(box1, box2)
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

        // Iterate through each box and select it if it doesn't overlap significantly with already
        // selected boxes
        for current in boxes {
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
    let output = outputs
        .get("output0")
        .ok_or_else(|| {
            YoloError::OrtExtractSensorError(ort::Error::new("the model has no `output0` output"))
        })?
        .try_extract_array::<f32>()
        .map_err(YoloError::OrtExtractSensorError)?
        .reversed_axes();
    let output = output.slice(s![.., .., 0]);

    // Turn the output tensor into bounding boxes
    let boxes = output
        .axis_iter(Axis(0))
        .filter_map(|row| {
            // The box (centre x, centre y, width, height), then one probability per class.
            let mut values = row.iter().copied();
            let [xc, yc, w, h] = [
                values.next()?,
                values.next()?,
                values.next()?,
                values.next()?,
            ];
            let (class_id, prob) = values
                .enumerate()
                .reduce(|accum, row| if row.1 > accum.1 { row } else { accum })
                .filter(|(_, prob)| *prob >= probability_threshold)?;

            // A class the labels don't cover is dropped rather than panicking.
            let label = labels.get(class_id)?.clone();

            #[expect(
                clippy::cast_precision_loss,
                reason = "image sides are far below f32's exact integer range (2^24)"
            )]
            let (width_scale, height_scale) = (raw_width as f32 / 640., raw_height as f32 / 640.);
            let (xc, w) = (xc * width_scale, w * width_scale);
            let (yc, h) = (yc * height_scale, h * height_scale);

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
