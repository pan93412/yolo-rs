//! Detects objects in a picture with a YOLO model and saves the picture with a box around each.

use std::path::PathBuf;
use std::time::Instant;

use anyhow::{Context, Result};
use clap::Parser;
use image::{Rgba, RgbaImage};
use yolo_rs::{BoundingBox, YoloEntityOutput, image_to_yolo_input_tensor, inference, model};

/// The box outline's width, in pixels.
const LINE_WIDTH: u32 = 4;

#[derive(Parser, Debug)]
#[command(version, about, long_about = None)]
struct Args {
    /// The YOLO model, exported to ONNX.
    model_path: PathBuf,
    /// The picture to detect objects in.
    picture_path: PathBuf,

    /// Where to save the picture with the boxes drawn on it.
    #[arg(long, short, default_value = "detected.png")]
    output: PathBuf,

    /// The lowest class probability a detection needs (0.5 by default).
    #[arg(long)]
    probability_threshold: Option<f32>,

    /// The overlap above which non-maximum suppression drops the weaker box (0.7 by default).
    #[arg(long)]
    iou_threshold: Option<f32>,
}

fn main() -> Result<()> {
    tracing_subscriber::fmt::init();

    let args = Args::parse();

    tracing::debug!("Initializing ONNX runtime…");
    // ONNX Runtime falls back to the CPU; ort's `cuda` feature adds `ort::ep::CUDA`.
    #[cfg(target_os = "macos")]
    let builder = ort::init().with_execution_providers([ort::ep::CoreML::default().build()]);
    #[cfg(not(target_os = "macos"))]
    let builder = ort::init();
    let environment = builder
        .build()
        .context("failed to initialize ONNX Runtime")?;

    tracing::info!("Loading image {}…", args.picture_path.display());
    let original_img = image::open(&args.picture_path)
        .with_context(|| format!("failed to open image {}", args.picture_path.display()))?;

    tracing::info!("Loading model {}…", args.model_path.display());
    let mut model = model::YoloModelSession::from_filename_v8(&environment, &args.model_path)
        .with_context(|| format!("failed to load model {}", args.model_path.display()))?;
    model.iou_threshold = args.iou_threshold;
    model.probability_threshold = args.probability_threshold;

    tracing::debug!("Converting image to tensor…");
    let input = image_to_yolo_input_tensor(&original_img);

    tracing::info!("Running inference…");
    let now = Instant::now();
    let entities = inference(&mut model, input.view())?;
    tracing::info!("Inference took {:?}", now.elapsed());

    let mut picture = original_img.to_rgba8();
    for YoloEntityOutput {
        bounding_box,
        label,
        confidence,
    } in &entities
    {
        tracing::info!(
            "Found entity {label:?} with confidence {confidence:.2} at ({:.2}, {:.2}) - ({:.2}, {:.2})",
            bounding_box.x1,
            bounding_box.y1,
            bounding_box.x2,
            bounding_box.y2
        );
        draw_box(&mut picture, bounding_box, color_of(label));
    }

    // Through `DynamicImage`, so formats without alpha (JPEG) get a picture they can encode.
    image::DynamicImage::ImageRgba8(picture)
        .save(&args.output)
        .with_context(|| format!("failed to save {}", args.output.display()))?;
    tracing::info!("Saved {}", args.output.display());

    Ok(())
}

/// The outline colour for an entity's label.
fn color_of(label: &str) -> Rgba<u8> {
    match label {
        "baseball bat" => Rgba([0x00, 0x20, 0xFF, 0xFF]),
        "baseball glove" => Rgba([0x40, 0xFF, 0x80, 0xFF]),
        _ => Rgba([0xFF, 0x20, 0x80, 0xFF]),
    }
}

/// Draws the outline of `bounding_box` on `picture`, clipped to the picture's edges.
fn draw_box(picture: &mut RgbaImage, bounding_box: &BoundingBox, color: Rgba<u8>) {
    let (width, height) = picture.dimensions();
    if width == 0 || height == 0 {
        return;
    }

    #[expect(
        clippy::cast_possible_truncation,
        clippy::cast_sign_loss,
        clippy::cast_precision_loss,
        reason = "each coordinate is clamped to the picture's pixels first"
    )]
    let clamp = |value: f32, max: u32| value.round().clamp(0., (max - 1) as f32) as u32;
    let (x1, x2) = (clamp(bounding_box.x1, width), clamp(bounding_box.x2, width));
    let (y1, y2) = (
        clamp(bounding_box.y1, height),
        clamp(bounding_box.y2, height),
    );

    for y in y1..=y2 {
        for x in x1..=x2 {
            let on_edge = x < x1 + LINE_WIDTH
                || x + LINE_WIDTH > x2
                || y < y1 + LINE_WIDTH
                || y + LINE_WIDTH > y2;
            if on_edge {
                picture.put_pixel(x, y, color);
            }
        }
    }
}
