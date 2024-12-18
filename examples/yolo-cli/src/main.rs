use std::path::PathBuf;

use anyhow::{Context, Result};
use clap::Parser;
use ort::execution_providers::{CUDAExecutionProvider, CoreMLExecutionProvider};
use raqote::{DrawOptions, DrawTarget, LineJoin, PathBuilder, SolidSource, Source, StrokeStyle};
use show_image::{AsImageView, WindowOptions, event};
use yolo_rs::{YoloEntityOutput, image_to_yolo_input_tensor, inference, model};

#[derive(Parser, Debug)]
#[command(version, about, long_about = None)]
struct Args {
    model_path: PathBuf,
    picture_path: PathBuf,

    #[arg(long)]
    probability_threshold: Option<f32>,

    #[arg(long)]
    iou_threshold: Option<f32>,
}

#[show_image::main]
fn main() -> Result<()> {
    tracing_subscriber::fmt::init();

    let args = Args::parse();

    tracing::debug!("Initializing ONNX runtime…");
    ort::init()
        .with_execution_providers([
            CUDAExecutionProvider::default().build(),
            CoreMLExecutionProvider::default().build(),
        ])
        .commit()?;

    tracing::info!("Loading image {:?}…", args.picture_path.display());
    let original_img = image::open(&args.picture_path)
        .with_context(|| format!("failed to open image {:?}", args.picture_path.display()))?;

    tracing::info!("Loading models {:?}…", args.model_path.display());
    let model = {
        let mut model = model::YoloModelSession::from_filename_v8(&args.model_path)
            .with_context(|| format!("failed to load model {:?}", args.model_path.display()))?;

        model.iou_threshold = args.iou_threshold;
        model.probability_threshold = args.probability_threshold;

        model
    };

    tracing::debug!("Converting image to tensor…");
    let input = image_to_yolo_input_tensor(&original_img);

    // Run YOLOv8 inference
    tracing::info!("Running inference…");

    let now = std::time::Instant::now();
    let result = inference(&model, input.view())?;
    tracing::info!("Inference took {:?}", now.elapsed());

    tracing::debug!("Drawing bounding boxes…");
    let (img_width, img_height) = (original_img.width(), original_img.height());

    let mut dt = DrawTarget::new(img_width as _, img_height as _);

    for YoloEntityOutput {
        bounding_box: bbox,
        label,
        confidence,
    } in result
    {
        tracing::info!(
            "Found entity {:?} with confidence {:.2} at ({:.2}, {:.2}) - ({:.2}, {:.2})",
            label,
            confidence,
            bbox.x1,
            bbox.y1,
            bbox.x2,
            bbox.y2
        );

        let mut pb = PathBuilder::new();
        pb.rect(bbox.x1, bbox.y1, bbox.x2 - bbox.x1, bbox.y2 - bbox.y1);
        let path = pb.finish();
        let color = match label.as_str() {
            "baseball bat" => SolidSource {
                r: 0x00,
                g: 0x10,
                b: 0x80,
                a: 0x80,
            },
            "baseball glove" => SolidSource {
                r: 0x20,
                g: 0x80,
                b: 0x40,
                a: 0x80,
            },
            _ => SolidSource {
                r: 0x80,
                g: 0x10,
                b: 0x40,
                a: 0x80,
            },
        };
        dt.stroke(
            &path,
            &Source::Solid(color),
            &StrokeStyle {
                join: LineJoin::Round,
                width: 4.,
                ..StrokeStyle::default()
            },
            &DrawOptions::new(),
        );
    }

    let overlay: show_image::Image = dt.into();

    tracing::info!("Displaying image…");
    let window = show_image::context()
        .run_function_wait(move |context| -> Result<_, String> {
            let mut window = context
                .create_window("ort + YOLOv8", WindowOptions {
                    size: Some([img_width, img_height]),
                    ..WindowOptions::default()
                })
                .map_err(|e| e.to_string())?;
            window.set_image(
                "baseball",
                &original_img.as_image_view().map_err(|e| e.to_string())?,
            );
            window.set_overlay(
                "yolo",
                &overlay.as_image_view().map_err(|e| e.to_string())?,
                true,
            );
            Ok(window.proxy())
        })
        .map_err(|e| anyhow::anyhow!(e))
        .context("failed to create window")?;

    for event in window
        .event_channel()
        .context("failed to get event channel")?
    {
        if let event::WindowEvent::KeyboardInput(event) = event {
            if event.input.key_code == Some(event::VirtualKeyCode::Escape)
                && event.input.state.is_pressed()
            {
                break;
            }
        }
    }

    Ok(())
}
