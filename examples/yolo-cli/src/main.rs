use std::fs;
use std::path::PathBuf;

use anyhow::{Context, Result};
use clap::Parser;
use ort::execution_providers::{CUDAExecutionProvider, CoreMLExecutionProvider};
use raqote::{DrawOptions, DrawTarget, LineJoin, PathBuilder, SolidSource, Source, StrokeStyle};
use show_image::{AsImageView, WindowOptions, event};
use yolo_rs::{
    YoloEntityOutput, YoloSegmentationOutput, image_to_yolo_input_tensor, inference,
    inference_segment, inference_segment_with_prompts, inference_with_prompts, model,
    prompt::YoloPromptEncoderSession,
};

#[derive(Parser, Debug)]
#[command(version, about, long_about = None)]
struct Args {
    model_path: PathBuf,
    picture_path: PathBuf,

    /// Override the class labels used to decode the model output.
    ///
    /// Pass this multiple times for exported YOLOE closed-set models, for example:
    /// --label person --label bus
    #[arg(long = "label")]
    labels: Vec<String>,

    /// Runtime open-vocabulary prompts for a promptable YOLOE ONNX model.
    #[arg(long = "prompt")]
    prompts: Vec<String>,

    /// Path to a prompt-encoder ONNX model that converts tokenized text to prompt embeddings.
    #[arg(long)]
    prompt_encoder_model: Option<PathBuf>,

    /// Path to a tokenizer JSON compatible with the prompt encoder.
    #[arg(long)]
    prompt_tokenizer: Option<PathBuf>,

    /// Read class labels from a UTF-8 text file, one label per line.
    #[arg(long)]
    labels_file: Option<PathBuf>,

    #[arg(long)]
    probability_threshold: Option<f32>,

    #[arg(long)]
    iou_threshold: Option<f32>,

    /// Skip opening the GUI window and only print detections.
    #[arg(long)]
    no_display: bool,
}

fn read_labels(args: &Args) -> Result<Option<Vec<String>>> {
    let mut labels = if let Some(path) = &args.labels_file {
        fs::read_to_string(path)
            .with_context(|| format!("failed to read labels file {:?}", path.display()))?
            .lines()
            .map(str::trim)
            .filter(|line| !line.is_empty())
            .map(ToOwned::to_owned)
            .collect::<Vec<_>>()
    } else {
        Vec::new()
    };

    labels.extend(args.labels.iter().cloned());

    if labels.is_empty() {
        Ok(None)
    } else {
        Ok(Some(labels))
    }
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
        .commit()
        .then_some(())
        .context("failed to commit ONNX Runtime")?;

    tracing::info!("Loading image {:?}…", args.picture_path.display());
    let original_img = image::open(&args.picture_path)
        .with_context(|| format!("failed to open image {:?}", args.picture_path.display()))?;

    tracing::info!("Loading models {:?}…", args.model_path.display());
    let labels = read_labels(&args)?;
    let prompt_mode = !args.prompts.is_empty();
    let mut model = {
        let mut model = if prompt_mode {
            model::YoloModelSession::from_filename(&args.model_path)
        } else if let Some(labels) = labels {
            model::YoloModelSession::from_filename_with_labels(
                &args.model_path,
                labels.into_iter(),
            )
        } else {
            model::YoloModelSession::from_filename_v8(&args.model_path)
        }
        .with_context(|| format!("failed to load model {:?}", args.model_path.display()))?;

        model.iou_threshold = args.iou_threshold;
        model.probability_threshold = args.probability_threshold;

        model
    };

    tracing::debug!("Converting image to tensor…");
    let input = image_to_yolo_input_tensor(&original_img);

    let prompt_embeddings = if prompt_mode {
        let prompt_encoder_model = args
            .prompt_encoder_model
            .as_ref()
            .context("--prompt-encoder-model is required when using --prompt")?;
        let prompt_tokenizer = args
            .prompt_tokenizer
            .as_ref()
            .context("--prompt-tokenizer is required when using --prompt")?;
        tracing::info!("Encoding runtime prompts…");
        let mut prompt_encoder = YoloPromptEncoderSession::from_files(
            prompt_encoder_model,
            prompt_tokenizer,
        )
        .with_context(|| {
            format!(
                "failed to load prompt encoder {:?} and tokenizer {:?}",
                prompt_encoder_model.display(),
                prompt_tokenizer.display()
            )
        })?;
        Some(prompt_encoder.encode(args.prompts.iter().map(String::as_str))?)
    } else {
        None
    };

    // Run YOLOv8 inference
    tracing::info!("Running inference…");

    let now = std::time::Instant::now();
    let has_mask_output = model.session.outputs().len() > 1;
    let result = if has_mask_output {
        let (img_width, img_height) = (original_img.width(), original_img.height());
        let mut dt = DrawTarget::new(img_width as _, img_height as _);
        let result = if let Some(prompt_embeddings) = prompt_embeddings.as_ref() {
            inference_segment_with_prompts(&mut model, input.view(), prompt_embeddings.view())?
        } else {
            inference_segment(&mut model, input.view())?
        };
        let data = dt.get_data_mut();
        for YoloSegmentationOutput { entity, mask } in &result {
            let fill = match entity.label.as_str() {
                "baseball bat" => 0x40001080u32,
                "baseball glove" => 0x40208040u32,
                _ => 0x40801040u32,
            };

            for (x, y, pixel) in mask.enumerate_pixels() {
                if pixel.0[0] > 0 {
                    let index = (y as usize) * (img_width as usize) + (x as usize);
                    data[index] = fill;
                }
            }
        }

        (result.into_iter().map(|output| output.entity).collect::<Vec<_>>(), dt)
    } else {
        let (img_width, img_height) = (original_img.width(), original_img.height());
        let dt = DrawTarget::new(img_width as _, img_height as _);
        (
            if let Some(prompt_embeddings) = prompt_embeddings.as_ref() {
                inference_with_prompts(&mut model, input.view(), prompt_embeddings.view())?
            } else {
                inference(&mut model, input.view())?
            },
            dt,
        )
    };

    tracing::info!("Inference took {:?}", now.elapsed());

    tracing::debug!("Drawing bounding boxes…");
    let (img_width, img_height) = (original_img.width(), original_img.height());
    let (result, mut dt) = result;

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

    if args.no_display {
        return Ok(());
    }

    let overlay: show_image::Image = dt.into();

    tracing::info!("Displaying image…");
    let window = show_image::context()
        .run_function_wait(move |context| -> Result<_, String> {
            let mut window = context
                .create_window(
                    "ort + YOLOv8",
                    WindowOptions {
                        size: Some([img_width, img_height]),
                        ..WindowOptions::default()
                    },
                )
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
        if let event::WindowEvent::KeyboardInput(event) = event
            && event.input.key_code == Some(event::VirtualKeyCode::Escape)
            && event.input.state.is_pressed()
        {
            break;
        }
    }

    Ok(())
}
