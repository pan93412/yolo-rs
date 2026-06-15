use std::path::Path;

use arcstr::ArcStr;
use ndarray::{Array2, Array3, ArrayView3, Axis};
use ort::{inputs, value::TensorRef};
use tokenizers::{PaddingParams, PaddingStrategy, Tokenizer, TruncationParams};

use crate::error::YoloError;

#[derive(Debug, Clone)]
pub struct YoloPromptEmbeddings {
    labels: Vec<ArcStr>,
    embeddings: Array3<f32>,
}

impl YoloPromptEmbeddings {
    pub fn try_new(
        labels: impl IntoIterator<Item = impl Into<ArcStr>>,
        embeddings: Array3<f32>,
    ) -> Result<Self, YoloError> {
        if embeddings.shape().len() != 3 || embeddings.shape()[0] != 1 {
            return Err(YoloError::InvalidPromptEmbeddingShape(embeddings.shape().to_vec()));
        }

        let labels = labels.into_iter().map(Into::into).collect::<Vec<_>>();
        if labels.is_empty() {
            return Err(YoloError::EmptyPromptList);
        }
        if labels.len() != embeddings.shape()[1] {
            return Err(YoloError::PromptLabelEmbeddingMismatch {
                labels: labels.len(),
                embeddings: embeddings.shape()[1],
            });
        }

        Ok(Self { labels, embeddings })
    }

    pub fn labels(&self) -> &[ArcStr] {
        &self.labels
    }

    pub fn embeddings(&self) -> &Array3<f32> {
        &self.embeddings
    }

    pub fn view(&self) -> YoloPromptEmbeddingsView<'_> {
        YoloPromptEmbeddingsView {
            labels: &self.labels,
            embeddings: self.embeddings.view(),
        }
    }
}

#[derive(Debug, Clone, Copy)]
pub struct YoloPromptEmbeddingsView<'a> {
    pub labels: &'a [ArcStr],
    pub embeddings: ArrayView3<'a, f32>,
}

#[derive(Debug)]
pub struct YoloPromptEncoderSession {
    session: ort::session::Session,
    tokenizer: Tokenizer,
    max_length: Option<usize>,
}

impl YoloPromptEncoderSession {
    pub fn from_files(
        model_path: impl AsRef<Path>,
        tokenizer_path: impl AsRef<Path>,
    ) -> Result<Self, YoloError> {
        let session = ort::session::Session::builder()
            .map_err(YoloError::OrtSessionBuildError)?
            .commit_from_file(model_path)
            .map_err(YoloError::OrtSessionLoadError)?;

        let tokenizer = Tokenizer::from_file(tokenizer_path.as_ref())
            .map_err(|error| YoloError::TokenizerLoadError(error.to_string()))?;

        Ok(Self {
            session,
            tokenizer,
            max_length: None,
        })
    }

    pub fn encode<'a>(
        &mut self,
        prompts: impl IntoIterator<Item = &'a str>,
    ) -> Result<YoloPromptEmbeddings, YoloError> {
        let labels = prompts
            .into_iter()
            .map(ArcStr::from)
            .collect::<Vec<_>>();
        if labels.is_empty() {
            return Err(YoloError::EmptyPromptList);
        }

        let prompt_texts = labels.iter().map(ToString::to_string).collect::<Vec<_>>();
        let mut tokenizer = self.tokenizer.clone();
        if let Some(max_length) = self.max_length {
            tokenizer
                .with_truncation(Some(TruncationParams {
                    max_length,
                    ..Default::default()
                }))
                .map_err(|error| YoloError::PromptTokenizationError(error.to_string()))?;
            tokenizer.with_padding(Some(PaddingParams {
                strategy: PaddingStrategy::Fixed(max_length),
                ..Default::default()
            }));
        } else {
            tokenizer.with_padding(Some(PaddingParams {
                strategy: PaddingStrategy::BatchLongest,
                ..Default::default()
            }));
        }

        let encodings = tokenizer
            .encode_batch(prompt_texts, true)
            .map_err(|error| YoloError::PromptTokenizationError(error.to_string()))?;
        let batch_size = encodings.len();
        let sequence_length = encodings.first().map(|encoding| encoding.len()).unwrap_or(0);

        let input_ids = Array2::from_shape_vec(
            (batch_size, sequence_length),
            encodings
                .iter()
                .flat_map(|encoding| encoding.get_ids().iter().map(|token| i64::from(*token)))
                .collect(),
        )
        .map_err(|_| YoloError::PromptTokenizationError("failed to build input_ids tensor".into()))?;
        let attention_mask = Array2::from_shape_vec(
            (batch_size, sequence_length),
            encodings
                .iter()
                .flat_map(|encoding| encoding.get_attention_mask().iter().map(|token| i64::from(*token)))
                .collect(),
        )
        .map_err(|_| YoloError::PromptTokenizationError("failed to build attention_mask tensor".into()))?;
        let token_type_ids = Array2::<i64>::zeros((batch_size, sequence_length));

        let output_name = self
            .session
            .outputs()
            .first()
            .map(|output| output.name().to_owned())
            .ok_or(YoloError::MissingModelOutput)?;
        let outputs = match self.session.inputs().len() {
            1 => {
                let input_name = self.session.inputs()[0].name().to_owned();
                let inputs = inputs![input_name.as_str() => TensorRef::from_array_view(input_ids.view()).map_err(YoloError::OrtInputError)?];
                self.session.run(inputs).map_err(YoloError::OrtInferenceError)?
            }
            2 => {
                let input_name = self.session.inputs()[0].name().to_owned();
                let mask_name = self.session.inputs()[1].name().to_owned();
                let inputs = inputs![
                    input_name.as_str() => TensorRef::from_array_view(input_ids.view()).map_err(YoloError::OrtInputError)?,
                    mask_name.as_str() => TensorRef::from_array_view(attention_mask.view()).map_err(YoloError::OrtInputError)?,
                ];
                self.session.run(inputs).map_err(YoloError::OrtInferenceError)?
            }
            3 => {
                let first_name = self.session.inputs()[0].name().to_owned();
                let second_name = self.session.inputs()[1].name().to_owned();
                let third_name = self.session.inputs()[2].name().to_owned();
                let inputs = inputs![
                    first_name.as_str() => TensorRef::from_array_view(input_ids.view()).map_err(YoloError::OrtInputError)?,
                    second_name.as_str() => TensorRef::from_array_view(attention_mask.view()).map_err(YoloError::OrtInputError)?,
                    third_name.as_str() => TensorRef::from_array_view(token_type_ids.view()).map_err(YoloError::OrtInputError)?,
                ];
                self.session.run(inputs).map_err(YoloError::OrtInferenceError)?
            }
            count => return Err(YoloError::UnsupportedPromptEncoderInputCount(count)),
        };
        let output = outputs[output_name.as_str()]
            .try_extract_array::<f32>()
            .map_err(YoloError::OrtExtractTensorError)?;

        let mut embeddings = if let Ok(output) = output.view().into_dimensionality::<ndarray::Ix2>() {
            output.to_owned().insert_axis(Axis(0))
        } else if let Ok(output) = output.view().into_dimensionality::<ndarray::Ix3>() {
            if output.shape()[0] != 1 {
                return Err(YoloError::InvalidPromptEncoderOutputShape(output.shape().to_vec()));
            }
            output.to_owned()
        } else {
            return Err(YoloError::InvalidPromptEncoderOutputShape(output.shape().to_vec()));
        };

        normalize_last_axis(&mut embeddings);
        YoloPromptEmbeddings::try_new(labels, embeddings)
    }
}

fn normalize_last_axis(embeddings: &mut Array3<f32>) {
    for mut prompt in embeddings.axis_iter_mut(Axis(1)) {
        let norm = prompt.iter().map(|value| value * value).sum::<f32>().sqrt();
        if norm > 0.0 {
            prompt.iter_mut().for_each(|value| *value /= norm);
        }
    }
}