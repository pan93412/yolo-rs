//! Load the YOLO model.

use std::path::Path;

use arcstr::ArcStr;

use crate::error::YoloError;

/// The YOLO model.
///
/// It is a wrapper around the ONNX runtime session and the YOLO labels.
#[derive(Debug)]
pub struct YoloModelSession {
    pub session: ort::session::Session,
    pub labels: Vec<ArcStr>,
}

impl YoloModelSession {
    /// Wrap a ONNX session to a [`YoloModelSession`].
    ///
    /// The `session` is the ONNX runtime session, and the `labels` are the YOLO labels.
    pub fn new(
        session: ort::session::Session,
        labels: impl Iterator<Item = impl Into<ArcStr>>,
    ) -> Self {
        Self {
            session,
            labels: labels.map(Into::into).collect(),
        }
    }

    /// Wrap a ONNX session to a [`YoloModelSession`] based on the labels of YOLO v8 (v11).
    pub fn new_v8(session: ort::session::Session) -> Self {
        const LABELS: &[ArcStr] = &[
            arcstr::literal!("person"),
            arcstr::literal!("bicycle"),
            arcstr::literal!("car"),
            arcstr::literal!("motorcycle"),
            arcstr::literal!("airplane"),
            arcstr::literal!("bus"),
            arcstr::literal!("train"),
            arcstr::literal!("truck"),
            arcstr::literal!("boat"),
            arcstr::literal!("traffic light"),
            arcstr::literal!("fire hydrant"),
            arcstr::literal!("stop sign"),
            arcstr::literal!("parking meter"),
            arcstr::literal!("bench"),
            arcstr::literal!("bird"),
            arcstr::literal!("cat"),
            arcstr::literal!("dog"),
            arcstr::literal!("horse"),
            arcstr::literal!("sheep"),
            arcstr::literal!("cow"),
            arcstr::literal!("elephant"),
            arcstr::literal!("bear"),
            arcstr::literal!("zebra"),
            arcstr::literal!("giraffe"),
            arcstr::literal!("backpack"),
            arcstr::literal!("umbrella"),
            arcstr::literal!("handbag"),
            arcstr::literal!("tie"),
            arcstr::literal!("suitcase"),
            arcstr::literal!("frisbee"),
            arcstr::literal!("skis"),
            arcstr::literal!("snowboard"),
            arcstr::literal!("sports ball"),
            arcstr::literal!("kite"),
            arcstr::literal!("baseball bat"),
            arcstr::literal!("baseball glove"),
            arcstr::literal!("skateboard"),
            arcstr::literal!("surfboard"),
            arcstr::literal!("tennis racket"),
            arcstr::literal!("bottle"),
            arcstr::literal!("wine glass"),
            arcstr::literal!("cup"),
            arcstr::literal!("fork"),
            arcstr::literal!("knife"),
            arcstr::literal!("spoon"),
            arcstr::literal!("bowl"),
            arcstr::literal!("banana"),
            arcstr::literal!("apple"),
            arcstr::literal!("sandwich"),
            arcstr::literal!("orange"),
            arcstr::literal!("broccoli"),
            arcstr::literal!("carrot"),
            arcstr::literal!("hot dog"),
            arcstr::literal!("pizza"),
            arcstr::literal!("donut"),
            arcstr::literal!("cake"),
            arcstr::literal!("chair"),
            arcstr::literal!("couch"),
            arcstr::literal!("potted plant"),
            arcstr::literal!("bed"),
            arcstr::literal!("dining table"),
            arcstr::literal!("toilet"),
            arcstr::literal!("tv"),
            arcstr::literal!("laptop"),
            arcstr::literal!("mouse"),
            arcstr::literal!("remote"),
            arcstr::literal!("keyboard"),
            arcstr::literal!("cell phone"),
            arcstr::literal!("microwave"),
            arcstr::literal!("oven"),
            arcstr::literal!("toaster"),
            arcstr::literal!("sink"),
            arcstr::literal!("refrigerator"),
            arcstr::literal!("book"),
            arcstr::literal!("clock"),
            arcstr::literal!("vase"),
            arcstr::literal!("scissors"),
            arcstr::literal!("teddy bear"),
            arcstr::literal!("hair drier"),
            arcstr::literal!("toothbrush"),
        ];

        Self {
            session,
            labels: LABELS.to_vec(),
        }
    }

    /// Load the YOLO ONNX model from a filename.
    ///
    /// You can use this function to load a YOLO v8 (v11) model from a file.
    /// The `filename` is the path to the ONNX model file.
    ///
    /// You can export the ONNX model file according to
    /// [Ultralytics' manual](https://docs.ultralytics.com/integrations/onnx/).
    pub fn from_filename_v8(filename: impl AsRef<Path>) -> Result<Self, YoloError> {
        let session = ort::session::Session::builder()
            .map_err(|e| YoloError::OrtSessionBuildError(e))?
            .commit_from_file(filename)
            .map_err(|e| YoloError::OrtSessionLoadError(e))?;

        Ok(Self::new_v8(session))
    }
}

impl AsRef<ort::session::Session> for YoloModelSession {
    fn as_ref(&self) -> &ort::session::Session {
        &self.session
    }
}
