---
default: major
---

# YoloModelSession::from_filename_v8 takes an ort Environment

yolo-rs now uses ort 2.0.0-rc.15, where every session is created in an `ort::environment::Environment`. Pass the one `ort::init().build()` returns: `YoloModelSession::from_filename_v8(&environment, path)`. Execution providers moved from `ort::execution_providers::CUDAExecutionProvider` to `ort::ep::CUDA` and the like, each behind its ort feature.
