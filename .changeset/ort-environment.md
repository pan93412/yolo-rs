---
default: major
---

# YoloModelSession::from_filename_v8 takes an ort Environment

yolo-rs now uses ort 2.0.0-rc.15, where every session is created in an `ort::environment::Environment`. Pass the one `ort::init().build()` returns: `YoloModelSession::from_filename_v8(&environment, path)`. Execution providers moved from `ort::execution_providers::CUDAExecutionProvider` to `ort::ep::CUDA` and the like, each behind its ort feature.

On macOS, ONNX Runtime from ort rc.15 needs the compiler runtime linked. If linking fails with `Undefined symbols … "___isPlatformVersionAtLeast"`, add `-Clink-arg=-fapple-link-rtlib` to your rustflags for Apple targets (README, "Building on macOS").
