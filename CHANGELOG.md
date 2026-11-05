# Changelog

All notable changes to yolo-rs. Versions follow [Semantic Versioning](https://semver.org/); before 1.0.0, a minor release may break the API. Each section comes from the change files in `.changeset/` ([RELEASING.md](RELEASING.md)). Releases up to 0.2.2 are described on [GitHub Releases](https://github.com/pan93412/yolo-rs/releases).

## 0.3.0 (2026-11-05)

### Breaking Changes

#### YoloModelSession::from_filename_v8 takes an ort Environment

yolo-rs now uses ort 2.0.0-rc.15, where every session is created in an `ort::environment::Environment`. Pass the one `ort::init().build()` returns: `YoloModelSession::from_filename_v8(&environment, path)`. Execution providers moved from `ort::execution_providers::CUDAExecutionProvider` to `ort::ep::CUDA` and the like, each behind its ort feature.

On macOS, ONNX Runtime from ort rc.15 needs the compiler runtime linked. If linking fails with `Undefined symbols … "___isPlatformVersionAtLeast"`, add `-Clink-arg=-fapple-link-rtlib` to your rustflags for Apple targets (README, "Building on macOS").

On macOS, sessions now run on CoreML by default: ort rc.15 lets ONNX Runtime pick the device, and on a Mac it always adds CoreML, even when only the CPU execution provider is registered. To stay on the CPU, build the session with `.with_auto_device(AutoDevicePolicy::PreferCPU)` and pass it to `YoloModelSession::new_v8` (README, "CoreML by default").

### Fixes

#### yolo-rs no longer turns on image's codecs

yolo-rs works on an already decoded `DynamicImage`, so it now depends on `image` without its default
features. This drops about 70 crates, among them the unmaintained `paste`. If your project decodes
images with `image` but turned its default features off, enable the formats you open (for example
`features = ["jpeg", "png"]`) on your own `image` dependency.

#### `inference` no longer panics on unexpected model outputs

A model with more classes than `YoloModelSession::labels` made `inference` panic on a detection of an
extra class; that detection is now dropped instead. A model without an `output0` output made it panic
too; it now returns `YoloError::OrtExtractSensorError`.

## 0.2.3 (2026-11-05)

### Features

#### yolo-rs is licensed under Apache-2.0 only

yolo-rs was dual-licensed MIT OR Apache-2.0; from this release on, it is Apache-2.0 only. Releases up to 0.2.2 keep their dual licence.
