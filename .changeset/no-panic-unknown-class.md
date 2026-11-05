---
default: patch
---

# `inference` no longer panics on unexpected model outputs

A model with more classes than `YoloModelSession::labels` made `inference` panic on a detection of an
extra class; that detection is now dropped instead. A model without an `output0` output made it panic
too; it now returns `YoloError::OrtExtractSensorError`.
