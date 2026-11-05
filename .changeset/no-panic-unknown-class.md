---
default: patch
---

# `inference` no longer panics on a class without a label

A model with more classes than `YoloModelSession::labels` made `inference` panic on a detection of an
extra class. That detection is now dropped instead.
