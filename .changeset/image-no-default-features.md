---
default: patch
---

# yolo-rs no longer turns on image's codecs

yolo-rs works on an already decoded `DynamicImage`, so it now depends on `image` without its default
features. This drops about 70 crates, among them the unmaintained `paste`. If your project decodes
images with `image` but turned its default features off, enable the formats you open (for example
`features = ["jpeg", "png"]`) on your own `image` dependency.
