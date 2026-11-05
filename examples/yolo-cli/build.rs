//! Links the compiler runtime that ONNX Runtime needs on Apple platforms.

fn main() {
    #[cfg(any(target_os = "macos", target_os = "ios", target_os = "tvos"))]
    println!("cargo:rustc-link-arg=-fapple-link-rtlib");
}
