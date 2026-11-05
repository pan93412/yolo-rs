//! Links the compiler runtime that ONNX Runtime needs on Apple platforms.

fn main() {
    #[cfg(any(target_os = "macos", target_os = "ios", target_os = "tvos"))]
    #[expect(
        clippy::print_stdout,
        reason = "build scripts talk to Cargo through stdout"
    )]
    {
        println!("cargo:rustc-link-arg=-fapple-link-rtlib");
    }
}
