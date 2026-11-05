{
  description = "A Nix-flake-based Rust development environment";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixpkgs-unstable";
    rust-overlay = {
      url = "github:oxalica/rust-overlay";
      inputs.nixpkgs.follows = "nixpkgs";
    };
  };

  outputs =
    inputs:
    let
      supportedSystems = [
        "x86_64-linux"
        "aarch64-linux"
        "x86_64-darwin"
        "aarch64-darwin"
      ];
      forEachSupportedSystem =
        f:
        inputs.nixpkgs.lib.genAttrs supportedSystems (
          system:
          f {
            pkgs = import inputs.nixpkgs {
              inherit system;
              overlays = [
                inputs.rust-overlay.overlays.default
                inputs.self.overlays.default
              ];
            };
          }
        );
    in
    {
      overlays.default = final: _: {
        # The newest nightly in the locked rust-overlay that has all these components.
        rustToolchain = final.rust-bin.selectLatestNightlyWith (
          toolchain:
          toolchain.minimal.override {
            extensions = [
              "rustfmt"
              "clippy"
              "rust-src"
              "rust-analyzer"
            ];
          }
        );
      };

      devShells = forEachSupportedSystem (
        { pkgs }:
        {
          default = pkgs.mkShell {
            packages = with pkgs; [
              rustToolchain
              openssl
              pkg-config
              cargo-edit
              cargo-deny

              nixfmt
              knope
            ];

            env = {
              # Required by rust-analyzer
              RUST_SRC_PATH = "${pkgs.rustToolchain}/lib/rustlib/src/rust/library";
            };
          };

          # `yolo export` for the Models workflow.
          export =
            let
              python = pkgs.python3;
            in
            pkgs.mkShell {
              packages = [
                (python.withPackages (p: [
                  # Its tests download datasets, which the build sandbox can't reach.
                  (p.ultralytics.overridePythonAttrs { doCheck = false; })
                  p.onnx
                  p.onnxslim
                  p.onnxruntime
                ]))
              ];
              # What an export depends on; the Models workflow caches the ONNX files under it.
              passthru.exportKey =
                with python.pkgs;
                "ultralytics-${ultralytics.version}-torch-${torch.version}-onnx-${onnx.version}-onnxslim-${onnxslim.version}";
            };
        }
      );
    };
}
