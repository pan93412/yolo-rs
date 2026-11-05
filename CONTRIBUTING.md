# Contributing

**Build this repository with nightly Rust.** `.cargo/config.toml` uses nightly-only settings, so
stable Cargo stops at the first command (``config profile `release` is not valid``). Either:

- run everything inside `nix develop`, which provides the toolchain CI uses (see "Toolchain"); or
- with rustup, run `rustup toolchain install nightly --component rustfmt clippy` and then
  `rustup override set nightly` in your clone.

Projects that depend on yolo-rs from crates.io are not affected: they build it with stable Rust.

## Developer Certificate of Origin

This project uses the [Developer Certificate of Origin 1.1](DCO), not a CLA. Sign off every commit
to certify that you wrote it or otherwise have the right to submit it under the project licence:

```sh
git commit -s
```

The DCO check rejects pull requests whose commits lack a `Signed-off-by` line matching the author. To sign off
commits you already made, run `git rebase --signoff master`. Commit messages follow
[Conventional Commits](https://www.conventionalcommits.org/) (`feat:`, `fix:`, `chore:`, `ci:`,
`docs:`).

## Toolchain

`nix develop` provides the newest nightly Rust in the locked rust-overlay, with rustfmt, clippy,
rust-src and rust-analyzer. `flake.lock` is the only place the compiler version is set; there is no
`rust-toolchain.toml`. Renovate updates the lock file weekly, which moves the nightly forward.

Nightly is needed for the formatting options in `rustfmt.toml` and for `-Zthreads=8` and
`trim-paths` in `.cargo/config.toml`. Cargo reads that file for every command in this repository,
which is why stable Rust cannot build it here. The published crate does not include it and builds
on stable Rust.

## Lints and formatting

Lints are set once for the workspace in `Cargo.toml`. All are `warn`, so unfinished work still
compiles locally, and `cargo lint` turns every warning into an error in CI.

- `unsafe_code` is denied and `non_ascii_idents` is forbidden.
- clippy `pedantic`, `cargo` and `nursery` are on.
- A selection of `restriction` lints keeps the library from panicking on its input (`unwrap_used`,
  `expect_used`, `indexing_slicing`, …; tests may unwrap), keeps errors' causes
  (`map_err_ignore`, `let_underscore_must_use`, …) and asks for a reason on every suppression:
  use `#[expect(lint, reason = "…")]`, which also fails once it is no longer needed.
- Every public item has documentation (`missing_docs`).

`rustfmt.toml` groups and orders imports, wraps comments at 100 columns and fails on code it
cannot format.

## Dependencies

`cargo deny check` enforces the policy in `deny.toml`:

- [RustSec](https://rustsec.org/) advisories (the database `cargo audit` reads) and yanked
  versions fail the check. Every ignored advisory carries its reason in `deny.toml`.
- Licences must be OSI-approved; crates come from crates.io only; no `*` versions.

Cargo does not pick a release younger than seven days unless it is already in the lock file
(`registry.global-min-publish-age` in `.cargo/config.toml`), which gives the community time to
catch a compromised release. An urgent security update overrides it for one command:
`CARGO_REGISTRY_GLOBAL_MIN_PUBLISH_AGE="0 days" cargo update -p <crate>`.

## Before you open a pull request

Run what CI runs, inside `nix develop`:

```sh
cargo fmt --all --check
cargo lint
cargo deny check
cargo test --workspace --all-features --locked
```

## Models

`tests/e2e.rs` runs the ONNX model in `YOLO_E2E_MODEL` on `examples/yolo-cli/data/baseball.jpg`
and checks that it finds the three players and the glove. `benches/inference.rs` times
preprocessing, and inference with NMS for each model in `YOLO_BENCH_MODELS` (comma-separated).
Without the variables, the test does nothing and the benchmark times only preprocessing.

```sh
nix develop .#export -c yolo export model=yolo11x.pt format=onnx end2end=False
YOLO_E2E_MODEL=yolo11x.onnx cargo test --test e2e
YOLO_BENCH_MODELS=yolo11x.onnx cargo bench --bench inference
```

The Models workflow does both for YOLOv8x, YOLO11x, YOLO12x and YOLO26x on pull requests that
change the code, on `master`, every Monday and by hand. The benchmark results are in each job's
summary.
