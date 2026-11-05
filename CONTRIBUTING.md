# Contributing

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

## Before you open a pull request

Run what CI runs, inside `nix develop`:

```sh
cargo fmt --all --check
cargo clippy --workspace --all-targets --all-features --locked -- --deny warnings
cargo test --workspace --locked
```

## Benchmarks

`benches/inference.rs` times preprocessing, and inference with NMS for each ONNX model listed in
`YOLO_BENCH_MODELS` (comma-separated):

```sh
YOLO_BENCH_MODELS=examples/yolo-cli/models/yolo11x.onnx cargo bench --bench inference
```

The Benchmark workflow runs it every Monday and by hand, with YOLO11x and the newest YOLO family in
ultralytics/assets, which it exports with `nix develop .#export -c yolo export … end2end=False`.
The results are in the run's summary.
