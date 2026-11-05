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
