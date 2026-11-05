# Contributing

## Developer Certificate of Origin

This project uses the [Developer Certificate of Origin 1.1](DCO), not a CLA. Sign off every commit
to certify that you wrote it or otherwise have the right to submit it under the project licence:

```sh
git commit -s
```

CI rejects pull requests whose commits lack a `Signed-off-by` line matching the author. To sign off
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
