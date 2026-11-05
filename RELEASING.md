# Releasing

## Change files

A pull request that changes what users of yolo-rs see adds a change file.
`nix develop -c knope document-change` asks for its level and summary and writes it to
`.changeset/`; add paragraphs under the summary when it needs them:

```md
---
default: minor
---

# YoloModelSession takes an ort Environment

`from_filename_v8` now takes the `ort::Environment` to create the session in.
```

Choose the level by the effect on users: `major` when they have to change their code, `minor` for
something new, `patch` for a fix. Before 1.0, a `major` change makes the next minor version (0.3.0
after 0.2.2) and a `minor` change the next patch. A change only to CI, tooling or the examples
needs none; label the pull request `changelog: none` when its title is `feat`, `fix` or `perf` and
users see no difference. The Changeset check enforces this. Keep each paragraph on one line: the
section becomes the release notes as it is.

## Cutting a release

1. Run the **Release** workflow from the Actions tab. Leave `version` empty to take the one the
   change files give, or set `prerelease: rc` for `X.Y.Z-rc.N`.
2. It opens `chore(release): vX.Y.Z` from `release/next`: `CHANGELOG.md` gains the new section,
   the change files are removed, and `Cargo.toml`, `Cargo.lock` and the example's dependency move
   to the new version. Edit `CHANGELOG.md` on that branch if the section should read differently.
3. Squash-merge it. The push to `master` publishes the crate to crates.io, then tags the merge
   commit `vX.Y.Z` and creates the GitHub release with the section as its notes.

## One-time setup

- **crates.io trusted publishing:** in the crate's settings on crates.io, add a trusted publisher
  for `pan93412/yolo-rs` with the workflow `release.yml`. No crates.io token is stored in GitHub.
- **`RELEASE_TOKEN`:** a fine-grained token for this repository with *Contents* and *Pull
  requests* read and write, stored as an Actions secret. A pull request opened with the default
  `GITHUB_TOKEN` starts no workflow, so its required checks would never run.
