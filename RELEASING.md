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
4. The CLI workflow then builds yolo-cli, which has the crate's version, for Linux (x86_64,
   arm64) and macOS (arm64), signs and notarizes the macOS binary, and attaches
   `yolo-cli-<target>.tar.gz`, `SHA256SUMS` and a build provenance attestation for each
   archive to the release. Approve the `apple` deployment when it asks.

## One-time setup

- **crates.io trusted publishing:** in the crate's settings on crates.io, add a trusted publisher
  for `pan93412/yolo-rs` with the workflow `release.yml` and the environment `crates-io`. No
  crates.io token is stored in GitHub.
- **`crates-io` environment:** in the repository's Settings → Environments, with deployment
  branches limited to `master`. It holds no secrets or variables.
- **`RELEASE_TOKEN`:** a fine-grained token for this repository with *Contents* and *Pull
  requests* read and write, stored as an Actions secret. A pull request opened with the default
  `GITHUB_TOKEN` starts no workflow, so its required checks would never run.
- **`apple` environment:** in Settings → Environments, with deployment branches limited to
  `master` and you as a required reviewer. Its secrets sign and notarize the macOS yolo-cli:
  - `APPLE_DEVELOPER_ID_P12`: a *Developer ID Application* certificate (developer.apple.com →
    Certificates) exported from Keychain Access with its private key as a `.p12`, then
    `base64 -i developer-id.p12 | pbcopy`.
  - `APPLE_DEVELOPER_ID_P12_PASSWORD`: the password chosen when exporting it.
  - `APPLE_API_KEY`: an App Store Connect API key (App Store Connect → Users and Access →
    Integrations → Team Keys, with the *Developer* role) encoded by
    `rcodesign encode-app-store-connect-api-key <issuer-id> <key-id> AuthKey_<key-id>.p8`; get
    rcodesign with `nix shell nixpkgs#rcodesign`. Paste the JSON it prints.

  Run the **CLI** workflow by hand from `master` to check them: it signs and notarizes the macOS
  binary without releasing anything.
