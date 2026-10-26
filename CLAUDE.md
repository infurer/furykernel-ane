<!--
SPDX-FileCopyrightText: 2026 Mika Tammi
SPDX-License-Identifier: BSD-3-Clause
-->

# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working
with code in this repository.

## Project overview

furykernel-ane is FuryKernel's Apple Neural Engine target. The ANE runs
whole Core ML programs, not SIMT kernels, so this target takes FuryKernel
programs (the core's layer of typed tensor dataflow) rather than kernels.
It depends only on the core library, furykernel. It holds:

- the target's capabilities, `AppleNE`: fp16 compute, the supported op set,
  the `(B, C, 1, S)` layout with its last axis 64-byte aligned, tensor
  dimension limits, static or enumerated shapes, a weight budget per program
  and no data-dependent control flow;
- the check of programs against them, through the core's gate;
- the rewrites the ANE prefers: linear layers as 1×1 convolutions, splits
  that keep values in fp16 range, chunking into several programs;
- the emitters: the Core ML program (an `.mlpackage` holding the MIL
  program), a map from its weights to FuryModel tensors with the bf16 →
  fp16 conversion, a manifest, and typed Rust bindings for runtimes.

The Core ML model format is coremltools': its protobuf definitions will be
vendored here under their BSD-3-Clause license. The emitters write it from
Haskell, so neither Python nor coremltools is needed.

It never runs anything on the ANE: it emits programs, which the consuming
runtime compiles with Core ML at run time. So it builds and is tested on
any system, and golden tests pin its output.

Like the core, it knows no models: programs for a particular model live with
the runtime that runs them.

## Commands

```sh
nix develop                # GHC, cabal, ghcid, hlint, cabal2nix, reuse
cabal build                # inside the dev shell
nix fmt
nix flake check --all-systems --no-update-lock-file
```

After changing `furykernel-ane.cabal`, regenerate `furykernel-ane.nix`:

```sh
cabal2nix . | sed 's|^  src = .*;$|  src = ./.;|' > furykernel-ane.nix
```

`nix flake check` evaluates every system but builds only the current
system's checks. The per-commit gate builds them for all systems; on a Mac
the Linux ones go to its remote builder. Flakes only see files git tracks,
so `git add` new files first:

```sh
nix fmt && git diff --exit-code &&
  nix flake check --all-systems --no-update-lock-file &&
  (
    for s in x86_64-linux aarch64-linux aarch64-darwin; do
      nix build --no-link --impure --expr \
        "builtins.attrValues (builtins.getFlake \"git+file://$PWD\").checks.$s" || exit 1
    done
  )
```

## Architecture

- `flake.nix`: flake-parts on nixos-26.05; systems x86_64-linux,
  aarch64-linux and aarch64-darwin.
- `furykernel-ane.cabal`, `src/`: the library, one package.
  `furykernel-ane.nix` is cabal2nix's output, committed so that evaluation
  never builds anything first.
- `nix/`: `package.nix` (the library, built with GHC 9.12 under
  `-Werror`), `checks.nix` (the build with its tests, hlint,
  cabal2nix-in-sync and `reuse lint`), `devshell.nix` and `treefmt.nix`
  (`nix fmt` and `checks.treefmt`).
- `.gitlab-ci.yml`: GitLab CI on x86_64-linux runners. `checks` runs
  `nix flake check --all-systems`, which builds x86_64-linux's checks, and
  builds both Linux systems' packages;
  `pins` runs `nix run --no-update-lock-file .#check-pins` (`nix/pins.nix`):
  every input of ours (the root's inputs but other owners' on GitHub)
  locked as a `github:infurer` revision on github.com, owners compared
  case-insensitively, at a commit on GitHub master, and a `flake.lock` to
  look in, so that it fails until a branch's
  dependencies are merged, published and locked there; `publish-github`,
  on master once those passed, pushes master to the public GitHub
  repository.

## Conventions

- Every file has SPDX copyright and license comments
  (`2026 Mika Tammi`, `BSD-3-Clause`). Files that can't hold comments are
  annotated in `REUSE.toml`.
- Every file type has a formatter in `nix/treefmt.nix`, enabled in the
  commit that adds the first file of that type. Haskell is formatted by
  fourmolu (`fourmolu.yaml`), cabal files by cabal-gild.
- GHC 9.12 (`haskell.packages.ghc912`, set once in `nix/package.nix`),
  GHC2024 and the warnings in `furykernel-ane.cabal`'s `warnings` stanza;
  `-Werror` only in the Nix build, so work in progress still loads in ghci.
  Modules have export lists.
- Nothing host-specific in the repository: no hostnames, absolute paths of
  anyone's machine, or names of private repositories and services. master
  is mirrored to a public GitHub repository in full.

## Git workflow

- One commit per reasonably sized feature. Docs (README, this file) change
  in the same commit as the feature they describe.
- Subjects in nixpkgs style, `<attr or area>: <summary>`. Short bodies: why,
  and facts that aren't obvious from the diff.
- Commit with `git commit -s`, and with an `Assisted-by:` trailer naming
  the AI assistant that helped, such as `Assisted-by: Claude Opus 5.5`,
  directly above the Signed-off-by. No other trailers.
- `flake.lock` changes only together with the input change that needs them.
- Every commit passes the per-commit gate (see Commands).
- Corrections to earlier commits go in as `git commit --fixup=<commit>` (or
  `--squash`) and are autosquashed before the merge request.
- v0.2 work happens on `initialv02` and reaches master through merge
  requests on the GitLab repository, which CI mirrors to GitHub. Never push
  to GitHub by hand. Mika signs the commits with GPG before the merge.

## Common mistakes to avoid

- `path:` flake inputs, or `flake.lock` updates riding along in unrelated
  commits.
- Committing `HANDOFF.md` or other session notes.
- Writing SPDX tag names followed by a colon in prose: `reuse lint` parses
  them anywhere in a file.
