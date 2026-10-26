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

## Conventions

- Every file has SPDX copyright and license comments
  (`2026 Mika Tammi`, `BSD-3-Clause`). Files that can't hold comments are
  annotated in `REUSE.toml`.
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
- Corrections to earlier commits go in as `git commit --fixup=<commit>` (or
  `--squash`) and are autosquashed before the merge request.
- v0.2 work happens on `initialv02` and reaches master through merge
  requests on the GitLab repository, which CI mirrors to GitHub. Never push
  to GitHub by hand. Mika signs the commits with GPG before the merge.

## Common mistakes to avoid

- Committing `HANDOFF.md` or other session notes.
- Writing SPDX tag names followed by a colon in prose: `reuse lint` parses
  them anywhere in a file.
