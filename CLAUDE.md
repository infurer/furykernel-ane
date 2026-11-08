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

- stages: the programs one prediction runs one after another (a tower's
  step), and how each program's sources are fed: by the host, by an
  earlier program, by a model's tensor, by a cache's fp16 mirror, or by the
  step's own rows;
- the ANE's limits (`Caps`, `appleNE`, as measured on an M3 Max under
  macOS 27.2: the channels of an op's output, the positions of a
  program's last axis, the bytes of a stage's weight file as the lowering
  writes it) and the gate that holds a stage to them and to what the
  lowering takes. The limits are necessary, not sufficient: Core ML's
  cost model can place a whole small program on the CPU though the ANE
  takes each of its ops (two decode rows over a mirror of 16384
  positions, a 1024-wide norm over 16 rows), so a runtime asserts its
  placement with `MLComputePlan`;
- the ANE's arithmetic, as observed there (`FuryKernel.Ane.Interp`'s
  `NeuralEngine` mode): a convolution's or a matmul's products each
  rounded to a multiple of 2^-16, ties away from zero, or dropped where
  their operands' exponents sum below -16, and summed in a fixed-point
  accumulator of \[-32768, 32768), any partial sum past it infinite
  whatever the result, the sum rounded to binary16, ties away
  (`neuralEngineDot`, bit for bit over every probe); so a product errs
  by up to 2^-17 whatever its size, about 2^-16·√(k/12) over a fan-in of
  k. The other ops keep binary16's whole range, add and sub rounded
  exactly, ties away; the ANE's own tanh is up to 0.58 % low for small
  arguments, its GELU (either mode) off by up to 0.006 near ±0.25, its
  exp by up to ten units in the last place;
- the lowering to one ML program in the `(B, C, 1, S)` layout: linear
  layers as 1×1 convolutions, split where wider than the ANE's channels;
  attention over mirrors or the step's own rows through masks, the rows
  cast to fp16 as their cache holds them (a cast Core ML fuses with the
  op that makes them), its probabilities times 2^8 before they weigh the
  values and the result times 2^-8, which keeps their products above the
  accumulator's grid over long contexts and so needs attention's values,
  a mirror's and the step's own, below 128 in magnitude (past that the
  sums overflow, to infinity); what the ANE can't do well (rotary angles,
  masks, gathers from big tables) asked of the host as inputs. Not done
  yet: each linear layer's weights scaled by a power of two the host
  chooses per output channel, k_o ≤ 15 - log2(‖w_o‖₂·√d·max|γ|) for an
  RMSNorm's output of gain γ (a partial sum is at most ‖w_o‖₂·‖x‖₂), the
  manifest's scale carrying 2^k and a host-filled scalar its inverse:
  that would lift small rows' products off the grid, so the conformance
  run would need no absolute floor. One scale for every layer isn't safe:
  it cuts the accumulator's headroom (at 2^4 the corpus's long sums
  overflowed, at 2^8 a drafting step);
- the emitters: the Core ML program (an `.mlpackage` holding the MIL
  program) and a manifest, the binding runtimes read: every input and what
  the host derives for it, every output, and the map from the weights'
  blobs to model tensors with the bf16 → fp16 conversion;
- references for tests: the host's derivations, and an f64 interpreter of
  the programs the lowering writes, which also computes as the ANE does,
  its dot products bit for bit.

The Core ML model format is coremltools': its protobuf definitions are
vendored here (`vendor/coremltools`, 9.0) under their BSD-3-Clause license.
The emitters write it from Haskell, so neither Python nor coremltools is
needed. Programs are specification version 10 with the opset `CoreML9`,
for macOS 26 and later: the newest that macOS 27.2's Core ML takes.

The library never runs anything on the ANE itself: it emits programs,
which the consuming runtime compiles with Core ML at run time. So it builds
and is tested on any system, and golden tests pin its output; the Neural
Engine conformance run (`test/ane/`) runs that output on an Apple silicon
Mac. What the repository records of the ANE was observed on one M3 Max,
under macOS 27.2.

Like the core, it knows no models: programs for a particular model live with
the runtime that runs them.

## Commands

```sh
nix develop                # GHC, cabal, ghcid, hlint, cabal2nix, reuse, protoc
cabal build                # inside the dev shell
nix fmt
nix flake check --all-systems --no-update-lock-file
nix run .#ane-conformance  # on an Apple silicon Mac: the corpus on its ANE
```

`flake.lock` pins the core, the `furykernel` input, as a
`github:infurer/furykernel` revision on its master, as
`nix run --no-update-lock-file .#check-pins` checks, which fails without
an input of ours.

After changing `furykernel-ane.cabal`, regenerate `furykernel-ane.nix`:

```sh
cabal2nix . | sed 's|^  src = .*;$|  src = ./.;|' > furykernel-ane.nix
```

When the encoding of ML programs changes on purpose, the tiny program's
golden test fails and writes the new bytes to the temporary directory;
copy them over `test/golden/tiny.mlmodel` and decode them into the golden
text, which `checks.protoc-decode` compares (the lowering's manifest golden,
`test/golden/draft_step.json`, works the same way, without decoding):

```sh
protoc --decode=CoreML.Specification.Model \
  --proto_path=vendor/coremltools/format vendor/coremltools/format/Model.proto \
  < test/golden/tiny.mlmodel > test/golden/tiny.txt
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

The Neural Engine conformance run needs a Mac's Neural Engine, so it is no
check and runs outside CI: in the per-commit gate, on an Apple silicon Mac,
at every commit from the one that adds it, with every case passing, and
again on each new Mac or macOS. `nix run .#ane-conformance` runs the corpus
of the working tree, uncommitted changes to tracked files included, so a
change can be held to it before it is committed;
`nix run .#ane-conformance -- DIR` runs the corpus in `DIR` instead.

## Architecture

- `flake.nix`: flake-parts on nixos-26.05; systems x86_64-linux,
  aarch64-linux and aarch64-darwin. The `furykernel` input is the core,
  whose package `nix/package.nix` builds this one against.

- `furykernel-ane.cabal`, `src/`: the library, one package.
  `furykernel-ane.nix` is cabal2nix's output, committed so that evaluation
  never builds anything first. The Core ML program writer:

  - `FuryKernel.Ane.Protobuf`: the protocol buffers wire format, encoding
    only;
  - `FuryKernel.Ane.CoreML`: the subset of the Core ML specification an ML
    program needs (Model.proto, FeatureTypes.proto, MIL.proto), its field
    numbers the vendored definitions', and a text form of programs for
    reviews of generated ones;
  - `FuryKernel.Ane.Blob`: the blob file of a program's weights (version 2),
    each blob's bytes given or its size reserved, for a runtime to write a
    model's tensors at the offsets its layout gives;
  - `FuryKernel.Ane.Package`: the `.mlpackage` directory, with fixed
    identifiers;
  - `FuryKernel.Ane.Half`: binary16, rounding to nearest even, and what a
    conversion from bf16 does to a tensor's range;
  - `FuryKernel.Ane.Mil`: a pure builder of ML programs (one function, one
    block), in fp16 or as an fp32 reference, whose weights can be parts of
    a model's tensors, reserved once each and listed in its weight map.

  The lowering of programs:

  - `FuryKernel.Ane.Stage`: stages, parts and feeds, the ANE's `Caps`, and
    `checkStage`, the gate (`CheckedStage`'s constructor is private);
  - `FuryKernel.Ane.Lower`: `lowerStage`, a checked stage to its ML program
    and manifest;
  - `FuryKernel.Ane.Manifest`: the manifest and its JSON;
  - `FuryKernel.Ane.Host`: the host's derivations as a reference (rotary
    tables, masks, mirrors, gathers, selections, weight blobs) and
    `evaluateStage`, a stage evaluated part by part by the core's f64
    evaluator;
  - `FuryKernel.Ane.Interp`: ML programs run in f64, for the ops the
    lowering writes, a cast to fp16 rounding to binary16 as the core's
    evaluator rounds an fp16 cache's stores.

- `test/`: tasty with hedgehog: the wire format against the protocol
  buffers documentation's examples, rounding to binary16 (every finite
  value back as itself), the blob file's layout and bytes, a tiny
  program against its golden bytes, and two stages lowered (a drafting
  step's, an image tower's) whose programs, interpreted on the host's
  inputs, compute what the core's evaluator does, with the manifest of one
  against its golden text. The stages are `Test.FuryKernel.Ane.Stages`', of
  any size, which the conformance corpus takes too.

- `test/ane/`: the Neural Engine conformance run (`apps.ane-conformance`).
  `Cases.hs` writes its corpus: the tests' stages at sizes Core ML places
  on the ANE (a drafting step whose output layer is two chunks, an image
  tower 384 wide over 128 rows), a layer of Gemma 4's larger image tower
  (1152 wide over 256 rows), and edge cases: norms over rows of zero, of
  magnitudes from 2^-8 to 4096 and near 32768, and of one large element;
  each head's norm at the shapes whose norm Core ML's CPU computed wrongly
  along axis 2, the two small ones on the CPU alone, as Core ML places
  them there anyway; attention over a mirror of 16384 positions, what no
  row sees holding large numbers, and attention over all 16384, the length
  the gate allows; a softcap saturating; argmaxes whose logits tie across
  the output layer's chunks; residuals of rows near 2^-6 and near 30000;
  sums whose running sums reach 30000 on their way to zero; attention over
  all 16384 positions with every probability 2^-14 (`flat_mirror`); and
  two cases held to the interpreter's model of the ANE bit for bit, which
  f64 differs from by design: products the ANE's sums keep on their grid
  of 2^-16, round away from zero, or drop by their operands' exponents
  (`products_grid`), and running sums that pass 32768 on the way to
  results that fit (`prefix_overflow`). The
  drafting step, the Gemma 4 layer and the mirrors, whose references take
  many minutes or gigabytes (the core's evaluator holds 15 to 40 GiB over
  a mirror), are `ane-corpus-large`, which `checks.ane-corpus` leaves out:
  its cases need 4 GiB. The data is a model's in scale: attention scores
  of a few units, values sharing a part, weights of unit gain. Each case
  is lowered by `lowerStage`, its inputs derived by `FuryKernel.Ane.Host`,
  in dyadic numbers, exact in every sum the evaluator makes, or in
  binary16 numbers of a full mantissa, and its outputs computed by the
  core's evaluator and by the interpreter, both in f64, and by the
  interpreter's `NeuralEngine` mode. The runner, Rust on objc2-core-ml,
  compiles each case with Core ML as a runtime does, asserts from
  `MLComputePlan` that every op runs on the ANE but the program's tail on
  the CPU (`reduce_argmax`, and the `concat` that joins an output layer's
  chunks for it), none of which the ANE reads again (an op the plan gives
  no device is fused into another), predicts it twice, the same bits each
  time, and holds each float output's rows within τ of their largest
  magnitude in both f64 references, or of 2^-9/τ if that is larger, so
  that a small row is held to 2^-9 absolutely: τ = 2^-7, but 2^-6 for an
  output made through a softcap, which the generator marks `softcap` from
  the stage's IR (rows below 2^-2, and below 2^-3, held absolutely). An
  output marked `model` is held to the interpreter's `NeuralEngine` mode
  bit for bit instead. A token must be the first largest of its predicted
  logits and of a reference logit within twice its logits' τ of the
  largest, or, of logits the corpus makes tie, the evaluator's. The norms
  run on the CPU too, held to 2^-5 of their rows' largest magnitudes or of
  2^-3. The comparison is unit-tested (`checks.ane-runner-tests`);
  `ANE_CONFORMANCE_KEEP=DIR` writes every output as predicted.

  The bounds are checked against the corpus rather than derived. There the
  ANE's worst row is within 0.55 of τ (`range_low`'s small rows, held
  absolutely; 0.36 elsewhere), a softcap's within 0.40 of its 2^-6 (the
  ANE's tanh is up to 0.58 % low for small arguments), and the CPU's within
  0.46 of its 2^-5: margins of 1.85 at the least. The ANE's errors come from
  its sums, which the interpreter's `NeuralEngine` mode computes as the ANE
  does (on `range_low`, `range_high`, `products_grid` and `prefix_overflow`
  its bits are the ANE's), and from its own tanh, GELU, exp, softmax and
  layer norm, which the mode computes exactly and rounds, so that it is
  nearer f64 than the ANE where they make an output (the drafting step's
  logits within 0.10 of their τ, the ANE's 0.30). Each product a sum takes
  errs by up to 2^-17 whatever its size, so a convolution's outputs err by
  about 2^-16·√(k/12) over a fan-in of k, hence the absolute 2^-9. Attention
  over 16384 keys is within 0.14 of τ (`full_mirror`; `flat_mirror` 0.09),
  as the lowering scales its probabilities by 2^8; unscaled, products of
  probabilities near 1/n with the values sat on the sums' grid, 0.72 of 2^-6
  over 16384 keys, and `flat_mirror` fails by 54τ. The sums overflow past
  32768 even where their results would fit (`prefix_overflow`), so the other
  cases keep them below (`long_sums`). Core ML's CPU sums in binary16, four
  to eight times as far from f64 as the ANE. Planted defects fall outside
  the bounds, most of them far: a softcap's tanh left out 511τ; a dropped
  mask infinitely far (the hidden values overflow the sums); attention's
  probabilities unscaled 54τ; attention's scale left out 14τ; rotary halves
  paired wrongly 5.7τ to 98τ, its sign on the wrong half 7.9τ to 98τ; an
  RMSNorm squaring its input scaled by 1/16 4.2τ to 158τ (infinite on the
  CPU), its epsilon times 1024 112τ; and the cap raised to 16640 puts the
  masked mirror on the CPU.

- `vendor/coremltools/`: the Core ML model format's definitions, as
  published.

- `nix/`: `default.nix` (the flake-parts modules below, as the flake
  imports them), `package.nix` (the library, built with GHC 9.12 under
  `-Werror`), `checks.nix` (the build with its tests, `protoc-decode`,
  which decodes the golden bytes with protoc against the vendored
  definitions and compares them with the golden text, hlint over the
  library and its tests, cabal2nix-in-sync, `reuse lint` and the dev
  shell), `devshell.nix`, `pins.nix` (`nix run .#check-pins`), `rust.nix`
  (crane and the Rust toolchain for the conformance runner, and rustfmt),
  `ane.nix` (the corpus, `checks.ane-corpus` on every system; the runner,
  the app, `checks.ane-runner-clippy` and `checks.ane-runner-tests` on
  aarch64-darwin) and `treefmt.nix` (`nix fmt` and `checks.treefmt`).

- `.gitlab-ci.yml`: GitLab CI on x86_64-linux runners. `checks` runs
  `nix flake check --all-systems`, which builds x86_64-linux's checks, and
  builds both Linux systems' packages;
  `pins` runs `nix run --no-update-lock-file .#check-pins` (`nix/pins.nix`):
  every input of ours (the root's inputs but other owners' on GitHub)
  locked as a `github:infurer` revision on github.com, owners compared
  case-insensitively, at a commit on GitHub master, and a `flake.lock` to
  look in, so that it fails until a branch's
  dependencies are merged, published and locked there; `publish-github`,
  on master once those passed, pushes master, and the tags it holds, to
  the public GitHub repository.

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
