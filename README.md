<!--
SPDX-FileCopyrightText: 2026 Mika Tammi
SPDX-License-Identifier: BSD-3-Clause
-->

# furykernel-ane

furykernel-ane is FuryKernel's Apple Neural Engine target: a Haskell
library that lowers FuryKernel programs, the core's layer of typed tensor
dataflow, to Core ML ML programs. The Neural Engine runs whole programs,
not kernels, so this target takes programs. The core, furykernel, is at
<https://github.com/infurer/furykernel>. It holds:

- the stages (`FuryKernel.Ane.Stage`): the programs one prediction runs,
  one after another, and how each one's sources are fed: by the host, by
  an earlier program, by a model's tensor, by a cache's fp16 mirror or by
  the step's own rows; the Neural Engine's limits, `appleNE`, and the gate
  that holds a stage to them;
- the lowering (`FuryKernel.Ane.Lower`): a checked stage as one ML program
  in an `.mlpackage`, specification version 10 with the opset `CoreML9`,
  and its manifest (`FuryKernel.Ane.Manifest`), which runtimes read: every
  input and what the host derives for it, every output, and each weight
  blob's part of a model tensor;
- references for tests: the host's derivations and a stage evaluated with
  the core's evaluator (`FuryKernel.Ane.Host`), and an interpreter of the
  ML programs it writes (`FuryKernel.Ane.Interp`), in f64, rounding each
  op's result to binary16, or keeping values as the Neural Engine was
  observed to.

It emits programs only. Runtimes compile them with Core ML at run time, on
macOS 26 or later, so the library builds and is tested on any system, and
golden tests pin its output. It writes the Core ML model format itself,
from the protobuf definitions vendored from coremltools 9.0
(`vendor/coremltools`), so neither Python nor coremltools is needed. Like
the core, it knows no models: a model's programs live with the runtime
that runs them.

## Status

Version 0.2, in development, with the core: the lowering covers what a
drafting step and an image tower use. What the documentation records of
the Neural Engine, its limits included, was observed on one M3 Max, under
macOS 27.2.

## Building and testing

```sh
nix build                  # the library, its test suite run in the check phase
nix develop                # GHC, cabal and the tools
cabal build                # inside the dev shell
nix flake check            # the build, its tests and every other check
nix run .#ane-conformance  # on an Apple silicon Mac: the conformance run
```

The Neural Engine conformance run needs an Apple silicon Mac, on macOS 26
or later, so it runs outside CI: in the per-commit gate, and again on each
new Mac or macOS. It lowers a corpus of stages (`test/ane`), runs it on
the Neural Engine as a runtime would, and holds every output to the core's
evaluator and to the interpreter, within 2^-7 of each row's largest
magnitude (2^-6 for an output made through a softcap).

## License

BSD 3-Clause, in `LICENSES/BSD-3-Clause.txt`. The vendored Core ML format
definitions are Apple's, under the same license, in
`vendor/coremltools/format/LICENSE.txt`.
