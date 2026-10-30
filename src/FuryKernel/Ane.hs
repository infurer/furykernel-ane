-- SPDX-FileCopyrightText: 2026 Mika Tammi
-- SPDX-License-Identifier: BSD-3-Clause

{- | furykernel-ane: the Apple Neural Engine target of FuryKernel.

The Core ML program writer:

* "FuryKernel.Ane.Protobuf": the protocol buffers wire format, encoding
  only;
* "FuryKernel.Ane.CoreML": the subset of the Core ML specification an ML
  program needs, and its encoding;
* "FuryKernel.Ane.Blob": the blob file of a program's weights;
* "FuryKernel.Ane.Package": the @.mlpackage@ directory Core ML compiles;
* "FuryKernel.Ane.Half": IEEE binary16, the ANE's compute type;
* "FuryKernel.Ane.Mil": a builder for ML programs, whose weights can be a
  model's tensors for a runtime to fill.

Programs lowered to it:

* "FuryKernel.Ane.Stage": the programs one prediction runs and how their
  sources are fed, the Neural Engine's limits, and the gate;
* "FuryKernel.Ane.Lower": a checked stage to one ML program and its
  manifest ("FuryKernel.Ane.Manifest");
* "FuryKernel.Ane.Host" and "FuryKernel.Ane.Interp": the host's side and
  the programs run in f64, references for tests.
-}
module FuryKernel.Ane () where
