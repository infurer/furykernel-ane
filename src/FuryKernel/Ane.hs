-- SPDX-FileCopyrightText: 2026 Mika Tammi
-- SPDX-License-Identifier: BSD-3-Clause

{- | furykernel-ane: the Apple Neural Engine target of FuryKernel.

The Core ML program writer, so far:

* "FuryKernel.Ane.Protobuf": the protocol buffers wire format, encoding
  only;
* "FuryKernel.Ane.CoreML": the subset of the Core ML specification an ML
  program needs, and its encoding;
* "FuryKernel.Ane.Blob": the blob file of a program's weights;
* "FuryKernel.Ane.Package": the @.mlpackage@ directory Core ML compiles;
* "FuryKernel.Ane.Half": IEEE binary16, the ANE's compute type;
* "FuryKernel.Ane.Mil": a builder for ML programs, whose weights can be a
  model's tensors for a runtime to fill.
-}
module FuryKernel.Ane () where
