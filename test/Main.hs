-- SPDX-FileCopyrightText: 2026 Mika Tammi
-- SPDX-License-Identifier: BSD-3-Clause

module Main (main) where

import Test.FuryKernel.Ane.BlobTest qualified as Blob
import Test.FuryKernel.Ane.HalfTest qualified as Half
import Test.FuryKernel.Ane.MilTest qualified as Mil
import Test.FuryKernel.Ane.ProtobufTest qualified as Protobuf
import Test.Tasty (defaultMain, testGroup)

main :: IO ()
main =
  defaultMain $
    testGroup
      "furykernel-ane"
      [Protobuf.tests, Half.tests, Blob.tests, Mil.tests]
