-- SPDX-FileCopyrightText: 2026 Mika Tammi
-- SPDX-License-Identifier: BSD-3-Clause

-- | The wire format against the protocol buffers documentation's examples.
module Test.FuryKernel.Ane.ProtobufTest (tests) where

import Data.ByteString.Lazy qualified as BL
import Data.Word (Word8)
import FuryKernel.Ane.Protobuf qualified as P
import Test.Tasty (TestTree, testGroup)
import Test.Tasty.HUnit (testCase, (@?=))

bytesOf :: P.Msg -> [Word8]
bytesOf = BL.unpack . P.encode

tests :: TestTree
tests =
  testGroup
    "Protobuf"
    [ testCase "varints" $ do
        bytesOf (P.uint 1 0) @?= [0x08, 0x00]
        bytesOf (P.uint 1 150) @?= [0x08, 0x96, 0x01]
        bytesOf (P.uint 2 300) @?= [0x10, 0xac, 0x02]
        bytesOf (P.bool 3 True) @?= [0x18, 0x01]
    , testCase "a negative int64 takes ten bytes" $
        bytesOf (P.int64 1 (-1)) @?= 0x08 : replicate 9 0xff <> [0x01]
    , testCase "length-delimited fields" $ do
        bytesOf (P.string 2 "testing") @?= [0x12, 0x07, 0x74, 0x65, 0x73, 0x74, 0x69, 0x6e, 0x67]
        bytesOf (P.message 3 (P.uint 1 150)) @?= [0x1a, 0x03, 0x08, 0x96, 0x01]
        bytesOf (P.packedVarints 4 [3, 270, 86942]) @?= [0x22, 0x06, 0x03, 0x8e, 0x02, 0x9e, 0xa7, 0x05]
        bytesOf (P.packedFloats 1 [1]) @?= [0x0a, 0x04, 0x00, 0x00, 0x80, 0x3f]
    , testCase "a map entry is its key as field 1 and its value as field 2" $
        bytesOf (P.mapEntry 2 (P.string 1 "a") (P.uint 2 1)) @?= [0x12, 0x05, 0x0a, 0x01, 0x61, 0x10, 0x01]
    ]
