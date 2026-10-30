-- SPDX-FileCopyrightText: 2026 Mika Tammi
-- SPDX-License-Identifier: BSD-3-Clause

-- | The blob file's layout, and its bytes where every blob was given.
module Test.FuryKernel.Ane.BlobTest (tests) where

import Data.ByteString qualified as B
import Data.ByteString.Lazy qualified as BL
import Data.Either (isLeft)
import FuryKernel.Ane.Blob
import Test.Tasty (TestTree, testGroup)
import Test.Tasty.HUnit (assertBool, testCase, (@?=))

tests :: TestTree
tests =
  testGroup
    "Blob"
    [ testCase "no blobs: the header alone" $
        fmap BL.unpack (blobsFile emptyBlobs)
          @?= Right ([0, 0, 0, 0, 2, 0, 0, 0] <> replicate 56 0)
    , testCase "each blob's record and data are 64-byte aligned" $ do
        let (a, b1) = addBlob BlobFloat16 (B.pack [1, 2]) emptyBlobs
            (b, b2) = reserveBlob BlobFloat16 100 b1
            (c, b3) = addBlob BlobFloat32 (B.replicate 64 7) b2
        (a, b, c) @?= (64, 192, 384)
        map (\p -> (p.placedRecord, p.placedData, p.placedSize)) (blobsLayout b3)
          @?= [(64, 128, 2), (192, 256, 100), (384, 448, 64)]
        assertBool "a reserved blob has no bytes yet" (isLeft (blobsFile b3))
    , testCase "a given blob's record and data" $ do
        let (_, blobs) = addBlob BlobFloat16 (B.pack [1, 2]) emptyBlobs
            file = either error BL.toStrict (blobsFile blobs)
            record = B.take 64 (B.drop 64 file)
        B.length file @?= 192
        B.take 8 file @?= B.pack [1, 0, 0, 0, 2, 0, 0, 0]
        B.unpack (B.take 24 record)
          @?= [0xef, 0xbe, 0xad, 0xde, 1, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0, 128, 0, 0, 0, 0, 0, 0, 0]
        B.take 2 (B.drop 128 file) @?= B.pack [1, 2]
    ]
