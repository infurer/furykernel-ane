-- SPDX-FileCopyrightText: 2026 Mika Tammi
-- SPDX-License-Identifier: BSD-3-Clause

-- | Rounding to binary16, and what a conversion from bf16 reports.
module Test.FuryKernel.Ane.HalfTest (tests) where

import Data.Bits (shiftR, (.&.))
import Data.ByteString qualified as B
import Data.Word (Word16)
import FuryKernel.Ane.Half (Range (..), bf16ToFloat, bf16ToHalf, floatToHalf, halfBytes, halfToDouble)
import Hedgehog (forAll, property, (===))
import Hedgehog.Gen qualified as Gen
import Hedgehog.Range qualified as Range
import Test.Tasty (TestTree, testGroup)
import Test.Tasty.HUnit (testCase, (@?=))
import Test.Tasty.Hedgehog (testProperty)

-- | A finite binary16's value, exactly.
halfValue :: Word16 -> Float
halfValue h = sign * magnitude
 where
  sign = if h .&. 0x8000 /= 0 then -1 else 1
  e = fromIntegral ((h `shiftR` 10) .&. 0x1f) :: Int
  m = fromIntegral (h .&. 0x3ff) :: Float
  magnitude
    | e == 0 = m * 2 ^^ (-24 :: Int)
    | otherwise = (1 + m / 1024) * 2 ^^ (e - 15)

-- | bf16 bytes, little-endian, of the given bits.
bf16 :: [Word16] -> B.ByteString
bf16 = B.pack . concatMap (\w -> [fromIntegral (w .&. 0xff), fromIntegral (w `shiftR` 8)])

tests :: TestTree
tests =
  testGroup
    "Half"
    [ testCase "known values, rounded to nearest, ties to even" $ do
        floatToHalf 1 @?= 0x3c00
        floatToHalf (-2) @?= 0xc000
        floatToHalf 0.5 @?= 0x3800
        floatToHalf 65504 @?= 0x7bff
        floatToHalf 65520 @?= 0x7c00
        floatToHalf (2 ^^ (-24 :: Int)) @?= 0x0001
        floatToHalf (2 ^^ (-25 :: Int)) @?= 0x0000
        floatToHalf (3 * 2 ^^ (-25 :: Int)) @?= 0x0002
        floatToHalf (1 + 2 ^^ (-11 :: Int)) @?= 0x3c00
        floatToHalf (1 + 3 * 2 ^^ (-11 :: Int)) @?= 0x3c02
        floatToHalf (1 / 0) @?= 0x7c00
        floatToHalf (0 / 0) .&. 0x7e00 @?= 0x7e00
    , testProperty "every finite binary16 comes back as itself" $ property $ do
        h <- forAll (Gen.filter (\w -> w .&. 0x7c00 /= 0x7c00) (Gen.word16 Range.constantBounded))
        floatToHalf (halfValue h) === h
    , testProperty "a binary16 read back is its value" $ property $ do
        h <- forAll (Gen.filter (\w -> w .&. 0x7c00 /= 0x7c00) (Gen.word16 Range.constantBounded))
        halfToDouble h === realToFrac (halfValue h)
    , testCase "fp16 bytes are little-endian" $
        halfBytes [1, -2] @?= B.pack [0x00, 0x3c, 0x00, 0xc0]
    , testCase "bf16 to fp16 says what the trip did" $ do
        -- 1, 999424 (past fp16), 1e-8 (below its least), 1e-6 (subnormal), NaN.
        let (bytes, r) = bf16ToHalf (bf16 [0x3f80, 0x4974, 0x322c, 0x3586, 0x7fc0])
        B.take 4 bytes @?= B.pack [0x00, 0x3c, 0x00, 0x7c]
        (r.rangeCount, r.rangeOverflow, r.rangeFlushed, r.rangeSubnormal, r.rangeNonFinite)
          @?= (5, 1, 1, 1, 1)
        r.rangeMaxAbs @?= 999424
    , testCase "bf16 to f32 is exact" $
        bf16ToFloat (bf16 [0x3f80]) @?= B.pack [0x00, 0x00, 0x80, 0x3f]
    ]
