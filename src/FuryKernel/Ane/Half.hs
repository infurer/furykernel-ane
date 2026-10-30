-- SPDX-FileCopyrightText: 2026 Mika Tammi
-- SPDX-License-Identifier: BSD-3-Clause

{- | IEEE binary16, the ANE's compute type: conversion from bf16 and f32 with
round-to-nearest-even, and the range statistics of a conversion, which say
whether a tensor survives the trip into fp16.
-}
module FuryKernel.Ane.Half (
  floatToHalf,
  halfBytes,
  Range (..),
  bf16ToHalf,
  bf16ToFloat,
) where

import Data.Bits (shiftL, shiftR, testBit, (.&.), (.|.))
import Data.ByteString qualified as B
import Data.ByteString.Internal qualified as BI
import Data.ByteString.Unsafe qualified as BU
import Data.Word (Word16, Word32)
import Foreign.Ptr (Ptr, castPtr)
import Foreign.Storable (peekElemOff, pokeElemOff)
import GHC.Float (castFloatToWord32, castWord32ToFloat)
import System.IO.Unsafe (unsafeDupablePerformIO)

-- | The fp16 bits nearest to an f32, ties to even; overflow gives infinity.
floatBitsToHalf :: Word32 -> Word16
floatBitsToHalf x
  | expo == 0xff = sign .|. 0x7c00 .|. (if mant /= 0 then 0x200 else 0)
  | e >= 0x1f = sign .|. 0x7c00
  | e <= 0 =
      if e < -10
        then sign
        else
          let m = mant .|. 0x800000
              s = 14 - e
           in sign .|. fromIntegral (roundShift m s)
  | otherwise =
      sign .|. fromIntegral (roundShift ((fromIntegral e `shiftL` 23) .|. mant) 13)
 where
  sign = fromIntegral ((x `shiftR` 16) .&. 0x8000) :: Word16
  expo = fromIntegral ((x `shiftR` 23) .&. 0xff) :: Int
  mant = x .&. 0x7fffff
  e = expo - 127 + 15
  -- m >> s, rounded to nearest, ties to even; a carry into the exponent is
  -- the correct result (up to infinity).
  roundShift :: Word32 -> Int -> Word32
  roundShift m s =
    let q = m `shiftR` s
        r = m .&. ((1 `shiftL` s) - 1)
        h = 1 `shiftL` (s - 1)
     in if r > h || (r == h && testBit q 0) then q + 1 else q

floatToHalf :: Float -> Word16
floatToHalf = floatBitsToHalf . castFloatToWord32

-- | Little-endian fp16 bytes of a list of floats.
halfBytes :: [Float] -> B.ByteString
halfBytes xs = BI.unsafeCreate (2 * length xs) $ \p ->
  mapM_ (\(i, x) -> pokeElemOff (castPtr p :: Ptr Word16) i (floatToHalf x)) (zip [0 ..] xs)

-- | What converting a tensor to fp16 did to it.
data Range = Range
  { rangeCount :: !Int
  , rangeMaxAbs :: !Float
  -- ^ Largest finite magnitude of the source.
  , rangeOverflow :: !Int
  -- ^ Finite values that became infinite (above 65504).
  , rangeSubnormal :: !Int
  -- ^ Values that became fp16 subnormals (below 2^-14), losing precision.
  , rangeFlushed :: !Int
  -- ^ Nonzero values that became zero (below 2^-25).
  , rangeNonFinite :: !Int
  -- ^ NaN or infinity already in the source.
  }
  deriving stock (Show)

-- | Little-endian bf16 bytes to little-endian fp16 bytes, with the range.
bf16ToHalf :: B.ByteString -> (B.ByteString, Range)
bf16ToHalf src = unsafeDupablePerformIO $
  BU.unsafeUseAsCString src $ \cs -> do
    let n = B.length src `div` 2
        sp = castPtr cs :: Ptr Word16
    (out, r) <- BI.createUptoN' (2 * n) $ \dp -> do
      let dp16 = castPtr dp :: Ptr Word16
          go !i !mx !ov !sub !fl !nf
            | i >= n = pure (Range n mx ov sub fl nf)
            | otherwise = do
                b <- peekElemOff sp i
                let w = fromIntegral b `shiftL` 16 :: Word32
                    f = castWord32ToFloat w
                    a = abs f
                    h = floatBitsToHalf w
                    finiteSrc = not (isNaN f || isInfinite f)
                    habs = h .&. 0x7fff
                pokeElemOff dp16 i h
                go
                  (i + 1)
                  (if finiteSrc && a > mx then a else mx)
                  (if finiteSrc && habs == 0x7c00 then ov + 1 else ov)
                  (if habs /= 0 && habs < 0x0400 then sub + 1 else sub)
                  (if a /= 0 && finiteSrc && habs == 0 then fl + 1 else fl)
                  (if finiteSrc then nf else nf + 1)
      r <- go 0 0 0 0 0 0
      pure (2 * n, r)
    pure (out, r)

-- | Little-endian bf16 bytes to little-endian f32 bytes, exactly.
bf16ToFloat :: B.ByteString -> B.ByteString
bf16ToFloat src = unsafeDupablePerformIO $
  BU.unsafeUseAsCString src $ \cs -> do
    let n = B.length src `div` 2
        sp = castPtr cs :: Ptr Word16
    BI.create (4 * n) $ \dp -> do
      let dp32 = castPtr dp :: Ptr Word32
          go !i
            | i >= n = pure ()
            | otherwise = do
                b <- peekElemOff sp i
                pokeElemOff dp32 i (fromIntegral b `shiftL` 16)
                go (i + 1)
      go 0
