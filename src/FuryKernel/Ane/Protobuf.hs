-- SPDX-FileCopyrightText: 2026 Mika Tammi
-- SPDX-License-Identifier: BSD-3-Clause

{- | The protocol buffers wire format, encoding only, for the few Core ML
messages the target writes. A message is the concatenation of its fields;
nested messages and maps are length-delimited fields.
-}
module FuryKernel.Ane.Protobuf (
  Msg,
  encode,
  uint,
  int64,
  bool,
  string,
  bytes,
  message,
  packedFloats,
  packedVarints,
  mapEntry,
) where

import Data.Bits (shiftL, shiftR, (.&.), (.|.))
import Data.ByteString qualified as B
import Data.ByteString.Builder qualified as BB
import Data.ByteString.Lazy qualified as BL
import Data.Int (Int64)
import Data.Word (Word64)

-- | The encoded fields of a message.
type Msg = BB.Builder

encode :: Msg -> BL.ByteString
encode = BB.toLazyByteString

varint :: Word64 -> BB.Builder
varint n
  | n < 0x80 = BB.word8 (fromIntegral n)
  | otherwise = BB.word8 (fromIntegral (n .&. 0x7f) .|. 0x80) <> varint (n `shiftR` 7)

key :: Int -> Word64 -> BB.Builder
key field wire = varint (fromIntegral field `shiftL` 3 .|. wire)

-- | A varint field (uint32, uint64, int32 for non-negative values, enums).
uint :: Int -> Word64 -> Msg
uint f v = key f 0 <> varint v

-- | An int64 field: two's complement, as a ten-byte varint when negative.
int64 :: Int -> Int64 -> Msg
int64 f v = uint f (fromIntegral v)

bool :: Int -> Bool -> Msg
bool f b = uint f (if b then 1 else 0)

lengthDelimited :: Int -> BL.ByteString -> Msg
lengthDelimited f bs =
  key f 2 <> varint (fromIntegral (BL.length bs)) <> BB.lazyByteString bs

string :: Int -> String -> Msg
string f = lengthDelimited f . BB.toLazyByteString . BB.stringUtf8

bytes :: Int -> B.ByteString -> Msg
bytes f = lengthDelimited f . BL.fromStrict

message :: Int -> Msg -> Msg
message f = lengthDelimited f . encode

packedFloats :: Int -> [Float] -> Msg
packedFloats f = lengthDelimited f . BB.toLazyByteString . foldMap BB.floatLE

packedVarints :: Int -> [Word64] -> Msg
packedVarints f = lengthDelimited f . BB.toLazyByteString . foldMap varint

{- | One entry of a map field: the generated entry message has the key as
field 1 and the value as field 2.
-}
mapEntry :: Int -> Msg -> Msg -> Msg
mapEntry f k v = message f (k <> v)
