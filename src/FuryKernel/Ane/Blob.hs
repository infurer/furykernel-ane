-- SPDX-FileCopyrightText: 2026 Mika Tammi
-- SPDX-License-Identifier: BSD-3-Clause

{- | The MIL blob storage file (@weights/weight.bin@), version 2, after
coremltools' MILBlob/Blob/StorageFormat.hpp:

* a 64-byte header: the blob count (u32), version 2 (u32), seven reserved
  u64;
* per blob, a 64-byte metadata record (the sentinel 0xDEADBEEF, the data
  type, the size in bytes, the data's offset, five reserved u64), then the
  data, each 64-byte aligned.

A program refers to a blob by the offset of its metadata record. A blob's
bytes may be given, or only its size reserved: a model's weights, which a
runtime writes into the file from the model's own tensors, at the offsets
'blobsLayout' gives.
-}
module FuryKernel.Ane.Blob (
  BlobType (..),
  blobTypeCode,
  BlobData (..),
  Blobs,
  emptyBlobs,
  addBlob,
  reserveBlob,
  Placed (..),
  blobsLayout,
  blobsFile,
) where

import Data.ByteString qualified as B
import Data.ByteString.Builder qualified as BB
import Data.ByteString.Lazy qualified as BL
import Data.Word (Word32, Word64)

data BlobType = BlobFloat16 | BlobFloat32
  deriving stock (Eq, Show)

-- | The type's code in a metadata record.
blobTypeCode :: BlobType -> Word32
blobTypeCode = \case
  BlobFloat16 -> 1
  BlobFloat32 -> 2

-- | A blob's contents: its bytes, or a size reserved for bytes to come.
data BlobData = Given B.ByteString | Reserved Word64
  deriving stock (Eq, Show)

dataSize :: BlobData -> Word64
dataSize = \case
  Given bs -> fromIntegral (B.length bs)
  Reserved n -> n

-- | Blobs in file order, reversed, and the offset of the next one.
data Blobs = Blobs [(BlobType, BlobData)] !Word64
  deriving stock (Eq, Show)

emptyBlobs :: Blobs
emptyBlobs = Blobs [] 64

align64 :: Word64 -> Word64
align64 n = (n + 63) `div` 64 * 64

-- | Appends a blob of these bytes, returning the offset a program refers to it by.
addBlob :: BlobType -> B.ByteString -> Blobs -> (Word64, Blobs)
addBlob t bs = append t (Given bs)

-- | Reserves a blob of this many bytes, returning its offset.
reserveBlob :: BlobType -> Word64 -> Blobs -> (Word64, Blobs)
reserveBlob t n = append t (Reserved n)

append :: BlobType -> BlobData -> Blobs -> (Word64, Blobs)
append t d (Blobs acc next) =
  (next, Blobs ((t, d) : acc) (align64 (next + 64 + dataSize d)))

-- | Where a blob sits: its metadata record's offset, the data's, its type and size.
data Placed = Placed
  { placedRecord :: Word64
  , placedData :: Word64
  , placedType :: BlobType
  , placedSize :: Word64
  , placedBytes :: Maybe B.ByteString
  -- ^ Its bytes, if they were given.
  }
  deriving stock (Eq, Show)

-- | Every blob, in file order, where it sits.
blobsLayout :: Blobs -> [Placed]
blobsLayout (Blobs acc _) = go 64 (reverse acc)
 where
  go _ [] = []
  go off ((t, d) : rest) =
    Placed off (off + 64) t (dataSize d) (given d) : go (align64 (off + 64 + dataSize d)) rest
  given = \case
    Given bs -> Just bs
    Reserved _ -> Nothing

{- | The file, if every blob's bytes were given: a runtime writes the
reserved ones itself.
-}
blobsFile :: Blobs -> Either String BL.ByteString
blobsFile blobs@(Blobs acc _) = do
  placed <- traverse given (blobsLayout blobs)
  pure (BB.toLazyByteString (header <> foldMap record placed))
 where
  given p = case placedBytes p of
    Just bs -> Right (p, bs)
    Nothing -> Left ("the blob at " <> show (placedRecord p) <> " was only reserved")
  header =
    BB.word32LE (fromIntegral (length acc)) <> BB.word32LE 2 <> foldMap BB.word64LE (replicate 7 0)
  record (p, bs) =
    let end = placedData p + placedSize p
        pad = fromIntegral (align64 end - end)
     in BB.word32LE 0xDEADBEEF
          <> BB.word32LE (blobTypeCode (placedType p))
          <> BB.word64LE (placedSize p)
          <> BB.word64LE (placedData p)
          <> foldMap BB.word64LE (replicate 5 0)
          <> BB.byteString bs
          <> BB.byteString (B.replicate pad 0)
