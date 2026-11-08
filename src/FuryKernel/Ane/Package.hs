-- SPDX-FileCopyrightText: 2026 Mika Tammi
-- SPDX-License-Identifier: BSD-3-Clause

{- | The @.mlpackage@ directory Core ML compiles: a manifest naming the
specification and the weights directory, the encoded model, and the blob
file. Its identifiers are fixed, so the same model makes the same package.
-}
module FuryKernel.Ane.Package (
  packageManifest,
  writePackage,
) where

import Data.ByteString.Lazy qualified as BL
import FuryKernel.Ane.Blob (Blobs, blobsFile)
import FuryKernel.Ane.CoreML (Model, encodeModel)
import System.Directory (createDirectoryIfMissing)
import System.FilePath ((</>))

-- Fixed identifiers keep the output deterministic.
specId, weightsId :: String
specId = "5F4A0B6E-2A44-4C3B-9E3D-6B0F6B9E1A01"
weightsId = "5F4A0B6E-2A44-4C3B-9E3D-6B0F6B9E1A02"

-- | The package's @Manifest.json@.
packageManifest :: String
packageManifest =
  unlines
    [ "{"
    , "    \"fileFormatVersion\": \"1.0.0\","
    , "    \"itemInfoEntries\": {"
    , "        \"" <> weightsId <> "\": {"
    , "            \"author\": \"com.apple.CoreML\","
    , "            \"description\": \"CoreML Model Weights\","
    , "            \"name\": \"weights\","
    , "            \"path\": \"com.apple.CoreML/weights\""
    , "        },"
    , "        \"" <> specId <> "\": {"
    , "            \"author\": \"com.apple.CoreML\","
    , "            \"description\": \"CoreML Model Specification\","
    , "            \"name\": \"model.mlmodel\","
    , "            \"path\": \"com.apple.CoreML/model.mlmodel\""
    , "        }"
    , "    },"
    , "    \"rootModelIdentifier\": \"" <> specId <> "\""
    , "}"
    ]

{- | Writes the package into @dir@; it fails if a blob's bytes weren't
given, which only a runtime has.
-}
writePackage :: FilePath -> Model -> Blobs -> IO ()
writePackage dir model blobs = do
  weights <- either fail pure (blobsFile blobs)
  let core = dir </> "Data" </> "com.apple.CoreML"
  createDirectoryIfMissing True (core </> "weights")
  BL.writeFile (core </> "model.mlmodel") (encodeModel model)
  BL.writeFile (core </> "weights" </> "weight.bin") weights
  writeFile (dir </> "Manifest.json") packageManifest
