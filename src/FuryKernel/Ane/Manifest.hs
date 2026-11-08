-- SPDX-FileCopyrightText: 2026 Mika Tammi
-- SPDX-License-Identifier: BSD-3-Clause

{- | What a runtime needs to run a lowered stage: its inputs and what each
holds, its outputs, and its weights' blobs, each the part of a model's
tensor a runtime writes there.

Every tensor is fp16 (an index, int32) in Core ML's shape, which the
runtime lays out as a pixel buffer: the last axis its width, the others
its height, each row 64-byte aligned. The ANE's layout is @(B, C, 1, S)@:
a value of @w@ elements per row over @m@ rows is @[1, w, 1, m]@.

The inputs the host derives, it derives as "FuryKernel.Ane.Host" does,
from the row integers and tensors named here.

Its JSON ('renderManifest') names the version of its format first, 1, so
that a runtime refuses a manifest of a format it doesn't read.
-}
module FuryKernel.Ane.Manifest (
  Manifest (..),
  PartInfo (..),
  Input (..),
  InputKind (..),
  MirrorSpec (..),
  MirrorLayout (..),
  RopeTable (..),
  MaskSpec (..),
  MaskKeys (..),
  GatherSpec (..),
  SelectionSpec (..),
  Output (..),
  OutputType (..),
  WeightBlob (..),
  manifestJson,
  renderManifest,
) where

import Data.Text (Text)
import Data.Text qualified as T
import Data.Word (Word64)
import FuryKernel.Json (Json (..), renderJson)
import Numeric.Natural (Natural)

data Manifest = Manifest
  { manifestStage :: Text
  , manifestParts :: [PartInfo]
  , manifestInputs :: [Input]
  , manifestOutputs :: [Output]
  , manifestWeights :: [WeightBlob]
  , manifestWeightFile :: Word64
  -- ^ The blob file's size in bytes.
  }
  deriving stock (Eq, Show)

-- | A part, as the stage ran it.
data PartInfo = PartInfo
  { partInfoName :: Text
  , partInfoProgram :: Text
  , partInfoRows :: Natural
  }
  deriving stock (Eq, Show)

data Input = Input
  { inputName :: Text
  , inputShape :: [Word64]
  , inputKind :: InputKind
  }
  deriving stock (Eq, Show)

data InputKind
  = -- | An activation from the engine, @[1, w, 1, m]@.
    Activation
  | -- | A mirror's keys, @[1, kvHeads, positions, headDim]@.
    MirrorKeys MirrorSpec
  | -- | A mirror's values, @[1, kvHeads, headDim, positions]@.
    MirrorValues MirrorSpec
  | -- | A rotary embedding's cosines, @[1, 1, width, m]@.
    RopeCos RopeTable
  | -- | Its sines.
    RopeSin RopeTable
  | {- | What each row's queries see, @[1, 1, keys, m]@ (or @[1, 1, keys, 1]@,
    shared): 0 where a row sees a key, -30000 where it doesn't.
    -}
    Mask MaskSpec
  | -- | Rows of a tensor gathered and summed, @[1, w, 1, m]@.
    Gathered GatherSpec
  | {- | The pooling of an earlier part's rows, @[1, 1, rows, m]@: at
    @(i, r)@ the factor times how often row @r@'s integers name @i@.
    -}
    Selection SelectionSpec
  deriving stock (Eq, Ord, Show)

{- | A cache's mirror: one sequence's keys or values, as the cache it
mirrors holds them (a ring of its positions, or paged in blocks of a size,
from the first position), its heads, their width, and the row integers
that name the sequence (its slot or its block table).
-}
data MirrorSpec = MirrorSpec
  { mirrorLayout :: MirrorLayout
  , mirrorKvHeads :: Natural
  , mirrorHeadDim :: Natural
  , mirrorSequence :: Text
  }
  deriving stock (Eq, Ord, Show)

data MirrorLayout
  = -- | A ring of this many positions: position @p@ at @p mod length@.
    MirrorRing Natural
  | -- | This many positions from the first, of a cache paged in blocks of this size.
    MirrorLinear Natural Natural
  deriving stock (Eq, Ord, Show)

{- | A rotary embedding's angles for each row's positions (the row
integers named), over a head of @width@ in @axes@ equal parts, each turned
by the row's position of its axis: in a part of width @d@, element @i@'s
pair is @i mod (d/2)@, and pair @j@ turns by @position · theta^(-2j/d)@
if @j < rotated@, else not at all (cosine 1, sine 0).
-}
data RopeTable = RopeTable
  { ropeTheta :: Double
  , ropeRotated :: Natural
  , ropeAxes :: Natural
  , ropeWidth :: Natural
  , ropePositions :: Text
  }
  deriving stock (Eq, Ord, Show)

{- | Attention's mask: a row at its position @p@ (the row integers named),
with a reach @r@ (named) or none (@r = p@), sees the positions from @p −
window + 1@ (or the first) up to @max p r@.
-}
data MaskSpec = MaskSpec
  { maskPositions :: Text
  , maskReach :: Maybe Text
  , maskWindow :: Maybe Natural
  , maskKeys :: MaskKeys
  , maskShared :: Bool
  -- ^ One mask for every row: the first row's.
  }
  deriving stock (Eq, Ord, Show)

-- | The keys a mask is over.
data MaskKeys
  = {- | A ring mirror's positions: at index @j@, the position @t ≡ j@ (mod
    the length) in the length up to the row's last.
    -}
    RingKeys Natural
  | -- | A linear mirror's: position @j@ at @j@.
    LinearKeys Natural
  | -- | The step's own rows, each at its position (the row integers named).
    RowKeys Text
  deriving stock (Eq, Ord, Show)

{- | Per row, the sum of the tensor's rows its integers name (the row
integers named), times the factor: @[1, width, 1, m]@.
-}
data GatherSpec = GatherSpec
  { gatherTensor :: Text
  , gatherTokens :: Text
  , gatherFactor :: Double
  }
  deriving stock (Eq, Ord, Show)

-- | A pooling of an earlier part's rows, by the row integers named.
data SelectionSpec = SelectionSpec
  { selectionTokens :: Text
  , selectionFactor :: Double
  }
  deriving stock (Eq, Ord, Show)

data Output = Output
  { outputName :: Text
  , outputShape :: [Word64]
  , outputType :: OutputType
  , outputPart :: Text
  , outputValue :: Text
  }
  deriving stock (Eq, Show)

data OutputType = OutputHalf | OutputIndex
  deriving stock (Eq, Show)

{- | A weight's blob: where its metadata record and data are in the blob
file, its size in bytes, and what a runtime writes there: the tensor's
elements from @offset@, as many as the shape has, times the scale, in
fp16.
-}
data WeightBlob = WeightBlob
  { blobRecord :: Word64
  , blobData :: Word64
  , blobSize :: Word64
  , blobTensor :: Text
  , blobOffset :: Word64
  , blobShape :: [Word64]
  , blobScale :: Double
  }
  deriving stock (Eq, Show)

manifestJson :: Manifest -> Json
manifestJson m =
  JObject
    [ ("version", JInt 1)
    , ("stage", JString m.manifestStage)
    , ("element", JString "fp16")
    , ("parts", JArray (map part m.manifestParts))
    , ("inputs", JArray (map input m.manifestInputs))
    , ("outputs", JArray (map output m.manifestOutputs))
    , ("weight_file", JInt (fromIntegral m.manifestWeightFile))
    , ("weights", JArray (map weight m.manifestWeights))
    ]
 where
  part p =
    JObject
      [ ("name", JString p.partInfoName)
      , ("program", JString p.partInfoProgram)
      , ("rows", nat p.partInfoRows)
      ]
  input i = JObject ([("name", JString i.inputName), ("shape", dims i.inputShape)] <> kind i.inputKind)
  kind = \case
    Activation -> [("kind", JString "activation")]
    MirrorKeys s -> ("kind", JString "keys") : mirror s
    MirrorValues s -> ("kind", JString "values") : mirror s
    RopeCos r -> ("kind", JString "rope_cos") : rope r
    RopeSin r -> ("kind", JString "rope_sin") : rope r
    Mask s ->
      [ ("kind", JString "mask")
      , ("positions", JString s.maskPositions)
      , ("reach", maybe JNull JString s.maskReach)
      , ("window", maybe JNull nat s.maskWindow)
      , ("shared", JBool s.maskShared)
      ]
        <> case s.maskKeys of
          RingKeys n -> [("keys", JString "ring"), ("length", nat n)]
          LinearKeys n -> [("keys", JString "linear"), ("length", nat n)]
          RowKeys p -> [("keys", JString "rows"), ("key_positions", JString p)]
    Gathered g ->
      [ ("kind", JString "gathered")
      , ("tensor", JString g.gatherTensor)
      , ("tokens", JString g.gatherTokens)
      , ("factor", double g.gatherFactor)
      ]
    Selection s ->
      [ ("kind", JString "selection")
      , ("tokens", JString s.selectionTokens)
      , ("factor", double s.selectionFactor)
      ]
  mirror s =
    [ ("sequence", JString s.mirrorSequence)
    , ("kv_heads", nat s.mirrorKvHeads)
    , ("head_dim", nat s.mirrorHeadDim)
    ]
      <> case s.mirrorLayout of
        MirrorRing n -> [("layout", JString "ring"), ("length", nat n)]
        MirrorLinear n b -> [("layout", JString "linear"), ("length", nat n), ("block_size", nat b)]
  rope r =
    [ ("positions", JString r.ropePositions)
    , ("theta", double r.ropeTheta)
    , ("rotated", nat r.ropeRotated)
    , ("axes", nat r.ropeAxes)
    , ("width", nat r.ropeWidth)
    ]
  output o =
    JObject
      [ ("name", JString o.outputName)
      , ("shape", dims o.outputShape)
      , ("type", JString (case o.outputType of OutputHalf -> "fp16"; OutputIndex -> "int32"))
      , ("part", JString o.outputPart)
      , ("value", JString o.outputValue)
      ]
  weight w =
    JObject
      [ ("record", JInt (fromIntegral w.blobRecord))
      , ("data", JInt (fromIntegral w.blobData))
      , ("size", JInt (fromIntegral w.blobSize))
      , ("tensor", JString w.blobTensor)
      , ("offset", JInt (fromIntegral w.blobOffset))
      , ("shape", dims w.blobShape)
      , ("scale", double w.blobScale)
      ]
  dims = JArray . map (JInt . fromIntegral)
  nat = JInt . fromIntegral
  -- The shortest decimal that reads back as the same double.
  double = JString . T.pack . show

renderManifest :: Manifest -> Text
renderManifest = renderJson . manifestJson
