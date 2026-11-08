-- SPDX-FileCopyrightText: 2026 Mika Tammi
-- SPDX-License-Identifier: BSD-3-Clause
{-# LANGUAGE GHC2024 #-}
{-# LANGUAGE OverloadedRecordDot #-}
{-# LANGUAGE OverloadedStrings #-}
{-# LANGUAGE StrictData #-}

{- | The corpus of the Neural Engine conformance run (apps.ane-conformance):
stages lowered by 'lowerStage' at sizes Core ML places on the Neural
Engine, each with what the host derives for it and its outputs as the
core's evaluator and FuryKernel.Ane.Interp compute them in f64.
@cases OUT@ writes a directory @OUT/<case>@ for each:

* @model.mlpackage@, its weights written in from the case's tensors, as a
  runtime writes them;
* @inputs.txt@, a line per input, @name d0,d1,…@, and @<name>.f16@, the
  input's binary16 bits in C order, as FuryKernel.Ane.Host derives it;
* @outputs.txt@, a line per output: @name float d0,d1,…@, then
  @softcap@ for one made through a softcap or @model@ for one held to the
  interpreter's model of the Neural Engine; @name token d0,d1,… logits@,
  a token taken from the output @logits@; or @name token d0,d1,… exact@,
  a token whose logits tie exactly where the corpus makes them tie;
* @eval/<name>.f64@, the evaluator's output, the stage's meaning;
  @interp/<name>.f64@, the lowered program's in f64; and
  @model/<name>.f64@, the lowered program's as the Neural Engine computes
  it, as far as it was observed ('NeuralEngine'): binary64, little-endian,
  C order;
* @units.txt@: @ane@, then @cpu@ for a case also run on the CPU.

Every number a case is given is a binary16 number, so the references
see what the Neural Engine is given. Two kinds: 'Dyadic' ones, @k / 2^j@
of seven bits, as the tests', exact in every sum the evaluator makes, so
the evaluator and the interpreter agree to f64's rounding; and 'Full'
ones, of eleven, weights at @1 / √fan-in@ and activations at magnitudes
from @2^-6@ to @2^4@, as a model's are.
-}
module Main (main) where

import Control.Monad (foldM, forM_, unless, when)
import Data.Bits (shiftR, xor)
import Data.ByteString qualified as BS
import Data.ByteString.Builder qualified as BB
import Data.ByteString.Lazy qualified as BL
import Data.List (intercalate, sort, transpose, zip4)
import Data.Map.Strict (Map)
import Data.Map.Strict qualified as Map
import Data.Set qualified as Set
import Data.Text (Text)
import Data.Text qualified as T
import Data.Word (Word32, Word64)
import FuryKernel.Ane.Blob (BlobType (..), Blobs, Placed (..), addBlob, blobsLayout, emptyBlobs)
import FuryKernel.Ane.Half (floatToHalf, halfToDouble)
import FuryKernel.Ane.Host
import FuryKernel.Ane.Interp (Precision (..), interpretValuesWith, tensor, tensorElems)
import FuryKernel.Ane.Lower
import FuryKernel.Ane.Manifest qualified as M
import FuryKernel.Ane.Mil (Built (..))
import FuryKernel.Ane.Package (writePackage)
import FuryKernel.Ane.Stage
import FuryKernel.Program.Build qualified as B
import FuryKernel.Program.Check (checkedProgram)
import FuryKernel.Program.Graph (AttendTo (..), AttnSpec (..), CacheElem (..), CacheLayout (..), Node (..), Op (Input, Softcap), Output (..), Program (..), RopePairs (..), RopeSpec (..), Source (..), opValues, programSource, wholeRow)
import FuryKernel.Type (ScalarType (..))
import GHC.Clock (getMonotonicTime)
import Numeric.Natural (Natural)
import System.Directory (createDirectoryIfMissing)
import System.Environment (getArgs)
import System.FilePath ((</>))
import System.IO (hFlush, stdout)
import Test.FuryKernel.Ane.Stages

------------------------------------------------------------------------------
-- Numbers

-- | splitmix64's numbers from a seed.
randoms :: Word64 -> [Word64]
randoms = go
 where
  go s =
    let s' = s + 0x9E3779B97F4A7C15
        z1 = (s' `xor` (s' `shiftR` 30)) * 0xBF58476D1CE4E5B9
        z2 = (z1 `xor` (z1 `shiftR` 27)) * 0x94D049BB133111EB
     in (z2 `xor` (z2 `shiftR` 31)) : go s'

-- | Uniform in (0, 1).
uniforms :: Word64 -> [Double]
uniforms seed = [(fromIntegral (w `shiftR` 11) + 0.5) / 2 ^ (53 :: Int) | w <- randoms seed]

-- | Standard normal, by Box and Muller.
normals :: Word64 -> [Double]
normals seed = pairs (uniforms seed)
 where
  pairs (u : v : rest) = let r = sqrt (negate (2 * log u)) in r * cos (2 * pi * v) : r * sin (2 * pi * v) : pairs rest
  pairs _ = []

-- | Rounded to binary16 and back, as a runtime writes a value into an fp16 tensor.
toHalf :: Double -> Double
toHalf = halfToDouble . floatToHalf . realToFrac

-- | @k / denominator@ for @|k| <= 64@.
dyadic :: Word64 -> Double -> [Double]
dyadic seed denominator = [fromIntegral ((fromIntegral (w `shiftR` 33) :: Int) `mod` 129 - 64) / denominator | w <- randoms seed]

-- | How a case's numbers are drawn.
data Fill = Dyadic | Full
  deriving stock (Eq, Show)

fillName :: Fill -> String
fillName = \case
  Dyadic -> ""
  Full -> "_full"

-- | A linear layer's weights, of about unit gain over this fan-in.
weightsOf :: Fill -> Word64 -> Natural -> Natural -> [Double]
weightsOf fill seed fanIn n = take (fromIntegral n) $ case fill of
  Dyadic -> dyadic seed (2 ^^ (round (logBase 2 (64 * sqrt (fromIntegral fanIn :: Double))) :: Int))
  Full -> map (toHalf . (/ sqrt (fromIntegral fanIn))) (normals seed)

-- | A norm's gains, about 1.
gainsOf :: Fill -> Word64 -> Natural -> [Double]
gainsOf fill seed n = take (fromIntegral n) $ case fill of
  Dyadic -> map (+ 1) (dyadic seed 1024)
  Full -> map (toHalf . (+ 1) . (/ 16)) (normals seed)

-- | Activations about this large (a power of two).
actsOf :: Fill -> Word64 -> Double -> Natural -> [Double]
actsOf fill seed scale n = take (fromIntegral n) $ case fill of
  Dyadic -> map (* scale) (dyadic seed 64)
  Full -> map (toHalf . (* scale)) (normals seed)

-- | @m@ rows of @w@.
rowsOf :: Natural -> [Double] -> [[Double]]
rowsOf w = go
 where
  go [] = []
  go xs = let (r, rest) = splitAt (fromIntegral w) xs in r : go rest

------------------------------------------------------------------------------
-- Cases

-- | How a token output is held.
data Token
  = -- | To the evaluator's where the logits' two largest are apart by more than the bound allows.
    Logits Text
  | -- | To the evaluator's, the logits tying exactly as the corpus made them.
    Exactly

data Case = Case
  { caseName :: String
  , caseStage :: Stage
  , caseData :: StageData
  , caseTokens :: [(Text, Token)]
  , caseAne :: Bool
  -- ^ Run on the Neural Engine: all but the shapes only the CPU is held to.
  , caseCpu :: Bool
  -- ^ Run on the CPU.
  , caseLarge :: Bool
  {- ^ Of the minutes or the memory its references take: written for the
  run, not by @checks.ane-corpus@.
  -}
  , caseModel :: Bool
  {- ^ Its float outputs held to the interpreter's model of the Neural
  Engine, bit for bit: it pins what the Neural Engine was observed to
  compute, where f64 differs by design.
  -}
  }

------------------------------------------------------------------------------
-- A drafting step

{- | A drafting step Core ML places on the Neural Engine, its output layer
of two chunks. Large: its references hold 7 GiB.
-}
aneDraft :: Draft
aneDraft =
  Draft
    { draftVocab = 4096
    , draftEmbed = 256
    , draftEmbedScale = 16
    , draftState = 256
    , draftWidth = 512
    , draftHeadDim = 64
    , draftRingHeads = 8
    , draftRingKv = 2
    , draftRingLength = 1024
    , draftWindow = 512
    , draftSequences = 2
    , draftPagedHeads = 8
    , draftPagedKv = 2
    , draftBlocks = 64
    , draftBlockSize = 64
    , draftRowBlocks = 16
    , draftPositions = 1024
    , draftMlp = 2048
    , draftOut = 20000
    , draftSoftcap = 30
    , draftRows = 4
    , draftEps = 2 ^^ (-20 :: Int)
    }

{- | Four rows of the ring's second sequence, the one the mirrors hold, at
positions 700 to 703, their hidden states from @2^-6@ to 16. What the
caches hold where no row sees, before the first row's window or past the
last row's position, is large: keys eight times, and values 64 times, as
large as the rest, so that a mask that leaks shows.
-}
draftDataOf :: Fill -> Draft -> StageData
draftDataOf fill d =
  StageData
    { sdInputs = Map.fromList [("hidden", [actsOf fill (100 + fromIntegral r) scale d.draftState | (r, scale) <- zip [0 .. rows - 1] [2 ^^ (-6 :: Int), 2 ^^ (-2 :: Int), 1, 16]])]
    , sdRowInts =
        Map.fromList
          [ ("token", [[fromIntegral ((r * 7919 + 13) `mod` fromIntegral d.draftVocab)] | r <- [0 .. rows - 1]])
          , ("position", [[fromIntegral (position r)] | r <- [0 .. rows - 1]])
          , ("seen", [[fromIntegral (position r - 1)] | r <- [0 .. rows - 1]])
          , ("slot", replicate rows [1])
          , ("blocks", replicate rows (map fromIntegral table))
          ]
    , sdCaches =
        Map.fromList
          [ ("k_ring", ring 200 8)
          , ("v_ring", ring 202 64)
          , ("k_paged", paged 204 8)
          , ("v_paged", paged 206 64)
          ]
    , sdTensors =
        Map.fromList
          [ ("embed", actsOf fill 1 (1 / 16) (d.draftVocab * d.draftEmbed))
          , ("pre_proj", weightsOf fill 2 (d.draftEmbed + d.draftState) (d.draftWidth * (d.draftEmbed + d.draftState)))
          , ("l0.norm", gainsOf fill 3 d.draftWidth)
          , ("l0.q_proj", weightsOf fill 4 d.draftWidth (ring' * d.draftWidth))
          , ("l0.q_norm", map (/ 8) (gainsOf fill 5 d.draftHeadDim))
          , ("l0.o_proj", weightsOf fill 6 ring' (d.draftWidth * ring'))
          , ("l1.q_proj", weightsOf fill 7 d.draftWidth (paged' * d.draftWidth))
          , ("l1.o_proj", weightsOf fill 8 paged' (d.draftWidth * paged'))
          , ("l1.gate", weightsOf fill 9 d.draftWidth (d.draftMlp * d.draftWidth))
          , ("l1.up", weightsOf fill 10 d.draftWidth (d.draftMlp * d.draftWidth))
          , ("l1.down", weightsOf fill 11 d.draftMlp (d.draftWidth * d.draftMlp))
          , ("l1.scalar", [0.75])
          , ("final_norm", gainsOf fill 12 d.draftWidth)
          , ("head", weightsOf fill 13 d.draftWidth (d.draftOut * d.draftWidth))
          , ("post", weightsOf fill 14 d.draftWidth (d.draftState * d.draftWidth))
          ]
    }
 where
  rows = fromIntegral d.draftRows :: Int
  ring' = d.draftRingHeads * d.draftHeadDim
  paged' = d.draftPagedHeads * d.draftHeadDim
  position r = 700 + r :: Int
  -- What some row sees: from the first row's window, the ring layer's,
  -- up to the last row's position before its own.
  firstSeen = position 0 - fromIntegral d.draftWindow
  lastSeen = position (rows - 1) - 1
  -- Sixteen of the pool's 64 blocks.
  table = [3, 17, 5, 42, 8, 60, 1, 33, 21, 12, 50, 7, 29, 44, 14, 38] :: [Int]
  dim = fromIntegral d.draftHeadDim :: Int
  -- The second slot's position j at its ring's j: no row is past the
  -- ring's length. The first slot's no row sees.
  ring seed junk =
    let n = d.draftSequences * d.draftRingKv * d.draftRingLength * d.draftHeadDim
        at = [(s, j) | s <- [0 .. fromIntegral d.draftSequences - 1 :: Int], _ <- [1 .. d.draftRingKv], j <- [0 .. fromIntegral d.draftRingLength - 1], _ <- [1 .. dim]]
     in [if s == 1 && j >= firstSeen && j <= lastSeen then x else y | ((s, j), x, y) <- zip3 at (actsOf fill seed 1 n) (actsOf fill (seed + 1) junk n)]
  -- A block of the table holds its positions; the paged layer has no
  -- window, and the blocks the table doesn't name no row sees.
  paged seed junk =
    let n = d.draftBlocks * d.draftBlockSize * d.draftPagedKv * d.draftHeadDim
        size = fromIntegral d.draftBlockSize :: Int
        at = [(b, i) | b <- [0 .. fromIntegral d.draftBlocks - 1], _ <- [1 .. d.draftPagedKv], i <- [0 .. size - 1], _ <- [1 .. dim]]
        seenAt b i = or [k * size + i <= lastSeen | (k, b') <- zip [0 :: Int ..] table, b' == b]
     in [if seenAt b i then x else y | ((b, i), x, y) <- zip3 at (actsOf fill seed 1 n) (actsOf fill (seed + 1) junk n)]

draftCase :: Fill -> Case
draftCase fill =
  Case
    { caseName = "draft_step" <> fillName fill
    , caseStage = draftStageOf aneDraft
    , caseData = draftDataOf fill aneDraft
    , caseTokens = [("next_token", Logits "logits")]
    , caseAne = True
    , caseCpu = False
    , caseLarge = True
    , caseModel = False
    }

------------------------------------------------------------------------------
-- An image's tower

-- | The tests' tower, 384 wide over 128 rows.
aneTower :: Tower
aneTower =
  Tower
    { towerPatch = 192
    , towerPlaces = 64
    , towerWidth = 384
    , towerHeads = 6
    , towerHeadDim = 64
    , towerAxisBound = 16
    , towerAhead = 127
    , towerMlp = Nothing
    , towerLayers = 2
    , towerRows = 128
    , towerSequences = 1
    , towerPoolBy = 4
    , towerPooled = 32
    , towerStdFactor = 2
    , towerOut = 1024
    , towerEps = 2 ^^ (-20 :: Int)
    }

-- | A layer of Gemma 4's larger image tower: 1152 wide, 16 heads of 72, an MLP of 4304, 256 rows.
gemmaTower :: Tower
gemmaTower =
  Tower
    { towerPatch = 768
    , towerPlaces = 64
    , towerWidth = 1152
    , towerHeads = 16
    , towerHeadDim = 72
    , towerAxisBound = 64
    , towerAhead = 255
    , towerMlp = Just 4304
    , towerLayers = 1
    , towerRows = 256
    , towerSequences = 1
    , towerPoolBy = 9
    , towerPooled = 28
    , towerStdFactor = 1
    , towerOut = 2816
    , towerEps = 2 ^^ (-20 :: Int)
    }

{- | An image of @real@ patches over the tower's rows, on a grid @side@
patches wide: rows past the last real one are padding, holding the first
patch's pixels, at their own positions.
-}
towerDataOf :: Fill -> Tower -> Int -> Int -> StageData
towerDataOf fill t real side =
  StageData
    { sdInputs = Map.fromList [("patches", take m (rowsOf t.towerPatch (actsOf fill 20 1 (fromIntegral real * t.towerPatch)) <> repeat (take (fromIntegral t.towerPatch) (actsOf fill 20 1 t.towerPatch))))]
    , sdRowInts =
        Map.fromList
          [ ("places", [[fromIntegral (i `mod` side), fromIntegral (half + (i `div` side) `mod` half)] | i <- [0 .. m - 1]])
          , ("patch", [[fromIntegral i] | i <- [0 .. m - 1]])
          , ("axes", [if i < real then [fromIntegral (i `mod` side), fromIntegral (i `div` side)] else [0, 0] | i <- [0 .. m - 1]])
          , ("reach", [[fromIntegral (max (real - 1) i)] | i <- [0 .. m - 1]])
          , ("slot", replicate m [0])
          , ("squares", [[fromIntegral (min (real - 1) (by * p + j)) | j <- [0 .. by - 1]] | p <- [0 .. fromIntegral t.towerPooled - 1]])
          ]
    , sdCaches = Map.empty
    , sdTensors =
        Map.fromList $
          [ ("patch_proj", weightsOf fill 21 t.towerPatch (t.towerWidth * t.towerPatch))
          , ("position_table", actsOf fill 22 (1 / 4) (t.towerPlaces * t.towerWidth))
          , ("std_bias", actsOf fill 23 (1 / 16) t.towerWidth)
          , ("std_scale", gainsOf fill 24 t.towerWidth)
          , ("projector", weightsOf fill 25 t.towerWidth (t.towerOut * t.towerWidth))
          ]
            <> concat [layer (T.pack ("l" <> show i <> ".")) (100 + 20 * fromIntegral i) | i <- [0 .. t.towerLayers - 1]]
    }
 where
  m = fromIntegral t.towerRows :: Int
  by = fromIntegral t.towerPoolBy :: Int
  half = fromIntegral t.towerPlaces `div` 2 :: Int
  inner = t.towerHeads * t.towerHeadDim
  wd = t.towerWidth
  layer l s =
    [ (l <> "norm", gainsOf fill (s + 1) wd)
    , (l <> "q_proj", weightsOf fill (s + 2) wd (inner * wd))
    , (l <> "k_proj", weightsOf fill (s + 3) wd (inner * wd))
    , (l <> "v_proj", weightsOf fill (s + 4) wd (inner * wd))
    , (l <> "q_norm", map (/ 2) (gainsOf fill (s + 5) t.towerHeadDim))
    , (l <> "k_norm", map (/ 2) (gainsOf fill (s + 6) t.towerHeadDim))
    , (l <> "o_proj", weightsOf fill (s + 7) inner (wd * inner))
    ]
      <> case t.towerMlp of
        Nothing -> []
        Just mlp ->
          [ (l <> "post_norm", gainsOf fill (s + 8) wd)
          , (l <> "mlp_norm", gainsOf fill (s + 9) wd)
          , (l <> "gate", weightsOf fill (s + 10) wd (mlp * wd))
          , (l <> "up", weightsOf fill (s + 11) wd (mlp * wd))
          , (l <> "down", weightsOf fill (s + 12) mlp (wd * mlp))
          , (l <> "mlp_post_norm", gainsOf fill (s + 13) wd)
          ]

towerCase :: String -> Tower -> Fill -> Int -> Int -> Bool -> Case
towerCase name t fill real side large =
  Case
    { caseName = name <> fillName fill
    , caseStage = towerStageOf t
    , caseData = towerDataOf fill t real side
    , caseTokens = []
    , caseAne = True
    , caseCpu = False
    , caseLarge = large
    , caseModel = False
    }

------------------------------------------------------------------------------
-- Edges

-- | An epsilon of Gemma 4's size, exact in binary16.
smallEps :: Double
smallEps = 2 ^^ (-20 :: Int)

-- | One part of one program fed by the host and the case's tensors, and its outputs.
single :: String -> Natural -> [(Text, Feed)] -> [Text] -> B.P () -> Stage
single name m feeds outs body =
  Stage
    { stageName = T.pack name
    , stageParts = [Part "p" (checked (B.program (T.pack ("e_" <> name)) body)) m feeds]
    , stageOutputs = [(o, "p", o) | o <- outs]
    }

-- | A row of a norm's input.
data NormRow
  = -- | Every element this.
    Same Double
  | -- | Normal, of this deviation.
    Normal Double
  | -- | From 28000 to 32000, signs alternating.
    Near32768
  | -- | Alternately this and its negation.
    Alternating Double
  | -- | This at element 5, and the deviation of the normal others.
    Peak Double Double

{- | Rows of a 1024-wide RMSNorm, as many as there are of them: binary16
numbers of a full mantissa, @k@ the row.
-}
normRowsOf :: [NormRow] -> [[Double]]
normRowsOf = zipWith row [0 ..]
 where
  w = 1024
  row :: Word64 -> NormRow -> [Double]
  row k = \case
    Same c -> replicate w c
    Normal scale -> actsOf Full (40 + k) scale (fromIntegral w)
    Near32768 -> [toHalf (sign * (28000 + 4000 * u)) | (sign, u) <- take w (zip (cycle [1, -1]) (uniforms (40 + k)))]
    Alternating c -> take w (cycle [c, -c])
    Peak p scale -> [if i == 5 then p else x | (i, x) <- zip [0 :: Int ..] (actsOf Full (40 + k) scale (fromIntegral w))]

{- | RMSNorm over 256 rows, these kinds in turn, and its output projected,
as a model's is, which also makes the program one Core ML places on the
Neural Engine.
-}
normCase :: String -> [NormRow] -> Case
normCase name kinds =
  Case
    { caseName = name
    , caseStage =
        single name 256 [("x", Host "x"), ("g", Tensor "g"), ("w", Tensor "w")] ["n", "y"] $ do
          x <- B.input "x" (wholeRow 1024) TF32
          g <- B.weight "g" [1024]
          w <- B.weight "w" [1024, 1024]
          n <- B.rmsnorm "n" smallEps (Just g) 1 x
          y <- B.linear "y" w 0 n
          B.output "n" n
          B.output "y" y
    , caseData =
        StageData
          { sdInputs = Map.fromList [("x", normRowsOf (take 256 (cycle kinds)))]
          , sdRowInts = Map.empty
          , sdCaches = Map.empty
          , sdTensors = Map.fromList [("g", gainsOf Full 30 1024), ("w", weightsOf Full 31 1024 (1024 * 1024))]
          }
    , caseTokens = []
    , caseAne = True
    , caseCpu = True
    , caseLarge = False
    , caseModel = False
    }

{- | RMSNorm over rows of the magnitudes the Neural Engine keeps to eight
bits or more: zero, then from @2^-8@ up to 4096 in steps, from 28000 to
32000, constants and alternating @±2^-6@. Below @2^-8@ it keeps fewer:
values under @2^-6@ on a grid of @2^-16@.
-}
normRows :: Case
normRows =
  normCase
    "norm_rows"
    ( [Same 0]
        <> [Normal (2 ^^ e) | e <- [-8, -7, -6, -5, -4, -2, 0, 4, 8, 12 :: Int]]
        <> [Near32768, Same 1, Same (-3), Alternating (2 ^^ (-6 :: Int)), Normal (1 / 3)]
    )

{- | Rows whose square sum is one element's: 20000, -30000, 1 and @2^-6@
among numbers of @2^-8@, @2^-6@ or zero, each row's peak 32 times its
root mean square.
-}
normPeaks :: Case
normPeaks = normCase "norm_peaks" [Peak 20000 (2 ^^ (-8 :: Int)), Peak (-30000) (2 ^^ (-6 :: Int)), Peak 1 0, Peak (2 ^^ (-6 :: Int)) 0]

{- | Each head's RMSNorm, of these heads, head width and rows: the shapes
whose norm Core ML's CPU computed wrongly as a concat normalised along
axis 2, now along the last.
-}
headNorm :: Natural -> Natural -> Natural -> Bool -> Case
headNorm h d m onAne =
  Case
    { caseName = "head_norm_" <> show h <> "x" <> show d
    , caseStage =
        single "head_norm" m [("x", Host "x"), ("w", Tensor "w"), ("g", Tensor "g")] ["y"] $ do
          x <- B.input "x" (wholeRow (h * d)) TF32
          w <- B.weight "w" [h * d, h * d]
          g <- B.weight "g" [d]
          q <- B.linear "q" w d x
          y <- B.rmsnorm "y" smallEps (Just g) 1 q
          B.output "y" y
    , caseData =
        StageData
          { sdInputs = Map.fromList [("x", rowsOf (h * d) (actsOf Full 50 1 (m * h * d)))]
          , sdRowInts = Map.empty
          , sdCaches = Map.empty
          , sdTensors = Map.fromList [("w", weightsOf Full 51 (h * d) (h * d * h * d)), ("g", gainsOf Full 52 d)]
          }
    , caseTokens = []
    , caseAne = onAne
    , caseCpu = True
    , caseLarge = False
    , caseModel = False
    }

{- | Attention of 16 rows over a ring's mirror of 16384 positions, the
most the Neural Engine takes, each row seeing the positions up to the
one this gives it (@seen@, the row): keys of unit deviation, and values
each head's common vector with a quarter of that in noise, so that
attention's average keeps its size. Past @hidden@, if given, what the
cache holds no row sees: keys eight times and values 64 times as large.
(Core ML places it on the Neural Engine from about 16 such rows on, the
program's size.) Large: the core's evaluator, keeping a bound for every
product, holds 15 to 40 GiB over it.
-}
mirrorCase :: String -> (Natural -> Natural) -> Maybe Natural -> Case
mirrorCase name seen hidden =
  Case
    { caseName = name
    , caseStage =
        single name 16 feeds ["y"] $ do
          x <- B.input "x" (wholeRow 512) TF32
          pos <- B.rowInts "position" 1 len
          seenBy <- B.rowInts "seen" 1 len
          slot <- B.rowInts "slot" 1 1
          wq <- B.weight "q_proj" [512, 512]
          wo <- B.weight "o_proj" [512, 512]
          k <- B.cacheOf CacheF16 "k_cache" (Ring 1 2 len 64)
          v <- B.cacheOf CacheF16 "v_cache" (Ring 1 2 len 64)
          q <- B.linear "q" wq 64 x
          qr <- B.rope "qr" (RopeSpec 1000000 16 1 RopeHalves) pos q
          att <- B.attend "att" (AttnSpec 8 2 64 Nothing 0.125 Nothing) qr AttendTo {attendKeys = k, attendValues = v, attendPositions = seenBy, attendSequences = slot}
          y <- B.linear "y" wo 0 att
          B.output "y" y
    , caseData =
        StageData
          { sdInputs = Map.fromList [("x", rowsOf 512 (actsOf Full 60 1 (16 * 512)))]
          , sdRowInts = Map.fromList [("position", [[fromIntegral (seen r)] | r <- [0 .. 15]]), ("seen", [[fromIntegral (seen r)] | r <- [0 .. 15]]), ("slot", replicate 16 [0])]
          , sdCaches = Map.fromList [("k", keys 61), ("v", values 63)]
          , sdTensors = Map.fromList [("q_proj", weightsOf Full 65 512 (512 * 512)), ("o_proj", weightsOf Full 66 512 (512 * 512))]
          }
    , caseTokens = []
    , caseAne = True
    , caseCpu = False
    , caseLarge = True
    , caseModel = False
    }
 where
  len = 16384
  feeds = [("x", Host "x"), ("position", Host "position"), ("seen", Host "seen"), ("slot", Host "slot"), ("q_proj", Tensor "q_proj"), ("o_proj", Tensor "o_proj"), ("k_cache", Mirror "k" len), ("v_cache", Mirror "v" len)]
  shown j = maybe True (j <) hidden
  keys seed = [if shown j then x else y | (j, x, y) <- zip3 positions (actsOf Full seed 1 n) (actsOf Full (seed + 1) 8 n)]
  values seed =
    concat
      [ [if shown j then toHalf (c + x / 4) else y | (j, c, x, y) <- zip4 headPositions (cycle common) (actsOf Full (seed + 10 + h) 1 n2) (actsOf Full (seed + 20 + h) 64 n2)]
      | h <- [0, 1]
      , let common = actsOf Full (seed + 30 + h) 1 64
      ]
  n = 2 * len * 64
  n2 = len * 64
  headPositions = [j | j <- [0 .. len - 1], _ <- [1 .. 64 :: Int]]
  positions = [j | _ <- [1 .. 2 :: Int], j <- headPositions]

-- | Rows seeing 512 to 8192 positions, in steps of 512, and what's past them large.
maskedMirror :: Case
maskedMirror = mirrorCase "masked_mirror" (\r -> 511 + 512 * r) (Just 8192)

-- | Rows seeing 16369 to 16384 positions: attention at the length the gate allows.
fullMirror :: Case
fullMirror = mirrorCase "full_mirror" (16368 +) Nothing

{- | An output layer of 32770 rows, so three chunks of 16384, 16384 and 2,
and rows whose largest logits tie exactly: across the first two chunks'
boundary (16383 and 16384), across all three (5, 16389 and 32768), alone
in the last chunk (32769), and within the first (7 and 9). Row @r@ of
@x@ is one on its @r@th quarter, so the norm makes it two there; a tying
row of the layer is 1/16 on that quarter, making its logit 8, where the
others, of 1 / √256, make about one.
-}
argmaxTies :: Case
argmaxTies =
  Case
    { caseName = "argmax_ties"
    , caseStage =
        Stage
          { stageName = "argmax_ties"
          , stageParts =
              [ Part
                  "p"
                  ( checked . B.program "e_argmax_ties" $ do
                      x <- B.input "x" (wholeRow 256) TF32
                      w <- B.weight "head" [vocab, 256]
                      n <- B.rmsnorm "n" smallEps Nothing 1 x
                      logits <- B.linear "logits" w 0 n
                      token <- B.argmax "token" logits
                      B.output "token" token
                  )
                  4
                  [("x", Host "x"), ("head", Tensor "head")]
              ]
          , stageOutputs = [("token", "p", "token")]
          }
    , caseData =
        StageData
          { sdInputs = Map.fromList [("x", [[if c `div` 64 == r then 1 else 0 | c <- [0 .. 255 :: Int]] | r <- [0 .. 3]])]
          , sdRowInts = Map.empty
          , sdCaches = Map.empty
          , sdTensors = Map.fromList [("head", concat (zipWith headRow [0 ..] noise))]
          }
    , caseTokens = [("token", Exactly)]
    , caseAne = True
    , caseCpu = False
    , caseLarge = False
    , caseModel = False
    }
 where
  vocab = 32770 :: Natural
  ties = [[16383, 16384], [5, 16389, 32768], [32769], [7, 9]] :: [[Int]]
  noise = rowsOf 256 (weightsOf Full 70 256 (vocab * 256))
  headRow v other = case [r | (r, t) <- zip [0 ..] ties, v `elem` t] of
    r : _ -> [if c `div` 64 == r then 1 / 16 else 0 | c <- [0 .. 255 :: Int]]
    [] -> other

{- | A softcap of 30 over logits of 256 rows, their magnitudes in turn
from a quarter of the cap to eight times it, so that the tanh saturates
in most: a projection of rows of @1/4@ to 64 of unit gain.
-}
softcap :: Case
softcap =
  Case
    { caseName = "softcap"
    , caseStage =
        single "softcap" 256 [("x", Host "x"), ("w", Tensor "w")] ["y"] $ do
          x <- B.input "x" (wholeRow 1024) TF32
          w <- B.weight "w" [2048, 1024]
          l <- B.linear "l" w 0 x
          y <- B.softcap "y" 30 l
          B.output "y" y
    , caseData =
        StageData
          { sdInputs = Map.fromList [("x", [actsOf Full (400 + fromIntegral r) scale 1024 | (r, scale) <- zip [0 :: Int ..] (take 256 (cycle [1 / 4, 4, 16, 32, 64]))])]
          , sdRowInts = Map.empty
          , sdCaches = Map.empty
          , sdTensors = Map.fromList [("w", weightsOf Full 410 1024 (2048 * 1024))]
          }
    , caseTokens = []
    , caseAne = True
    , caseCpu = False
    , caseLarge = False
    , caseModel = False
    }

{- | A linear layer and its residual over 256 rows of one end of the
Neural Engine's range, @(name, scales, off)@: rows of these magnitudes
in turn, and a layer of half the identity and @off@ times weights of
@1 / √1024@ elsewhere. At the low end, rows of @2^-7@ to @2^-5@, where
it keeps values coarsely, their products with the weights above its
@2^-16@; at the high end, rows of 20000, whose sum with the layer's half
of them is about 30000, below the 32768 it can't hold.
-}
rangeCase :: String -> [Double] -> Double -> Case
rangeCase name scales off =
  Case
    { caseName = name
    , caseStage =
        single name 256 [("x", Host "x"), ("w", Tensor "w")] ["y"] $ do
          x <- B.input "x" (wholeRow 1024) TF32
          w <- B.weight "w" [1024, 1024]
          h <- B.linear "h" w 0 x
          y <- B.add "y" x h
          B.output "y" y
    , caseData =
        StageData
          { sdInputs = Map.fromList [("x", zipWith row [0 ..] (take 256 (cycle scales)))]
          , sdRowInts = Map.empty
          , sdCaches = Map.empty
          , sdTensors = Map.fromList [("w", [toHalf (if i == j then 0.5 else off * e) | (i, j, e) <- zip3 (concatMap (replicate 1024) [0 .. 1023 :: Int]) (cycle [0 .. 1023]) (weightsOf Full 80 1024 (1024 * 1024))])]
          }
    , caseTokens = []
    , caseAne = True
    , caseCpu = False
    , caseLarge = False
    , caseModel = False
    }
 where
  -- Of one magnitude, signs mixed: within 5% of it.
  row :: Word64 -> Double -> [Double]
  row r scale = [toHalf (s * scale * (0.95 + 0.05 * u)) | (s, u) <- take 1024 (zip (cycle [1, -1]) (uniforms (90 + r)))]

rangeLow, rangeHigh :: Case
rangeLow = rangeCase "range_low" [2 ^^ (-7 :: Int), 2 ^^ (-6 :: Int), 2 ^^ (-5 :: Int), 1] 1
rangeHigh = rangeCase "range_high" [1, 4096, 20000] (1 / 64)

{- | Sums whose terms are far larger than the sum, their running sums
below 32768: every weight is @2^-10@, and each row of @x@ holds @a@ and
@-a@ equally often (or 1 for one @a@), by halves, alternately or in an
order of its own, for @a@ of 4000, 8000 and 15000, so that by halves a
sum reaches @2a@. (Past 32768 the Neural Engine's sums overflow, as its
values do, whatever they come to.)
-}
longSums :: Case
longSums =
  Case
    { caseName = "long_sums"
    , caseStage =
        single "long_sums" 64 [("x", Host "x"), ("w", Tensor "w")] ["y"] $ do
          x <- B.input "x" (wholeRow 4096) TF32
          w <- B.weight "w" [1024, 4096]
          y <- B.linear "y" w 0 x
          B.output "y" y
    , caseData =
        StageData
          { sdInputs = Map.fromList [("x", [kind r | r <- [0 .. 63 :: Int]])]
          , sdRowInts = Map.empty
          , sdCaches = Map.empty
          , sdTensors = Map.fromList [("w", replicate (1024 * 4096) (2 ^^ (-10 :: Int)))]
          }
    , caseTokens = []
    , caseAne = True
    , caseCpu = False
    , caseLarge = False
    , caseModel = False
    }
 where
  kind r =
    let a = [4000, 8000, 15000] !! (r `mod` 3)
     in case (r `div` 3) `mod` 4 of
          0 -> [if c < 2048 then a else -a | c <- [0 .. 4095 :: Int]]
          1 -> take 4096 (cycle [a, -a])
          2 -> [if c < 2047 then a else if c == 2047 then 1 else -a | c <- [0 .. 4095 :: Int]]
          -- Equally many of each, in the order of the row's own random numbers.
          _ ->
            let us = take 4096 (uniforms (fromIntegral (300 + r)))
                median = sort us !! 2048
             in [if u < median then a else -a | u <- us]

------------------------------------------------------------------------------
-- Writing

{- | A stage's outputs made through a softcap: by one, or from what one
made, in their part or an earlier one.
-}
softcapped :: Stage -> [Text]
softcapped st = [o | (o, pn, on) <- st.stageOutputs, (pn, on) `Set.member` through]
 where
  through = foldl' part Set.empty st.stageParts
  part acc pt =
    let prog = checkedProgram pt.partProgram
        made = foldl' (node acc pt prog) Set.empty prog.programNodes
     in foldr Set.insert acc [(pt.partName, o.outputName) | o <- prog.programOutputs, o.outputValue `Set.member` made]
  node earlier pt prog made n =
    let fromSoftcap = case n.nodeOp of
          Softcap {} -> True
          Input src -> case programSource prog src >>= partFeed pt . (.sourceName) of
            Just (Earlier pn on) -> (pn, on) `Set.member` earlier
            _ -> False
          op -> any (`Set.member` made) (opValues op)
     in if fromSoftcap then foldr Set.insert made n.nodeOutputs else made

{- | Attention of 16 rows over a ring's mirror of 16384 positions, all of
which they see, the keys alike, so that every probability is 2^-14: the
values each head's common vector, its elements of magnitudes from 2^-3 to
2^-1, a sixteenth of that in noise. Every product of a value with a
probability is within four steps of the 2^-16 grid the Neural Engine
sums on, or under it, dropped: the lowering's scaling of the
probabilities by 2^8 is what keeps them. Large, as the other mirrors.
-}
flatMirror :: Case
flatMirror =
  base
    { caseName = "flat_mirror"
    , caseData =
        baseData
          { sdRowInts = Map.fromList [("position", replicate 16 [len - 1]), ("seen", replicate 16 [len - 1]), ("slot", replicate 16 [0])]
          , sdCaches = Map.fromList [("k", keys), ("v", values)]
          }
    }
 where
  base = fullMirror
  baseData = base.caseData
  len = 16384 :: Word32
  keys = concat [concat (replicate (fromIntegral len) (actsOf Full (70 + h) 1 64)) | h <- [0, 1]]
  values =
    concat
      [ [toHalf (c * (1 + z / 16)) | (c, z) <- zip (cycle common) (take (fromIntegral len * 64) (normals (75 + h)))]
      | h <- [0, 1 :: Word64]
      , let common = [toHalf (sg * (0.125 + 0.375 * u)) | (sg, u) <- zip (cycle [1, -1, -1, 1]) (take 64 (uniforms (72 + h)))]
      ]

{- | A linear layer's products where the Neural Engine's sums keep them on
a grid of 2^-16, or drop them: weights of exponent -8 and random
mantissas, of 1.5·2^-9, and of 1.5·2^-8, by thirds of fan-in 1024 or so;
rows by fours of inputs of exponents -8 to -10 (products of exponent sums
-16 to -18, all but the first dropped), of 2^-7 (products of 1.5·2^-16,
ties, rounded away from zero), of 1.5·2^-9 (sums of -17 and less: every
product dropped, though some are over 2^-16), and of exponent -6, signs
random. Held to the model, from which f64 differs by design.
-}
productsGrid :: Case
productsGrid =
  linearCase "products_grid" [row r | r <- [0 .. 255]] ws
 where
  ws = [toHalf (sg * w o i) | ((o, i), sg) <- zip [(o, i) | o <- [0 .. 1023 :: Int], i <- [0 .. 1023 :: Int]] (signs 81)]
  w o i
    | i < 512 = (1 + mantissa (o * 1024 + i)) * 2 ^^ (-8 :: Int)
    | i < 768 = 1.5 * 2 ^^ (-9 :: Int)
    | otherwise = 1.5 * 2 ^^ (-8 :: Int)
  row r = [toHalf (sg * x r i) | (i, sg) <- zip [0 .. 1023 :: Int] (signs (82 + fromIntegral r))]
  x r i = case r `mod` 4 of
    0 -> (1 + mantissa (r * 1024 + i)) * 2 ^^ negate (8 + i `mod` 3)
    1 -> 2 ^^ (-7 :: Int)
    2 -> 1.5 * 2 ^^ (-9 :: Int)
    _ -> (1 + mantissa (r * 1024 + i)) * 2 ^^ (-6 :: Int)
  mantissa k = fromIntegral ((k * 2654435761 + 40503) `mod` 1024) / 1024
  signs seed = [if u < 0.5 then 1 else -1 | u <- uniforms seed]

{- | Sums whose running sums pass 32768 on the way to results that fit:
weights of 30 to 34 by output, inputs of ±1 and ±2 in halves, quarters,
and +2 then -2 at breakpoints from 400 to 647, so that a row's peak
@2bc@ crosses 32768 along its outputs. The Neural Engine gives infinity
wherever a partial sum passes it, as the model does. Held to the model.
-}
prefixOverflow :: Case
prefixOverflow =
  linearCase "prefix_overflow" [row r | r <- [0 .. 255]] [toHalf (30 + fromIntegral o * 4 / 1024) | o <- [0 .. 1023 :: Int], _ <- [0 .. 1023 :: Int]]
 where
  row :: Int -> [Double]
  row = \case
    0 -> replicate 1024 1
    1 -> replicate 1024 (-1)
    2 -> replicate 512 1 <> replicate 512 (-1)
    3 -> replicate 768 1 <> replicate 256 (-1)
    4 -> replicate 640 2 <> replicate 384 (-2)
    5 -> take 1024 (cycle [2, -2])
    6 -> replicate 384 (-2) <> replicate 640 2
    r -> let b = 400 + (r - 7) in replicate b 2 <> replicate (1024 - b) (-2)

-- | A linear layer of 1024 inputs to 1024 outputs over 256 rows, held to the model.
linearCase :: String -> [[Double]] -> [Double] -> Case
linearCase name xs ws =
  Case
    { caseName = name
    , caseStage =
        single name 256 [("x", Host "x"), ("w", Tensor "w")] ["y"] $ do
          x <- B.input "x" (wholeRow 1024) TF32
          w <- B.weight "w" [1024, 1024]
          y <- B.linear "y" w 0 x
          B.output "y" y
    , caseData =
        StageData
          { sdInputs = Map.fromList [("x", xs)]
          , sdRowInts = Map.empty
          , sdCaches = Map.empty
          , sdTensors = Map.fromList [("w", ws)]
          }
    , caseTokens = []
    , caseAne = True
    , caseCpu = False
    , caseLarge = False
    , caseModel = True
    }

corpus :: [Case]
corpus =
  [draftCase Dyadic, draftCase Full, towerCase "image_tower" aneTower Dyadic 120 12 False, towerCase "image_tower" aneTower Full 120 12 False, towerCase "tower_layer_26b" gemmaTower Full 256 16 True]
    <> [normRows, normPeaks, headNorm 6 64 128 False, headNorm 5 64 128 False, headNorm 16 72 256 True, maskedMirror, fullMirror, flatMirror, argmaxTies, softcap, rangeLow, rangeHigh, longSums, productsGrid, prefixOverflow]

orFail :: String -> Either Text a -> IO a
orFail what = either (\e -> fail (what <> ": " <> T.unpack e)) pure

dims :: [Word64] -> String
dims = intercalate "," . map show

-- | Every blob's bytes: the program's own, and the case's tensors for the ones only reserved.
filledBlobs :: Built -> Map Word64 [Double] -> Either Text Blobs
filledBlobs built weights = foldM fill emptyBlobs (blobsLayout built.builtBlobs)
 where
  fill acc p = do
    bytes <- case (p.placedBytes, Map.lookup p.placedRecord weights) of
      (Just bs, _) -> Right bs
      (Nothing, Just xs) -> Right (BL.toStrict (BB.toLazyByteString (foldMap (element p.placedType) xs)))
      (Nothing, Nothing) -> Left ("no weights for the blob at " <> T.pack (show p.placedRecord))
    let (at, acc') = addBlob p.placedType bytes acc
    unless (at == p.placedRecord && fromIntegral (BS.length bytes) == p.placedSize) $
      Left ("the blob at " <> T.pack (show p.placedRecord) <> " doesn't fit its place")
    pure acc'
  element = \case
    BlobFloat16 -> BB.word16LE . floatToHalf . realToFrac
    BlobFloat32 -> BB.floatLE . realToFrac

writeCase :: FilePath -> Case -> IO ()
writeCase out c = do
  started <- getMonotonicTime
  let what = c.caseName
      dir = out </> what
  checkedS <- either (\fs -> fail (what <> ": the gate: " <> show fs)) pure (checkStage appleNE c.caseStage)
  let l = lowerStage checkedS
      manifest = l.loweredManifest
  derived <- orFail what (hostInputs manifest c.caseData)
  weights <- Map.map (map toHalf) <$> orFail what (hostWeights manifest c.caseData)
  blobs <- orFail what (filledBlobs l.loweredBuilt weights)
  writePackage (dir </> "model.mlpackage") l.loweredBuilt.builtModel blobs
  -- What the host derives, rounded to binary16 as it writes it.
  let inputs = Map.map (map toHalf) derived
  forM_ manifest.manifestInputs $ \i ->
    BL.writeFile (dir </> T.unpack i.inputName <> ".f16") (BB.toLazyByteString (foldMap (BB.word16LE . floatToHalf . realToFrac) (inputs Map.! i.inputName)))
  writeFile (dir </> "inputs.txt") (unlines [T.unpack i.inputName <> " " <> dims i.inputShape | i <- manifest.manifestInputs])
  let given = Map.fromList [(T.unpack i.inputName, tensor (map fromIntegral i.inputShape) (inputs Map.! i.inputName)) | i <- manifest.manifestInputs]
  values <- orFail what (interpretValuesWith Exact l.loweredBuilt.builtModel given weights)
  kept <- orFail what (interpretValuesWith NeuralEngine l.loweredBuilt.builtModel given weights)
  evaluated <- orFail what (evaluateStage checkedS c.caseData)
  mapM_ (createDirectoryIfMissing True . (dir </>)) ["eval", "interp", "model"]
  forM_ manifest.manifestOutputs $ \o -> do
    let name = T.unpack o.outputName
        count = fromIntegral (product o.outputShape)
    interpreted <- maybe (fail (what <> ": the interpreter made no " <> name)) (pure . tensorElems) (Map.lookup name values)
    modelled <- maybe (fail (what <> ": the interpreter made no " <> name)) (pure . tensorElems) (Map.lookup name kept)
    -- An output of shape [1, w, 1, m] or [1, h, d, m] is m rows of w or h·d.
    meant <- maybe (fail (what <> ": the evaluator made no " <> name)) (pure . concat . transpose) (Map.lookup o.outputName evaluated)
    when (any ((/= count) . length) [interpreted, modelled, meant]) $ fail (what <> ": " <> name <> " isn't of its shape")
    forM_ [("eval", meant), ("interp", interpreted), ("model", modelled)] $ \(sub, xs) ->
      BL.writeFile (dir </> sub </> name <> ".f64") (BB.toLazyByteString (foldMap BB.doubleLE xs))
  writeFile (dir </> "outputs.txt") . unlines $
    [ name <> " " <> kind <> " " <> dims o.outputShape <> rule
    | o <- manifest.manifestOutputs
    , let name = T.unpack o.outputName
          (kind, rule) = case (o.outputType, lookup o.outputName c.caseTokens) of
            (M.OutputIndex, Just (Logits n)) -> ("token", " " <> T.unpack n)
            (M.OutputIndex, Just Exactly) -> ("token", " exact")
            (M.OutputIndex, Nothing) -> error (what <> ": a token output of no rule")
            (M.OutputHalf, _)
              | c.caseModel -> ("float", " model")
              | o.outputName `elem` softcapped c.caseStage -> ("float", " softcap")
              | otherwise -> ("float", "")
    ]
  writeFile (dir </> "units.txt") (unlines (["ane" | c.caseAne] <> ["cpu" | c.caseCpu]))
  took <- subtract started <$> getMonotonicTime
  putStrLn (what <> ": " <> show (length manifest.manifestInputs) <> " inputs, " <> show (length manifest.manifestOutputs) <> " outputs, a weight file of " <> show manifest.manifestWeightFile <> " bytes, in " <> show (round took :: Int) <> " s")
  hFlush stdout

{- | @cases OUT@ writes every case, @cases --small OUT@ the ones
@checks.ane-corpus@ writes, @cases --large OUT@ the others, and @cases
--only NAME,… OUT@ the ones named.
-}
main :: IO ()
main =
  getArgs >>= \case
    [out] -> write out corpus
    ["--small", out] -> write out (filter (not . (.caseLarge)) corpus)
    ["--large", out] -> write out (filter (.caseLarge) corpus)
    ["--only", names, out] -> do
      let wanted = map T.unpack (T.splitOn "," (T.pack names))
      case filter (`notElem` map (.caseName) corpus) wanted of
        [] -> write out (filter ((`elem` wanted) . (.caseName)) corpus)
        unknown -> fail ("no cases " <> unwords unknown)
    _ -> fail "usage: cases [--small | --large | --only NAME,...] OUT"
 where
  write out cs = do
    createDirectoryIfMissing True out
    mapM_ (writeCase out) cs
