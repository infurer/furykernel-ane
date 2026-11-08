-- SPDX-FileCopyrightText: 2026 Mika Tammi
-- SPDX-License-Identifier: BSD-3-Clause

{- | The stages the tests lower and the conformance corpus runs, of any
size: a step that drafts (a token gathered beside a state, attention over
a ring's mirror with a window and over a paged cache's, an output layer
and its argmax), and an image's tower (patches projected and placed,
layers of self-attention over the step's own rows with two rotary axes,
the rows pooled, standardized and projected). The tests take them small
('smallDraft', 'smallTower'); test/ane/Cases.hs takes them at sizes Core
ML places on the Neural Engine.
-}
module Test.FuryKernel.Ane.Stages (
  checked,
  eps,
  Draft (..),
  smallDraft,
  draftStageOf,
  Tower (..),
  smallTower,
  towerStageOf,
) where

import Data.Text (Text)
import Data.Text qualified as T
import FuryKernel.Ane.Stage
import FuryKernel.Program.Build qualified as B
import FuryKernel.Program.Check (CheckedProgram, checkProgram)
import FuryKernel.Program.Graph (AttendTo (..), AttnSpec (..), CacheElem (..), CacheLayout (..), Program, Reach (..), RopePairs (..), RopeSpec (..), StoreAt (..), wholeRow)
import FuryKernel.Type (ScalarType (..))
import Numeric.Natural (Natural)

checked :: Program -> CheckedProgram
checked p = either (\fs -> error ("a test program fails its gate: " <> show fs)) id (checkProgram p)

-- | An epsilon exact in fp16.
eps :: Double
eps = 2 ^^ (-10 :: Int)

------------------------------------------------------------------------------
-- A drafting step

-- | A drafting step's sizes.
data Draft = Draft
  { draftVocab :: Natural
  -- ^ The tokens' bound: the embedding's rows.
  , draftEmbed :: Natural
  -- ^ The embedding's width.
  , draftEmbedScale :: Double
  , draftState :: Natural
  -- ^ The state's width, taken and made.
  , draftWidth :: Natural
  -- ^ The layers' width.
  , draftHeadDim :: Natural
  , draftRingHeads :: Natural
  , draftRingKv :: Natural
  , draftRingLength :: Natural
  -- ^ The ring's positions, all of which its mirror holds.
  , draftWindow :: Natural
  , draftSequences :: Natural
  -- ^ The ring's slots.
  , draftPagedHeads :: Natural
  , draftPagedKv :: Natural
  , draftBlocks :: Natural
  -- ^ The paged cache's pool.
  , draftBlockSize :: Natural
  , draftRowBlocks :: Natural
  -- ^ The blocks a row's table names, which its mirror holds.
  , draftPositions :: Natural
  -- ^ The positions' bound.
  , draftMlp :: Natural
  , draftOut :: Natural
  -- ^ The output layer's rows.
  , draftSoftcap :: Double
  , draftRows :: Natural
  , draftEps :: Double
  }
  deriving stock (Show)

-- | The tests' drafting step.
smallDraft :: Draft
smallDraft =
  Draft
    { draftVocab = 32
    , draftEmbed = 8
    , draftEmbedScale = 4
    , draftState = 8
    , draftWidth = 6
    , draftHeadDim = 4
    , draftRingHeads = 2
    , draftRingKv = 1
    , draftRingLength = 6
    , draftWindow = 4
    , draftSequences = 2
    , draftPagedHeads = 4
    , draftPagedKv = 2
    , draftBlocks = 8
    , draftBlockSize = 4
    , draftRowBlocks = 4
    , draftPositions = 16
    , draftMlp = 12
    , draftOut = 40
    , draftSoftcap = 4
    , draftRows = 2
    , draftEps = eps
    }

draftIn :: Draft -> Program
draftIn d = B.program "t_in" $ do
  tokens <- B.rowInts "tokens" 1 d.draftVocab
  state <- B.input "state" (wholeRow d.draftState) TF32
  table <- B.weight "embed" [d.draftVocab, d.draftEmbed]
  proj <- B.weight "proj" [d.draftWidth, d.draftEmbed + d.draftState]
  e <- B.gather "embedded" table tokens d.draftEmbedScale
  joined <- B.concatenate "joined" e state
  x <- B.linear "x" proj 0 joined
  B.output "x" x

-- | Every pair of a head turns.
ringLayer :: Draft -> Program
ringLayer d = B.program "t_ring" $ do
  x <- B.input "x" (wholeRow d.draftWidth) TF32
  pos <- B.rowInts "position" 1 d.draftPositions
  seen <- B.rowInts "seen" 1 d.draftPositions
  slot <- B.rowInts "slot" 1 d.draftSequences
  norm <- B.weight "norm" [d.draftWidth]
  wq <- B.weight "q_proj" [d.draftRingHeads * d.draftHeadDim, d.draftWidth]
  qNorm <- B.weight "q_norm" [d.draftHeadDim]
  wo <- B.weight "o_proj" [d.draftWidth, d.draftRingHeads * d.draftHeadDim]
  k <- B.cacheOf CacheF16 "k_cache" (Ring d.draftSequences d.draftRingKv d.draftRingLength d.draftHeadDim)
  v <- B.cacheOf CacheF16 "v_cache" (Ring d.draftSequences d.draftRingKv d.draftRingLength d.draftHeadDim)
  a <- B.rmsnorm "a" d.draftEps (Just norm) 1 x
  q <- B.linear "q" wq d.draftHeadDim a
  qn <- B.rmsnorm "qn" d.draftEps (Just qNorm) 1 q
  qr <- B.rope "qr" (RopeSpec 10000 (d.draftHeadDim `div` 2) 1 RopeHalves) pos qn
  att <- B.attend "att" (AttnSpec d.draftRingHeads d.draftRingKv d.draftHeadDim (Just d.draftWindow) 1 Nothing) qr AttendTo {attendKeys = k, attendValues = v, attendPositions = seen, attendSequences = slot}
  o <- B.linear "o" wo 0 att
  y <- B.add "y" x o
  B.output "x" y

-- | Half the pairs of a head turn; scores are scaled by @1 / √headDim@.
pagedLayer :: Draft -> Program
pagedLayer d = B.program "t_paged" $ do
  x <- B.input "x" (wholeRow d.draftWidth) TF32
  pos <- B.rowInts "position" 1 d.draftPositions
  seen <- B.rowInts "seen" 1 d.draftPositions
  blocks <- B.rowInts "blocks" d.draftRowBlocks d.draftBlocks
  wq <- B.weight "q_proj" [d.draftPagedHeads * d.draftHeadDim, d.draftWidth]
  wo <- B.weight "o_proj" [d.draftWidth, d.draftPagedHeads * d.draftHeadDim]
  wg <- B.weight "gate" [d.draftMlp, d.draftWidth]
  wu <- B.weight "up" [d.draftMlp, d.draftWidth]
  wd <- B.weight "down" [d.draftWidth, d.draftMlp]
  sc <- B.weight "scalar" [1]
  k <- B.cacheOf CacheF16 "k_cache" (Paged d.draftBlocks d.draftBlockSize d.draftPagedKv d.draftHeadDim)
  v <- B.cacheOf CacheF16 "v_cache" (Paged d.draftBlocks d.draftBlockSize d.draftPagedKv d.draftHeadDim)
  q <- B.linear "q" wq d.draftHeadDim x
  qr <- B.rope "qr" (RopeSpec 100 (d.draftHeadDim `div` 4) 1 RopeHalves) pos q
  att <- B.attend "att" (AttnSpec d.draftPagedHeads d.draftPagedKv d.draftHeadDim Nothing (1 / sqrt (fromIntegral d.draftHeadDim)) Nothing) qr AttendTo {attendKeys = k, attendValues = v, attendPositions = seen, attendSequences = blocks}
  o <- B.linear "o" wo 0 att
  g <- B.linear "g" wg 0 x
  u <- B.linear "u" wu 0 x
  act <- B.geluMul "act" g u
  m <- B.linear "m" wd 0 act
  y1 <- B.add "y1" x o
  y <- B.add "y" y1 m
  z <- B.scaleBy "z" sc y
  B.output "x" z

draftOutOf :: Draft -> Program
draftOutOf d = B.program "t_out" $ do
  x <- B.input "x" (wholeRow d.draftWidth) TF32
  finalNorm <- B.weight "final_norm" [d.draftWidth]
  headW <- B.weight "head" [d.draftOut, d.draftWidth]
  post <- B.weight "post" [d.draftState, d.draftWidth]
  n <- B.rmsnorm "n" d.draftEps (Just finalNorm) 2 x
  logits <- B.linear "logits" headW 0 n
  capped <- B.softcap "capped" d.draftSoftcap logits
  token <- B.argmax "token" capped
  state <- B.linear "state" post 0 n
  B.output "token" token
  B.output "state" state
  B.output "logits" capped

{- | The step: its embedding, a ring's layer, a paged cache's, and its
output, over 'draftRows' rows.
-}
draftStageOf :: Draft -> Stage
draftStageOf d =
  Stage
    { stageName = "draft_step"
    , stageParts =
        [ Part "in" (checked (draftIn d)) d.draftRows [("tokens", Host "token"), ("state", Host "hidden"), ("embed", Tensor "embed"), ("proj", Tensor "pre_proj")]
        , Part
            "ring"
            (checked (ringLayer d))
            d.draftRows
            [ ("x", Earlier "in" "x")
            , ("position", Host "position")
            , ("seen", Host "seen")
            , ("slot", Host "slot")
            , ("norm", Tensor "l0.norm")
            , ("q_proj", Tensor "l0.q_proj")
            , ("q_norm", Tensor "l0.q_norm")
            , ("o_proj", Tensor "l0.o_proj")
            , ("k_cache", Mirror "k_ring" d.draftRingLength)
            , ("v_cache", Mirror "v_ring" d.draftRingLength)
            ]
        , Part
            "paged"
            (checked (pagedLayer d))
            d.draftRows
            [ ("x", Earlier "ring" "x")
            , ("position", Host "position")
            , ("seen", Host "seen")
            , ("blocks", Host "blocks")
            , ("q_proj", Tensor "l1.q_proj")
            , ("o_proj", Tensor "l1.o_proj")
            , ("gate", Tensor "l1.gate")
            , ("up", Tensor "l1.up")
            , ("down", Tensor "l1.down")
            , ("scalar", Tensor "l1.scalar")
            , ("k_cache", Mirror "k_paged" (d.draftRowBlocks * d.draftBlockSize))
            , ("v_cache", Mirror "v_paged" (d.draftRowBlocks * d.draftBlockSize))
            ]
        , Part "out" (checked (draftOutOf d)) d.draftRows [("x", Earlier "paged" "x"), ("final_norm", Tensor "final_norm"), ("head", Tensor "head"), ("post", Tensor "post")]
        ]
    , stageOutputs = [("next_token", "out", "token"), ("next_state", "out", "state"), ("logits", "out", "logits")]
    }

------------------------------------------------------------------------------
-- An image's tower

-- | An image tower's sizes.
data Tower = Tower
  { towerPatch :: Natural
  -- ^ A patch's pixels.
  , towerPlaces :: Natural
  -- ^ The position table's rows: the places' bound.
  , towerWidth :: Natural
  , towerHeads :: Natural
  , towerHeadDim :: Natural
  , towerAxisBound :: Natural
  -- ^ The rotary axes' positions' bound.
  , towerAhead :: Natural
  -- ^ How far past its own position a row's reach may go.
  , towerMlp :: Maybe Natural
  {- ^ A layer's gate and up rows, if it has an MLP: then each block's
  output is normed before its residual add, as Gemma 4's are.
  -}
  , towerLayers :: Int
  , towerRows :: Natural
  -- ^ The image's rows.
  , towerSequences :: Natural
  -- ^ The rows' sequences, a ring's slot each.
  , towerPoolBy :: Natural
  -- ^ The rows pooled into one.
  , towerPooled :: Natural
  -- ^ The rows pooled.
  , towerStdFactor :: Double
  , towerOut :: Natural
  -- ^ The projector's rows.
  , towerEps :: Double
  }
  deriving stock (Show)

-- | The tests' tower over an image of eight rows.
smallTower :: Tower
smallTower =
  Tower
    { towerPatch = 12
    , towerPlaces = 10
    , towerWidth = 8
    , towerHeads = 2
    , towerHeadDim = 4
    , towerAxisBound = 3
    , towerAhead = 4
    , towerMlp = Nothing
    , towerLayers = 2
    , towerRows = 8
    , towerSequences = 1
    , towerPoolBy = 2
    , towerPooled = 3
    , towerStdFactor = 2
    , towerOut = 10
    , towerEps = eps
    }

towerEmbed :: Tower -> Program
towerEmbed t = B.program "v_embed" $ do
  patches <- B.input "patches" (wholeRow t.towerPatch) TF32
  places <- B.rowInts "places" 2 t.towerPlaces
  proj <- B.weight "patch_proj" [t.towerWidth, t.towerPatch]
  table <- B.weight "positions" [t.towerPlaces, t.towerWidth]
  p <- B.linear "projected" proj 0 patches
  g <- B.gather "placed" table places 1
  x <- B.add "x" p g
  B.output "x" x

-- | A layer's weights, by the names its part feeds them under.
towerWeights :: Tower -> [Text]
towerWeights t =
  ["norm", "q_proj", "k_proj", "v_proj", "q_norm", "k_norm", "o_proj"]
    <> maybe [] (const ["post_norm", "mlp_norm", "gate", "up", "down", "mlp_post_norm"]) t.towerMlp

towerLayerOf :: Tower -> Program
towerLayerOf t = B.program "v_layer" $ do
  let m = t.towerRows
      inner = t.towerHeads * t.towerHeadDim
  x <- B.input "x" (wholeRow t.towerWidth) TF32
  patch <- B.rowInts "patch" 1 m
  axes <- B.rowInts "axes" 2 t.towerAxisBound
  reach <- B.rowInts "reach" 1 m
  slot <- B.rowInts "slot" 1 t.towerSequences
  norm <- B.weight "norm" [t.towerWidth]
  wq <- B.weight "q_proj" [inner, t.towerWidth]
  wk <- B.weight "k_proj" [inner, t.towerWidth]
  wv <- B.weight "v_proj" [inner, t.towerWidth]
  qNorm <- B.weight "q_norm" [t.towerHeadDim]
  kNorm <- B.weight "k_norm" [t.towerHeadDim]
  wo <- B.weight "o_proj" [t.towerWidth, inner]
  kc <- B.cacheOf CacheF16 "k_cache" (Ring t.towerSequences t.towerHeads m t.towerHeadDim)
  vc <- B.cacheOf CacheF16 "v_cache" (Ring t.towerSequences t.towerHeads m t.towerHeadDim)
  a <- B.rmsnorm "a" t.towerEps (Just norm) 1 x
  q <- B.linear "q" wq t.towerHeadDim a
  k <- B.linear "k" wk t.towerHeadDim a
  v <- B.linear "v" wv t.towerHeadDim a
  qn <- B.rmsnorm "qn" t.towerEps (Just qNorm) 1 q
  kn <- B.rmsnorm "kn" t.towerEps (Just kNorm) 1 k
  vn <- B.rmsnorm "vn" t.towerEps Nothing 1 v
  let spec = RopeSpec 100 (t.towerHeadDim `div` 4) 2 RopeHalves
  qr <- B.rope "qr" spec axes qn
  kr <- B.rope "kr" spec axes kn
  B.kvStore "store_k" StoreAt {storeCache = kc, storePositions = patch, storeSequences = slot} kr
  B.kvStore "store_v" StoreAt {storeCache = vc, storePositions = patch, storeSequences = slot} vn
  att <- B.attend "att" (AttnSpec t.towerHeads t.towerHeads t.towerHeadDim Nothing 1 (Just (Reach reach.srcId t.towerAhead))) qr AttendTo {attendKeys = kc, attendValues = vc, attendPositions = patch, attendSequences = slot}
  o <- B.linear "o" wo 0 att
  case t.towerMlp of
    Nothing -> do
      y <- B.add "y" x o
      B.output "x" y
    Just mlp -> do
      postNorm <- B.weight "post_norm" [t.towerWidth]
      mlpNorm <- B.weight "mlp_norm" [t.towerWidth]
      wg <- B.weight "gate" [mlp, t.towerWidth]
      wu <- B.weight "up" [mlp, t.towerWidth]
      wd <- B.weight "down" [t.towerWidth, mlp]
      mlpPostNorm <- B.weight "mlp_post_norm" [t.towerWidth]
      on <- B.rmsnorm "on" t.towerEps (Just postNorm) 1 o
      x1 <- B.add "x1" x on
      mi <- B.rmsnorm "mi" t.towerEps (Just mlpNorm) 1 x1
      g <- B.linear "g" wg 0 mi
      u <- B.linear "u" wu 0 mi
      act <- B.geluMul "act" g u
      mo <- B.linear "mo" wd 0 act
      mn <- B.rmsnorm "mn" t.towerEps (Just mlpPostNorm) 1 mo
      y <- B.add "y" x1 mn
      B.output "x" y

towerPool :: Tower -> Program
towerPool t = B.program "v_pool" $ do
  squares <- B.rowInts "squares" t.towerPoolBy t.towerRows
  encoded <- B.rowsF32 "encoded" [t.towerRows, t.towerWidth]
  bias <- B.weight "std_bias" [t.towerWidth]
  scale <- B.weight "std_scale" [t.towerWidth]
  proj <- B.weight "projector" [t.towerOut, t.towerWidth]
  pooled <- B.gather "pooled" encoded squares (1 / fromIntegral t.towerPoolBy)
  standard <- B.standardize "standardized" bias scale t.towerStdFactor pooled
  normed <- B.rmsnorm "normed" t.towerEps Nothing 1 standard
  soft <- B.linear "soft" proj 0 normed
  B.output "soft" soft

{- | The tower: the image's rows embedded, 'towerLayers' layers (their
tensors @l0.@, @l1.@, …), and the pooled rows projected.
-}
towerStageOf :: Tower -> Stage
towerStageOf t =
  Stage
    { stageName = "image_tower"
    , stageParts =
        [Part "embed" (checked (towerEmbed t)) t.towerRows [("patches", Host "patches"), ("places", Host "places"), ("patch_proj", Tensor "patch_proj"), ("positions", Tensor "position_table")]]
          <> [layerPart i | i <- [0 .. t.towerLayers - 1]]
          <> [ Part
                 "pool"
                 (checked (towerPool t))
                 t.towerPooled
                 [ ("squares", Host "squares")
                 , ("encoded", Earlier (layerName (t.towerLayers - 1)) "x")
                 , ("std_bias", Tensor "std_bias")
                 , ("std_scale", Tensor "std_scale")
                 , ("projector", Tensor "projector")
                 ]
             ]
    , stageOutputs = [("soft", "pool", "soft")]
    }
 where
  layerName i = if i < 0 then "embed" else T.pack ("layer" <> show i)
  layerPart i =
    Part
      (layerName i)
      (checked (towerLayerOf t))
      t.towerRows
      ( [ ("x", Earlier (layerName (i - 1)) "x")
        , ("patch", Host "patch")
        , ("axes", Host "axes")
        , ("reach", Host "reach")
        , ("slot", Host "slot")
        , ("k_cache", Own FirstRowMasks)
        , ("v_cache", Own FirstRowMasks)
        ]
          <> [(w, Tensor (T.pack ("l" <> show i <> ".") <> w)) | w <- towerWeights t]
      )
