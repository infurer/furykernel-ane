-- SPDX-FileCopyrightText: 2026 Mika Tammi
-- SPDX-License-Identifier: BSD-3-Clause

{- | Stages lowered, held to their programs' meaning: each stage evaluated
part by part by the core's f64 evaluator, against its lowered program run
by the f64 interpreter on what the host derives for it. Weights, inputs
and caches are small dyadic numbers, exact in fp16, so the two differ by
f64 rounding alone.

Two stages: one of a step that drafts (a token gathered beside a state,
attention over a ring's mirror with a window and over a paged cache's,
an output layer wider than the caps' channels, its argmax), and one of an
image's tower (patches projected and placed, two layers of
self-attention over the step's own rows with two rotary axes, the rows
pooled, standardized and projected), whose padding rows change nothing.
-}
module Test.FuryKernel.Ane.LowerTest (tests) where

import Control.Exception (IOException, try)
import Control.Monad (foldM, forM_, unless)
import Data.List (nub)
import Data.Map.Strict (Map)
import Data.Map.Strict qualified as Map
import Data.Text (Text)
import Data.Text qualified as T
import Data.Text.IO qualified as TIO
import FuryKernel.Ane.CoreML (Arg (..), DataType (..), Function (..), Immediate (..), Model (..), Op (..), TensorType (..), Value (..))
import FuryKernel.Ane.Host
import FuryKernel.Ane.Interp (interpretValues, tensor, tensorElems)
import FuryKernel.Ane.Interp qualified as I
import FuryKernel.Ane.Lower
import FuryKernel.Ane.Manifest
import FuryKernel.Ane.Mil (Built (..))
import FuryKernel.Ane.Stage
import FuryKernel.Program.Build qualified as B
import FuryKernel.Program.Check (CheckedProgram, checkProgram, checkedProgram)
import FuryKernel.Program.Eval (Kept (..))
import FuryKernel.Program.Graph (AttendTo (..), AttnSpec (..), CacheElem (..), CacheLayout (..), Node (..), Op (..), Program (..), Reach (..), RopePairs (..), RopeSpec (..), Source (..), StoreAt (..), programSource, wholeRow)
import FuryKernel.Type (ScalarType (..))
import Numeric.Natural (Natural)
import System.Directory (getTemporaryDirectory)
import System.FilePath ((</>))
import Test.Tasty (TestTree, testGroup)
import Test.Tasty.HUnit (Assertion, assertBool, assertFailure, testCase, (@?=))

-- | Small caps: outputs of 16 channels at most.
caps :: Caps
caps = appleNE {capsChannels = 16}

-- | An epsilon exact in fp16.
eps :: Double
eps = 2 ^^ (-10 :: Int)

checked :: Program -> CheckedProgram
checked p = either (\fs -> error ("a test program fails its gate: " <> show fs)) id (checkProgram p)

-- | Deterministic dyadic numbers: @k / denominator@ for @|k| <= 64@.
dyadic :: Int -> Double -> Int -> [Double]
dyadic seed denominator n = take n [fromIntegral (x `mod` 129 - 64) / denominator | x <- drop 1 (iterate lcg seed)]
 where
  lcg x = (x * 1103515245 + 12345) `mod` 2147483648

------------------------------------------------------------------------------
-- A drafting step

draftIn :: Program
draftIn = B.program "t_in" $ do
  tokens <- B.rowInts "tokens" 1 32
  state <- B.input "state" (wholeRow 8) TF32
  table <- B.weight "embed" [32, 8]
  proj <- B.weight "proj" [6, 16]
  e <- B.gather "embedded" table tokens 4
  joined <- B.concatenate "joined" e state
  x <- B.linear "x" proj 0 joined
  B.output "x" x

ringLayer :: Program
ringLayer = B.program "t_ring" $ do
  x <- B.input "x" (wholeRow 6) TF32
  pos <- B.rowInts "position" 1 16
  seen <- B.rowInts "seen" 1 16
  slot <- B.rowInts "slot" 1 2
  norm <- B.weight "norm" [6]
  wq <- B.weight "q_proj" [8, 6]
  qNorm <- B.weight "q_norm" [4]
  wo <- B.weight "o_proj" [6, 8]
  k <- B.cacheOf CacheF16 "k_cache" (Ring 2 1 6 4)
  v <- B.cacheOf CacheF16 "v_cache" (Ring 2 1 6 4)
  a <- B.rmsnorm "a" eps (Just norm) 1 x
  q <- B.linear "q" wq 4 a
  qn <- B.rmsnorm "qn" eps (Just qNorm) 1 q
  qr <- B.rope "qr" (RopeSpec 10000 2 1 RopeHalves) pos qn
  att <- B.attend "att" (AttnSpec 2 1 4 (Just 4) 1 Nothing) qr AttendTo {attendKeys = k, attendValues = v, attendPositions = seen, attendSequences = slot}
  o <- B.linear "o" wo 0 att
  y <- B.add "y" x o
  B.output "x" y

pagedLayer :: Program
pagedLayer = B.program "t_paged" $ do
  x <- B.input "x" (wholeRow 6) TF32
  pos <- B.rowInts "position" 1 16
  seen <- B.rowInts "seen" 1 16
  blocks <- B.rowInts "blocks" 4 8
  wq <- B.weight "q_proj" [16, 6]
  wo <- B.weight "o_proj" [6, 16]
  wg <- B.weight "gate" [12, 6]
  wu <- B.weight "up" [12, 6]
  wd <- B.weight "down" [6, 12]
  sc <- B.weight "scalar" [1]
  k <- B.cacheOf CacheF16 "k_cache" (Paged 8 4 2 4)
  v <- B.cacheOf CacheF16 "v_cache" (Paged 8 4 2 4)
  q <- B.linear "q" wq 4 x
  qr <- B.rope "qr" (RopeSpec 100 1 1 RopeHalves) pos q
  att <- B.attend "att" (AttnSpec 4 2 4 Nothing 0.5 Nothing) qr AttendTo {attendKeys = k, attendValues = v, attendPositions = seen, attendSequences = blocks}
  o <- B.linear "o" wo 0 att
  g <- B.linear "g" wg 0 x
  u <- B.linear "u" wu 0 x
  act <- B.geluMul "act" g u
  m <- B.linear "m" wd 0 act
  y1 <- B.add "y1" x o
  y <- B.add "y" y1 m
  z <- B.scaleBy "z" sc y
  B.output "x" z

draftOut :: Program
draftOut = B.program "t_out" $ do
  x <- B.input "x" (wholeRow 6) TF32
  finalNorm <- B.weight "final_norm" [6]
  headW <- B.weight "head" [40, 6]
  post <- B.weight "post" [8, 6]
  n <- B.rmsnorm "n" eps (Just finalNorm) 2 x
  logits <- B.linear "logits" headW 0 n
  capped <- B.softcap "capped" 4 logits
  token <- B.argmax "token" capped
  state <- B.linear "state" post 0 n
  B.output "token" token
  B.output "state" state
  B.output "logits" capped

draftStage :: Stage
draftStage =
  Stage
    { stageName = "draft_step"
    , stageParts =
        [ Part "in" (checked draftIn) 2 [("tokens", Host "token"), ("state", Host "hidden"), ("embed", Tensor "embed"), ("proj", Tensor "pre_proj")]
        , Part
            "ring"
            (checked ringLayer)
            2
            [ ("x", Earlier "in" "x")
            , ("position", Host "position")
            , ("seen", Host "seen")
            , ("slot", Host "slot")
            , ("norm", Tensor "l0.norm")
            , ("q_proj", Tensor "l0.q_proj")
            , ("q_norm", Tensor "l0.q_norm")
            , ("o_proj", Tensor "l0.o_proj")
            , ("k_cache", Mirror "k_ring" 6)
            , ("v_cache", Mirror "v_ring" 6)
            ]
        , Part
            "paged"
            (checked pagedLayer)
            2
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
            , ("k_cache", Mirror "k_paged" 16)
            , ("v_cache", Mirror "v_paged" 16)
            ]
        , Part "out" (checked draftOut) 2 [("x", Earlier "paged" "x"), ("final_norm", Tensor "final_norm"), ("head", Tensor "head"), ("post", Tensor "post")]
        ]
    , stageOutputs = [("next_token", "out", "token"), ("next_state", "out", "state"), ("logits", "out", "logits")]
    }

draftData :: StageData
draftData =
  StageData
    { sdInputs = Map.fromList [("hidden", rows 2 8 (dyadic 1 32 16))]
    , sdRowInts =
        Map.fromList
          [ ("token", [[3], [17]])
          , ("position", [[9], [10]])
          , ("seen", [[8], [9]])
          , ("slot", [[1], [1]])
          , ("blocks", [[3, 0, 5, 1], [3, 0, 5, 1]])
          ]
    , sdCaches =
        Map.fromList
          [ ("k_ring", dyadic 2 16 48)
          , ("v_ring", dyadic 3 16 48)
          , ("k_paged", dyadic 4 16 256)
          , ("v_paged", dyadic 5 16 256)
          ]
    , sdTensors =
        Map.fromList
          [ ("embed", dyadic 6 64 256)
          , ("pre_proj", dyadic 7 64 96)
          , ("l0.norm", dyadic 8 32 6)
          , ("l0.q_proj", dyadic 9 64 48)
          , ("l0.q_norm", dyadic 10 32 4)
          , ("l0.o_proj", dyadic 11 64 48)
          , ("l1.q_proj", dyadic 12 64 96)
          , ("l1.o_proj", dyadic 13 64 96)
          , ("l1.gate", dyadic 14 64 72)
          , ("l1.up", dyadic 15 64 72)
          , ("l1.down", dyadic 16 64 72)
          , ("l1.scalar", [0.75])
          , ("final_norm", dyadic 17 32 6)
          , ("head", dyadic 18 64 240)
          , ("post", dyadic 19 64 48)
          ]
    }

------------------------------------------------------------------------------
-- An image's tower

towerEmbed :: Program
towerEmbed = B.program "v_embed" $ do
  patches <- B.input "patches" (wholeRow 12) TF32
  places <- B.rowInts "places" 2 10
  proj <- B.weight "patch_proj" [8, 12]
  table <- B.weight "positions" [10, 8]
  p <- B.linear "projected" proj 0 patches
  g <- B.gather "placed" table places 1
  x <- B.add "x" p g
  B.output "x" x

{- | A layer for rows of up to this many sequences, a ring's slot each,
over an image of this many rows.
-}
towerLayerOf :: Natural -> Natural -> Program
towerLayerOf sequences m = B.program "v_layer" $ do
  x <- B.input "x" (wholeRow 8) TF32
  patch <- B.rowInts "patch" 1 m
  axes <- B.rowInts "axes" 2 3
  reach <- B.rowInts "reach" 1 m
  slot <- B.rowInts "slot" 1 sequences
  norm <- B.weight "norm" [8]
  wq <- B.weight "q_proj" [8, 8]
  wk <- B.weight "k_proj" [8, 8]
  wv <- B.weight "v_proj" [8, 8]
  qNorm <- B.weight "q_norm" [4]
  kNorm <- B.weight "k_norm" [4]
  wo <- B.weight "o_proj" [8, 8]
  kc <- B.cacheOf CacheF16 "k_cache" (Ring sequences 2 m 4)
  vc <- B.cacheOf CacheF16 "v_cache" (Ring sequences 2 m 4)
  a <- B.rmsnorm "a" eps (Just norm) 1 x
  q <- B.linear "q" wq 4 a
  k <- B.linear "k" wk 4 a
  v <- B.linear "v" wv 4 a
  qn <- B.rmsnorm "qn" eps (Just qNorm) 1 q
  kn <- B.rmsnorm "kn" eps (Just kNorm) 1 k
  vn <- B.rmsnorm "vn" eps Nothing 1 v
  let spec = RopeSpec 100 1 2 RopeHalves
  qr <- B.rope "qr" spec axes qn
  kr <- B.rope "kr" spec axes kn
  B.kvStore "store_k" StoreAt {storeCache = kc, storePositions = patch, storeSequences = slot} kr
  B.kvStore "store_v" StoreAt {storeCache = vc, storePositions = patch, storeSequences = slot} vn
  att <- B.attend "att" (AttnSpec 2 2 4 Nothing 1 (Just (Reach reach.srcId 4))) qr AttendTo {attendKeys = kc, attendValues = vc, attendPositions = patch, attendSequences = slot}
  o <- B.linear "o" wo 0 att
  y <- B.add "y" x o
  B.output "x" y

towerPool :: Natural -> Program
towerPool m = B.program "v_pool" $ do
  squares <- B.rowInts "squares" 2 m
  encoded <- B.rowsF32 "encoded" [m, 8]
  bias <- B.weight "std_bias" [8]
  scale <- B.weight "std_scale" [8]
  proj <- B.weight "projector" [10, 8]
  pooled <- B.gather "pooled" encoded squares 0.5
  standard <- B.standardize "standardized" bias scale 2 pooled
  normed <- B.rmsnorm "normed" eps Nothing 1 standard
  soft <- B.linear "soft" proj 0 normed
  B.output "soft" soft

-- | The tower over an image of this many rows.
towerStage :: Natural -> Stage
towerStage = towerStageWith 1

-- | 'towerStage', its layers' rows of up to this many sequences.
towerStageWith :: Natural -> Natural -> Stage
towerStageWith sequences m =
  Stage
    { stageName = "image_tower"
    , stageParts =
        [ Part "embed" (checked towerEmbed) m [("patches", Host "patches"), ("places", Host "places"), ("patch_proj", Tensor "patch_proj"), ("positions", Tensor "position_table")]
        , layerPart "layer0" "embed" "l0"
        , layerPart "layer1" "layer0" "l1"
        , Part
            "pool"
            (checked (towerPool m))
            3
            [ ("squares", Host "squares")
            , ("encoded", Earlier "layer1" "x")
            , ("std_bias", Tensor "std_bias")
            , ("std_scale", Tensor "std_scale")
            , ("projector", Tensor "projector")
            ]
        ]
    , stageOutputs = [("soft", "pool", "soft")]
    }
 where
  layerPart name from l =
    Part
      name
      (checked (towerLayerOf sequences m))
      m
      ( [ ("x", Earlier from "x")
        , ("patch", Host "patch")
        , ("axes", Host "axes")
        , ("reach", Host "reach")
        , ("slot", Host "slot")
        , ("k_cache", Own FirstRowMasks)
        , ("v_cache", Own FirstRowMasks)
        ]
          <> [(w, Tensor (l <> "." <> w)) | w <- ["norm", "q_proj", "k_proj", "v_proj", "q_norm", "k_norm", "o_proj"]]
      )

{- | An image of five patches over this many rows: rows past the fifth are
padding, holding the first patch's pixels, at their own positions.
-}
towerData :: Int -> StageData
towerData m =
  StageData
    { sdInputs = Map.fromList [("patches", take m (rows 5 12 (dyadic 20 32 60) <> repeat (take 12 (dyadic 20 32 12))))]
    , sdRowInts =
        Map.fromList
          [ ("places", take m [[c, 5 + r] | (c, r) <- [(0, 0), (1, 0), (2, 0), (0, 1), (1, 1)] <> repeat (0, 0)])
          , ("patch", map pure [0 .. fromIntegral m - 1])
          , ("axes", take m ([[0, 0], [1, 0], [2, 0], [0, 1], [1, 1]] <> repeat [0, 0]))
          , ("reach", [[max 4 r] | r <- [0 .. fromIntegral m - 1]])
          , ("slot", replicate m [0])
          , ("squares", [[0, 1], [2, 3], [4, 4]])
          ]
    , sdCaches = Map.empty
    , sdTensors =
        Map.fromList $
          [ ("patch_proj", dyadic 21 64 96)
          , ("position_table", dyadic 22 32 80)
          , ("std_bias", dyadic 23 32 8)
          , ("std_scale", dyadic 24 32 8)
          , ("projector", dyadic 25 64 80)
          ]
            <> concat
              [ [ (l <> ".norm", dyadic (s + 1) 32 8)
                , (l <> ".q_proj", dyadic (s + 2) 64 64)
                , (l <> ".k_proj", dyadic (s + 3) 64 64)
                , (l <> ".v_proj", dyadic (s + 4) 64 64)
                , (l <> ".q_norm", dyadic (s + 5) 32 4)
                , (l <> ".k_norm", dyadic (s + 6) 32 4)
                , (l <> ".o_proj", dyadic (s + 7) 64 64)
                ]
              | (l, s) <- [("l0", 30), ("l1", 40)]
              ]
    }

------------------------------------------------------------------------------

rows :: Int -> Int -> [Double] -> [[Double]]
rows m w xs = [take w (drop (r * w) xs) | r <- [0 .. m - 1]]

lowered :: Stage -> Lowered
lowered s = either (\fs -> error ("a test stage fails its gate: " <> show fs)) lowerStage (checkStage caps s)

{- | Every value of the stage's lowered program, interpreted on what the
host derives, as rows: a value @[1, w, 1, m]@ or @[1, h, d, m]@ is @m@
rows of @w@ or @h·d@.
-}
runLowered :: Stage -> StageData -> Either Text (Map Text [[Double]])
runLowered s sd = do
  let l = lowered s
  ins <- hostInputs l.loweredManifest sd
  ws <- hostWeights l.loweredManifest sd
  let shapes = Map.fromList [(i.inputName, map fromIntegral i.inputShape) | i <- l.loweredManifest.manifestInputs]
  values <- interpretValues l.loweredBuilt.builtModel (Map.fromList [(T.unpack n, tensor (shapes Map.! n) xs) | (n, xs) <- Map.toList ins]) ws
  pure $
    Map.fromList
      [ (T.pack n, [[xs !! (c * m + r) | c <- [0 .. h * w - 1]] | r <- [0 .. m - 1]])
      | (n, t) <- Map.toList values
      , let xs = tensorElems t
      , [1, h, w, m] <- [t.tensorShape]
      ]

{- | Per part, the rows its lowered program stored into the step's own
caches: each store's value cast to fp16, named after the part and the
cache.
-}
stored :: Stage -> Map Text [[Double]] -> Map Text Kept
stored s values =
  Map.fromList
    [ (p.partName, Kept Map.empty (Map.fromList kept))
    | p <- s.stageParts
    , let prog = checkedProgram p.partProgram
          kept =
            [ (src.sourceName, rowsOf)
            | n <- prog.programNodes
            , KvStore (StoreAt c _ _) _ <- [n.nodeOp]
            , Just src <- [programSource prog c]
            , Just (Own _) <- [partFeed p src.sourceName]
            , Just rowsOf <- [Map.lookup (p.partName <> "_" <> src.sourceName) values]
            ]
    , not (null kept)
    ]

closeTo :: Text -> [[Double]] -> [[Double]] -> Assertion
closeTo name want got = do
  unless (length want == length got && and (zipWith (\a b -> length a == length b) want got)) $
    assertFailure (T.unpack name <> ": shapes differ: " <> show (map length want) <> " and " <> show (map length got))
  forM_ (zip3 [0 :: Int ..] want got) $ \(r, ws, gs) ->
    forM_ (zip3 [0 :: Int ..] ws gs) $ \(c, w, g) ->
      unless (abs (w - g) <= 1e-9 * max 1 (abs w)) $
        assertFailure (T.unpack name <> " at row " <> show r <> ", element " <> show c <> ": " <> show w <> " evaluated, " <> show g <> " lowered")

{- | The stage's outputs, evaluated (reading what the lowered program stored
into its own caches) and lowered, agree: the first rows of an output, as
many as given, or all.
-}
agrees :: Stage -> StageData -> [(Text, Int)] -> Assertion
agrees s sd only = do
  got <- either (assertFailure . T.unpack) pure (runLowered s sd)
  want <- either (assertFailure . T.unpack) pure (evaluateStageStored (stored s got) (either (error . show) id (checkStage caps s)) sd)
  forM_ (Map.toList want) $ \(n, w) -> do
    let keep = maybe id take (lookup n only)
    case Map.lookup n got of
      Nothing -> assertFailure ("no lowered output " <> T.unpack n)
      Just g -> closeTo n (keep w) (keep g)

golden :: FilePath
golden = "test/golden/draft_step.json"

findings :: Stage -> [Text]
findings = findingsWith caps

findingsWith :: Caps -> Stage -> [Text]
findingsWith c s = either (map (\f -> f.findingWhere <> ": " <> f.findingMessage)) (const []) (checkStage c s)

------------------------------------------------------------------------------
-- Names

-- | Two parts whose values' generated names meet: @a@'s @b_c@ and @a_b@'s @c@.
meetingNames :: Stage
meetingNames =
  Stage
    "names"
    [ Part "a" (checked (one "t_a" "b_c")) 2 [("x", Host "x"), ("w", Tensor "w1")]
    , Part "a_b" (checked (one "t_ab" "c")) 2 [("x", Earlier "a" "y"), ("w", Tensor "w2")]
    ]
    [("y", "a_b", "y")]
 where
  one name value = B.program name $ do
    x <- B.input "x" (wholeRow 8) TF32
    w <- B.weight "w" [8, 8]
    y <- B.linear value w 0 x
    B.output "y" y

-- | Every name the program's main function has: its inputs and every op's output.
programNames :: Lowered -> [String]
programNames l = case lookup "main" l.loweredBuilt.builtModel.modelFunctions of
  Just fn -> map fst fn.fnInputs <> [n | o <- fn.fnOperations, (n, _) <- o.opOutputs]
  Nothing -> []

-- | Whether every op reads only names made before it: the inputs or an earlier op's.
resolves :: Lowered -> Bool
resolves l = case lookup "main" l.loweredBuilt.builtModel.modelFunctions of
  Just fn -> go (map fst fn.fnInputs) fn.fnOperations
  Nothing -> False
 where
  go _ [] = True
  go known (o : rest) =
    and [n `elem` known | (_, as) <- o.opInputs, ArgName n <- as] && go (map fst o.opOutputs <> known) rest

------------------------------------------------------------------------------
-- The caps

{- | Two linear layers over 16384 channels, of these outputs, then this
many scales, each by a tensor of its own.
-}
wideStage :: Natural -> Natural -> Int -> Stage
wideStage a b scales =
  Stage
    "wide"
    [Part "p" (checked prog) 1 ([("x", Host "x"), ("w1", Tensor "w1"), ("w2", Tensor "w2")] <> [(s, Tensor s) | s <- names])]
    [("y", "p", "y")]
 where
  names = [T.pack ("s" <> show i) | i <- [1 .. scales]]
  prog = B.program "t_wide" $ do
    x <- B.input "x" (wholeRow 16384) TF32
    w1 <- B.weight "w1" [a, 16384]
    w2 <- B.weight "w2" [b, a]
    h <- B.linear "h" w1 0 x
    y <- B.linear "y0" w2 0 h
    z <- foldM (\v s -> B.weight s [1] >>= \w -> B.scaleBy ("z_" <> s) w v) y names
    B.output "y" z

-- | A linear layer over this many rows.
rowsStage :: Natural -> Stage
rowsStage n =
  Stage
    "rows"
    [Part "p" (checked prog) n [("x", Host "x"), ("w", Tensor "w")]]
    [("y", "p", "y")]
 where
  prog = B.program "t_rows" $ do
    x <- B.input "x" (wholeRow 8) TF32
    w <- B.weight "w" [8, 8]
    y <- B.linear "y" w 0 x
    B.output "y" y

-- | One gamma at two factors, so two blobs of it.
twoFactors :: Stage
twoFactors =
  Stage
    "factors"
    [Part "p" (checked prog) 2 [("x", Host "x"), ("g", Tensor "g")]]
    [("y", "p", "y"), ("z", "p", "z")]
 where
  prog = B.program "t_factors" $ do
    x <- B.input "x" (wholeRow 8) TF32
    g <- B.weight "g" [8]
    y <- B.rmsnorm "y" eps (Just g) 1 x
    z <- B.rmsnorm "z" eps (Just g) 2 x
    B.output "y" y
    B.output "z" z

tests :: TestTree
tests =
  testGroup
    "Lower"
    [ testCase "a drafting step lowered computes what its programs mean" $
        agrees draftStage draftData []
    , testCase "an image's tower lowered computes what its programs mean" $
        agrees (towerStage 5) (towerData 5) []
    , testCase "an image's padding rows change none of its soft tokens, interpreted in f64" $ do
        exact <- either (assertFailure . T.unpack) pure (runLowered (towerStage 5) (towerData 5))
        padded <- either (assertFailure . T.unpack) pure (runLowered (towerStage 8) (towerData 8))
        case (Map.lookup "soft" exact, Map.lookup "soft" padded) of
          (Just e, Just p) -> closeTo "soft" e p
          _ -> assertFailure "no soft tokens"
    , testCase "a wide layer is split into the caps' channels" $ do
        let l = lowered draftStage
            heads = [w | w <- l.loweredManifest.manifestWeights, w.blobTensor == "head"]
        map (\w -> (w.blobOffset, w.blobShape)) heads @?= [(0, [16, 6, 1, 1]), (96, [16, 6, 1, 1]), (192, [8, 6, 1, 1])]
    , testCase "a norm's factor goes into its gamma" $ do
        let l = lowered draftStage
        [w.blobScale | w <- l.loweredManifest.manifestWeights, w.blobTensor == "final_norm"] @?= [2]
    , testCase "the host's inputs: shared where the parts share them" $ do
        let l = lowered draftStage
        map (\i -> (i.inputName, i.inputShape)) l.loweredManifest.manifestInputs
          @?= [ ("hidden", [1, 8, 1, 2])
              , ("gathered0", [1, 8, 1, 2])
              , ("rope_cos0", [1, 1, 4, 2])
              , ("rope_sin0", [1, 1, 4, 2])
              , ("k_ring", [1, 1, 6, 4])
              , ("v_ring", [1, 1, 4, 6])
              , ("mask0", [1, 1, 6, 2])
              , ("rope_cos1", [1, 1, 4, 2])
              , ("rope_sin1", [1, 1, 4, 2])
              , ("k_paged", [1, 2, 16, 4])
              , ("v_paged", [1, 2, 4, 16])
              , ("mask1", [1, 1, 16, 2])
              ]
        let t = lowered (towerStage 6)
        [(i.inputName, i.inputShape) | i <- t.loweredManifest.manifestInputs, i.inputName `elem` ["mask0", "selection0"]]
          @?= [("mask0", [1, 1, 6, 1]), ("selection0", [1, 1, 6, 3])]
    , testCase "the manifest is its golden text" $ do
        let got = renderManifest (lowered draftStage).loweredManifest
        want <- either (const Nothing) Just <$> (try (TIO.readFile golden) :: IO (Either IOException Text))
        unless (Just got == want) $ do
          tmp <- getTemporaryDirectory
          TIO.writeFile (tmp </> "draft_step.json") got
          assertFailure ("the manifest changed; it is in " <> tmp </> "draft_step.json, to replace " <> golden)
    , testCase "the gate names what the ANE doesn't run" $ do
        let routed = B.program "t_routed" $ do
              x <- B.input "x" (wholeRow 8) TF32
              r <- B.weight "router" [4, 8]
              logits <- B.linear "logits" r 0 x
              (ids, _) <- B.route "route" 2 Nothing logits
              B.output "ids" ids
            s =
              Stage
                "bad"
                [Part "p" (checked routed) 1 [("x", Host "x"), ("router", Earlier "q" "y")]]
                [("ids", "p", "ids"), ("x", "p", "ids")]
        findings s
          @?= [ "p.router: A weight can't be fed by the part q's y."
              , "p.logits: A linear layer's weight is a model's tensor."
              , "p.route_ids: The ANE runs no routing."
              , "x: An output named as an input."
              ]
        let narrowRing = draftStage {stageParts = [p {partFeeds = map shorten p.partFeeds} | p <- draftStage.stageParts]}
            shorten (n, Mirror m 6) = (n, Mirror m 5)
            shorten f = f
        assertBool "a ring's mirror is its ring" ("ring.k_cache: A ring's mirror is its ring, of 6 positions, not 5." `elem` findings narrowRing)
        let swapped = draftStage {stageParts = [p {partFeeds = map swap p.partFeeds} | p <- draftStage.stageParts, p.partName /= "paged"]}
            swap ("v_cache", Mirror _ len) = ("v_cache", Mirror "k_ring" len)
            swap f = f
        let reservedName = draftStage {stageParts = [p {partFeeds = map rename p.partFeeds} | p <- draftStage.stageParts]}
            rename ("state", Host _) = ("state", Host "state")
            rename f = f
        assertBool "a name Core ML reserves" ("in.state: The host's name state is a word Core ML's program parser reserves." `elem` findings reservedName)
        assertBool "a mirror is keys or values" ("k_ring: One mirror, read as keys and values, or by attention of other heads." `elem` findings swapped)
        -- Rows of two sequences over the step's own rows: the lowering's
        -- masks would let each see the other's.
        findings (towerStageWith 2 4)
          @?= [ "layer0.att: Attention over the step's own rows is of one sequence: its sequences are row integers below 1, not 2."
              , "layer1.att: Attention over the step's own rows is of one sequence: its sequences are row integers below 1, not 2."
              ]
    , testCase "the interpreter's fp16, and the ANE's activations as observed" $ do
        let inf = 1 / 0 :: Double
        -- binary16: to nearest, ties to even; past its largest, infinite.
        map (I.roundAs I.Binary16) [1 + 2 ^^ (-11 :: Int), 1 + 3 * 2 ^^ (-11 :: Int), 65519, 65520]
          @?= [1, 1 + 2 ^^ (-9 :: Int), 65504, inf]
        -- The ANE's ops but its sums: binary16, ties away from zero, its
        -- whole range.
        map (I.roundAs I.NeuralEngine) [1 + 2 ^^ (-11 :: Int), -(1 + 2 ^^ (-11 :: Int)), 2 ^^ (-25 :: Int), 40000, 65519, 65520]
          @?= [1 + 2 ^^ (-10 :: Int), -(1 + 2 ^^ (-10 :: Int)), 2 ^^ (-24 :: Int), 40000, 65504, inf]
        -- Its sums: products by their operands' exponents, dropped below -16
        -- (1.5·2^-9 · 1.5·2^-8 too, though over 2^-16); rounded to 2^-16,
        -- ties away; past 32768 on the way, infinite; the sum ties away.
        map
          I.neuralEngineDot
          [ [(2 ^^ (-8 :: Int), 2 ^^ (-8 :: Int))]
          , [(2 ^^ (-8 :: Int), 2 ^^ (-9 :: Int))]
          , [(1.5 * 2 ^^ (-9 :: Int), 1.5 * 2 ^^ (-8 :: Int))]
          , [(5 * 2 ^^ (-9 :: Int), 2 ^^ (-8 :: Int)), (-(5 * 2 ^^ (-9 :: Int)), 2 ^^ (-8 :: Int)), (5 * 2 ^^ (-9 :: Int), 2 ^^ (-8 :: Int))]
          , [(16384, 1), (16384, 1), (-16384, 1), (-16384, 1)]
          , [(-16384, 1), (-16384, 1)]
          , [(1, 1), (2 ^^ (-11 :: Int), 1)]
          ]
          @?= [2 ^^ (-16 :: Int), 0, 0, 3 * 2 ^^ (-16 :: Int), inf, -32768, 1 + 2 ^^ (-10 :: Int)]
        -- A lowered stage in fp16 rounds, and stays within fp16 of f64.
        let l = lowered draftStage
            run p = do
              ins <- hostInputs l.loweredManifest draftData
              ws <- hostWeights l.loweredManifest draftData
              let shapes = Map.fromList [(i.inputName, map fromIntegral i.inputShape) | i <- l.loweredManifest.manifestInputs]
              I.interpretValuesWith p l.loweredBuilt.builtModel (Map.fromList [(T.unpack n, tensor (shapes Map.! n) xs) | (n, xs) <- Map.toList ins]) ws
        case (run I.Exact, run I.Binary16) of
          (Right exact, Right half) -> do
            let out = tensorElems . (Map.! "next_state")
                worst = maximum (zipWith (\a b -> abs (a - b)) (out exact) (out half))
                size = maximum (map abs (out exact))
            assertBool ("fp16 moved next_state by " <> show (worst / size) <> " of its size") (worst > 0 && worst <= 1e-2 * size)
          (e, h) -> assertFailure (show (either Just (const Nothing) e, either Just (const Nothing) h))
    , testCase "the gate counts the weights the lowering reserves" $ do
        forM_ [draftStage, towerStage 6, towerStage 8, twoFactors] $ \s -> do
          let l = lowered s
          stageWeightFile caps s @?= fromIntegral l.loweredManifest.manifestWeightFile
        stageWeights caps twoFactors @?= [Reservation "g" 0 [1, 8, 1, 1] 1, Reservation "g" 0 [1, 8, 1, 1] 2]
    , testCase "the gate holds a stage to the ANE's measured caps" $ do
        -- Data of 2^30 - 32768 bytes in 2 blobs, then in 256 more of 2 bytes:
        -- 32768 bytes of records and padding more, past 2^30. And data of 2^30.
        findingsWith appleNE (wideStage 16384 16383 0) @?= []
        findingsWith appleNE (wideStage 16384 16383 256)
          @?= ["wide: The stage's weight file takes 1073742016 bytes, more than the ANE's 1073741824."]
        findingsWith appleNE (wideStage 16384 16384 0)
          @?= ["wide: The stage's weight file takes 1073742016 bytes, more than the ANE's 1073741824."]
        findings (rowsStage 16384) @?= []
        findings (rowsStage 16385) @?= ["p: A part of 16385 rows, more than the ANE's 16384 positions."]
        let longPaged = draftStage {stageParts = [if p.partName == "paged" then p {partFeeds = map longer p.partFeeds} else p | p <- draftStage.stageParts]}
            longer (n, Mirror m _) = (n, Mirror m 16385)
            longer f = f
        assertBool "a mirror past the ANE's positions" ("paged.k_cache: A mirror of 16385 positions, more than the ANE's 16384." `elem` findings longPaged)
    , testCase "every name of a lowered program is its own, and read after it is made" $ do
        let outs s = s.stageOutputs
            cases =
              [ ("an output named as a derived input", draftStage {stageOutputs = ("mask0", "out", "token") : drop 1 (outs draftStage)})
              , ("a host input named as a generated const", draftStage {stageParts = [p {partFeeds = [(n, case f of Host "hidden" -> Host "w_3"; _ -> f) | (n, f) <- p.partFeeds]} | p <- draftStage.stageParts]})
              , ("an output named as a part's value", draftStage {stageOutputs = ("ring_q", "out", "token") : drop 1 (outs draftStage)})
              , ("one value as two outputs", draftStage {stageOutputs = outs draftStage <> [("state_again", "out", "state")]})
              , ("two parts' values named alike", meetingNames)
              ]
        forM_ cases $ \(what, s) -> do
          let l = lowered s
              ns = programNames l
          assertBool (what <> ": " <> show [n | n <- nub ns, length (filter (== n) ns) > 1]) (length (nub ns) == length ns)
          assertBool (what <> ": a name read before it is made") (resolves l)
          [o.outputName | o <- l.loweredManifest.manifestOutputs] @?= [o | (o, _, _) <- s.stageOutputs]
    , testCase "a head's norm moves its axis last, a row's normalizes its channels" $ do
        let axesOf l = case lookup "main" l.loweredBuilt.builtModel.modelFunctions of
              Just fn ->
                let consts = [(n, v) | o <- fn.fnOperations, o.opType == "const", [(n, _)] <- [o.opOutputs], Just v <- [lookup "val" o.opAttributes]]
                 in [v | o <- fn.fnOperations, o.opType == "layer_norm", Just [ArgName a] <- [lookup "axes" o.opInputs], Just v <- [lookup a consts]]
              Nothing -> []
        forM_ [draftStage, towerStage 6] $ \s -> do
          let axes = axesOf (lowered s)
          assertBool (show axes) (not (null axes) && all (`elem` [Immediate (TensorType Int32 [1]) (Ints [1]), Immediate (TensorType Int32 [1]) (Ints [3])]) axes)
    , testCase "a lowered stage is fp16, its argmax int32" $ do
        let l = lowered draftStage
        [(o.outputName, o.outputType) | o <- l.loweredManifest.manifestOutputs]
          @?= [("next_token", OutputIndex), ("next_state", OutputHalf), ("logits", OutputHalf)]
        length l.loweredBuilt.builtModel.modelInputs @?= 12
    ]
