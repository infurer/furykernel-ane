-- SPDX-FileCopyrightText: 2026 Mika Tammi
-- SPDX-License-Identifier: BSD-3-Clause

{- | The host's side of a lowered stage, as a reference: what it derives
for each of the stage's inputs, and what it writes into each weight's
blob, from what a step of the stage's programs is given ('StageData'). A
runtime derives the same; tests hold the lowering to the programs'
meaning through it: the stage evaluated part by part ('evaluateStage')
against its lowered program run on these inputs.

The host's rules:

* a mirror holds one sequence: every row that reads it is of the
  sequence the first row names, and its positions from the window's start
  to the row's last are in the mirror (a ring as long as the cache's, a
  linear mirror as long as the sequence);
* a cache of a step's own rows holds what the step stores: every position
  a row sees is a row's of the step, and the step's rows are one
  sequence's (the gate holds its sequences' row integers below 1), since
  the lowering masks them by position alone;
* every row the stage computes holds finite values, the rows nobody reads
  included (an image's padding takes a real patch's pixels): a mask hides
  keys behind a large negative number, which a NaN would see through;
* attention's values are below 128 in magnitude, a mirror's the host
  writes and the step's own rows of them alike: attention weighs them by
  its probabilities times 2^8, and the ANE's sums overflow past 32768,
  to infinity.
-}
module FuryKernel.Ane.Host (
  StageData (..),
  evaluateStage,
  evaluateStageStored,
  hostInputs,
  hostWeights,
) where

import Control.Monad (foldM, forM, forM_, unless, when)
import Data.Array.Unboxed (UArray, listArray, (!))
import Data.List (transpose)
import Data.Map.Strict (Map)
import Data.Map.Strict qualified as Map
import Data.Maybe (fromMaybe)
import Data.Text (Text)
import Data.Word (Word32, Word64)
import FuryKernel.Ane.Lower (maskFill)
import FuryKernel.Ane.Manifest qualified as M
import FuryKernel.Ane.Stage
import FuryKernel.Program.Check (checkedProgram)
import FuryKernel.Program.Eval (Bindings (..), Evaluation (..), Kept, evaluateStored)
import FuryKernel.Program.Eval qualified as Eval
import FuryKernel.Program.Graph

{- | What a step of a stage is given, by the stage's names: the host's
activations (a list per row) and row integers, the caches its mirrors
copy (each as the programs' cache holds it, under its mirror's name), and
the model's tensors (row-major, widened).
-}
data StageData = StageData
  { sdInputs :: Map Text [[Double]]
  , sdRowInts :: Map Text [[Word32]]
  , sdCaches :: Map Text [Double]
  , sdTensors :: Map Text [Double]
  }

lookupE :: Text -> Text -> Map Text a -> Either Text a
lookupE what n m = maybe (Left ("No " <> what <> " " <> n <> ".")) Right (Map.lookup n m)

-- | A cache or a table as an array, read element by element.
indexed :: [Double] -> UArray Int Double
indexed xs = listArray (0, length xs - 1) xs

{- | The stage's outputs, each part evaluated in f64 by the core's
evaluator, the parts' outputs feeding later parts.
-}
evaluateStage :: CheckedStage -> StageData -> Either Text (Map Text [[Double]])
evaluateStage = evaluateStageStored Map.empty

{- | 'evaluateStage', given per part what its lowered program stored into
the step's own caches: the core's evaluator rounds what a program stores
into a cache to the cache's element, fp16 for a cache of fp16
('CacheF16', as the ANE keeps the rows it attends over), which the
lowering's cast to fp16 gives too. With the rows stored, the parts read
what the lowered program read.
-}
evaluateStageStored :: Map Text Kept -> CheckedStage -> StageData -> Either Text (Map Text [[Double]])
evaluateStageStored kept checked sd = do
  outs <- foldM part Map.empty stage.stageParts
  fmap Map.fromList . forM stage.stageOutputs $ \(o, pn, on) ->
    (o,) <$> maybe (Left ("No output " <> on <> " of " <> pn <> ".")) Right (Map.lookup (pn, on) outs)
 where
  stage = checkedStage checked
  part outs p = do
    let prog = checkedProgram p.partProgram
        fed s = maybe (Left ("No feed for " <> s.sourceName <> ".")) Right (partFeed p s.sourceName)
        earlier pn on = maybe (Left ("No output " <> on <> " of " <> pn <> ".")) Right (Map.lookup (pn, on) outs)
    bound <- forM prog.programSources $ \s -> do
      f <- fed s
      case (s.sourceKind, f) of
        (SInput {}, Host n) -> (\v -> mempty {bInputs = [(s.sourceName, v)]}) <$> lookupE "input" n sd.sdInputs
        (SInput {}, Earlier pn on) -> (\v -> mempty {bInputs = [(s.sourceName, v)]}) <$> earlier pn on
        (SRowInts _ _, Host n) -> (\v -> mempty {bInts = [(s.sourceName, v)]}) <$> lookupE "row integers" n sd.sdRowInts
        (SWeight {}, Tensor t) -> (\v -> mempty {bWeights = [(s.sourceName, v)]}) <$> lookupE "tensor" t sd.sdTensors
        (SWeight {}, Earlier pn on) -> (\v -> mempty {bWeights = [(s.sourceName, concat v)]}) <$> earlier pn on
        (SCache l _, Mirror n _) -> (\v -> mempty {bCaches = [(s.sourceName, unitsOf l v)]}) <$> lookupE "cache" n sd.sdCaches
        (SCache l _, Own _) -> pure mempty {bCaches = [(s.sourceName, unitsOf l (replicate (cacheSize l) 0))]}
        _ -> Left ("The source " <> s.sourceName <> " is fed as the gate lets nothing be.")
    let b = mconcat bound
        bindings =
          Bindings
            { bindRows = fromIntegral p.partRows
            , bindInputs = Map.fromList b.bInputs
            , bindRowInts = Map.fromList b.bInts
            , bindWeights = Map.fromList b.bWeights
            , bindTables = Map.empty
            , bindCaches = Map.fromList b.bCaches
            }
    let run = maybe Eval.evaluate evaluateStored (Map.lookup p.partName kept)
    e <- either (\m -> Left (p.partName <> ": " <> m)) Right (run p.partProgram bindings)
    pure (Map.union outs (Map.fromList [((p.partName, o), v) | (o, v) <- Map.toList e.evalOutputs]))

data Bound = Bound
  { bInputs :: [(Text, [[Double]])]
  , bInts :: [(Text, [[Word32]])]
  , bWeights :: [(Text, [Double])]
  , bCaches :: [(Text, Map Int [Double])]
  }

instance Semigroup Bound where
  Bound a b c d <> Bound a' b' c' d' = Bound (a <> a') (b <> b') (c <> c') (d <> d')

instance Monoid Bound where
  mempty = Bound [] [] [] []

cacheSize :: CacheLayout -> Int
cacheSize = \case
  Ring s h n d -> fromIntegral (s * h * n * d)
  Paged b n h d -> fromIntegral (b * n * h * d)

-- | A cache's elements as its slots or blocks, from the first.
unitsOf :: CacheLayout -> [Double] -> Map Int [Double]
unitsOf layout xs = Map.fromList (zip [0 ..] (takeWhile (not . null) (map (take n) (iterate (drop n) xs))))
 where
  n = fromIntegral $ case layout of
    Ring _ h len d -> h * len * d
    Paged _ size h d -> size * h * d

{- | Every input of the lowered stage, as the host derives it: its
elements in the order of its shape's axes, the last fastest.
-}
hostInputs :: M.Manifest -> StageData -> Either Text (Map Text [Double])
hostInputs manifest sd = Map.fromList <$> mapM one manifest.manifestInputs
 where
  one i = (i.inputName,) <$> either (\m -> Left (i.inputName <> ": " <> m)) Right (derive i)
  ints n = lookupE "row integers" n sd.sdRowInts
  firstInt n = do
    rs <- ints n
    case rs of
      (x : _) : _ -> pure (fromIntegral x :: Int)
      _ -> Left ("No first row in " <> n <> ".")
  derive i = case (i.inputKind, map fromIntegral i.inputShape :: [Int]) of
    (M.Activation, [1, w, 1, m]) -> do
      rows <- lookupE "input" i.inputName sd.sdInputs
      unless (length rows == m && all ((== w) . length) rows) $ Left "Not its rows."
      pure (concat (transpose rows))
    (M.MirrorKeys spec, [1, h, n, d]) -> do
      at <- mirrorIndex spec i.inputName
      cache <- indexed <$> lookupE "cache" i.inputName sd.sdCaches
      pure [maybe 0 (cache !) (at kv j e) | kv <- [0 .. h - 1], j <- [0 .. n - 1], e <- [0 .. d - 1]]
    (M.MirrorValues spec, [1, h, d, n]) -> do
      at <- mirrorIndex spec i.inputName
      cache <- indexed <$> lookupE "cache" i.inputName sd.sdCaches
      pure [maybe 0 (cache !) (at kv j e) | kv <- [0 .. h - 1], e <- [0 .. d - 1], j <- [0 .. n - 1]]
    (M.RopeCos t, [1, 1, d, m]) -> rope cos t d m
    (M.RopeSin t, [1, 1, d, m]) -> rope sin t d m
    (M.Mask spec, [1, 1, keys, m]) -> mask spec keys m
    (M.Gathered g, [1, w, 1, m]) -> do
      table <- indexed <$> lookupE "tensor" g.gatherTensor sd.sdTensors
      toks <- ints g.gatherTokens
      unless (length toks == m) $ Left "Not its rows."
      pure [g.gatherFactor * sum [table ! (fromIntegral t * w + c) | t <- ts] | c <- [0 .. w - 1], ts <- toks]
    (M.Selection s, [1, 1, rowsIn, m]) -> do
      toks <- ints s.selectionTokens
      unless (length toks == m) $ Left "Not its rows."
      pure [s.selectionFactor * fromIntegral (length (filter ((== r) . fromIntegral) ts)) | r <- [0 .. rowsIn - 1], ts <- toks]
    _ -> Left "An input of a shape its kind doesn't take."

  -- Where a mirror's element (head, position, element) is in the cache.
  mirrorIndex spec name = do
    let h = fromIntegral spec.mirrorKvHeads :: Int
        d = fromIntegral spec.mirrorHeadDim :: Int
    case spec.mirrorLayout of
      M.MirrorRing n -> do
        slot <- firstInt spec.mirrorSequence
        let len = fromIntegral n :: Int
        pure (\kv j e -> Just (((slot * h + kv) * len + j) * d + e))
      M.MirrorLinear _ b -> do
        rows <- ints spec.mirrorSequence
        table <- case rows of
          t : _ -> pure (map fromIntegral t :: [Int])
          [] -> Left ("No block table in " <> name <> ".")
        let block = fromIntegral b :: Int
        pure $ \kv j e -> case drop (j `div` block) table of
          blk : _ -> Just (((blk * h + kv) * block + j `mod` block) * d + e)
          [] -> Nothing

  rope f t d m = do
    rows <- ints t.ropePositions
    unless (length rows == m) $ Left "Not its rows."
    let axes = fromIntegral t.ropeAxes :: Int
        part = d `div` axes
        half = part `div` 2
        angle ps i =
          let a = i `div` part
              pair = (i `mod` part) `mod` half
           in if pair < fromIntegral t.ropeRotated
                then fromIntegral (ps !! a) * t.ropeTheta ** negate (fromIntegral (2 * pair) / fromIntegral part)
                else 0
    pure [f (angle ps i) | i <- [0 .. d - 1], ps <- rows]

  -- Each row's first integer.
  firsts n =
    ints n
      >>= mapM
        ( \case
            x : _ -> pure (fromIntegral x :: Int)
            [] -> Left ("A row of no integers in " <> n <> ".")
        )

  mask spec keys m = do
    positions <- firsts spec.maskPositions
    reaches <- maybe (pure positions) firsts spec.maskReach
    let rowsSeen = if spec.maskShared then 1 else m
    unless (length positions >= rowsSeen) $ Left "Fewer rows than the mask's."
    keyPositions <- case spec.maskKeys of
      M.RowKeys kp -> Just <$> firsts kp
      _ -> pure Nothing
    cols <- forM (take rowsSeen (zip positions reaches)) $ \(p, r) -> do
      let hi = max p r :: Int
          first = maybe 0 (\w -> max 0 (p - fromIntegral w + 1)) spec.maskWindow
      case spec.maskKeys of
        M.RingKeys n -> do
          let len = fromIntegral n
          when (hi - first + 1 > len) $ Left "A row sees more positions than the ring keeps."
          pure [let t = hi - ((hi - j) `mod` len) in t >= first && t >= 0 | j <- [0 .. keys - 1]]
        M.LinearKeys n -> do
          when (hi >= fromIntegral n) $ Left "A row reaches past the mirror."
          pure [j >= first && j <= hi | j <- [0 .. keys - 1]]
        M.RowKeys _ -> do
          let qs = fromMaybe [] keyPositions
          forM_ [first .. hi] $ \t -> unless (t `elem` qs) $ Left "A row sees a position no row of the step stores."
          pure [q >= first && q <= hi | q <- qs]
    let seenBy = [listArray (0, length col - 1) col :: UArray Int Bool | col <- cols]
    pure [if seen then 0 else maskFill | j <- [0 .. keys - 1], col <- seenBy, let seen = col ! j]

{- | Each weight blob's elements, by its record's offset: the tensor's from
the blob's offset, times its scale.
-}
hostWeights :: M.Manifest -> StageData -> Either Text (Map Word64 [Double])
hostWeights manifest sd = Map.fromList <$> mapM one manifest.manifestWeights
 where
  one w = do
    t <- lookupE "tensor" w.blobTensor sd.sdTensors
    let count = fromIntegral (product w.blobShape)
        xs = take count (drop (fromIntegral w.blobOffset) t)
    unless (length xs == count) $ Left ("The tensor " <> w.blobTensor <> " is shorter than its blob.")
    pure (w.blobRecord, map (* w.blobScale) xs)
