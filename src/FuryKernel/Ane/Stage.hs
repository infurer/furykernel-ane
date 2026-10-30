-- SPDX-FileCopyrightText: 2026 Mika Tammi
-- SPDX-License-Identifier: BSD-3-Clause

{- | Stages: the programs one prediction runs, one after another, and how
each of their sources is fed.

The ANE runs a whole ML program per prediction, so the programs of a
tower's step (a draft step's input, its layers and its output; an image's
embedding, its encoder's layers and its pooling) are lowered into one. A
'Part' is one program of the stage, at a number of rows, with a 'Feed' for
each of its sources:

* an activation comes from the host ('Host': a prediction input of that
  name) or is an earlier part's output ('Earlier');
* row integers come from the host ('Host'). The ANE never reads them: the
  host derives from them what the ANE reads instead, rotary tables, masks,
  gathered rows ("FuryKernel.Ane.Host");
* a weight is a model's tensor ('Tensor'), or the rows of an earlier
  part's output ('Earlier'), which a gather pools;
* a KV cache is a mirror the host keeps of one sequence's keys or values,
  in fp16, of a number of positions ('Mirror'), or the step's own rows,
  which it stores and attends over at once ('Own': self-attention).

Host names belong to the stage: parts fed by the same name read the same
input. 'checkStage' holds a stage to what the ANE runs ('Caps') and to
what the lowering takes; the lowering takes only checked stages.
-}
module FuryKernel.Ane.Stage (
  -- * Stages
  Stage (..),
  Part (..),
  Feed (..),
  Masks (..),
  partFeed,

  -- * The Neural Engine's limits
  Caps (..),
  appleNE,
  Reservation (..),
  stageWeights,
  stageWeightFile,
  linearChunks,
  gammaShape,

  -- * The gate
  StageFinding (..),
  CheckedStage,
  checkedStage,
  stageCaps,
  checkStage,
) where

import Control.Monad (forM_, unless, when)
import Control.Monad.Trans.State.Strict (State, execState, modify')
import Data.Char (isAsciiLower, isAsciiUpper, isDigit)
import Data.Containers.ListUtils (nubOrd)
import Data.List (find, nub)
import Data.Map.Strict qualified as Map
import Data.Maybe (isNothing, listToMaybe)
import Data.Text (Text)
import Data.Text qualified as T
import FuryKernel.IR (Name)
import FuryKernel.Program.Check (CheckedProgram, checkedProgram)
import FuryKernel.Program.Graph
import FuryKernel.Type (ScalarType (..))
import Numeric.Natural (Natural)

------------------------------------------------------------------------------
-- Stages

data Stage = Stage
  { stageName :: Text
  , stageParts :: [Part]
  -- ^ In the order they run.
  , stageOutputs :: [(Text, Text, Name)]
  -- ^ The prediction's outputs: each a name, and the part and output it is.
  }

data Part = Part
  { partName :: Text
  , partProgram :: CheckedProgram
  , partRows :: Natural
  , partFeeds :: [(Name, Feed)]
  -- ^ How each of the program's sources is fed, by its name.
  }

data Feed
  = -- | The host's input of this name.
    Host Text
  | -- | An earlier part's output: the part and the output's name.
    Earlier Text Name
  | -- | A model's tensor, by name.
    Tensor Text
  | {- | A cache's mirror, named, of this many positions: one sequence's,
    in fp16. A ring cache's mirror is its sequence's ring, as long; a paged
    cache's holds the positions from the first.
    -}
    Mirror Text Natural
  | -- | A cache of the step's own rows: stored, then attended over.
    Own Masks
  deriving stock (Eq, Show)

{- | Which keys the rows of an 'Own' cache see: each row its own, or every
row what the first row sees, the host's one mask for all (an image's
patches, all of which see the image; rows after it are padding, which the
host doesn't read).
-}
data Masks = RowMasks | FirstRowMasks
  deriving stock (Eq, Show)

-- | How a part feeds a source of its program.
partFeed :: Part -> Name -> Maybe Feed
partFeed part n = lookup n part.partFeeds

------------------------------------------------------------------------------
-- Caps

{- | What the Neural Engine runs, as the lowering takes it.

Apple documents none of these limits. Those of 'appleNE' were measured: a
program past its length or weight cap is one Core ML's placement
('MLComputePlan') puts off the ANE, onto the CPU or GPU; its channels are
the lowering's choice.

The caps are necessary, not sufficient: Core ML's cost model can put a
whole program on the CPU though the ANE takes each of its ops, as it did,
on an M3 Max under macOS 27.2, with two decode rows of attention over a
mirror of 16384 positions (16 rows went to the ANE) and with a 1024-wide
RMSNorm over 16 rows (over 256, with a projection after it, to the ANE).
A runtime asserts the placement it needs with @MLComputePlan@.

Nor do they hold the values, which the gate doesn't see: the ANE's sums
overflow past 32768, to infinity, whatever their result (see
"FuryKernel.Ane.Lower"), and attention scales its probabilities by 2^8
before they weigh the values, so attention's values, a mirror's and the
step's own, must be below 128 in magnitude.
-}
data Caps = Caps
  { capsChannels :: Natural
  {- ^ The most channels an op's output has: a wider linear layer becomes
  several, each of at most this many outputs.
  -}
  , capsLength :: Natural
  {- ^ The most positions a program's last axis has: a part's rows, and a
  mirror's positions.
  -}
  , capsWeightBytes :: Natural
  {- ^ The most bytes a stage's weight file takes ('stageWeightFile'): its
  header, and a record and the data of each blob its program reserves
  ('stageWeights', one for each part of a model's tensor at each factor),
  each padded to 64 bytes.
  -}
  }
  deriving stock (Eq, Show)

{- | The Neural Engine as measured on an M3 Max under macOS 27.2, where
Core ML placed each of these programs on the ANE and the next one up off
it:

* outputs of up to 16384 channels a linear layer. This one is the lowering's
  choice, not the ANE's: convs of 65536 output or input channels were
  placed there too, so splitting at 16384 is harmless rather than needed;
* up to 16384 positions a last axis: a conv over 16385 rows was supported
  only on the CPU, and attention over a mirror of 16640 positions only on
  the CPU and GPU;
* a weight file of up to 2^30 bytes a stage: a program whose weight file
  was 1073741824 bytes was placed on the ANE, the same weights in one blob
  more, 1073741888 bytes, on the CPU, every op of it;

Other Macs may differ: M1's are to be measured when work on it starts.
-}
appleNE :: Caps
appleNE =
  Caps
    { capsChannels = 16384
    , capsLength = 16384
    , capsWeightBytes = 2 ^ (30 :: Int)
    }

{- | A blob of a program's weights the lowering reserves: a model's tensor,
from an element on, in a shape, its elements times a factor. It reserves
one for each such distinct part, so a tensor read whole and in parts, or
at two factors, takes two.
-}
data Reservation = Reservation
  { resTensor :: Text
  , resFrom :: Natural
  , resShape :: [Natural]
  , resScale :: Double
  }
  deriving stock (Eq, Ord, Show)

{- | The blobs the lowering of a stage reserves, distinct: each linear
layer's weight in its chunks ('linearChunks'), each norm's gamma at its
factor ('gammaShape'), each scale's and each standardization's tensors.
A table only gathered from is the host's to read, so none of it.
-}
stageWeights :: Caps -> Stage -> [Reservation]
stageWeights caps stage =
  nubOrd
    [ r
    | p <- stage.stageParts
    , let prog = checkedProgram p.partProgram
          tensor sid = do
            src <- programSource prog sid
            Tensor t <- partFeed p src.sourceName
            pure t
          width vid = maybe 0 (.valueShape.shapeWidth) (programValue prog vid)
    , n <- prog.programNodes
    , r <- case n.nodeOp of
        Linear w _ x
          | Just t <- tensor w
          , [o] <- n.nodeOutputs ->
              let ins = width x
               in [Reservation t (from * ins) [k, ins, 1, 1] 1 | (from, k) <- linearChunks caps (width o)]
        RmsNorm _ (Just g) f x
          | Just t <- tensor g
          , Just v <- programValue prog x ->
              [Reservation t 0 (gammaShape v.valueShape) f]
        ScaleBy w _ | Just t <- tensor w -> [Reservation t 0 [1, 1, 1, 1] 1]
        Standardize b s _ x ->
          [Reservation t 0 [1, width x, 1, 1] 1 | Just t <- [tensor b, tensor s]]
        _ -> []
    ]

{- | The bytes of a stage's weight file, as the lowering writes it: a 64-byte
header, and per blob ('stageWeights') a 64-byte record and its fp16
elements, padded to 64 bytes.
-}
stageWeightFile :: Caps -> Stage -> Natural
stageWeightFile caps stage = 64 + sum [64 + align64 (2 * product r.resShape) | r <- stageWeights caps stage]
 where
  align64 n = (n + 63) `div` 64 * 64

{- | A linear layer of this many outputs in chunks of at most the caps'
channels: each chunk's first output and its number of outputs.
-}
linearChunks :: Caps -> Natural -> [(Natural, Natural)]
linearChunks caps outs =
  [(b, min outs (b + c) - b) | b <- takeWhile (< outs) [0, c ..]]
 where
  c = caps.capsChannels

{- | The shape of a norm's gamma over a value of this shape: along the
channels of a whole row, else along the segment.
-}
gammaShape :: Shape -> [Natural]
gammaShape sh
  | sh.shapeSegment == sh.shapeWidth = [1, sh.shapeSegment, 1, 1]
  | otherwise = [1, 1, sh.shapeSegment, 1]

------------------------------------------------------------------------------
-- The gate

data StageFinding = StageFinding
  { findingWhere :: Text
  -- ^ The part (or the stage) and the source or value.
  , findingMessage :: Text
  }
  deriving stock (Eq, Show)

-- | A stage that passed the gate, for these caps.
data CheckedStage = CheckedStage Caps Stage

checkedStage :: CheckedStage -> Stage
checkedStage (CheckedStage _ s) = s

stageCaps :: CheckedStage -> Caps
stageCaps (CheckedStage c _) = c

checkStage :: Caps -> Stage -> Either [StageFinding] CheckedStage
checkStage caps stage = case reverse (execState run []) of
  [] -> Right (CheckedStage caps stage)
  fs -> Left fs
 where
  run = do
    whole
    mapM_ part (zip [0 ..] stage.stageParts)
    hostNames
    outputs
    weightBytes

  bad :: Text -> Text -> State [StageFinding] ()
  bad w m = modify' (StageFinding w m :)

  parts = stage.stageParts

  whole = do
    unless (identifier stage.stageName) $
      bad stage.stageName "A stage's name is an identifier: letters, digits and underscores."
    when (null parts) $ bad stage.stageName "A stage has parts."
    forM_ [n | (n, c) <- Map.toList (Map.fromListWith (+) [(p.partName, 1 :: Int) | p <- parts]), c > 1] $ \n ->
      bad n "Two parts of one name."

  part (i, p) = do
    let prog = checkedProgram p.partProgram
        at = p.partName
    unless (identifier p.partName) $ bad at "A part's name is an identifier."
    when (p.partRows == 0) $ bad at "A part has rows."
    when (p.partRows > caps.capsLength) $
      bad at ("A part of " <> showT p.partRows <> " rows, more than the ANE's " <> showT caps.capsLength <> " positions.")
    -- Every source fed, and only the program's.
    forM_ prog.programSources $ \s -> case partFeed p s.sourceName of
      Nothing -> bad (at <> "." <> s.sourceName) "No feed for the source."
      Just f -> feed i p s f
    forM_ [n | (n, _) <- p.partFeeds, isNothing (find ((== n) . (.sourceName)) prog.programSources)] $ \n ->
      bad (at <> "." <> n) "A feed for no source of the program."
    forM_ [n | (n, c) <- Map.toList (Map.fromListWith (+) [(n, 1 :: Int) | (n, _) <- p.partFeeds]), c > 1] $ \n ->
      bad (at <> "." <> n) "Two feeds for one source."
    mapM_ (node p prog) prog.programNodes
    wide p prog

  earlier i name = find ((== name) . (.partName)) (take i parts)

  feed i p s f = do
    let at = p.partName <> "." <> s.sourceName
    case (s.sourceKind, f) of
      (SInput sh t, Host n) -> do
        unless (t == TF32) $ bad at "An activation is f32 (the ANE takes it in fp16)."
        hostName at n
        when (sh.shapeWidth > caps.capsChannels) $ bad at "An activation wider than the ANE's channels."
      (SInput sh _, Earlier e o) -> case earlier i e of
        Nothing -> bad at ("No earlier part " <> e <> ".")
        Just ep -> case outputValue ep o of
          Nothing -> bad at ("The part " <> e <> " has no output " <> o <> ".")
          Just v -> do
            unless (v.valueShape.shapeWidth == sh.shapeWidth) $
              bad at ("The output " <> e <> "." <> o <> " is " <> showT v.valueShape.shapeWidth <> " wide, the input " <> showT sh.shapeWidth <> ".")
            unless (ep.partRows == p.partRows) $
              bad at ("The part " <> e <> " runs " <> showT ep.partRows <> " rows, this one " <> showT p.partRows <> ".")
            unless (v.valueType == TF32) $ bad at "An activation from an earlier part is f32."
      (SRowInts _ _, Host n) -> hostName at n
      (SWeight _ TBF16 _, Tensor _) -> pure ()
      (SWeight [rows, width] TF32 _, Earlier e o) -> case earlier i e of
        Nothing -> bad at ("No earlier part " <> e <> ".")
        Just ep -> case outputValue ep o of
          Nothing -> bad at ("The part " <> e <> " has no output " <> o <> ".")
          Just v -> do
            unless (v.valueShape.shapeWidth == width && ep.partRows == rows) $
              bad at ("The rows are " <> showT rows <> " of " <> showT width <> ", the output " <> showT ep.partRows <> " of " <> showT v.valueShape.shapeWidth <> ".")
            unless (all (onlyGathers s.sourceId) (checkedProgram p.partProgram).programNodes) $
              bad at "Rows from an earlier part are only gathered."
      (STable {}, _) -> bad at "The ANE runs no tables of experts."
      (SCache layout _, Mirror n len) -> do
        hostName at n
        when (len > caps.capsLength) $
          bad at ("A mirror of " <> showT len <> " positions, more than the ANE's " <> showT caps.capsLength <> ".")
        case layout of
          Ring _ _ ringLen _ ->
            unless (len == ringLen) $
              bad at ("A ring's mirror is its ring, of " <> showT ringLen <> " positions, not " <> showT len <> ".")
          Paged {} -> when (len == 0) $ bad at "A mirror has positions."
      (SCache {}, Own _) -> pure ()
      (_, _) -> bad at ("A " <> kindWord s.sourceKind <> " can't be fed by " <> feedWord f <> ".")

  -- A node that reads the source only as the table it gathers from.
  onlyGathers sid n = case n.nodeOp of
    Gather t _ _ | t == sid -> True
    o -> sid `notElem` opSources o

  node p prog n = do
    let at = p.partName <> "." <> n.nodeName
        sourceFeed sid = programSource prog sid >>= partFeed p . (.sourceName)
        sourceName sid = maybe "?" (.sourceName) (programSource prog sid)
    case n.nodeOp of
      LinearRouted {} -> bad at "The ANE runs no routed experts."
      Route {} -> bad at "The ANE runs no routing."
      CombineRanks {} -> bad at "The ANE runs no routing."
      Linear w _ _ -> case sourceFeed w of
        Just (Tensor _) -> pure ()
        _ -> bad at "A linear layer's weight is a model's tensor."
      KvStore (StoreAt c _ _) _ -> case sourceFeed c of
        Just (Own _) -> do
          let stores = [() | m <- prog.programNodes, KvStore (StoreAt c' _ _) _ <- [m.nodeOp], c' == c]
          when (length stores > 1) $ bad at ("The step's own cache " <> sourceName c <> " is stored once.")
        _ -> bad at "The ANE stores only into a cache of the step's own rows."
      Attend a _ (AttendTo k v _ sq) -> case (sourceFeed k, sourceFeed v) of
        (Just (Mirror kn _), Just (Mirror vn _)) ->
          when (kn == vn) $ bad at "Keys and values are mirrored apart."
        (Just (Own _), Just (Own _)) -> do
          let storeOf c = listToMaybe [(pos, n') | n' <- prog.programNodes, KvStore (StoreAt c' pos _) _ <- [n'.nodeOp], c' == c]
          case (storeOf k, storeOf v) of
            (Just (pk, sk), Just (pv, sv)) -> do
              unless (pk == pv) $ bad at "The step stores its keys and values at the same positions."
              unless (sk.nodeId < n.nodeId && sv.nodeId < n.nodeId) $ bad at "The step stores its own keys and values before it attends."
            _ -> bad at "The step stores its own keys and values."
          unless (sourceFeed k == sourceFeed v) $ bad at "Its keys and values are seen alike."
          when (a.attnKvHeads == 0) $ bad at "No heads."
          -- The lowering masks the step's own rows by position alone.
          case (.sourceKind) <$> programSource prog sq of
            Just (SRowInts _ 1) -> pure ()
            Just (SRowInts _ b) -> bad at ("Attention over the step's own rows is of one sequence: its sequences are row integers below 1, not " <> showT b <> ".")
            _ -> bad at "Attention over the step's own rows is of one sequence: its sequences are row integers below 1."
        _ -> bad at "Keys and values are both mirrors or both the step's own."
      Gather t _ _ -> case sourceFeed t of
        Just (Tensor _) -> pure ()
        Just (Earlier _ _) -> pure ()
        _ -> bad at "A gather reads a model's tensor or an earlier part's rows."
      Standardize b s _ _ -> forM_ [b, s] $ \w -> case sourceFeed w of
        Just (Tensor _) -> pure ()
        _ -> bad at "Standardizing's weights are a model's tensors."
      RmsNorm _ (Just g) _ _ -> case sourceFeed g of
        Just (Tensor _) -> pure ()
        _ -> bad at "A norm's gamma is a model's tensor."
      ScaleBy w _ -> case sourceFeed w of
        Just (Tensor _) -> pure ()
        _ -> bad at "A scale is a model's tensor."
      _ -> pure ()

  -- A value wider than an op's outputs: a linear layer's, split, read by
  -- what works on its parts.
  wide p prog = forM_ prog.programValues $ \v ->
    when (v.valueShape.shapeWidth > caps.capsChannels) $ do
      let at = p.partName <> "." <> v.valueName
          made = producerOf prog v.valueId
          readers = [n | n <- prog.programNodes, v.valueId `elem` opValues n.nodeOp]
      case made of
        Just Node {nodeOp = Linear {}} -> pure ()
        Just Node {nodeOp = Softcap {}} -> pure ()
        _ -> bad at ("A value wider than the ANE's " <> showT caps.capsChannels <> " channels, which no linear layer made.")
      forM_ readers $ \r -> case r.nodeOp of
        Softcap {} -> pure ()
        ArgMax {} -> pure ()
        _ -> bad at ("A value wider than the ANE's channels, read by " <> r.nodeName <> ".")
      forM_ (laterReaders p v) $ \r -> bad at ("A value wider than the ANE's channels, read by the part " <> r <> ".")

  laterReaders p v =
    [ q.partName
    | q <- parts
    , (_, Earlier e o) <- q.partFeeds
    , e == p.partName
    , Just v' <- [outputValue p o]
    , v'.valueId == v.valueId
    ]

  outputValue p o = do
    let prog = checkedProgram p.partProgram
    out <- find ((== o) . (.outputName)) prog.programOutputs
    programValue prog out.outputValue

  hostName at n = do
    unless (identifier n) $ bad at ("The host's name " <> n <> " is not an identifier.")
    when (reserved n) $ bad at ("The host's name " <> n <> " is a word Core ML's program parser reserves.")

  -- A host name is one kind of thing, of one shape.
  hostNames = do
    let uses =
          [ (n, (kindWord s.sourceKind, use s.sourceKind p))
          | p <- parts
          , s <- (checkedProgram p.partProgram).programSources
          , Just f <- [partFeed p s.sourceName]
          , n <- case f of
              Host n -> [n]
              Mirror n _ -> [n]
              _ -> []
          ]
        use kind p = case kind of
          SInput sh _ -> [p.partRows, sh.shapeWidth]
          SRowInts k _ -> [p.partRows, k]
          _ -> []
    forM_ (nub (map fst uses)) $ \n -> do
      let these = nub [u | (n', u) <- uses, n' == n]
      when (length these > 1) $
        bad n ("The host's " <> n <> " is fed as " <> T.intercalate " and " [k <> " " <> showT s | (k, s) <- these] <> ".")
    -- Mirrors of one name are one cache's, the same length.
    let mirrors =
          [ (n, (len, layoutWord l))
          | p <- parts
          , s <- (checkedProgram p.partProgram).programSources
          , SCache l _ <- [s.sourceKind]
          , Just (Mirror n len) <- [partFeed p s.sourceName]
          ]
    forM_ (nub (map fst mirrors)) $ \n ->
      when (length (nub [m | (n', m) <- mirrors, n' == n]) > 1) $
        bad n "One mirror, of different lengths or layouts."
    -- A mirror is keys or values, and of one cache's heads.
    let roles =
          [ (n, (role, a.attnKvHeads, a.attnHeadDim))
          | p <- parts
          , let prog = checkedProgram p.partProgram
          , n' <- prog.programNodes
          , Attend a _ (AttendTo k v _ _) <- [n'.nodeOp]
          , (role, c) <- [("keys" :: Text, k), ("values", v)]
          , Just src <- [programSource prog c]
          , Just (Mirror n _) <- [partFeed p src.sourceName]
          ]
    forM_ (nub (map fst roles)) $ \n ->
      when (length (nub [r | (n', r) <- roles, n' == n]) > 1) $
        bad n "One mirror, read as keys and values, or by attention of other heads."

  outputs = do
    forM_ stage.stageOutputs $ \(o, pn, on) -> do
      unless (identifier o) $ bad o "An output's name is an identifier."
      when (reserved o) $ bad o "An output's name is a word Core ML's program parser reserves."
      case find ((== pn) . (.partName)) parts of
        Nothing -> bad o ("No part " <> pn <> ".")
        Just p -> when (isNothing (outputValue p on)) $ bad o ("The part " <> pn <> " has no output " <> on <> ".")
    forM_ [n | (n, c) <- Map.toList (Map.fromListWith (+) [(o, 1 :: Int) | (o, _, _) <- stage.stageOutputs]), c > 1] $ \n ->
      bad n "Two outputs of one name."
    let hosts = nub [n | p <- parts, (_, f) <- p.partFeeds, n <- hostOf f]
        hostOf = \case
          Host n -> [n]
          Mirror n _ -> [n]
          _ -> []
    forM_ [o | (o, _, _) <- stage.stageOutputs, o `elem` hosts] $ \o ->
      bad o "An output named as an input."

  weightBytes = do
    let file = stageWeightFile caps stage
    when (file > caps.capsWeightBytes) $
      bad stage.stageName ("The stage's weight file takes " <> showT file <> " bytes, more than the ANE's " <> showT caps.capsWeightBytes <> ".")

kindWord :: SourceKind -> Text
kindWord = \case
  SInput {} -> "activation"
  SRowInts {} -> "row integers"
  SWeight _ TF32 _ -> "table of rows"
  SWeight {} -> "weight"
  STable {} -> "table"
  SCache {} -> "cache"

feedWord :: Feed -> Text
feedWord = \case
  Host n -> "the host's " <> n
  Earlier p o -> "the part " <> p <> "'s " <> o
  Tensor t -> "the tensor " <> t
  Mirror n _ -> "the mirror " <> n
  Own _ -> "the step's own rows"

layoutWord :: CacheLayout -> Text
layoutWord = \case
  Ring {} -> "ring"
  Paged {} -> "paged"

{- | Words Core ML's ML program parser takes for its own, which a name of
a program's input or output can't be: its types and element types, and
its literals, as coremlc refused them under macOS 27.2. It rejects an
input named @state@ (the state type of ML programs) as not an
identifier; @fp8@ alone it takes.
-}
reserved :: Text -> Bool
reserved n =
  n
    `elem` [ "any"
           , "bool"
           , "string"
           , "tensor"
           , "state"
           , "list"
           , "dict"
           , "tuple"
           , "func"
           , "program"
           , "true"
           , "false"
           , "bf16"
           , "fp8e4m3fn"
           , "fp8e5m2"
           , "fp16"
           , "fp32"
           , "fp64"
           , "int4"
           , "int8"
           , "int16"
           , "int32"
           , "int64"
           , "uint1"
           , "uint2"
           , "uint3"
           , "uint4"
           , "uint6"
           , "uint8"
           , "uint16"
           , "uint32"
           , "uint64"
           ]

-- | Letters, digits and underscores, not first a digit: a Core ML feature's name.
identifier :: Text -> Bool
identifier t = case T.uncons t of
  Just (c, rest) -> (letter c || c == '_') && T.all (\x -> letter x || isDigit x || x == '_') rest
  Nothing -> False
 where
  letter x = isAsciiLower x || isAsciiUpper x

showT :: (Show a) => a -> Text
showT = T.pack . show
