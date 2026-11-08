-- SPDX-FileCopyrightText: 2026 Mika Tammi
-- SPDX-License-Identifier: BSD-3-Clause

{- | A checked stage lowered to one ML program for the Neural Engine, with
the manifest a runtime runs it by.

Values take the ANE's @(B, C, 1, S)@ layout: @w@ elements per row over
@m@ rows are @[1, w, 1, m]@, and a value in segments of @s@ (heads) is
@[1, w/s, s, m]@ when an operation works per segment. The lowering, per
operation:

* a linear layer is a 1×1 convolution; one with more outputs than the
  ANE's channels is several, each of at most that many, which only
  elementwise work and an argmax read;
* RMSNorm is a layer norm of its input beside the input's negation, cut
  back to the input's half, its gamma times the norm's factor: it squares
  nothing in fp16, and small inputs stay as they are, where a scale that
  kept squares in range would move them where the ANE's values are
  coarse. A head's norm (along axis 2, in the per-head layout) moves the
  axis last around it, by transposes: Core ML's CPU computes the layer
  norm of a concat along axis 2 wrongly at some shapes, as observed under
  macOS 27.2 (heads 0, 1, 4 and 5 of @[1, 6, 64, 128]@ NaN or wrong),
  where the ANE computed both right. A whole row's (along axis 1) stays,
  right on the CPU over a sweep of widths and rows. Its epsilon is held
  in fp16 as it is, 1e-6 as the subnormal 1.0133e-6;
* rotary embeddings multiply by the host's cosines and sines, which hold
  each row's angles ('RopeCos', 'RopeSin');
* attention over a cache reads the cache's mirror, the host's fp16 copy of
  one sequence's keys and values, through the host's mask of what each row
  sees; attention over the step's own rows (a cache it stores into, then
  attends over) is self-attention over the rows, through a mask likewise,
  the rows cast to fp16 as the cache holds them: an identity on the
  ANE's fp16, which Core ML fuses with the op making the rows (that op
  loses its device in MLComputePlan, the cast carrying its cost), where
  the core's evaluator rounds a
  cache's stores. Its probabilities are scaled by 2^8 before they weigh
  the values, and the result by 2^-8, both exact: probabilities near
  @1/n@ times values put the ANE's products on its sums' grid of 2^-16
  (below), or under it, which over 16384 keys erred by 0.72 of the
  conformance bound and now errs by 0.07. As the probabilities sum to 1,
  each partial sum is within 2^8 of the values' largest magnitude, so the
  values, a mirror's and the step's own, must be below 128 in magnitude:
  from there the accumulator overflows, and the result is infinite;
* a gather from a model's tensor is the host's ('Gathered'); one from an
  earlier part's rows is a product with the host's selection of them
  ('Selection');
* the rest is elementwise, or a reduction for an argmax. Argmaxes come
  last in their part: Core ML runs them on the CPU, and the ops after
  them there too (a draft step's next state, otherwise), and nothing reads
  an index but the engine.

The weights are the model's tensors, their blobs reserved and listed in
the manifest for a runtime to fill. The program is fp16 throughout,
indices int32. The ANE sums a convolution's and a matmul's products in
fixed point, as observed bit for bit on an M3 Max under macOS 27.2
('FuryKernel.Ane.Interp.neuralEngineDot'): each product rounded to a
multiple of 2^-16, ties away from zero, or dropped where its operands'
exponents sum below -16, the products summed in a 32-bit accumulator of
@[-32768, 32768)@, a partial sum past it infinite, and the sum rounded
to binary16, ties away. So a linear layer's partial sums overflow at
32768, not 65520, whatever its result; its outputs below 2^-6 are
coarser than binary16; and each product errs by up to 2^-17 whatever
its size, about @2^-16 * sqrt (k / 12)@ over a fan-in of @k@: values
meant to keep their precision stay well above 2^-6. The ANE's other ops
keep binary16's whole range.

Not done yet: a linear layer's weights scaled by @2^k@ in their blob,
and its output by @2^-k@, would raise its products @2^k@ above the
grid, and the conformance run's rows below 2^-2 would need no absolute
floor (@k = 4@ took a residual of rows of 2^-7 from 0.54 of the bound to
0.05). A @k@ for every layer isn't safe, as it cuts the accumulator's
headroom by @2^k@: at 2^4 the conformance run's long sums overflowed, at
2^8 a drafting step and a softcap's projection. Each output channel
needs its own: a partial sum is at most @‖w_o‖₂·‖x‖₂@, an RMSNorm's
output of gain γ over @d@ elements has @‖x‖₂ ≤ √d·max|γ|@, so
@k_o ≤ 15 - log2 (‖w_o‖₂·√d·max|γ|)@. Only the host has the weights
and the gains: it would choose the scales, the manifest's scale carrying
@2^k@ and a host-filled scalar weight its inverse.
-}
module FuryKernel.Ane.Lower (
  Lowered (..),
  lowerStage,
  maskFill,
) where

import Control.Monad (forM, unless)
import Control.Monad.Trans.Class (lift)
import Control.Monad.Trans.State.Strict (StateT, execStateT, gets, modify')
import Data.List (find, partition)
import Data.Map.Strict (Map)
import Data.Map.Strict qualified as Map
import Data.Maybe (fromMaybe)
import Data.Set qualified as Set
import Data.Text (Text)
import Data.Text qualified as T
import Data.Word (Word64)
import FuryKernel.Ane.Blob (Placed (..), blobsLayout)
import FuryKernel.Ane.CoreML (DataType (..), Feature (..), Model (..), TensorType (..))
import FuryKernel.Ane.Manifest qualified as M
import FuryKernel.Ane.Mil
import FuryKernel.Ane.Stage
import FuryKernel.Program.Check (checkedProgram)
import FuryKernel.Program.Graph
import Numeric.Natural (Natural)

-- | A stage lowered: its ML program, blobs and weights, and its manifest.
data Lowered = Lowered
  { loweredBuilt :: Built
  , loweredManifest :: M.Manifest
  }

-- | What a mask holds where a row doesn't see a key.
maskFill :: Double
maskFill = -30000

{- | What attention's probabilities are multiplied by before they weigh
the values: 2^8, so that the ANE's sums keep a value's product with a
probability near @1/n@ well above their grid, and every partial sum below
32768 while the values are below 128 in magnitude.
-}
probabilityScale :: Double
probabilityScale = 256

------------------------------------------------------------------------------
-- State

-- | A value's lowered forms: flat, @[1, w, 1, m]@; in segments; or in chunks of channels.
data Forms = Forms
  { formFlat :: Maybe Var
  , formSegments :: Maybe Var
  , formChunks :: Maybe [Var]
  }

data St = St
  { stInputs :: [M.Input]
  -- ^ Reversed.
  , stInputVars :: Map Text Var
  , stDerived :: Map M.InputKind Text
  -- ^ The inputs the lowering asked the host for, by what they hold.
  , stParts :: Map Text (Program, Natural, Map ValueId Forms)
  -- ^ The parts lowered so far, their values.
  }

type L = StateT St Mil

-- What a part's lowering has at hand.
data Ctx = Ctx
  { ctxPart :: Part
  , ctxProgram :: Program
  , ctxRows :: Natural
  , ctxCaps :: Caps
  }

data PartSt = PartSt
  { psValues :: Map ValueId Forms
  , psOwn :: Map SourceId (Var, SourceId)
  -- ^ The step's own caches: the rows they hold, and their positions' source.
  }

type LP = StateT PartSt L

mil :: Mil a -> LP a
mil = lift . lift

lowerStage :: CheckedStage -> Lowered
lowerStage checked = Lowered built manifest
 where
  stage = checkedStage checked
  caps = stageCaps checked
  hosts =
    Set.fromList
      [ n
      | p <- stage.stageParts
      , (_, f) <- p.partFeeds
      , n <- case f of
          Host n -> [n]
          Mirror n _ -> [n]
          _ -> []
      ]
  run :: Mil ([Var], St)
  run = do
    -- One namespace: the stage's own names first, each generated one clear of them.
    reserve (map T.unpack (Set.toList hosts) <> [T.unpack o | (o, _, _) <- stage.stageOutputs])
    st <- execStateT (mapM_ (lowerPart caps) stage.stageParts) (St [] Map.empty Map.empty Map.empty)
    outs <- forM stage.stageOutputs $ \(o, pn, on) -> do
      let (prog, rows, values) = fromMaybe (error "lowerStage: an output of no part") (Map.lookup pn st.stParts)
          v = fromMaybe (error "lowerStage: no such output") (find ((== on) . (.outputName)) prog.programOutputs)
          forms = fromMaybe (error "lowerStage: an output never made") (Map.lookup v.outputValue values)
          val = fromMaybe (error "lowerStage: an output of no value") (programValue prog v.outputValue)
      x <- case (forms.formFlat, forms.formChunks, forms.formSegments) of
        (Just f, _, _) -> pure f
        (_, Just cs, _) -> concatAxis cs 1
        (_, _, Just s) -> reshape s (flatShape val rows)
        _ -> error "lowerStage: a value of no form"
      output (T.unpack o) x
    pure (outs, st)
  (built, final) = runMilWith Float16 run
  outVars = [Var f.featureName f.featureType | f <- built.builtModel.modelOutputs]
  placed = Map.fromList [(p.placedRecord, p) | p <- blobsLayout built.builtBlobs]
  manifest =
    M.Manifest
      { M.manifestStage = stage.stageName
      , M.manifestParts =
          [ M.PartInfo p.partName (checkedProgram p.partProgram).programName p.partRows
          | p <- stage.stageParts
          ]
      , M.manifestInputs = reverse final.stInputs
      , M.manifestOutputs =
          [ M.Output
              { M.outputName = o
              , M.outputShape = v.varType.ttShape
              , M.outputType = if v.varType.ttDataType == Int32 then M.OutputIndex else M.OutputHalf
              , M.outputPart = pn
              , M.outputValue = on
              }
          | ((o, pn, on), v) <- zip stage.stageOutputs outVars
          ]
      , M.manifestWeights =
          [ M.WeightBlob
              { M.blobRecord = w.weightRecord
              , M.blobData = maybe 0 (.placedData) p
              , M.blobSize = maybe 0 (.placedSize) p
              , M.blobTensor = T.pack w.weightTensor
              , M.blobOffset = w.weightOffset
              , M.blobShape = w.weightShape
              , M.blobScale = w.weightScale
              }
          | w <- built.builtWeights
          , let p = Map.lookup w.weightRecord placed
          ]
      , M.manifestWeightFile = case reverse (blobsLayout built.builtBlobs) of
          p : _ -> align64 (p.placedData + p.placedSize)
          [] -> 64
      }

align64 :: Word64 -> Word64
align64 n = (n + 63) `div` 64 * 64

------------------------------------------------------------------------------
-- Parts

lowerPart :: Caps -> Part -> L ()
lowerPart caps part = do
  let prog = checkedProgram part.partProgram
      ctx = Ctx part prog part.partRows caps
      (indices, rest) = partition isArgMax prog.programNodes
      isArgMax n = case n.nodeOp of
        ArgMax _ -> True
        _ -> False
  ps <- execStateT (mapM_ (lowerNode ctx) (rest <> indices)) (PartSt Map.empty Map.empty)
  modify' (\st -> st {stParts = Map.insert part.partName (prog, part.partRows, ps.psValues) st.stParts})

valueOf :: Ctx -> ValueId -> Value
valueOf ctx v = fromMaybe (error "lower: a value of no program") (programValue ctx.ctxProgram v)

sourceOf :: Ctx -> SourceId -> Source
sourceOf ctx s = fromMaybe (error "lower: a source of no program") (programSource ctx.ctxProgram s)

feedOf :: Ctx -> SourceId -> Feed
feedOf ctx s = fromMaybe (error "lower: a source fed by nothing") (partFeed ctx.ctxPart (sourceOf ctx s).sourceName)

hostOf :: Ctx -> SourceId -> Text
hostOf ctx s = case feedOf ctx s of
  Host n -> n
  _ -> error "lower: row integers not the host's"

flatShape :: Value -> Natural -> [Word64]
flatShape v rows = [1, fromIntegral v.valueShape.shapeWidth, 1, fromIntegral rows]

segShape :: Value -> Natural -> [Word64]
segShape v rows =
  [ 1
  , fromIntegral (segmentCount v.valueShape)
  , fromIntegral v.valueShape.shapeSegment
  , fromIntegral rows
  ]

-- | A value flat.
flat :: Ctx -> ValueId -> LP Var
flat ctx vid = do
  forms <- formsOf vid
  case (forms.formFlat, forms.formSegments, forms.formChunks) of
    (Just f, _, _) -> pure f
    (_, Just s, _) -> do
      f <- mil (reshape s (flatShape (valueOf ctx vid) ctx.ctxRows))
      remember vid forms {formFlat = Just f}
      pure f
    (_, _, Just cs) -> mil (concatAxis cs 1)
    _ -> error "lower: a value of no form"

-- | A value in its segments.
segments :: Ctx -> ValueId -> LP Var
segments ctx vid = do
  forms <- formsOf vid
  case (forms.formSegments, forms.formFlat) of
    (Just s, _) -> pure s
    (_, Just f) -> do
      s <- mil (reshape f (segShape (valueOf ctx vid) ctx.ctxRows))
      remember vid forms {formSegments = Just s}
      pure s
    _ -> do
      f <- flat ctx vid
      mil (reshape f (segShape (valueOf ctx vid) ctx.ctxRows))

-- | A value in its chunks of channels: one, if it is narrow.
chunks :: Ctx -> ValueId -> LP [Var]
chunks ctx vid = do
  forms <- formsOf vid
  case forms.formChunks of
    Just cs -> pure cs
    Nothing -> pure <$> flat ctx vid

formsOf :: ValueId -> LP Forms
formsOf vid = gets (fromMaybe (error "lower: a value read before it is made") . Map.lookup vid . (.psValues))

remember :: ValueId -> Forms -> LP ()
remember vid forms = modify' (\ps -> ps {psValues = Map.insert vid forms ps.psValues})

-- | A node's result, named after its part and value where the last op made it.
made :: Ctx -> ValueId -> (Maybe Var -> Maybe Var -> Maybe [Var] -> Forms) -> Var -> LP ()
made ctx vid form v = do
  let val = valueOf ctx vid
      n = T.unpack (ctx.ctxPart.partName <> "_" <> val.valueName)
  v' <- mil (unique n >>= (`named` v))
  remember vid (form (Just v') Nothing Nothing)

asFlat, asSegments :: Maybe Var -> Maybe Var -> Maybe [Var] -> Forms
asFlat v _ _ = Forms v Nothing Nothing
asSegments v _ _ = Forms Nothing v Nothing

weightVar :: Ctx -> SourceId -> Word64 -> [Word64] -> Double -> LP Var
weightVar ctx s from shape scale = case feedOf ctx s of
  Tensor t -> mil (weight (T.unpack t) from shape scale)
  _ -> error "lower: a weight not a model's tensor"

------------------------------------------------------------------------------
-- Inputs

-- | The host's input of this name, or one already asked for.
hostInput :: Text -> [Word64] -> M.InputKind -> LP Var
hostInput name shape kind = do
  known <- lift (gets (Map.lookup name . (.stInputVars)))
  case known of
    Just v -> do
      unless (v.varType.ttShape == shape) $
        error ("lower: the host's " <> T.unpack name <> " asked for in two shapes")
      pure v
    Nothing -> do
      v <- mil (input (T.unpack name) (TensorType Float16 shape))
      lift . modify' $ \st ->
        st
          { stInputs = M.Input name shape kind : st.stInputs
          , stInputVars = Map.insert name v st.stInputVars
          }
      pure v

-- | An input the host derives: one for each thing it holds, named after its kind.
derived :: Text -> [Word64] -> M.InputKind -> LP Var
derived prefix shape kind = do
  known <- lift (gets (Map.lookup kind . (.stDerived)))
  case known of
    Just n -> hostInput n shape kind
    Nothing -> do
      n <- lift (freshName prefix)
      lift (modify' (\st -> st {stDerived = Map.insert kind n st.stDerived}))
      hostInput n shape kind

-- | Core ML's GELU mode for the IR's form, one for each form.
geluMode :: Gelu -> String
geluMode = \case
  GeluTanh -> "TANH_APPROXIMATION"

-- | A derived input's name: the prefix and the first number no name of the program has.
freshName :: Text -> L Text
freshName prefix = lift (T.pack <$> numbered (T.unpack prefix))

------------------------------------------------------------------------------
-- Nodes

lowerNode :: Ctx -> Node -> LP ()
lowerNode ctx n = case (n.nodeOp, n.nodeOutputs) of
  (Input s, [o]) -> case feedOf ctx s of
    Host name -> do
      v <- hostInput name (flatShape (valueOf ctx o) ctx.ctxRows) M.Activation
      remember o (Forms (Just v) Nothing Nothing)
    Earlier pn on -> do
      (prog, _, values) <- lift (gets (fromMaybe (error "lower: no earlier part") . Map.lookup pn . (.stParts)))
      let out = fromMaybe (error "lower: no earlier output") (find ((== on) . (.outputName)) prog.programOutputs)
          forms = fromMaybe (error "lower: an earlier output never made") (Map.lookup out.outputValue values)
      -- The earlier value's forms, flat first: its rows are this part's.
      f <- case (forms.formFlat, forms.formSegments) of
        (Just f, _) -> pure f
        (_, Just sg) -> mil (reshape sg (flatShape (valueOf ctx o) ctx.ctxRows))
        _ -> error "lower: an earlier value of no form"
      remember o (Forms (Just f) Nothing Nothing)
    _ -> error "lower: an input fed by neither the host nor a part"
  (Linear w _ x, [o]) -> do
    xf <- flat ctx x
    let out = valueOf ctx o
        ins = shapeWidth (valueOf ctx x).valueShape
    ys <- forM (linearChunks ctx.ctxCaps out.valueShape.shapeWidth) $ \(b, k) -> do
      wv <- weightVar ctx w (fromIntegral (b * ins)) [fromIntegral k, fromIntegral ins, 1, 1] 1
      mil (conv1x1 xf wv)
    case ys of
      [y] -> made ctx o asFlat y
      _ -> remember o (Forms Nothing Nothing (Just ys))
  (RmsNorm eps g f x, [o]) -> do
    let val = valueOf ctx x
        whole = val.valueShape.shapeSegment == val.valueShape.shapeWidth
    xv <- if whole then flat ctx x else segments ctx x
    gamma <- case g of
      Just gs -> Just <$> weightVar ctx gs 0 (map fromIntegral (gammaShape val.valueShape)) f
      Nothing -> pure Nothing
    y <- mil (rmsNorm xv (if whole then 1 else 2) gamma (realToFrac eps))
    y' <- case g of
      Nothing | f /= 1 -> mil (mul y =<< constScalar (realToFrac f))
      _ -> pure y
    made ctx o (if whole then asFlat else asSegments) y'
  (Add a b, [o]) -> do
    av <- flat ctx a
    bv <- flat ctx b
    made ctx o asFlat =<< mil (add av bv)
  (Concat a b, [o]) -> do
    av <- flat ctx a
    bv <- flat ctx b
    made ctx o asFlat =<< mil (concatAxis [av, bv] 1)
  (ScaleBy w x, [o]) -> do
    xv <- flat ctx x
    wv <- weightVar ctx w 0 [1, 1, 1, 1] 1
    made ctx o asFlat =<< mil (mul xv wv)
  (GatedGelu form x, [o]) -> do
    xv <- segments ctx x
    let val = valueOf ctx x
        seg = fromIntegral val.valueShape.shapeSegment
    y <- mil $ do
      a <- sliceAxis xv 2 0 (seg `div` 2)
      b <- sliceAxis xv 2 (seg `div` 2) seg
      ga <- gelu a (geluMode form)
      mul ga b
    made ctx o asSegments y
  (GeluMul form a b, [o]) -> do
    av <- flat ctx a
    bv <- flat ctx b
    made ctx o asFlat =<< mil (gelu av (geluMode form) >>= (`mul` bv))
  (Rope r pos x, [o]) -> do
    -- The halves are the one pairing lowered; another must say how it pairs.
    case r.ropePairs of
      RopeHalves -> pure ()
    xv <- segments ctx x
    let val = valueOf ctx x
        d = val.valueShape.shapeSegment
        table =
          M.RopeTable
            { M.ropeTheta = r.ropeTheta
            , M.ropeRotated = r.ropeRotated
            , M.ropeAxes = r.ropeAxes
            , M.ropeWidth = d
            , M.ropePositions = hostOf ctx pos
            }
        shape = [1, 1, fromIntegral d, fromIntegral ctx.ctxRows]
    cosV <- derived "rope_cos" shape (M.RopeCos table)
    sinV <- derived "rope_sin" shape (M.RopeSin table)
    let part = fromIntegral (d `div` r.ropeAxes) :: Word64
        half = part `div` 2
    y <- mil $ do
      neg <- mul xv =<< constScalar (-1)
      pieces <- fmap concat . forM [0 .. fromIntegral r.ropeAxes - 1] $ \a -> do
        let base = a * part
        x2 <- sliceAxis neg 2 (base + half) (base + part)
        x1 <- sliceAxis xv 2 base (base + half)
        pure [x2, x1]
      rotated <- concatAxis pieces 2
      c <- mul xv cosV
      s <- mul rotated sinV
      add c s
    made ctx o asSegments y
  (Softcap c x, [o]) -> do
    xs <- chunks ctx x
    ys <- forM xs $ \xv -> mil $ do
      cv <- constScalar (realToFrac c)
      t <- tanhOp =<< realDiv xv cv
      mul t cv
    case ys of
      [y] -> made ctx o asFlat y
      _ -> remember o (Forms Nothing Nothing (Just ys))
  (KvStore (StoreAt c pos _) x, []) -> case feedOf ctx c of
    Own _ -> do
      -- The rows as the cache holds them, in fp16, named after the part and cache.
      let cache = sourceOf ctx c
      rows <- segments ctx x
      held <- mil $ do
        cast <- castFp16 rows
        name <- unique (T.unpack (ctx.ctxPart.partName <> "_" <> cache.sourceName))
        named name cast
      modify' (\ps -> ps {psOwn = Map.insert c (held, pos) ps.psOwn})
    _ -> error "lower: a store into a cache not the step's own"
  (Attend a q (AttendTo k v pos slot), [o]) -> attend ctx a q k v pos slot o
  (Gather t tok f, [o]) -> case feedOf ctx t of
    Tensor tensor -> do
      let spec = M.GatherSpec tensor (hostOf ctx tok) f
      g <- derived "gathered" (flatShape (valueOf ctx o) ctx.ctxRows) (M.Gathered spec)
      remember o (Forms (Just g) Nothing Nothing)
    Earlier pn on -> do
      (prog, rowsIn, values) <- lift (gets (fromMaybe (error "lower: no earlier part") . Map.lookup pn . (.stParts)))
      let out = fromMaybe (error "lower: no earlier output") (find ((== on) . (.outputName)) prog.programOutputs)
          forms = fromMaybe (error "lower: an earlier output never made") (Map.lookup out.outputValue values)
          width = fromIntegral (valueOf ctx o).valueShape.shapeWidth
          spec = M.SelectionSpec (hostOf ctx tok) f
      e <- case (forms.formFlat, forms.formSegments) of
        (Just e, _) -> pure e
        (_, Just s) -> mil (reshape s [1, width, 1, fromIntegral rowsIn])
        _ -> error "lower: an earlier value of no form"
      sel <- derived "selection" [1, 1, fromIntegral rowsIn, fromIntegral ctx.ctxRows] (M.Selection spec)
      y <- mil $ do
        e' <- reshape e [1, 1, width, fromIntegral rowsIn]
        p <- matmul e' False sel False
        reshape p [1, width, 1, fromIntegral ctx.ctxRows]
      made ctx o asFlat y
    _ -> error "lower: a gather from neither a tensor nor a part"
  (Standardize b s f x, [o]) -> do
    xv <- flat ctx x
    let w = fromIntegral (valueOf ctx x).valueShape.shapeWidth
    bv <- weightVar ctx b 0 [1, w, 1, 1] 1
    sv <- weightVar ctx s 0 [1, w, 1, 1] 1
    y <- mil $ do
      scaled <- if f == 1 then pure xv else mul xv =<< constScalar (realToFrac f)
      centred <- sub scaled bv
      mul centred sv
    made ctx o asFlat y
  (ArgMax x, [o]) -> do
    xs <- chunks ctx x
    whole <- case xs of
      [xv] -> pure xv
      _ -> mil (concatAxis xs 1)
    made ctx o asFlat =<< mil (reduceArgmax whole 1)
  (op', _) -> error ("lower: " <> show op' <> " has no lowering; the stage's gate lets none through")

{- | Attention: per group of query heads that share a key head, the
scores of the group's queries against every key (a mirror's positions, or
the step's own rows), the mask added, a softmax over the keys, and the
values weighted by it, the probabilities times 'probabilityScale' and the
result divided by it (see the module's description).
-}
attend :: Ctx -> AttnSpec -> ValueId -> SourceId -> SourceId -> SourceId -> SourceId -> ValueId -> LP ()
attend ctx a q k v pos slot o = do
  qv <- segments ctx q
  let heads = a.attnHeads
      kvh = a.attnKvHeads
      d = a.attnHeadDim
      rep = heads `div` kvh
      m = ctx.ctxRows
      w64 :: Natural -> Word64
      w64 = fromIntegral
      reach = hostOf ctx . (.reachSource) <$> a.attnReach
  qs <- if a.attnScale == 1 then pure qv else mil (mul qv =<< constScalar (realToFrac a.attnScale))
  qb <- mil (reshape qs [w64 kvh, w64 rep, w64 d, w64 m])
  out <- case (feedOf ctx k, feedOf ctx v) of
    (Mirror kn len, Mirror vn _) -> do
      let kSource = sourceOf ctx k
          layout = case kSource.sourceKind of
            SCache l _ -> l
            _ -> error "lower: attention over no cache"
          sequence' = case feedOf ctx slot of
            Host n -> n
            _ -> error "lower: a mirror's sequence not the host's"
          mirrorLayout = case layout of
            Ring {} -> M.MirrorRing len
            Paged _ block _ _ -> M.MirrorLinear len block
          spec = M.MirrorSpec mirrorLayout kvh d sequence'
          keys = case layout of
            Ring {} -> M.RingKeys len
            Paged {} -> M.LinearKeys len
          mask = M.MaskSpec (hostOf ctx pos) reach a.attnWindow keys False
      kIn <- hostInput kn [1, w64 kvh, w64 len, w64 d] (M.MirrorKeys spec)
      vIn <- hostInput vn [1, w64 kvh, w64 d, w64 len] (M.MirrorValues spec)
      maskIn <- derived "mask" [1, 1, w64 len, w64 m] (M.Mask mask)
      mil $ do
        kb <- reshape kIn [w64 kvh, 1, w64 len, w64 d]
        scores <- matmul kb False qb False
        p <- scaled probabilityScale =<< (`softmax` 2) =<< add scores maskIn
        vb <- reshape vIn [w64 kvh, 1, w64 d, w64 len]
        scaled (1 / probabilityScale) =<< matmul vb False p False
    (Own masks, Own _) -> do
      own <- gets (.psOwn)
      let (kSeg, storedAt) = fromMaybe (error "lower: own keys never stored") (Map.lookup k own)
          (vSeg, _) = fromMaybe (error "lower: own values never stored") (Map.lookup v own)
          shared = masks == FirstRowMasks
          mask = M.MaskSpec (hostOf ctx pos) reach a.attnWindow (M.RowKeys (hostOf ctx storedAt)) shared
      maskIn <- derived "mask" [1, 1, w64 m, if shared then 1 else w64 m] (M.Mask mask)
      mil $ do
        kb <- reshape kSeg [w64 kvh, 1, w64 d, w64 m]
        scores <- matmul kb True qb False
        p <- scaled probabilityScale =<< (`softmax` 2) =<< add scores maskIn
        vb <- reshape vSeg [w64 kvh, 1, w64 d, w64 m]
        scaled (1 / probabilityScale) =<< matmul vb False p False
    _ -> error "lower: attention over neither mirrors nor the step's own rows"
  y <- mil (reshape out [1, w64 heads, w64 d, w64 m])
  made ctx o asSegments y
 where
  scaled f x = mul x =<< constScalar f
