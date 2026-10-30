-- SPDX-FileCopyrightText: 2026 Mika Tammi
-- SPDX-License-Identifier: BSD-3-Clause

{- | ML programs run in f64: the ops the lowering writes, each as exact as
a double computes it (a cast to fp16 rounds to binary16, as the core's
evaluator rounds an fp16 cache's stores), so that tests hold a lowered
stage to its programs' meaning without Core ML. Constants are what the
program holds (an fp16 immediate is its fp16 value); weights are given
per blob.

It also runs them in fp16 ('interpretValuesWith'): each op's result
rounded to binary16, ties to even, as an fp16 program declares it; or as
the ANE was observed to compute them ('NeuralEngine').

A reference written for clarity, for small shapes.
-}
module FuryKernel.Ane.Interp (
  Tensor (..),
  tensor,
  tensorElems,
  Precision (..),
  roundAs,
  neuralEngineDot,
  interpret,
  interpretValues,
  interpretValuesWith,
) where

import Control.Monad (foldM, forM, unless, when)
import Data.Array.Unboxed (UArray, bounds, elems, listArray, (!))
import Data.Bits (shiftL, (.|.))
import Data.ByteString qualified as B
import Data.Map.Strict (Map)
import Data.Map.Strict qualified as Map
import Data.Text (Text)
import Data.Text qualified as T
import Data.Word (Word64)
import FuryKernel.Ane.CoreML
import FuryKernel.Ane.Half (halfToDouble)
import FuryKernel.Numerics (narrowF16Double, widenF16)
import GHC.Float (float2Double)

-- | A tensor: its shape, and its elements in order, the last axis fastest.
data Tensor = Tensor
  { tensorShape :: [Int]
  , tensorData :: UArray Int Double
  }
  deriving stock (Eq, Show)

tensor :: [Int] -> [Double] -> Tensor
tensor shape xs = Tensor shape (listArray (0, length xs - 1) xs)

tensorElems :: Tensor -> [Double]
tensorElems t = let (lo, hi) = bounds t.tensorData in if hi < lo then [] else elems t.tensorData

data Val = VTensor Tensor | VString String

{- | How the interpreter rounds each op's fp16 result.

* 'Exact': not at all, f64 throughout;
* 'Binary16': to binary16, to nearest with ties to even;
* 'NeuralEngine': as the ANE computes, as observed on an M3 Max under
  macOS 27.2 with every finite fp16 as an operand: a convolution's or a
  matmul's dot products as the ANE sums them, bit for bit
  ('neuralEngineDot'), and every other op's result rounded to binary16,
  to nearest with ties away from zero, in binary16's whole range: what
  the ANE's add and sub compute exactly, and its multiplication by a
  scalar but on subnormal results (up to 0.75 units in the last place).
  Weights keep their fp16, constants are left as the program holds them.

It models the ANE only so far. The ANE's own exp, tanh, GELU, softmax
and layer norm are approximations, which the model computes exactly and
rounds: tanh is up to 0.58 % low below 0.06 in magnitude (0.18 % at 1,
exact from 8), GELU (either of Core ML's modes) off by up to 0.006 near
±0.25, exp by up to ten units in the last place from 4 up, a division
by a scalar by up to one, softmax's probabilities by up to 0.7 % (2^-9
of each row's largest), and a layer norm's outputs by up to 2^-8 of
their row's largest. And the ANE fuses ops, which may keep more than
the model's rounding of each op's result.
-}
data Precision = Exact | Binary16 | NeuralEngine
  deriving stock (Eq, Show)

-- | A value as the precision rounds an op's fp16 result.
roundAs :: Precision -> Double -> Double
roundAs = \case
  Exact -> id
  Binary16 -> half
  NeuralEngine -> halfAway
 where
  half = float2Double . widenF16 . narrowF16Double

{- | The program's main function run on its inputs, by name, and its
weights, by the offsets of their blobs' records: its outputs, by name.
-}
interpret :: Model -> Map String Tensor -> Map Word64 [Double] -> Either Text (Map String Tensor)
interpret model inputs weights = do
  values <- interpretValues model inputs weights
  fn <- mainOf model
  fmap Map.fromList . forM fn.fnOutputs $ \n -> case Map.lookup n values of
    Just t -> pure (n, t)
    Nothing -> Left ("No output " <> T.pack n <> ".")

mainOf :: Model -> Either Text Function
mainOf model = case lookup "main" model.modelFunctions of
  Just f -> Right f
  Nothing -> Left "No main function."

-- | Every tensor the program computes, by name, its inputs and outputs included.
interpretValues :: Model -> Map String Tensor -> Map Word64 [Double] -> Either Text (Map String Tensor)
interpretValues = interpretValuesWith Exact

{- | 'interpretValues', each op's fp16 result rounded as the precision
says; a const's as the program holds it.
-}
interpretValuesWith :: Precision -> Model -> Map String Tensor -> Map Word64 [Double] -> Either Text (Map String Tensor)
interpretValuesWith precision model inputs weights = do
  fn <- mainOf model
  env0 <- fmap Map.fromList . forM fn.fnInputs $ \(n, ty) -> case Map.lookup n inputs of
    Just t -> do
      unless (t.tensorShape == map fromIntegral ty.ttShape) $
        Left ("The input " <> T.pack n <> " is " <> showT t.tensorShape <> ", not " <> showT ty.ttShape <> ".")
      pure (n, VTensor t)
    Nothing -> Left ("No input " <> T.pack n <> ".")
  env <- foldM step env0 fn.fnOperations
  pure (Map.mapMaybe (\case VTensor t -> Just t; VString _ -> Nothing) env)
 where
  step env o = do
    out <- either (\m -> Left (T.pack o.opType <> " " <> outName o <> ": " <> m)) Right (run env o)
    pure (Map.insert (T.unpack (outName o)) (rounded o out) env)
  rounded o = \case
    VTensor t
      | precision == NeuralEngine
      , o.opType `elem` ["conv", "matmul"] ->
          VTensor t
      | precision /= Exact
      , o.opType /= "const"
      , [(_, ty)] <- o.opOutputs
      , ty.ttDataType == Float16 ->
          VTensor (mapT (roundAs precision) t)
    v -> v
  outName o = case o.opOutputs of
    (n, _) : _ -> T.pack n
    [] -> "?"

  run env o = case o.opType of
    "const" -> case lookup "val" o.opAttributes of
      Just v -> constant v
      Nothing -> Left "A const of no value."
    "identity" -> VTensor <$> arg env o "x"
    "add" -> binary (+)
    "sub" -> binary (-)
    "mul" -> binary (*)
    "real_div" -> binary (/)
    "tanh" -> VTensor . mapT tanh <$> arg env o "x"
    "cast" -> do
      x <- arg env o "x"
      dtype <- str env o "dtype"
      unless (dtype == "fp16") $ Left ("A cast to " <> T.pack dtype <> ".")
      pure (VTensor (mapT (float2Double . widenF16 . narrowF16Double) x))
    "rsqrt" -> do
      x <- arg env o "x"
      e <- scalar =<< arg env o "epsilon"
      pure (VTensor (mapT (\v -> 1 / sqrt (v + e)) x))
    "gelu" -> do
      x <- arg env o "x"
      mode <- str env o "mode"
      unless (mode == "TANH_APPROXIMATION") $ Left ("GELU's mode " <> T.pack mode <> ".")
      pure (VTensor (mapT (\v -> 0.5 * v * (1 + tanh (sqrt (2 / pi) * (v + 0.044715 * v * v * v)))) x))
    "layer_norm" -> do
      x <- arg env o "x"
      axes <- map (norm x) <$> (ints =<< arg env o "axes")
      e <- scalar =<< arg env o "epsilon"
      pure (VTensor (layerNorm axes e x))
    "reduce_mean" -> do
      x <- arg env o "x"
      axes <- map (norm x) <$> (ints =<< arg env o "axes")
      keep <- (/= 0) <$> (scalar =<< arg env o "keep_dims")
      unless keep $ Left "Only kept dimensions."
      pure (VTensor (reduce (\xs -> sum xs / fromIntegral (length xs)) axes x))
    "reduce_argmax" -> do
      x <- arg env o "x"
      axis <- norm x . round <$> (scalar =<< arg env o "axis")
      keep <- (/= 0) <$> (scalar =<< arg env o "keep_dims")
      unless keep $ Left "Only kept dimensions."
      pure (VTensor (reduce argMax [axis] x))
    "softmax" -> do
      x <- arg env o "x"
      axis <- norm x . round <$> (scalar =<< arg env o "axis")
      pure (VTensor (softmax axis x))
    "conv" -> do
      x <- arg env o "x"
      w <- arg env o "weight"
      groups <- round <$> (scalar =<< arg env o "groups")
      case (x.tensorShape, w.tensorShape) of
        ([b, cin, h, wd], [cout, cin', 1, 1])
          | cin == cin'
          , groups == (1 :: Int) ->
              pure . VTensor . tensor [b, cout, h, wd] $
                [ dot [(at w [oc, ic, 0, 0], at x [bi, ic, hi, wi]) | ic <- [0 .. cin - 1]]
                | bi <- [0 .. b - 1]
                , oc <- [0 .. cout - 1]
                , hi <- [0 .. h - 1]
                , wi <- [0 .. wd - 1]
                ]
        _ -> Left "Only 1×1 convolutions of one group."
    "transpose" -> do
      x <- arg env o "x"
      perm <- ints =<< arg env o "perm"
      let shape = [x.tensorShape !! a | a <- perm]
          source ix = [i | j <- [0 .. length perm - 1], (i, a) <- zip ix perm, a == j]
      pure . VTensor . tensor shape $ [at x (source ix) | ix <- indices shape]
    "reshape" -> do
      x <- arg env o "x"
      shape <- ints =<< arg env o "shape"
      unless (product shape == product x.tensorShape) $ Left "A reshape to another size."
      pure (VTensor x {tensorShape = shape})
    "slice_by_index" -> do
      x <- arg env o "x"
      begin <- ints =<< arg env o "begin"
      end <- ints =<< arg env o "end"
      let shape = zipWith (-) end begin
      pure . VTensor . tensor shape $ [at x (zipWith (+) begin ix) | ix <- indices shape]
    "concat" -> do
      xs <- args env o "values"
      axis <- round <$> (scalar =<< arg env o "axis")
      case xs of
        [] -> Left "Nothing to concatenate."
        x : _ -> do
          let a = norm x axis
              widths = map ((!! a) . (.tensorShape)) xs
              shape = [if i == a then sum widths else d | (i, d) <- zip [0 ..] x.tensorShape]
              starts = scanl (+) 0 widths
              pick ix =
                let k = ix !! a
                    (part, from) = last [(t, s) | (t, s) <- zip xs starts, s <= k]
                 in at part [if i == a then j - from else j | (i, j) <- zip [0 ..] ix]
          pure (VTensor (tensor shape (map pick (indices shape))))
    "matmul" -> do
      x <- arg env o "x"
      y <- arg env o "y"
      tx <- (/= 0) <$> (scalar =<< arg env o "transpose_x")
      ty <- (/= 0) <$> (scalar =<< arg env o "transpose_y")
      matmul dot x tx y ty
    other -> Left ("No such op here: " <> T.pack other <> ".")
   where
    dot = if precision == NeuralEngine then neuralEngineDot else sum . map (uncurry (*))
    binary f = do
      x <- arg env o "x"
      y <- arg env o "y"
      VTensor <$> broadcastWith f x y

  constant = \case
    Immediate ty imm -> case imm of
      Floats xs -> pure (VTensor (tensor (shapeOf ty) (map realToFrac xs)))
      Ints xs -> pure (VTensor (tensor (shapeOf ty) (map fromIntegral xs)))
      Bools xs -> pure (VTensor (tensor (shapeOf ty) [if b then 1 else 0 | b <- xs]))
      Strings [s] -> pure (VString s)
      Strings _ -> Left "Strings, not one."
      Bytes bs -> pure (VTensor (tensor (shapeOf ty) (halves bs)))
    BlobFile ty _ off -> case Map.lookup off weights of
      Just xs -> do
        when (length xs /= product (shapeOf ty)) $ Left ("The blob at " <> showT off <> " is the wrong size.")
        pure (VTensor (tensor (shapeOf ty) xs))
      Nothing -> Left ("No weights for the blob at " <> showT off <> ".")
  shapeOf ty = map fromIntegral ty.ttShape
  halves bs =
    [ halfToDouble (fromIntegral (B.index bs i) .|. (fromIntegral (B.index bs (i + 1)) `shiftL` 8))
    | i <- [0, 2 .. B.length bs - 2]
    ]

  arg env o name =
    args env o name >>= \case
      [t] -> pure t
      _ -> Left ("Not one " <> T.pack name <> ".")
  args env o name = case lookup name o.opInputs of
    Just as -> forM as $ \case
      ArgName n -> case Map.lookup n env of
        Just (VTensor t) -> pure t
        Just (VString _) -> Left (T.pack n <> " is a string.")
        Nothing -> Left ("No value " <> T.pack n <> ".")
      ArgValue _ -> Left "Inline values aren't written here."
    Nothing -> Left ("No argument " <> T.pack name <> ".")
  str env o name = case lookup name o.opInputs of
    Just [ArgName n] -> case Map.lookup n env of
      Just (VString s) -> pure s
      _ -> Left (T.pack name <> " is not a string.")
    _ -> Left ("No argument " <> T.pack name <> ".")
  scalar t = case tensorElems t of
    [v] -> pure v
    _ -> Left "Not a scalar."
  ints t = pure (map round (tensorElems t))
  norm t a = if a < 0 then a + length t.tensorShape else a

------------------------------------------------------------------------------
-- Arithmetic

showT :: (Show a) => a -> Text
showT = T.pack . show

mapT :: (Double -> Double) -> Tensor -> Tensor
mapT f t = tensor t.tensorShape (map f (tensorElems t))

-- | Every index of a shape, the last axis fastest.
indices :: [Int] -> [[Int]]
indices = foldr (\d rest -> [i : r | i <- [0 .. d - 1], r <- rest]) [[]]

offset :: [Int] -> [Int] -> Int
offset shape ix = foldl' (\acc (d, i) -> acc * d + i) 0 (zip shape ix)

at :: Tensor -> [Int] -> Double
at t ix = t.tensorData ! offset t.tensorShape ix

broadcastWith :: (Double -> Double -> Double) -> Tensor -> Tensor -> Either Text Tensor
broadcastWith f x y = do
  let r = max (length x.tensorShape) (length y.tensorShape)
      pad s = replicate (r - length s) 1 <> s
      sx = pad x.tensorShape
      sy = pad y.tensorShape
  shape <- forM (zip sx sy) $ \(a, b) ->
    if a == b || a == 1 || b == 1 then pure (max a b) else Left ("Shapes " <> showT sx <> " and " <> showT sy <> " don't broadcast.")
  let pick s ix = [if d == 1 then 0 else i | (d, i) <- zip s ix]
      x' = x {tensorShape = sx}
      y' = y {tensorShape = sy}
  pure (tensor shape [f (at x' (pick sx ix)) (at y' (pick sy ix)) | ix <- indices shape])

{- | Layer normalisation over axes: each element less its group's mean,
over the square root of the group's population variance plus epsilon.
-}
layerNorm :: [Int] -> Double -> Tensor -> Tensor
layerNorm axes e t = tensor t.tensorShape [(at t ix - at means (key ix)) / sqrt (at vars (key ix) + e) | ix <- indices t.tensorShape]
 where
  mean xs = sum xs / fromIntegral (length xs)
  means = reduce mean axes t
  vars = reduce (\xs -> let m = mean xs in mean [(v - m) * (v - m) | v <- xs]) axes t
  key ix = [if i `elem` axes then 0 else j | (i, j) <- zip [0 ..] ix]

-- | A reduction over axes, each kept as an axis of one.
reduce :: ([Double] -> Double) -> [Int] -> Tensor -> Tensor
reduce f axes t = tensor shape [f [at t (merge ix r) | r <- indices reduced] | ix <- indices shape]
 where
  shape = [if i `elem` axes then 1 else d | (i, d) <- zip [0 ..] t.tensorShape]
  reduced = [d | (i, d) <- zip [0 ..] t.tensorShape, i `elem` axes]
  merge = go 0
   where
    go _ [] _ = []
    go i (a : as) rs
      | i `elem` axes, q : qs <- rs = q : go (i + 1) as qs
      | otherwise = a : go (i + 1) as rs

-- | The index of the first largest element.
argMax :: [Double] -> Double
argMax xs = fromIntegral (fst (foldl1 (\(i, a) (j, b) -> if b > a then (j, b) else (i, a)) (zip [0 :: Int ..] xs)))

-- | Each line's largest element and sum of exponentials taken once.
softmax :: Int -> Tensor -> Tensor
softmax axis t = tensor t.tensorShape [exp (at t ix - at tops (line ix)) / at totals (line ix) | ix <- indices t.tensorShape]
 where
  tops = reduce maximum [axis] t
  totals = tensor tops.tensorShape [sum [exp (at t (replace ix j) - at tops ix) | j <- [0 .. t.tensorShape !! axis - 1]] | ix <- indices tops.tensorShape]
  line ix = replace ix 0
  replace ix j = [if i == axis then j else v | (i, v) <- zip [0 ..] ix]

{- | A dot product as the ANE sums a convolution's or a matmul's, the
operands binary16 (observed bit for bit on an M3 Max under macOS 27.2,
over fan-ins of 64 to 16384 and operands of every magnitude): a product
whose operands' exponents sum below -16 is dropped, a subnormal's
counting as -14, so that @1.5·2^-9 · 1.5·2^-8@ is dropped though it is
over 2^-16; the others are each rounded to a multiple of 2^-16, to
nearest with ties away from zero, and summed in order in a fixed-point
accumulator of @[-32768, 32768)@, any partial sum past it infinite,
whatever the sum comes to; the sum is rounded to binary16, ties away.
-}
neuralEngineDot :: [(Double, Double)] -> Double
neuralEngineDot = go 0
 where
  go :: Int -> [(Double, Double)] -> Double
  go acc [] = halfAway (fromIntegral acc / 65536)
  go acc ((a, b) : rest)
    | a == 0 || b == 0 || expOf a + expOf b < -16 = go acc rest
    | otherwise =
        let acc' = acc + away (a * b * 65536)
         in if acc' >= 2 ^ (31 :: Int) || acc' < negate (2 ^ (31 :: Int))
              then fromIntegral (signum acc') / 0
              else go acc' rest
  expOf v = max (-14) (exponent v - 1)
  away t = let r = truncate (abs t + 0.5) in if t < 0 then negate r else r

-- | To binary16, to nearest with ties away from zero; infinite from 65520 on.
halfAway :: Double -> Double
halfAway v
  | v == 0 || isNaN v || isInfinite v = v
  | abs r >= 65520 = signum v / 0
  | otherwise = r
 where
  q = 2 ^^ (max (-14) (exponent v - 1) - 10) :: Double
  t = v / q
  r = fromIntegral (let n = truncate (abs t + 0.5) :: Integer in if t < 0 then negate n else n) * q

matmul :: ([(Double, Double)] -> Double) -> Tensor -> Bool -> Tensor -> Bool -> Either Text Val
matmul dot x tx y ty = do
  let lastTwo t = case splitAt (length t.tensorShape - 2) t.tensorShape of
        (batch, [r, c']) -> Right (batch, r, c')
        _ -> Left "A matmul of fewer than two axes."
  (bx, a, b) <- lastTwo x
  (by, c, d) <- lastTwo y
  let (rows, inner) = if tx then (b, a) else (a, b)
      (inner', cols) = if ty then (d, c) else (c, d)
      r = max (length bx) (length by)
      pad s = replicate (r - length s) 1 <> s
      bx' = pad bx
      by' = pad by
  unless (inner == inner') $ Left ("Inner sizes " <> showT inner <> " and " <> showT inner' <> ".")
  batch <- forM (zip bx' by') $ \(p, q) ->
    if p == q || p == 1 || q == 1 then pure (max p q) else Left "Batches that don't broadcast."
  let xAt bi i k = at x {tensorShape = bx' <> [a, b]} (pick bx' bi <> (if tx then [k, i] else [i, k]))
      yAt bi k j = at y {tensorShape = by' <> [c, d]} (pick by' bi <> (if ty then [j, k] else [k, j]))
      pick s ix = [if n == 1 then 0 else i | (n, i) <- zip s ix]
  pure . VTensor . tensor (batch <> [rows, cols]) $
    [ dot [(xAt bi i k, yAt bi k j) | k <- [0 .. inner - 1]]
    | bi <- indices batch
    , i <- [0 .. rows - 1]
    , j <- [0 .. cols - 1]
    ]
