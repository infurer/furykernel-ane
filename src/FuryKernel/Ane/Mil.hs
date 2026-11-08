-- SPDX-FileCopyrightText: 2026 Mika Tammi
-- SPDX-License-Identifier: BSD-3-Clause

{- | A builder for ML programs: one function, one block, ops appended in
order, each parameter a named const, as coremltools emits them. Weights go
to the blob file: given bytes, or a model's tensor, whose blob is only
reserved, and listed in the weight map ('Weight') for a runtime to fill
from the model's own files. The helpers know just enough shape arithmetic
for the ops a transformer step needs, in the ANE's @(B, C, 1, S)@ layout
or otherwise.

Programs are Core ML specification version 10 with the opset @CoreML9@,
which need macOS 26 or later: the newest that macOS 27.2's Core ML takes,
whose compiler refuses version 11 as greater than its maximum and doesn't
know an opset @CoreML10@. Building is pure: the same program always makes
the same bytes.
-}
module FuryKernel.Ane.Mil (
  -- * Building
  Mil,
  Var (..),
  shapeOf,
  Weight (..),
  Built (..),
  runMil,
  runMilWith,
  elementType,
  input,
  output,
  named,
  reserve,
  unique,
  numbered,

  -- * Constants
  constBlob,
  weight,
  constScalar,
  constInts,
  constInt,
  constBool,
  constStr,

  -- * Ops
  op,
  opOf,
  add,
  sub,
  mul,
  realDiv,
  reduceMean,
  reduceArgmax,
  rsqrt,
  softmax,
  gelu,
  tanhOp,
  castFp16,
  conv1x1,
  reshape,
  sliceAxis,
  concatAxis,
  matmul,
  rmsNorm,
  layerNorm,
  transpose,
) where

import Control.Monad.Trans.State.Strict (State, gets, modify', runState, state)
import Data.Bits (shiftR)
import Data.ByteString qualified as B
import Data.Int (Int32)
import Data.Map.Strict (Map)
import Data.Map.Strict qualified as Map
import Data.Maybe (fromMaybe, isJust)
import Data.Set (Set)
import Data.Set qualified as Set
import Data.Word (Word64)
import FuryKernel.Ane.Blob (BlobType (..), Blobs, addBlob, emptyBlobs, reserveBlob)
import FuryKernel.Ane.CoreML
import FuryKernel.Numerics (narrowF16Double)
import FuryKernel.Type (F16 (..))

-- | A value of the program: its name and type.
data Var = Var {varName :: String, varType :: TensorType}
  deriving stock (Eq, Show)

shapeOf :: Var -> [Word64]
shapeOf = ttShape . varType

{- | A weight a runtime writes into the blob file: the offset of its blob's
record, the model's tensor it comes from and the element of the tensor it
starts at, its shape (as many elements as the shape has, in the tensor's
order), and a factor its elements are multiplied by on the way, converted
to the program's element type.
-}
data Weight = Weight
  { weightRecord :: Word64
  , weightTensor :: String
  , weightOffset :: Word64
  , weightShape :: [Word64]
  , weightScale :: Double
  }
  deriving stock (Eq, Show)

-- | A program built: the model, its blob file and the weights to fill in it.
data Built = Built
  { builtModel :: Model
  , builtBlobs :: Blobs
  , builtWeights :: [Weight]
  }
  deriving stock (Eq, Show)

data St = St
  { stType :: DataType
  , stCounter :: Int
  , stOps :: [Op]
  -- ^ Reversed.
  , stBlobs :: Blobs
  , stInputs :: [(String, TensorType)]
  -- ^ Reversed.
  , stWeights :: [Weight]
  -- ^ Reversed.
  , stWeightVars :: Map (String, Word64, [Word64], Double) Var
  -- ^ Each weight's const, which every use of it shares.
  , stTaken :: Set String
  -- ^ Every name of the program so far, and those reserved for it.
  , stRenamed :: Map String String
  -- ^ The values whose op an output or 'named' renamed: the new name.
  }

newtype Mil a = Mil (State St a)
  deriving newtype (Functor, Applicative, Monad)

{- | Builds a program whose inputs, weights and scalars are of this element
type (fp16 for the ANE, fp32 for a reference), its outputs those returned.
-}
runMil :: DataType -> Mil [Var] -> Built
runMil t build = fst (runMilWith t ((,()) <$> build))

-- | 'runMil', with what the build returns besides its outputs.
runMilWith :: DataType -> Mil ([Var], a) -> (Built, a)
runMilWith t (Mil build) =
  let ((outs, other), st) = runState build (St t 0 [] emptyBlobs [] [] Map.empty Set.empty Map.empty)
      ins = reverse st.stInputs
      fn =
        Function
          { fnInputs = ins
          , fnOpset = "CoreML9"
          , fnOutputs = map (.varName) outs
          , fnOperations = reverse st.stOps
          }
      model =
        Model
          { modelSpecificationVersion = 10
          , modelInputs = [Feature n ty | (n, ty) <- ins]
          , modelOutputs = [Feature v.varName v.varType | v <- outs]
          , modelFunctions = [("main", fn)]
          }
   in (Built model st.stBlobs (reverse st.stWeights), other)

elementType :: Mil DataType
elementType = Mil (gets (.stType))

-- | A name of this prefix and the counter's next number that no name of the program has.
fresh :: String -> Mil String
fresh prefix = Mil $ state $ \st ->
  case [(n, k) | k <- [st.stCounter ..], let n = prefix <> "_" <> show k, n `Set.notMember` st.stTaken] of
    (n, k) : _ -> (n, st {stCounter = k + 1, stTaken = Set.insert n st.stTaken})
    [] -> error "Mil: no free name"

{- | Names the program will have, given before it is built (its inputs and
outputs), which every name made after keeps clear of.
-}
reserve :: [String] -> Mil ()
reserve ns = Mil (modify' (\st -> st {stTaken = foldr Set.insert st.stTaken ns}))

{- | This name if no name of the program has it, else the first of it with
@_1@, @_2@ and so on that none has; taken from then on.
-}
unique :: String -> Mil String
unique n = Mil $ state $ \st ->
  case [c | c <- n : [n <> "_" <> show k | k <- [1 :: Int ..]], c `Set.notMember` st.stTaken] of
    c : _ -> (c, st {stTaken = Set.insert c st.stTaken})
    [] -> error "Mil: no free name"

-- | The prefix and the first number from 0 that no name of the program has; taken from then on.
numbered :: String -> Mil String
numbered prefix = Mil $ state $ \st ->
  case [n | k <- [0 :: Int ..], let n = prefix <> show k, n `Set.notMember` st.stTaken] of
    n : _ -> (n, st {stTaken = Set.insert n st.stTaken})
    [] -> error "Mil: no free name"

emit :: Op -> Mil ()
emit o = Mil (modify' (\st -> st {stOps = o : st.stOps}))

nameAttr :: String -> (String, Value)
nameAttr n = ("name", Immediate (TensorType String []) (Strings [n]))

-- | A program input of this name and type.
input :: String -> TensorType -> Mil Var
input n t = do
  Mil (modify' (\st -> st {stInputs = (n, t) : st.stInputs, stTaken = Set.insert n st.stTaken}))
  pure (Var n t)

{- | The value as a program output of this name: its op renamed, if no op
reads it and no output renamed it already (the value under its old name
is gone then), else an identity of it under its name now. The name is
the caller's to keep clear ('reserve').
-}
output :: String -> Var -> Mil Var
output new v = do
  Mil (modify' (\st -> st {stTaken = Set.insert new st.stTaken}))
  now <- Mil (gets (Map.lookup v.varName . (.stRenamed)))
  done <- case now of
    Just _ -> pure Nothing
    Nothing -> unread new v
  case done of
    Just v' -> do
      Mil (modify' (\st -> st {stRenamed = Map.insert v.varName new st.stRenamed}))
      pure v'
    Nothing -> do
      let t = v.varType
      emit
        Op
          { opType = "identity"
          , opInputs = [("x", [ArgName (fromMaybe v.varName now)])]
          , opOutputs = [(new, t)]
          , opAttributes = [nameAttr new]
          }
      pure (Var new t)

{- | The value under this name, if the last op appended made it (and so
nothing reads it yet): names that tell what a program's ops compute, in
the places Core ML reports them. Otherwise the value as it was.
-}
named :: String -> Var -> Mil Var
named new v = fromMaybe v <$> renamed new v

-- | The value's op renamed, wherever it is, if no op reads the value.
unread :: String -> Var -> Mil (Maybe Var)
unread new v = do
  ops <- Mil (gets (.stOps))
  let readsIt o = any (\(_, as) -> ArgName v.varName `elem` as) o.opInputs
      rename o = case o.opOutputs of
        [(old, t)] | old == v.varName -> Just (o {opOutputs = [(new, t)], opAttributes = nameAttr new : drop 1 o.opAttributes}, t)
        _ -> Nothing
  case break (isJust . rename) ops of
    (later, o : earlier)
      | not (any readsIt later)
      , Just (o', t) <- rename o -> do
          Mil (modify' (\st -> st {stOps = later <> (o' : earlier)}))
          pure (Just (Var new t))
    _ -> pure Nothing

renamed :: String -> Var -> Mil (Maybe Var)
renamed new v = do
  ops <- Mil (gets (.stOps))
  case ops of
    o : rest
      | [(old, t)] <- o.opOutputs
      , old == v.varName -> do
          let o' = o {opOutputs = [(new, t)], opAttributes = nameAttr new : drop 1 o.opAttributes}
          Mil (modify' (\st -> st {stOps = o' : rest}))
          pure (Just (Var new t))
    _ -> pure Nothing

constValue :: String -> TensorType -> Value -> Mil Var
constValue prefix t v = do
  n <- fresh prefix
  emit Op {opType = "const", opInputs = [], opOutputs = [(n, t)], opAttributes = [nameAttr n, ("val", v)]}
  pure (Var n t)

blobPath :: FilePath
blobPath = "@model_path/weights/weight.bin"

blobType :: DataType -> BlobType
blobType = \case
  Float32 -> BlobFloat32
  _ -> BlobFloat16

{- | A tensor in the blob file, from little-endian bytes of the program's
element type.
-}
constBlob :: [Word64] -> B.ByteString -> Mil Var
constBlob shape bs = do
  dt <- elementType
  off <- Mil $ state $ \st -> let (o, b) = addBlob (blobType dt) bs st.stBlobs in (o, st {stBlobs = b})
  let t = TensorType dt shape
  constValue "blob" t (BlobFile t blobPath off)

elementSize :: DataType -> Word64
elementSize = \case
  Float32 -> 4
  Int32 -> 4
  _ -> 2

{- | A model's tensor, from this element on, in this shape, its elements
times this factor; its blob reserved for a runtime to fill. The same
weight asked for again is the same const.
-}
weight :: String -> Word64 -> [Word64] -> Double -> Mil Var
weight tensor from shape scale = do
  let key = (tensor, from, shape, scale)
  Mil (gets (Map.lookup key . (.stWeightVars))) >>= \case
    Just v -> pure v
    Nothing -> do
      dt <- elementType
      off <- Mil $ state $ \st ->
        let (o, b) = reserveBlob (blobType dt) (product shape * elementSize dt) st.stBlobs
         in (o, st {stBlobs = b, stWeights = Weight o tensor from shape scale : st.stWeights})
      let t = TensorType dt shape
      v <- constValue "w" t (BlobFile t blobPath off)
      Mil (modify' (\st -> st {stWeightVars = Map.insert key v st.stWeightVars}))
      pure v

{- | A scalar of the program's element type, rounded to it once, to
nearest with ties to even.
-}
constScalar :: Double -> Mil Var
constScalar x = do
  dt <- elementType
  let t = TensorType dt []
      imm = case dt of
        Float32 -> Floats [realToFrac x]
        _ -> let F16 h = narrowF16Double x in Bytes (B.pack [fromIntegral h, fromIntegral (h `shiftR` 8)])
  constValue "scalar" t (Immediate t imm)

constInts :: [Int32] -> Mil Var
constInts xs = constValue "ints" t (Immediate t (Ints xs))
 where
  t = TensorType Int32 [fromIntegral (length xs)]

constInt :: Int32 -> Mil Var
constInt x = constValue "int" t (Immediate t (Ints [x]))
 where
  t = TensorType Int32 []

constBool :: Bool -> Mil Var
constBool x = constValue "bool" t (Immediate t (Bools [x]))
 where
  t = TensorType Bool []

constStr :: String -> Mil Var
constStr x = constValue "str" t (Immediate t (Strings [x]))
 where
  t = TensorType String []

-- | An op with one output of the given shape, in its first input's element type.
op :: String -> [(String, [Var])] -> [Word64] -> Mil Var
op ty ins shape = do
  dt <- case ins of
    (_, v : _) : _ -> pure v.varType.ttDataType
    _ -> elementType
  opOf ty ins (TensorType dt shape)

-- | An op with one output of the given type.
opOf :: String -> [(String, [Var])] -> TensorType -> Mil Var
opOf ty ins t = do
  n <- fresh ty
  emit
    Op
      { opType = ty
      , opInputs = [(k, map (ArgName . (.varName)) vs) | (k, vs) <- ins]
      , opOutputs = [(n, t)]
      , opAttributes = [nameAttr n]
      }
  pure (Var n t)

broadcast :: [Word64] -> [Word64] -> [Word64]
broadcast a b = zipWith max (pad a) (pad b)
 where
  r = max (length a) (length b)
  pad xs = replicate (r - length xs) 1 <> xs

binary :: String -> Var -> Var -> Mil Var
binary ty x y = op ty [("x", [x]), ("y", [y])] (broadcast (shapeOf x) (shapeOf y))

add, sub, mul, realDiv :: Var -> Var -> Mil Var
add = binary "add"
sub = binary "sub"
mul = binary "mul"
realDiv = binary "real_div"

reduceMean :: Var -> [Int] -> Mil Var
reduceMean x axes = do
  a <- constInts (map fromIntegral axes)
  k <- constBool True
  op
    "reduce_mean"
    [("x", [x]), ("axes", [a]), ("keep_dims", [k])]
    [if i `elem` axes then 1 else d | (i, d) <- zip [0 ..] (shapeOf x)]

{- | The index of the largest element along an axis, kept as an axis of
one: int32.
-}
reduceArgmax :: Var -> Int -> Mil Var
reduceArgmax x axis = do
  a <- constInt (fromIntegral axis)
  k <- constBool True
  opOf
    "reduce_argmax"
    [("x", [x]), ("axis", [a]), ("keep_dims", [k])]
    (TensorType Int32 [if i == axis then 1 else d | (i, d) <- zip [0 ..] (shapeOf x)])

-- | @1 / sqrt (x + epsilon)@.
rsqrt :: Var -> Double -> Mil Var
rsqrt x eps = do
  e <- constScalar eps
  op "rsqrt" [("x", [x]), ("epsilon", [e])] (shapeOf x)

softmax :: Var -> Int -> Mil Var
softmax x axis = do
  a <- constInt (fromIntegral axis)
  op "softmax" [("x", [x]), ("axis", [a])] (shapeOf x)

tanhOp :: Var -> Mil Var
tanhOp x = op "tanh" [("x", [x])] (shapeOf x)

{- | @x@ as binary16, to nearest, ties to even, whatever the program's
element type: an identity on the fp16 tensors the ANE computes in, which
Core ML fuses with the op that makes @x@, that says where a program
rounds to fp16.
-}
castFp16 :: Var -> Mil Var
castFp16 x = do
  d <- constStr "fp16"
  opOf "cast" [("x", [x]), ("dtype", [d])] (TensorType Float16 (shapeOf x))

-- | GELU; the mode is @EXACT@, @TANH_APPROXIMATION@ or @SIGMOID_APPROXIMATION@.
gelu :: Var -> String -> Mil Var
gelu x mode = do
  md <- constStr mode
  op "gelu" [("x", [x]), ("mode", [md])] (shapeOf x)

{- | A linear layer as a 1x1 convolution: x @[B, Cin, 1, S]@, the weight
@[Cout, Cin, 1, 1]@.
-}
conv1x1 :: Var -> Var -> Mil Var
conv1x1 x w = do
  strides <- constInts [1, 1]
  padType <- constStr "valid"
  pad <- constInts [0, 0, 0, 0]
  dil <- constInts [1, 1]
  groups <- constInt 1
  let out = case (shapeOf x, shapeOf w) of
        ([b, _, h, s], cout : _) -> [b, cout, h, s]
        _ -> []
  op
    "conv"
    [ ("x", [x])
    , ("weight", [w])
    , ("strides", [strides])
    , ("pad_type", [padType])
    , ("pad", [pad])
    , ("dilations", [dil])
    , ("groups", [groups])
    ]
    out

reshape :: Var -> [Word64] -> Mil Var
reshape x shape = do
  s <- constInts (map fromIntegral shape)
  op "reshape" [("x", [x]), ("shape", [s])] shape

-- | The elements @[begin, end)@ along one axis.
sliceAxis :: Var -> Int -> Word64 -> Word64 -> Mil Var
sliceAxis x axis begin end = do
  let shape = shapeOf x
  b <- constInts [if i == axis then fromIntegral begin else 0 | i <- [0 .. length shape - 1]]
  e <- constInts [fromIntegral (if i == axis then end else d) | (i, d) <- zip [0 ..] shape]
  op
    "slice_by_index"
    [("x", [x]), ("begin", [b]), ("end", [e])]
    [if i == axis then end - begin else d | (i, d) <- zip [0 ..] shape]

concatAxis :: [Var] -> Int -> Mil Var
concatAxis xs axis = do
  a <- constInt (fromIntegral axis)
  il <- constBool False
  let shape = case xs of
        x : _ -> [if i == axis then sum (map ((!! axis) . shapeOf) xs) else d | (i, d) <- zip [0 ..] (shapeOf x)]
        [] -> []
  op "concat" [("values", xs), ("axis", [a]), ("interleave", [il])] shape

-- | A batched matmul over the last two axes, the batch axes broadcast.
matmul :: Var -> Bool -> Var -> Bool -> Mil Var
matmul x tx y ty = do
  a <- constBool tx
  b <- constBool ty
  let sx = shapeOf x
      sy = shapeOf y
      lastTwo s t = case drop (length s - 2) s of
        [r, c] -> if t then (c, r) else (r, c)
        _ -> (0, 0)
      (rows, _) = lastTwo sx tx
      (_, cols) = lastTwo sy ty
      batch = broadcast (take (length sx - 2) sx) (take (length sy - 2) sy)
  op "matmul" [("x", [x]), ("y", [y]), ("transpose_x", [a]), ("transpose_y", [b])] (batch <> [rows, cols])

{- | RMSNorm along an axis: @x * rsqrt (mean (x^2) + eps)@, times @w@ if
given, as the layer norm of @x@ beside @-x@ along the axis, cut back to
@x@'s half: beside its negation, @x@ has mean 0 and variance
@mean (x^2)@. It squares nothing in the program, so no input is scaled
to keep a square in fp16's range, which would move small inputs where
the ANE's values are coarse.

Along axis 2 of four (a head's elements, in the per-head layout), it
moves the axis last around the norm, by transposes: Core ML's CPU
computes the layer norm of a concat along axis 2 wrongly at some shapes,
as observed under macOS 27.2 (heads 0, 1, 4 and 5 of @[1, 6, 64, 128]@
NaN or wrong), and right along the last axis.
-}
rmsNorm :: Var -> Int -> Maybe Var -> Double -> Mil Var
rmsNorm x axis w eps = do
  y <-
    if axis == 2 && length (shapeOf x) == 4
      then do
        xt <- transpose x [0, 1, 3, 2]
        yt <- besideItsNegation xt 3
        transpose yt [0, 1, 3, 2]
      else besideItsNegation x axis
  maybe (pure y) (mul y) w
 where
  besideItsNegation v a = do
    neg <- mul v =<< constScalar (-1)
    both <- concatAxis [v, neg] a
    normed <- layerNorm both [a] eps
    sliceAxis normed a 0 (shapeOf v !! a)

-- | The axes in a new order: the result's axis @i@ is the input's @perm !! i@.
transpose :: Var -> [Int] -> Mil Var
transpose x perm = do
  p <- constInts (map fromIntegral perm)
  op "transpose" [("x", [x]), ("perm", [p])] [shapeOf x !! a | a <- perm]

{- | Layer normalisation along axes, without a gamma or beta:
@(x - mean x) / sqrt (var x + eps)@, the variance the population's.
-}
layerNorm :: Var -> [Int] -> Double -> Mil Var
layerNorm x axes eps = do
  a <- constInts (map fromIntegral axes)
  e <- constScalar eps
  op "layer_norm" [("x", [x]), ("axes", [a]), ("epsilon", [e])] (shapeOf x)
