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
  elementType,
  input,
  output,

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
  add,
  sub,
  mul,
  reduceMean,
  rsqrt,
  softmax,
  gelu,
  conv1x1,
  reshape,
  sliceAxis,
  concatAxis,
  matmul,
  rmsNorm,
  layerNorm,
) where

import Control.Monad.Trans.State.Strict (State, gets, modify', runState, state)
import Data.ByteString qualified as B
import Data.Int (Int32)
import Data.Word (Word64)
import FuryKernel.Ane.Blob (BlobType (..), Blobs, addBlob, emptyBlobs, reserveBlob)
import FuryKernel.Ane.CoreML
import FuryKernel.Ane.Half (halfBytes)

-- | A value of the program: its name and type.
data Var = Var {varName :: String, varType :: TensorType}
  deriving stock (Eq, Show)

shapeOf :: Var -> [Word64]
shapeOf = ttShape . varType

{- | A weight a runtime writes into the blob file: the offset of its blob's
record, the model's tensor it comes from, its shape, and a factor its
elements are multiplied by on the way, converted to the program's element
type.
-}
data Weight = Weight
  { weightRecord :: Word64
  , weightTensor :: String
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
  }

newtype Mil a = Mil (State St a)
  deriving newtype (Functor, Applicative, Monad)

{- | Builds a program whose inputs, weights and scalars are of this element
type (fp16 for the ANE, fp32 for a reference), its outputs those returned.
-}
runMil :: DataType -> Mil [Var] -> Built
runMil t (Mil build) =
  let (outs, st) = runState build (St t 0 [] emptyBlobs [] [])
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
   in Built model st.stBlobs (reverse st.stWeights)

elementType :: Mil DataType
elementType = Mil (gets (.stType))

fresh :: String -> Mil String
fresh prefix = Mil $ state $ \st ->
  (prefix <> "_" <> show st.stCounter, st {stCounter = st.stCounter + 1})

emit :: Op -> Mil ()
emit o = Mil (modify' (\st -> st {stOps = o : st.stOps}))

nameAttr :: String -> (String, Value)
nameAttr n = ("name", Immediate (TensorType String []) (Strings [n]))

-- | A program input of this name and type.
input :: String -> TensorType -> Mil Var
input n t = do
  Mil (modify' (\st -> st {stInputs = (n, t) : st.stInputs}))
  pure (Var n t)

{- | The value as a program output of this name: its op renamed, if it was
the last one appended, else an identity of it.
-}
output :: String -> Var -> Mil Var
output new v = do
  ops <- Mil (gets (.stOps))
  case ops of
    o : rest
      | [(old, t)] <- o.opOutputs
      , old == v.varName -> do
          let o' = o {opOutputs = [(new, t)], opAttributes = nameAttr new : drop 1 o.opAttributes}
          Mil (modify' (\st -> st {stOps = o' : rest}))
          pure (Var new t)
    _ -> do
      let t = v.varType
      emit
        Op
          { opType = "identity"
          , opInputs = [("x", [ArgName v.varName])]
          , opOutputs = [(new, t)]
          , opAttributes = [nameAttr new]
          }
      pure (Var new t)

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

{- | A model's tensor of this shape, its elements times this factor, its
blob reserved for a runtime to fill.
-}
weight :: String -> [Word64] -> Double -> Mil Var
weight tensor shape scale = do
  dt <- elementType
  off <- Mil $ state $ \st ->
    let (o, b) = reserveBlob (blobType dt) (product shape * elementSize dt) st.stBlobs
     in (o, st {stBlobs = b, stWeights = Weight o tensor shape scale : st.stWeights})
  let t = TensorType dt shape
  constValue "w" t (BlobFile t blobPath off)

-- | A scalar of the program's element type.
constScalar :: Float -> Mil Var
constScalar x = do
  dt <- elementType
  let t = TensorType dt []
      imm = case dt of
        Float32 -> Floats [x]
        _ -> Bytes (halfBytes [x])
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
  n <- fresh ty
  dt <- case ins of
    (_, v : _) : _ -> pure v.varType.ttDataType
    _ -> elementType
  let t = TensorType dt shape
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

add, sub, mul :: Var -> Var -> Mil Var
add = binary "add"
sub = binary "sub"
mul = binary "mul"

reduceMean :: Var -> [Int] -> Mil Var
reduceMean x axes = do
  a <- constInts (map fromIntegral axes)
  k <- constBool True
  op
    "reduce_mean"
    [("x", [x]), ("axes", [a]), ("keep_dims", [k])]
    [if i `elem` axes then 1 else d | (i, d) <- zip [0 ..] (shapeOf x)]

-- | @1 / sqrt (x + epsilon)@.
rsqrt :: Var -> Float -> Mil Var
rsqrt x eps = do
  e <- constScalar eps
  op "rsqrt" [("x", [x]), ("epsilon", [e])] (shapeOf x)

softmax :: Var -> Int -> Mil Var
softmax x axis = do
  a <- constInt (fromIntegral axis)
  op "softmax" [("x", [x]), ("axis", [a])] (shapeOf x)

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
-}
rmsNorm :: Var -> Int -> Maybe Var -> Float -> Mil Var
rmsNorm x axis w eps = do
  neg <- mul x =<< constScalar (-1)
  both <- concatAxis [x, neg] axis
  normed <- layerNorm both [axis] eps
  y <- sliceAxis normed axis 0 (shapeOf x !! axis)
  maybe (pure y) (mul y) w

{- | Layer normalisation along axes, without a gamma or beta:
@(x - mean x) / sqrt (var x + eps)@, the variance the population's.
-}
layerNorm :: Var -> [Int] -> Float -> Mil Var
layerNorm x axes eps = do
  a <- constInts (map fromIntegral axes)
  e <- constScalar eps
  op "layer_norm" [("x", [x]), ("axes", [a]), ("epsilon", [e])] (shapeOf x)
