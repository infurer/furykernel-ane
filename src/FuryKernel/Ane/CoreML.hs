-- SPDX-FileCopyrightText: 2026 Mika Tammi
-- SPDX-License-Identifier: BSD-3-Clause

{- | The subset of the Core ML model specification that an ML program needs,
with its encoding. Field numbers are those of coremltools' Model.proto,
FeatureTypes.proto and MIL.proto (vendor/coremltools). A text form of
programs ('renderModel') is for people: reviews of generated programs.
-}
module FuryKernel.Ane.CoreML (
  DataType (..),
  TensorType (..),
  Immediate (..),
  Value (..),
  Arg (..),
  Op (..),
  Function (..),
  Feature (..),
  Model (..),
  encodeModel,
  renderModel,
) where

import Data.Bits (shiftL, (.|.))
import Data.ByteString qualified as B
import Data.ByteString.Lazy qualified as BL
import Data.Int (Int32)
import Data.Text (Text)
import Data.Text qualified as T
import Data.Word (Word64)
import FuryKernel.Ane.Half (halfToDouble)
import FuryKernel.Ane.Protobuf qualified as P

-- | Element types, as MIL.proto's DataType.
data DataType = Bool | String | Float16 | Float32 | Int32
  deriving stock (Eq, Show)

milDataType :: DataType -> Word64
milDataType = \case
  Bool -> 1
  String -> 2
  Float16 -> 10
  Float32 -> 11
  Int32 -> 23

-- | A tensor type with a static shape; @[]@ is a scalar.
data TensorType = TensorType {ttDataType :: DataType, ttShape :: [Word64]}
  deriving stock (Eq, Show)

-- | Immediate tensor contents. fp16 values are raw little-endian bytes.
data Immediate = Floats [Float] | Ints [Int32] | Bools [Bool] | Strings [String] | Bytes B.ByteString
  deriving stock (Eq, Show)

data Value
  = Immediate TensorType Immediate
  | -- | A tensor in the weight file, at the offset of its blob metadata.
    BlobFile TensorType FilePath Word64
  deriving stock (Eq, Show)

-- | An operation input: a named value, or an inline one.
data Arg = ArgName String | ArgValue Value
  deriving stock (Eq, Show)

data Op = Op
  { opType :: String
  , opInputs :: [(String, [Arg])]
  , opOutputs :: [(String, TensorType)]
  , opAttributes :: [(String, Value)]
  }
  deriving stock (Eq, Show)

-- | A function with one block, specialised for its opset.
data Function = Function
  { fnInputs :: [(String, TensorType)]
  , fnOpset :: String
  , fnOutputs :: [String]
  , fnOperations :: [Op]
  }
  deriving stock (Eq, Show)

-- | A model input or output: a multi-array.
data Feature = Feature {featureName :: String, featureType :: TensorType}
  deriving stock (Eq, Show)

data Model = Model
  { modelSpecificationVersion :: Word64
  , modelInputs :: [Feature]
  , modelOutputs :: [Feature]
  , modelFunctions :: [(String, Function)]
  }
  deriving stock (Eq, Show)

-- TensorType: dataType 1, rank 2, dimensions 3 (Dimension.constant 1, its size 1).
tensorType :: TensorType -> P.Msg
tensorType (TensorType dt shape) =
  P.uint 1 (milDataType dt)
    <> (if null shape then mempty else P.int64 2 (fromIntegral (length shape)))
    <> foldMap (P.message 3 . P.message 1 . P.uint 1) shape

-- ValueType: tensorType 1.
valueType :: TensorType -> P.Msg
valueType = P.message 1 . tensorType

-- NamedValueType: name 1, type 2.
namedValueType :: (String, TensorType) -> P.Msg
namedValueType (n, t) = P.string 1 n <> P.message 2 (valueType t)

-- TensorValue: floats 1, ints 2, bools 3, strings 4, bytes 7; each holds
-- its values as field 1.
tensorValue :: Immediate -> P.Msg
tensorValue = \case
  Floats xs -> P.message 1 (P.packedFloats 1 xs)
  Ints xs -> P.message 2 (P.packedVarints 1 (map fromIntegral xs))
  Bools xs -> P.message 3 (P.packedVarints 1 (map (\b -> if b then 1 else 0) xs))
  Strings xs -> P.message 4 (foldMap (P.string 1) xs)
  Bytes b -> P.message 7 (P.bytes 1 b)

-- Value: type 2, immediateValue 3 (its tensor 1), blobFileValue 5 (fileName 1,
-- offset 2).
value :: Value -> P.Msg
value = \case
  Immediate t imm -> P.message 2 (valueType t) <> P.message 3 (P.message 1 (tensorValue imm))
  BlobFile t path off -> P.message 2 (valueType t) <> P.message 5 (P.string 1 path <> P.uint 2 off)

-- Argument: arguments 1, each a Binding with name 1 or value 2.
argument :: [Arg] -> P.Msg
argument = foldMap (P.message 1 . binding)
 where
  binding (ArgName n) = P.string 1 n
  binding (ArgValue v) = P.message 2 (value v)

-- Operation: type 1, inputs 2 (map to Argument), outputs 3, attributes 5
-- (map to Value).
operation :: Op -> P.Msg
operation o =
  P.string 1 (opType o)
    <> foldMap (\(k, as) -> P.mapEntry 2 (P.string 1 k) (P.message 2 (argument as))) (opInputs o)
    <> foldMap (P.message 3 . namedValueType) (opOutputs o)
    <> foldMap (\(k, v) -> P.mapEntry 5 (P.string 1 k) (P.message 2 (value v))) (opAttributes o)

-- Function: inputs 1, opset 2, block_specializations 3 (map to Block: outputs
-- 2, operations 3).
function :: Function -> P.Msg
function f =
  foldMap (P.message 1 . namedValueType) (fnInputs f)
    <> P.string 2 (fnOpset f)
    <> P.mapEntry 3 (P.string 1 (fnOpset f)) (P.message 2 block)
 where
  block = foldMap (P.string 2) (fnOutputs f) <> foldMap (P.message 3 . operation) (fnOperations f)

-- FeatureDescription: name 1, type 3 (FeatureType.multiArrayType 5:
-- ArrayFeatureType with shape 1, dataType 2).
feature :: Feature -> P.Msg
feature (Feature n (TensorType dt shape)) =
  P.string 1 n <> P.message 3 (P.message 5 (P.packedVarints 1 shape <> P.uint 2 (arrayDataType dt)))
 where
  arrayDataType = \case
    Float32 -> 65568
    Float16 -> 65552
    Int32 -> 131104
    _ -> 0

-- Model: specificationVersion 1, description 2 (input 1, output 10),
-- mlProgram 502 (Program: version 1, functions 2).
encodeModel :: Model -> BL.ByteString
encodeModel m =
  P.encode
    ( P.uint 1 (modelSpecificationVersion m)
        <> P.message
          2
          (foldMap (P.message 1 . feature) (modelInputs m) <> foldMap (P.message 10 . feature) (modelOutputs m))
        <> P.message
          502
          (P.int64 1 1 <> foldMap (\(n, f) -> P.mapEntry 2 (P.string 1 n) (P.message 2 (function f))) (modelFunctions m))
    )

{- | A program as text, one line per op, in the order the block runs them:
each output with its type, the op and its named arguments; a const's
value inline, or its blob's offset. Deterministic, as the encoding is.
-}
renderModel :: Model -> Text
renderModel m = T.unlines (concatMap fn (modelFunctions m))
 where
  fn (name, f) =
    ["function " <> T.pack name <> "(" <> T.intercalate ", " (map typed (fnInputs f)) <> ")"]
      <> map op (fnOperations f)
      <> ["  return (" <> T.intercalate ", " (map (("%" <>) . T.pack) (fnOutputs f)) <> ")"]
  typed (n, t) = "%" <> T.pack n <> ": " <> tensorText t
  tensorText (TensorType dt shape) =
    dtText dt <> "[" <> T.intercalate ", " (map (T.pack . show) shape) <> "]"
  dtText = \case
    Bool -> "bool"
    String -> "string"
    Float16 -> "fp16"
    Float32 -> "fp32"
    Int32 -> "int32"
  op o =
    "  "
      <> T.intercalate ", " (map typed (opOutputs o))
      <> " = "
      <> T.pack (opType o)
      <> "("
      <> T.intercalate ", " (map input (opInputs o) <> [T.pack k <> " = " <> valueText v | (k, v) <- opAttributes o, k /= "name"])
      <> ")"
  input (k, as) =
    T.pack k <> " = " <> case as of
      [a] -> arg a
      _ -> "(" <> T.intercalate ", " (map arg as) <> ")"
  arg = \case
    ArgName n -> "%" <> T.pack n
    ArgValue v -> valueText v
  valueText = \case
    Immediate _ imm -> case imm of
      Floats xs -> list (map (T.pack . show) xs)
      Ints xs -> list (map (T.pack . show) xs)
      Bools xs -> list [if b then "true" else "false" | b <- xs]
      Strings xs -> list (map (T.pack . show) xs)
      Bytes b -> list (map (T.pack . show) (halves b))
    BlobFile _ _ off -> "blob(" <> T.pack (show off) <> ")"
  list = \case
    [x] -> x
    xs -> "[" <> T.intercalate ", " xs <> "]"
  halves b =
    [ halfToDouble (fromIntegral (B.index b i) .|. (fromIntegral (B.index b (i + 1)) `shiftL` 8))
    | i <- [0, 2 .. B.length b - 2]
    ]
