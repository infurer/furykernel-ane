-- SPDX-FileCopyrightText: 2026 Mika Tammi
-- SPDX-License-Identifier: BSD-3-Clause

{- | ML programs built: a tiny one's encoding against its golden bytes (which
@checks.protoc-decode@ decodes against coremltools' definitions), the
outputs' names, and a model's weights reserved and listed.
-}
module Test.FuryKernel.Ane.MilTest (tests) where

import Control.Exception (IOException, try)
import Data.ByteString qualified as B
import Data.ByteString.Lazy qualified as BL
import FuryKernel.Ane.Blob (Placed (..), blobsLayout)
import FuryKernel.Ane.CoreML
import FuryKernel.Ane.Mil
import System.Directory (getTemporaryDirectory)
import System.FilePath ((</>))
import Test.Tasty (TestTree, testGroup)
import Test.Tasty.HUnit (assertFailure, testCase, (@?=))

-- | @y = x * 2 + 1@ over four fp16 values.
tiny :: Built
tiny = runMil Float16 $ do
  x <- input "x" (TensorType Float16 [1, 4])
  two <- constScalar 2
  one <- constScalar 1
  y <- add one =<< mul x two
  pure <$> output "y" y

golden :: FilePath
golden = "test/golden/tiny.mlmodel"

tests :: TestTree
tests =
  testGroup
    "Mil"
    [ testCase "a tiny program's bytes are its golden ones" $ do
        want <- either (const Nothing) Just <$> (try (B.readFile golden) :: IO (Either IOException B.ByteString))
        let got = BL.toStrict (encodeModel tiny.builtModel)
        if Just got == want
          then pure ()
          else do
            tmp <- getTemporaryDirectory
            B.writeFile (tmp </> "tiny.mlmodel") got
            assertFailure ("the encoding changed; it is in " <> tmp </> "tiny.mlmodel, to replace " <> golden)
    , testCase "an output names its op, or an identity of an earlier value" $ do
        let built = runMil Float32 $ do
              x <- input "x" (TensorType Float32 [2])
              y <- mul x x
              z <- add x x
              b <- output "b" z
              a <- output "a" y
              pure [a, b]
            ops = case built.builtModel.modelFunctions of
              [(_, f)] -> f.fnOperations
              _ -> []
        map (\o -> (o.opType, map fst o.opOutputs)) ops
          @?= [("mul", ["mul_0"]), ("add", ["b"]), ("identity", ["a"])]
    , testCase "a model's weights are reserved and listed" $ do
        let built = runMil Float16 $ do
              x <- input "x" (TensorType Float16 [1, 4, 1, 1])
              w <- weight "layers.0.q_proj" [8, 4, 1, 1] 1
              _ <- weight "layers.0.norm" [4] 0.0625
              y <- conv1x1 x w
              pure <$> output "y" y
        built.builtWeights
          @?= [ Weight 64 "layers.0.q_proj" [8, 4, 1, 1] 1
              , Weight 192 "layers.0.norm" [4] 0.0625
              ]
        map (\p -> (p.placedSize, p.placedBytes)) (blobsLayout built.builtBlobs)
          @?= [(64, Nothing), (8, Nothing)]
    ]
