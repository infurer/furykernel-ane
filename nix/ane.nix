# SPDX-FileCopyrightText: 2026 Mika Tammi
# SPDX-License-Identifier: BSD-3-Clause
#
# `nix run .#ane-conformance`: the Neural Engine conformance run, outside
# CI, on an Apple silicon Mac. test/ane/Cases.hs writes the corpus
# (ane-corpus): the tests' stages at sizes Core ML places on the Neural
# Engine, a layer of Gemma 4's larger image tower, and edge cases, each
# lowered by `lowerStage`, with what the host derives for it and the
# outputs of the core's evaluator and of the interpreter, in f64. The
# runner (ane-runner, test/ane) compiles each with Core ML, asserts where
# its ops run, predicts it twice and holds its outputs to both by the bound
# CLAUDE.md states. The corpus is a check on every system too,
# `checks.ane-corpus`: its generator, compiled under -Wall -Werror, needs
# only GHC. The check leaves out the cases whose references take many
# minutes or many gigabytes (ane-corpus-large: the drafting step, the
# Gemma 4 layer, and the mirrors, over which the core's evaluator holds
# 15 to 40 GiB), which only the run writes: the check's need 4 GiB.
{
  perSystem = {
    aneArgs,
    aneArtifacts,
    config,
    craneLib,
    haskellPackages,
    lib,
    pkgs,
    system,
    ...
  }:
    lib.mkMerge [
      (let
        ghc = haskellPackages.ghcWithPackages (_: [config.packages.furykernel-ane]);
        # The generator and the tests' stages, which it shares.
        src = lib.fileset.toSource {
          root = ../test;
          fileset = lib.fileset.unions [../test/ane/Cases.hs ../test/Test/FuryKernel/Ane/Stages.hs];
        };
        # Its cases, `--small` or `--large`; a large nursery spares it most
        # of its collections.
        corpus = which:
          pkgs.runCommand "furykernel-ane-corpus${lib.optionalString (which == "large") "-large"}" {
            nativeBuildInputs = [ghc];
          } ''
            ghc -O -Wall -Werror -XGHC2024 -XOverloadedRecordDot -XOverloadedStrings -XStrictData \
              -rtsopts "-with-rtsopts=-A256m" \
              -i${src} -outputdir "$TMPDIR/build" -o "$TMPDIR/cases" ${src}/ane/Cases.hs
            "$TMPDIR/cases" --${which} $out
          '';
      in {
        packages.ane-corpus = corpus "small";
        packages.ane-corpus-large = corpus "large";
        checks.ane-corpus = config.packages.ane-corpus;
      })
      (lib.mkIf (system == "aarch64-darwin") {
        packages.ane-runner = craneLib.buildPackage (aneArgs
          // {
            cargoArtifacts = aneArtifacts;
            doCheck = false;
          });
        checks.ane-runner-clippy = craneLib.cargoClippy (aneArgs
          // {
            cargoArtifacts = aneArtifacts;
            cargoClippyExtraArgs = "--all-targets -- --deny warnings";
          });
        # The runner's comparison, which needs no Neural Engine: its unit tests.
        checks.ane-runner-tests = craneLib.cargoTest (aneArgs
          // {
            cargoArtifacts = aneArtifacts;
          });
        apps.ane-conformance = {
          type = "app";
          program = lib.getExe (pkgs.writeShellApplication {
            name = "ane-conformance";
            text = let
              # Both of the corpus's parts, as one directory of cases.
              whole = pkgs.symlinkJoin {
                name = "furykernel-ane-corpus-whole";
                paths = [config.packages.ane-corpus config.packages.ane-corpus-large];
              };
            in ''
              echo "chip: $(/usr/sbin/sysctl -n machdep.cpu.brand_string)"
              # The corpus this flake writes, or the one given.
              ${config.packages.ane-runner}/bin/furykernel-ane-conformance "''${1:-${whole}}"
            '';
          });
          meta.description = "Run every case of the Neural Engine conformance corpus on this Mac";
        };
      })
    ];
}
