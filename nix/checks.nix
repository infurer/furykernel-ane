# SPDX-FileCopyrightText: 2026 Mika Tammi
# SPDX-License-Identifier: BSD-3-Clause
#
# Repository-wide checks beyond treefmt's own.
{self, ...}: {
  perSystem = {
    config,
    pkgs,
    ...
  }: {
    checks = {
      # The library and its tests, under -Werror.
      build = pkgs.haskell.lib.compose.doCheck config.packages.furykernel-ane;

      # The tiny program's golden bytes, which the tests hold the encoder to,
      # decoded by protoc against coremltools' own definitions: they must say
      # what the golden text says.
      protoc-decode =
        pkgs.runCommand "protoc-decode" {
          nativeBuildInputs = [pkgs.protobuf pkgs.diffutils];
        } ''
          protoc --decode=CoreML.Specification.Model \
            --proto_path=${self}/vendor/coremltools/format \
            ${self}/vendor/coremltools/format/Model.proto \
            < ${self}/test/golden/tiny.mlmodel > decoded.txt
          diff -u ${self}/test/golden/tiny.txt decoded.txt
          touch $out
        '';

      # The hints `nix fmt` cannot apply on its own.
      hlint =
        pkgs.runCommand "hlint" {
          nativeBuildInputs = [pkgs.hlint];
        } ''
          hlint --hint=${self}/.hlint.yaml ${self}/src ${self}/test
          touch $out
        '';

      # furykernel-ane.nix is cabal2nix's output, except for its `src`, which
      # cabal2nix sets to whatever path it was given.
      cabal2nix-in-sync =
        pkgs.runCommand "cabal2nix-in-sync" {
          nativeBuildInputs = [pkgs.cabal2nix pkgs.diffutils];
        } ''
          cabal2nix ${self} | sed 's|^  src = .*;$|  src = ./.;|' > want.nix
          if ! diff -u ${self}/furykernel-ane.nix want.nix; then
            echo "furykernel-ane.nix is out of date with furykernel-ane.cabal. Regenerate it:" >&2
            echo "  cabal2nix . | sed 's|^  src = .*;\$|  src = ./.;|' > furykernel-ane.nix" >&2
            exit 1
          fi
          touch $out
        '';

      # Every file carries SPDX copyright and license information.
      reuse =
        pkgs.runCommand "reuse-lint" {
          nativeBuildInputs = [pkgs.reuse];
        } ''
          cp -r ${self}/. src
          cd src
          reuse lint
          touch $out
        '';

      # `nix develop`'s shell, the gate's tools in it.
      devshell = config.devShells.default;
    };
  };
}
