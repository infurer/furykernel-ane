# SPDX-FileCopyrightText: 2026 Mika Tammi
# SPDX-License-Identifier: BSD-3-Clause
#
# The target library, built with GHC 9.12 (nixos-26.05's haskell.packages.ghc912,
# ahead of nixpkgs' default 9.10, which is moving to 9.12). Every module
# takes its Haskell package set from the `haskellPackages` argument set
# here. `../furykernel-ane.nix` is cabal2nix's output, committed rather
# than produced by callCabal2nix at evaluation time: that would be
# import-from-derivation, and evaluating the flake would have to build
# something first. `checks.cabal2nix-in-sync` keeps it in step with the
# .cabal file.
{
  perSystem = {
    lib,
    pkgs,
    ...
  }: let
    hlib = pkgs.haskell.lib.compose;
    haskellPackages = pkgs.haskell.packages.ghc912;

    # Only the files the build reads, so that result links or editor
    # droppings don't change its inputs.
    src = lib.fileset.toSource {
      root = ../.;
      fileset = lib.fileset.unions [
        ../LICENSES
        ../README.md
        ../furykernel-ane.cabal
        ../src
        ../test
      ];
    };

    # -Werror belongs to the gate, not to day-to-day work: a module in
    # progress with an unused binding should still load in ghci.
    furykernel-ane = lib.pipe (haskellPackages.callPackage ../furykernel-ane.nix {}) [
      (hlib.overrideSrc {inherit src;})
      (hlib.appendConfigureFlags ["--ghc-option=-Werror"])
    ];
  in {
    _module.args.haskellPackages = haskellPackages;

    packages = {
      inherit furykernel-ane;
      default = furykernel-ane;
    };
  };
}
