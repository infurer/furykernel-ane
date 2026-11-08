# SPDX-FileCopyrightText: 2026 Mika Tammi
# SPDX-License-Identifier: BSD-3-Clause
#
# `nix develop`: GHC with the library's dependencies, cabal, and the tools
# the per-commit gate uses.
{
  perSystem = {
    config,
    haskellPackages,
    pkgs,
    ...
  }: {
    devShells.default = haskellPackages.shellFor {
      packages = _: [config.packages.furykernel-ane];
      nativeBuildInputs = [
        config.treefmt.build.wrapper
        pkgs.cabal-install
        pkgs.cabal2nix
        pkgs.haskellPackages.ghcid
        pkgs.hlint
        # protoc, for decoding the golden program's bytes (CLAUDE.md).
        pkgs.protobuf
        pkgs.reuse
      ];
    };
  };
}
