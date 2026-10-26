# SPDX-FileCopyrightText: 2026 Mika Tammi
# SPDX-License-Identifier: BSD-3-Clause
#
# `nix develop`: the tools the per-commit gate uses.
{
  perSystem = {
    config,
    pkgs,
    ...
  }: {
    devShells.default = pkgs.mkShell {
      packages = [
        config.treefmt.build.wrapper
        pkgs.reuse
      ];
    };
  };
}
