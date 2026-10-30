# SPDX-FileCopyrightText: 2026 Mika Tammi
# SPDX-License-Identifier: BSD-3-Clause
#
# `nix fmt` and `checks.treefmt`. Every file type in the repository gets a
# formatter here in the same commit that adds the first file of that type.
{inputs, ...}: {
  imports = [
    inputs.treefmt-nix.flakeModule
  ];

  perSystem = {
    treefmt = {
      projectRootFile = "flake.nix";

      programs = {
        alejandra.enable = true;
        cabal-gild.enable = true;
        deadnix.enable = true;
        # Configured by fourmolu.yaml.
        fourmolu.enable = true;
        statix.enable = true;
        taplo.enable = true;
        mdformat = {
          enable = true;
          # GitHub-flavoured Markdown: plain mdformat would mangle tables.
          plugins = ps: [ps.mdformat-gfm];
        };
        yamlfmt.enable = true;
      };

      # Files without a formatter, and cabal2nix's output, which
      # checks.cabal2nix-in-sync compares with a fresh run byte for byte.
      settings.global.excludes = [
        ".gitignore"
        "LICENSES/*"
        "flake.lock"
        "furykernel-ane.nix"
        # Third-party files, kept as they were published.
        "vendor/*"
        # The golden tests' bytes and text, which the tests write and compare
        # as they are.
        "test/golden/*"
      ];
    };
  };
}
