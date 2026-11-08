# SPDX-FileCopyrightText: 2026 Mika Tammi
# SPDX-License-Identifier: BSD-3-Clause
{
  description = "furykernel-ane: the Apple Neural Engine target of FuryKernel";

  inputs = {
    # The Neural Engine conformance runner's Rust (nix/rust.nix), as the core's own.
    crane.follows = "furykernel/crane";
    flake-parts = {
      url = "github:hercules-ci/flake-parts";
      inputs.nixpkgs-lib.follows = "nixpkgs";
    };
    furykernel = {
      url = "github:infurer/furykernel";
      inputs = {
        flake-parts.follows = "flake-parts";
        nixpkgs.follows = "nixpkgs";
        treefmt-nix.follows = "treefmt-nix";
      };
    };
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-26.05";
    rust-overlay.follows = "furykernel/rust-overlay";
    treefmt-nix = {
      url = "github:numtide/treefmt-nix";
      inputs.nixpkgs.follows = "nixpkgs";
    };
  };

  outputs = inputs @ {flake-parts, ...}:
    flake-parts.lib.mkFlake {inherit inputs;} {
      imports = [./nix];
      # The target emits Core ML programs and never links Core ML, so it
      # builds wherever GHC does: on the Macs where it is developed and on the
      # Linux machines of CI.
      systems = [
        "x86_64-linux"
        "aarch64-linux"
        "aarch64-darwin"
      ];
    };
}
