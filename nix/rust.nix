# SPDX-FileCopyrightText: 2026 Mika Tammi
# SPDX-License-Identifier: BSD-3-Clause
#
# The Rust toolchain from rust-overlay (test/ane/rust-toolchain.toml) and
# crane, for the Neural Engine conformance runner, and rustfmt for `nix
# fmt`. One buildDepsOnly produces the runner's dependencies'
# cargoArtifacts, which its build, its clippy and its tests reuse; they are
# built on aarch64-darwin only (nix/ane.nix), since the runner links Core
# ML.
{inputs, ...}: {
  perSystem = {
    lib,
    system,
    ...
  }: let
    pkgs = import inputs.nixpkgs {
      inherit system;
      overlays = [inputs.rust-overlay.overlays.default];
    };
    rustToolchain = pkgs.rust-bin.fromRustupToolchainFile ../test/ane/rust-toolchain.toml;
    craneLib = (inputs.crane.mkLib pkgs).overrideToolchain rustToolchain;
    aneArgs = {
      pname = "furykernel-ane-conformance";
      version = "0.0.0";
      strictDeps = true;
      src = lib.fileset.toSource {
        root = ../test/ane;
        fileset = lib.fileset.unions [../test/ane/Cargo.toml ../test/ane/Cargo.lock ../test/ane/src];
      };
      cargoVendorDir = craneLib.vendorCargoDeps {cargoLock = ../test/ane/Cargo.lock;};
    };
  in {
    _module.args = {
      inherit craneLib aneArgs rustToolchain;
      aneArtifacts = craneLib.buildDepsOnly aneArgs;
    };
  };
}
