# SPDX-FileCopyrightText: 2026 Mika Tammi
# SPDX-License-Identifier: BSD-3-Clause
#
# Releases to Hackage. `nix run .#release -- VERSION` starts one: a signed
# tag, on a release commit with a merge request when the version changes
# (release.sh). Master's pipeline finishes it with `nix run
# .#publish-releases`: a pre-release goes to Hackage as the candidate of its
# version, the version itself to the index, each with its documentation, and
# the tags go to GitHub (publish-releases.sh). Both scripts start with
# hackage.sh. These files are the same in every Haskell repository of the
# project; they take the package from packages.default.
{
  perSystem = {
    config,
    pkgs,
    ...
  }: let
    hlib = pkgs.haskell.lib.compose;
    package = config.packages.default;
    pkgid = "${package.pname}-${package.version}";

    # The source tarball, as `cabal sdist` packs the package's sources once
    # `cabal check` has nothing to say about them.
    sdist =
      pkgs.runCommand "${pkgid}.tar.gz" {
        nativeBuildInputs = [pkgs.cabal-install];
        inherit (package) src;
      } ''
        cp -r "$src" source
        chmod -R u+w source
        cd source
        export HOME=$TMPDIR
        cabal check > "$TMPDIR/check" 2>&1 || true
        if ! grep -Fqx "No errors or warnings could be found in the package." "$TMPDIR/check"; then
          cat "$TMPDIR/check" >&2
          exit 1
        fi
        cabal sdist --output-directory "$TMPDIR/sdist"
        mv "$TMPDIR/sdist/${pkgid}.tar.gz" "$out"
      '';

    # The package built and tested from that tarball, as Hackage's users
    # build it: a file the build or the tests need that the tarball lacks
    # fails here.
    fromSdist = hlib.doCheck (hlib.overrideSrc {src = sdist;} package);

    # Its documentation as Hackage shows it (haddock --for-hackage: links to
    # the dependencies' pages on Hackage, the sources, quick jump), with the
    # Hoogle listing NAME.txt, which Hackage's search reads and the PVP check
    # of the next release compares.
    docs = hlib.documentationTarball fromSdist;

    # What a release uploads, PKGID.tar.gz and PKGID-docs.tar.gz. Building
    # it builds and tests the package from the tarball too.
    hackage = pkgs.runCommand "${pkgid}-hackage" {inherit fromSdist;} ''
      mkdir "$out"
      ln -s ${sdist} "$out/${pkgid}.tar.gz"
      ln -s ${docs}/${pkgid}-docs.tar.gz "$out/${pkgid}-docs.tar.gz"
    '';

    # Signing goes through git, with the user's own gpg (gpg.program).
    release = pkgs.writeShellApplication {
      name = "release";
      runtimeInputs = [
        pkgs.coreutils
        pkgs.curl
        pkgs.findutils
        pkgs.gawk
        pkgs.git
        pkgs.glab
        pkgs.gnused
        pkgs.gnutar
        pkgs.gzip
        pkgs.jq
      ];
      text = builtins.readFile ./hackage.sh + builtins.readFile ./release.sh;
    };
    # The tarballs come from the caller's nix (packages.hackage), the runner's
    # in CI.
    publish-releases = pkgs.writeShellApplication {
      name = "publish-releases";
      runtimeInputs = [
        pkgs.coreutils
        pkgs.curl
        pkgs.findutils
        pkgs.gawk
        pkgs.git
        pkgs.gnupg
        pkgs.gnused
        pkgs.gnutar
        pkgs.gzip
        pkgs.jq
        pkgs.openssh
      ];
      text = builtins.readFile ./hackage.sh + builtins.readFile ./publish-releases.sh;
    };
  in {
    packages.hackage = hackage;

    apps = {
      release = {
        type = "app";
        program = pkgs.lib.getExe release;
        meta.description = "Start a release: a signed tag, on a release commit with its merge request when the version changes";
      };
      publish-releases = {
        type = "app";
        program = pkgs.lib.getExe publish-releases;
        meta.description = "Release the release tags merged into master: candidates and versions to Hackage, tags to GitHub (--check: all but the upload)";
      };
    };

    checks.release = pkgs.callPackage ./release-test.nix {
      inherit release publish-releases;
    };
  };
}
