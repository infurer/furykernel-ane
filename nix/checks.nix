# SPDX-FileCopyrightText: 2026 Mika Tammi
# SPDX-License-Identifier: BSD-3-Clause
#
# Repository-wide checks beyond treefmt's own.
{self, ...}: {
  perSystem = {pkgs, ...}: {
    checks = {
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
    };
  };
}
