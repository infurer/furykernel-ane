# SPDX-FileCopyrightText: 2026 Mika Tammi
# SPDX-License-Identifier: BSD-3-Clause
#
# `nix run .#check-pins`, CI's pins job: every input of ours locked at a
# commit on GitHub master. It needs the network (GitHub's masters), so it
# is an app rather than a check.
{
  perSystem = {pkgs, ...}: {
    apps.check-pins = {
      type = "app";
      program = pkgs.lib.getExe (pkgs.writeShellApplication {
        name = "check-pins";
        runtimeInputs = [pkgs.git pkgs.jq];
        text = ''
          if [ ! -f flake.lock ]; then
            echo "no flake.lock: nothing is pinned" >&2
            exit 1
          fi
          status=0
          ours=0
          # The root's own inputs: each is ours unless it comes from someone
          # else's GitHub, so that one locked as path:, git+file: or a GitLab
          # URL is looked at too. GitHub's owners are case-insensitive.
          for node in $(jq -r '.nodes.root.inputs[] | strings' flake.lock); do
            original=$(jq -c --arg node "$node" '.nodes[$node].original' flake.lock)
            if [ "$(jq -r '.type' <<<"$original")" = github ] && [ "$(jq -r '.owner // "" | ascii_downcase' <<<"$original")" != infurer ]; then
              continue
            fi
            ours=$((ours + 1))
            locked=$(jq -c --arg node "$node" '.nodes[$node].locked' flake.lock)
            type=$(jq -r '.type' <<<"$locked")
            owner=$(jq -r '.owner // "" | ascii_downcase' <<<"$locked")
            repo=$(jq -r '.repo // ""' <<<"$locked")
            rev=$(jq -r '.rev // ""' <<<"$locked")
            if [ "$type" != github ] || [ "$owner" != infurer ]; then
              echo "$node: locked as $type, not as a github:infurer revision" >&2
              status=1
              continue
            fi
            # A host of GitHub Enterprise's, in either, is not github.com.
            host=$(jq -r '.host // "github.com"' <<<"$original")
            locked_host=$(jq -r '.host // "github.com"' <<<"$locked")
            if [ "$host" != github.com ] || [ "$locked_host" != github.com ]; then
              echo "$node: on $host, locked on $locked_host, not on github.com" >&2
              status=1
              continue
            fi
            scratch=$(mktemp -d)
            git -C "$scratch" init --quiet
            if ! git -C "$scratch" fetch --quiet --filter=blob:none "https://github.com/infurer/$repo" master; then
              echo "$node: no master to fetch from github.com/infurer/$repo" >&2
              status=1
            elif ! git -C "$scratch" merge-base --is-ancestor "$rev" FETCH_HEAD 2>/dev/null; then
              echo "$node: $rev is not on github.com/infurer/$repo master" >&2
              status=1
            else
              echo "$node: $rev, on github.com/infurer/$repo master"
            fi
            rm -rf "$scratch"
          done
          if [ "$ours" = 0 ]; then
            echo "no input of ours in flake.lock" >&2
            status=1
          fi
          exit "$status"
        '';
      });
      meta.description = "Check that the inputs of ours are on GitHub master";
    };
  };
}
