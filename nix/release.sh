# SPDX-FileCopyrightText: 2026 Mika Tammi
# SPDX-License-Identifier: BSD-3-Clause
#
# release [--no-push] [--yes] [--skip-pvp-check] VERSION
# release --abort VERSION
#
# Starts a release of the repository's package at VERSION, BASE or
# BASE-STAGE.N (hackage.sh): a signed tag vVERSION, which master's pipeline
# releases (publish-releases), a pre-release as the candidate of BASE on
# Hackage and BASE itself on Hackage's index.
#
# - A pre-release of the version master's package has tags master's head.
# - A pre-release of a newer BASE tags a release commit, which changes the
#   package's version to BASE and nothing else. Commit and tag go to GitLab
#   with a merge request that merges when its pipeline passes, and master's
#   pipeline then releases the tag.
# - BASE itself promotes BASE's newest pre-release, once Hackage's candidate
#   holds that pre-release's files: its tag goes on that pre-release's
#   commit, so Hackage's index gets what the candidate showed. A BASE that
#   had no pre-release is refused: every release is a candidate first.
# Tags get no pipeline of their own, so for a tag on a commit master has
# already, the script starts a master pipeline.
#
# Before it signs, a pre-release checks its BASE against the last version
# on Hackage's index by the PVP, from their Hoogle listings: a declaration
# gone or changed needs a new major version (A.B), one added a new minor
# version (C). The script refuses a smaller step. A function whose
# behaviour changed needs a new major version too, which no listing shows,
# so that stays the releaser's call. --skip-pvp-check shows the change but
# lets the release through, for listings that differ while the API doesn't,
# as across Haddock versions.
#
# The release commit skips the per-commit gate: it changes only the version
# in what master has passed, so the script checks that edit instead. The
# script works in a worktree and directories of its own, so the checkout's
# state doesn't matter.
#
# --abort VERSION closes the release's merge request and deletes its branch
# and tag, for a release that won't be merged. --no-push and --yes are for
# the tests: everything up to the push, without asking.

prog=release

usage() {
  echo "usage: release [--no-push] [--yes] [--skip-pvp-check] VERSION" >&2
  echo "       release --abort VERSION" >&2
}

push=1
ask=1
abort=0
pvp=1
version=
while [ "$#" -gt 0 ]; do
  case "$1" in
  --no-push)
    push=0
    shift
    ;;
  --yes)
    ask=0
    shift
    ;;
  --abort)
    abort=1
    shift
    ;;
  --skip-pvp-check)
    pvp=0
    shift
    ;;
  -h | --help)
    usage
    exit 0
    ;;
  -*)
    usage
    exit 2
    ;;
  *)
    if [ -n "$version" ]; then
      usage
      exit 2
    fi
    version=$1
    shift
    ;;
  esac
done
if [ -z "$version" ]; then
  usage
  exit 2
fi

if ! parsed=$(parse_version "$version"); then
  echo "release: $version is not a version such as 1.2.3.4 or 1.2.3.4-alpha.1 (alpha, beta, rc)" >&2
  exit 2
fi
read -r base stage _ <<<"$parsed"
tag=v$version
branch=release/$tag

root=$(git rev-parse --show-toplevel)
cd "$root"
origin=$(git remote get-url origin)
repo=$(basename "$origin" .git)

# The GitLab host and project, from origin's URL: scp-like
# (git@host:group/name.git), ssh:// or https://.
gitlab() {
  local rest
  case "$origin" in
  *://*)
    rest=${origin#*://}
    rest=${rest#*@}
    host=${rest%%/*}
    host=${host%%:*}
    project=${rest#*/}
    ;;
  *:*)
    rest=${origin#*@}
    host=${rest%%:*}
    project=${rest#*:}
    ;;
  *)
    echo "release: origin ($origin) is not a GitLab repository's URL" >&2
    exit 1
    ;;
  esac
  project=${project%.git}
  api=projects/${project//\//%2F}
}

if [ "$abort" = 1 ]; then
  gitlab
  git fetch --quiet origin master
  released=$(git ls-remote origin "refs/tags/$tag^{}" | cut -f1)
  if [ -n "$released" ] && git merge-base --is-ancestor "$released" origin/master 2>/dev/null; then
    echo "release: $tag is merged into master, which releases it: nothing to abort" >&2
    exit 1
  fi
  for iid in $(glab api --hostname "$host" "$api/merge_requests?state=opened&source_branch=${branch//\//%2F}" | jq -r '.[].iid'); do
    glab api --hostname "$host" --method PUT "$api/merge_requests/$iid" -f state_event=close >/dev/null
    echo "release: closed merge request !$iid"
  done
  for ref in "refs/heads/$branch" "refs/tags/$tag"; do
    if [ -n "$(git ls-remote origin "$ref")" ]; then
      git push --quiet origin --delete "$ref"
      echo "release: deleted $ref on GitLab"
    fi
  done
  git tag -d "$tag" >/dev/null 2>&1 || true
  git branch -D "$branch" >/dev/null 2>&1 || true
  exit 0
fi

if [ "$push" = 1 ]; then
  gitlab
fi
git fetch --quiet --tags origin
if git rev-parse -q --verify "refs/tags/$tag" >/dev/null; then
  echo "release: $tag exists already" >&2
  exit 1
fi
if git rev-parse -q --verify "refs/heads/$branch" >/dev/null; then
  echo "release: the branch $branch exists already (release --abort $version removes it)" >&2
  exit 1
fi

work=$(mktemp -d)
finished=0
cleanup() {
  git -C "$root" worktree remove --force "$work/tree" 2>/dev/null || true
  rm -rf "$work"
  if [ "$finished" = 0 ]; then
    git -C "$root" tag -d "$tag" >/dev/null 2>&1 || true
  fi
  if [ "$finished" = 0 ] || [ "$push" = 1 ]; then
    git -C "$root" branch -D "$branch" >/dev/null 2>&1 || true
  fi
}
trap cleanup EXIT

# Master's package, and what Hackage has of it.
mkdir "$work/master"
git archive origin/master | tar -x -C "$work/master"
cabal=$(cabal_file "$work/master")
name=$(cabal_field "$cabal" name)
current=$(cabal_field "$cabal" version)
if version_lt "$base" "$current"; then
  echo "release: master's $name is at $current, newer than $base" >&2
  exit 1
fi
published=$(index_versions "$name")
if grep -Fqx "$base" <<<"$published"; then
  echo "release: $name $base is on Hackage already" >&2
  exit 1
fi

# The pre-releases of BASE that have tags, and the newest of them.
pre_releases=$(git tag --list "v$base-*" | sed 's/^v//' | sort_versions)
newest=$(tail -n 1 <<<"$pre_releases")
if [ "$stage" != release ]; then
  if [ -n "$newest" ] && ! version_lt "$newest" "$version"; then
    echo "release: v$newest is newer than $version" >&2
    exit 1
  fi
elif [ -z "$newest" ]; then
  echo "release: $base has had no pre-release: release $base-alpha.1 first, and look at its candidate on Hackage" >&2
  exit 1
elif ! candidate_is "$name-$base" "$(git rev-parse "v$newest^{commit}")"; then
  echo "release: Hackage's candidate of $name $base isn't v$newest yet: has master's pipeline released it?" >&2
  exit 1
fi

# What the release signs: a tag on a commit that master has, or a release
# commit with its tag.
commit=
if [ "$stage" = release ]; then
  commit=$(git rev-parse "v$newest^{commit}")
  echo "release: $tag promotes v$newest, on $(git log -1 --format='%h %s' "$commit")"
elif [ "$base" = "$current" ]; then
  commit=$(git rev-parse origin/master)
  tree=$work/master
  echo "release: $tag is master's head, $(git log -1 --format='%h %s' "$commit")"
else
  git worktree add --quiet -b "$branch" "$work/tree" origin/master
  tree=$work/tree
  cd "$tree"
  cabal=$(basename "$cabal")

  # The edit: the version in the .cabal file and in cabal2nix's output.
  awk -v new="$base" '/^version:/ { $0 = "version: " new } { print }' "$cabal" >"$work/edited"
  cat "$work/edited" >"$cabal"
  awk -v old="$current" -v new="$base" '$0 == "  version = \"" old "\";" { $0 = "  version = \"" new "\";" } { print }' "$name.nix" >"$work/edited"
  cat "$work/edited" >"$name.nix"

  # The check of the edit, which stands in for the gate: both files at the
  # new version, and no other file or line changed.
  fail() {
    echo "release: $*" >&2
    git --no-pager diff >&2
    exit 1
  }
  [ "$(cabal_field "$cabal" version)" = "$base" ] || fail "$cabal is not at $base"
  grep -Fqx "  version = \"$base\";" "$name.nix" || fail "$name.nix is not at $base"
  changed=$(git diff --name-only | sort | tr '\n' ' ')
  [ "$changed" = "$(printf '%s\n' "$cabal" "$name.nix" | sort | tr '\n' ' ')" ] ||
    fail "files changed besides the version's: $changed"
  other=$(git diff -U0 | grep -E '^[-+]' | grep -Ev '^(\+\+\+|---) ' | grep -Ev '^[-+](version: |  version = ")' || true)
  [ -z "$other" ] || fail "lines changed besides the version: $other"
fi

# The PVP check of a pre-release, against the last version on Hackage's
# index before BASE: that version's Hoogle listing on Hackage, and the one
# a release of the tree would upload.
previous=$({
  printf '%s\n' "$published"
  echo "$base"
} | sort_versions | awk -v base="$base" '$0 == base { print last; exit } { last = $0 }')
if [ "$stage" != release ] && [ -n "$previous" ]; then
  tarballs "$tree"
  status=
  if ! status=$(curl -sS -L -o "$work/previous.txt" -w '%{http_code}' "$(fresh_url "package/$name-$previous/docs/$name.txt")") || [ "$status" != 200 ]; then
    echo "release: no Hoogle listing of $name $previous on Hackage (${status:-no answer})" >&2
    exit 1
  fi
  tar -xzOf "$docs" "$pkgid-docs/$name.txt" >"$work/next.txt"
  echo "release: the API from $name $previous to $base:"
  needs=$(api_change "$work/previous.txt" "$work/next.txt")
  step=$(bump "$previous" "$base")
  if [ "$(bump_rank "$step")" -lt "$(bump_rank "$needs")" ]; then
    if [ "$pvp" = 1 ]; then
      echo "release: by the PVP, that change needs a new $needs version, and $base is a $step step from $previous" >&2
      exit 1
    fi
    echo "release: by the PVP, that change needs a new $needs version; --skip-pvp-check lets $base, a $step step, through" >&2
  fi
fi

if [ -z "$commit" ]; then
  git --no-pager diff --stat
  git --no-pager diff
  question="Sign the release commit and $tag, and push them for the merge request?"
else
  question="Sign $tag and push it, and start master's pipeline to release it?"
fi
if [ "$ask" = 1 ]; then
  read -r -p "$question [y/N] " answer
  case "$answer" in
  y | Y | yes) ;;
  *)
    echo "release: stopped, nothing pushed" >&2
    exit 1
    ;;
  esac
fi

if [ -n "$commit" ]; then
  git tag --sign --message "$repo $version" "$tag" "$commit"
  if [ "$push" = 0 ]; then
    finished=1
    echo "release: $tag, not pushed"
    exit 0
  fi
  git push --quiet origin "refs/tags/$tag"
  finished=1
  pipeline=$(glab api --hostname "$host" --method POST "$api/pipeline" -f ref=master)
  echo "release: $tag is on GitLab; master's pipeline $(jq -r .web_url <<<"$pipeline") releases it"
  exit 0
fi

git commit --quiet --all --signoff --gpg-sign --message "$repo: $base"
git tag --sign --message "$repo $version" "$tag"
commit=$(git rev-parse HEAD)

if [ "$push" = 0 ]; then
  finished=1
  echo "release: $tag on $branch, not pushed"
  exit 0
fi

git push --quiet --atomic origin "refs/heads/$branch" "refs/tags/$tag"
finished=1
mr=$(glab api --hostname "$host" --method POST "$api/merge_requests" \
  -f source_branch="$branch" -f target_branch=master -f title="$repo: $base" \
  -F remove_source_branch=true \
  -f description="The release commit of $tag, $name's version becoming $base. Once merged, master's pipeline uploads it to Hackage as the candidate of $base and pushes $tag to GitHub.")
iid=$(jq -r .iid <<<"$mr")
url=$(jq -r .web_url <<<"$mr")

# Auto-merge waits for the merge request's pipeline, which must exist first:
# without one, there would be nothing to wait for.
pipeline=
for _ in $(seq 60); do
  pipeline=$(glab api --hostname "$host" "$api/merge_requests/$iid" |
    jq -r --arg sha "$commit" 'select(.head_pipeline.sha == $sha) | .head_pipeline.id')
  [ -z "$pipeline" ] || break
  sleep 5
done
if [ -z "$pipeline" ]; then
  echo "release: $url has no pipeline after 5 minutes: set it to auto-merge in GitLab" >&2
  exit 1
fi
glab api --hostname "$host" --method PUT "$api/merge_requests/$iid/merge" \
  -F auto_merge=true -f sha="$commit" -F should_remove_source_branch=true >/dev/null
echo "release: $url merges when its pipeline passes; master's pipeline then releases $tag"
